"""
Fundamentals Cache Service for stock fundamental data caching.

Provides intelligent caching of stock fundamental data (PE ratios, market cap,
institutional ownership, margins, etc.) with weekly refresh to minimize API calls.
Uses Redis for hot cache and database for persistence.
"""
import logging
import math
import inspect
from typing import Any, Optional, Dict, Callable
from datetime import UTC, datetime, date
from sqlalchemy.orm import Session

from app.domain.providers.data_plan import (
    DATASET_FUNDAMENTALS,
    provider_data_plan_registry,
)

try:
    import redis  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in desktop packaging
    redis = Any  # type: ignore

from ..database import SessionLocal
from ..models.stock import StockFundamental
from .universe_classification import backfill_universe_classification
from ..config import settings
from .errors import CacheRefreshError
from .fundamentals_completeness import (
    compute_completeness_score,
    derive_field_provenance,
)
from .field_capability_registry import field_capability_registry
from .fx_service import FXQuote, FXService, get_fx_service
from .institutional_ownership_service import InstitutionalOwnershipService
from .redis_pool import get_redis_client, is_redis_enabled
from .cache.market_cache_policy import MarketAwareCachePolicy, market_cache_policy
from .cache.redis_codec import decode_dict, encode_dict
from .universe_row_facts import (
    UniverseRowFactsResolver,
    normalize_universe_text,
)

logger = logging.getLogger(__name__)

def _assign_if_present(record: Any, attr: str, data: Dict[str, Any], key: str) -> None:
    """Write ``data[key]`` only when non-None; prevents partial refresh
    payloads from wiping previously stored market-state fields."""
    if key in data and data[key] is not None:
        setattr(record, attr, data[key])


_RECOMMENDATION_SCORE_BY_KEY: dict[str, float | None] = {
    "strong_buy": 1.0,
    "buy": 2.0,
    "hold": 3.0,
    "underperform": 4.0,
    "sell": 4.0,
    "strong_sell": 5.0,
    "none": None,
}

# StockFundamental columns carried in the fundamentals payload, in read order.
# Each is read back under its own name and copied from the same payload key on
# write, except where the sets below say otherwise.
_PAYLOAD_COLUMNS = (
    "market_cap", "shares_outstanding", "avg_volume", "relative_volume",
    "pe_ratio", "forward_pe", "peg_ratio", "price_to_book", "price_to_sales",
    "price_to_cash", "price_to_fcf", "ev_ebitda", "ev_sales", "target_price",
    "eps_current", "eps_next_y", "eps_next_5y", "eps_next_q",
    "eps_growth_quarterly", "eps_growth_annual", "eps_growth_yy",
    "revenue_current", "revenue_growth", "sales_past_5y", "sales_growth_yy", "sales_growth_qq",
    "recent_quarter_date", "previous_quarter_date", "growth_reporting_cadence",
    "growth_metric_basis", "growth_comparable_period_date", "growth_reference_gap_days",
    "next_earnings_date", "event_calendar_as_of_date", "recent_earnings",
    "profit_margin", "operating_margin", "gross_margin", "roe", "roa", "roic",
    "current_ratio", "quick_ratio", "debt_to_equity", "lt_debt_to_equity",
    "insider_ownership", "insider_transactions", "institutional_ownership",
    "institutional_transactions", "institutional_change",
    "short_float", "short_ratio", "short_interest",
    "beta", "rsi_14", "atr_14", "sma_20", "sma_50", "sma_200",
    "volatility_week", "volatility_month",
    "perf_week", "perf_month", "perf_quarter", "perf_half_year", "perf_year", "perf_ytd",
    "dividend_ttm", "dividend_yield", "payout_ratio",
    "week_52_high", "week_52_high_distance", "week_52_low", "week_52_low_distance",
    "sector", "industry", "country", "employees", "ipo_date",
    "description_yfinance", "description_finviz", "recommendation",
    "eps_5yr_cagr", "eps_q1_yoy", "eps_q2_yoy", "eps_raw_score", "eps_rating", "eps_years_available",
    "data_source", "finviz_snapshot_revision", "finviz_snapshot_at",
    "yahoo_profile_refreshed_at", "yahoo_statements_refreshed_at", "technicals_refreshed_at",
    "field_completeness_score", "field_provenance",
    "market_cap_usd", "adv_usd", "fx_metadata",
)
_WRITE_KEY_BY_COLUMN = {
    "eps_growth_quarterly": "eps_growth_qq",
    "eps_growth_annual": "eps_growth_yy",
}
# An update writes these only for a non-None payload value, so a partial
# refresh keeps the stored market state.
_KEEP_ON_NONE_COLUMNS = frozenset({
    "market_cap", "shares_outstanding", "avg_volume", "market_cap_usd", "adv_usd", "fx_metadata",
})
# Payloads carry these as ISO strings.
_TIMESTAMP_COLUMNS = frozenset({
    "finviz_snapshot_at", "yahoo_profile_refreshed_at", "yahoo_statements_refreshed_at",
    "technicals_refreshed_at",
})
# _store_in_database writes these itself; nothing writes institutional_change.
_UNCOPIED_COLUMNS = frozenset({
    "next_earnings_date", "event_calendar_as_of_date", "recent_earnings", "ipo_date", "data_source",
    "field_completeness_score", "field_provenance", "institutional_change",
})


def _fundamentals_from_record(record: StockFundamental) -> Dict[str, Any]:
    fundamentals = {column: getattr(record, column) for column in _PAYLOAD_COLUMNS}
    for column in _TIMESTAMP_COLUMNS:
        if fundamentals[column]:
            fundamentals[column] = fundamentals[column].isoformat()
    # Alias for CANSLIM compatibility
    fundamentals["eps_growth_qq"] = record.eps_growth_quarterly
    # Epoch format for compatibility
    fundamentals["first_trade_date"] = (
        int(datetime.combine(record.ipo_date, datetime.min.time()).timestamp())
        if record.ipo_date else None
    )
    fundamentals["description"] = (
        fundamentals["description_finviz"] or fundamentals["description_yfinance"]
    )
    return fundamentals


class FundamentalsCacheService:
    """
    Service for caching stock fundamental data with weekly updates.

    Strategy:
    - Store fundamental data in Redis with 7-day TTL for fast access
    - Store full fundamental data in StockFundamental table
    - Fetch only when missing or stale (> 7 days old)
    - Graceful fallback: Redis → Database → Live API
    """

    # Redis keys
    REDIS_KEY_PREFIX = "fundamentals:"
    REDIS_KEY_FORMAT = "fundamentals:{symbol}"
    REDIS_KEY_FALLBACK_COUNT = "fundamentals:on_demand_fallback_count"

    # TTL settings
    CACHE_TTL_SECONDS = 604800  # 7 days (weekly refresh)
    MAX_AGE_DAYS = 7  # Max age before data is considered stale
    REQUIRED_SCAN_KEYS = (
        "eps_rating",
        "ipo_date",
        "first_trade_date",
        "sector",
        "industry",
        "eps_growth_qq",
        "sales_growth_qq",
        "eps_growth_yy",
        "sales_growth_yy",
        "market_cap",
        "avg_volume",
    )

    @staticmethod
    def _coerce_datetime(value):
        """Best-effort conversion for snapshot provenance timestamps."""
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _coerce_date(value: Any) -> Optional[date]:
        """Best-effort conversion for date-only payload fields like ipo_date."""
        if value is None or value == "":
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                return None
        return None

    def _normalize_payload_for_storage(self, data: Dict) -> Dict:
        """Normalize payload types before writing to Redis/DB."""
        normalized = dict(data)
        if "ipo_date" in normalized:
            normalized["ipo_date"] = self._coerce_date(normalized.get("ipo_date"))
        if "recommendation" in normalized:
            normalized["recommendation"] = self._normalize_recommendation(
                normalized.get("recommendation")
            )
        return normalized

    @staticmethod
    def _normalize_recommendation(value: Any) -> float | None:
        """Coerce provider recommendation strings into the numeric DB contract."""
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                return numeric
            logger.warning("Unsupported analyst recommendation value %r; storing NULL", value)
            return None

        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in _RECOMMENDATION_SCORE_BY_KEY:
            return _RECOMMENDATION_SCORE_BY_KEY[normalized]

        try:
            numeric = float(normalized)
            if math.isfinite(numeric):
                return numeric
            logger.warning("Unsupported analyst recommendation value %r; storing NULL", value)
            return None
        except ValueError:
            logger.warning("Unsupported analyst recommendation value %r; storing NULL", value)
            return None

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session_factory: Optional[Callable[[], Session]] = None,
        fx_service: Optional[FXService] = None,
        cache_policy: MarketAwareCachePolicy = market_cache_policy,
        row_facts_resolver: Optional[UniverseRowFactsResolver] = None,
    ):
        """Initialize fundamentals cache service.

        ``fx_service`` is injectable for tests; in production callers
        typically rely on the process-wide default via ``get_fx_service()``.
        """
        self._session_factory = session_factory or SessionLocal
        self._fx_service = fx_service
        self._cache_policy = cache_policy
        self._row_facts_resolver = row_facts_resolver or UniverseRowFactsResolver(
            self._session_factory
        )
        if redis_client:
            self._redis_client = redis_client
        else:
            # Use shared connection pool for efficiency
            self._redis_client = get_redis_client()
            if self._redis_client:
                logger.debug("Connected to Redis for fundamentals caching (using shared pool)")
            elif is_redis_enabled():
                logger.warning("Redis connection failed. Will use database fallback.")
            else:
                logger.info("Redis disabled for this runtime. Using database fallback.")

    def get_fundamentals(
        self,
        symbol: str,
        force_refresh: bool = False,
        market: Optional[str] = None,
        cache_only: bool = False,
    ) -> Optional[Dict]:
        """
        Get fundamental data with caching and graceful fallback.

        Args:
            symbol: Stock ticker symbol
            force_refresh: Force fetch from yfinance, bypass cache
            market: Optional market code (US/HK/JP/TW/CN). Callers that already
                hold the ``StockUniverse`` row should pass this to avoid an
                extra per-symbol DB lookup on cache misses. ``None`` falls
                back to a best-effort DB resolve in ``_fetch_and_cache``.
            cache_only: Never reach a provider: serve cached data even when
                stale or incomplete, and None on a miss.

        Returns:
            Dict with fundamental metrics or None if unavailable

        Logic:
        1. Check Redis for cached data (if not force_refresh)
        2. Check database for cached data (if Redis miss)
        3. If data is stale or missing, fetch from yfinance
        4. Update cache with fresh data
        5. Return fundamentals dict
        """
        if force_refresh and cache_only:
            raise ValueError("force_refresh and cache_only are mutually exclusive")
        if force_refresh:
            logger.info(f"Force refresh requested for {symbol}")
            return self._fetch_and_cache(symbol, market=market)

        # Try Redis first (fast path)
        if self._redis_client:
            try:
                redis_key = self._redis_data_key(symbol, market)
                # A legacy (pickled) payload decodes to None: a miss.
                fundamentals = decode_dict(self._redis_client.get(redis_key))

                if fundamentals:
                    if self._needs_db_enrichment(fundamentals):
                        db_data, last_update = self._get_from_database(symbol)
                        if db_data is not None and self._is_data_fresh(last_update):
                            fundamentals = self._merge_fundamentals(fundamentals, db_data)
                            self._store_in_redis_for_market(symbol, fundamentals, market=market)
                    if not cache_only and self._needs_on_demand_enrichment(fundamentals):
                        logger.info(
                            "Cache HIT but incomplete for %s fundamentals - fetching on-demand enrichment",
                            symbol,
                        )
                        enriched = self._fetch_and_cache(symbol, market=market)
                        if enriched is not None:
                            return enriched
                    if self._ensure_field_availability_metadata(
                        symbol, fundamentals, market, convert_fx=not cache_only
                    ):
                        self._store_in_redis_for_market(symbol, fundamentals, market=market)
                    logger.debug(f"Cache HIT for {symbol} (Redis)")
                    return fundamentals
            except (TypeError, ValueError, RuntimeError, OSError) as exc:
                logger.warning(
                    "Redis read error for %s fundamentals: %s",
                    symbol,
                    exc,
                    extra={
                        "event": "cache_refresh_failed",
                        "path": "fundamentals_cache_service.get_fundamentals",
                        "pipeline": None,
                        "run_id": None,
                        "symbol": symbol,
                        "error_code": "fundamentals_cache_redis_read_failed",
                    },
                )
            except Exception as exc:
                logger.warning(
                    "Unexpected Redis read error for %s fundamentals: %s",
                    symbol,
                    exc,
                    extra={
                        "event": "cache_refresh_failed",
                        "path": "fundamentals_cache_service.get_fundamentals",
                        "pipeline": None,
                        "run_id": None,
                        "symbol": symbol,
                        "error_code": "fundamentals_cache_redis_read_unexpected",
                    },
                    exc_info=True,
                )

        # Try database (slower but persistent)
        cached_data, last_update = self._get_from_database(symbol)

        if cached_data is not None:
            # Check if data is fresh
            fresh = self._is_data_fresh(last_update)
            if fresh or cache_only:
                if not cache_only and self._needs_on_demand_enrichment(cached_data):
                    logger.info(
                        "Cache HIT but incomplete for %s fundamentals in database - fetching on-demand enrichment",
                        symbol,
                    )
                    enriched = self._fetch_and_cache(symbol, market=market)
                    if enriched is not None:
                        return enriched
                self._ensure_field_availability_metadata(
                    symbol, cached_data, market, convert_fx=not cache_only
                )
                logger.info(f"Cache HIT for {symbol} (Database, updated: {last_update})")

                # Also store in Redis for faster next access. Never a stale
                # row: every Redis reader treats a hit as fresh.
                if fresh:
                    self._store_in_redis_for_market(symbol, cached_data, market=market)

                return cached_data
            else:
                # Data is stale - fetch fresh data
                logger.info(f"Cache HIT but STALE for {symbol} (last update: {last_update}) - fetching fresh data")
                return self._fetch_and_cache(symbol, market=market)

        if cache_only:
            return None

        # No cached data - fetch from yfinance
        logger.info(f"Cache MISS for {symbol} - fetching fresh data")
        return self._fetch_and_cache(symbol, market=market)

    def _get_from_database(self, symbol: str) -> tuple[Optional[Dict], Optional[datetime]]:
        """
        Get cached fundamental data from database.

        Returns:
            Tuple of (fundamentals_dict, last_update_datetime) or (None, None)
        """
        db = self._session_factory()

        try:
            # Query StockFundamental table
            record = db.query(StockFundamental).filter(
                StockFundamental.symbol == symbol
            ).first()

            if not record:
                logger.debug(f"No cached data for {symbol} in database")
                return None, None

            fundamentals = _fundamentals_from_record(record)
            fundamentals["field_availability"] = (
                field_capability_registry.derive_ownership_sentiment_availability(
                    fundamentals, self._resolve_market(symbol)
                )
            )

            # Get last update timestamp
            last_update = record.updated_at

            logger.debug(f"Retrieved {symbol} from database (last update: {last_update})")
            return fundamentals, last_update

        except Exception as e:
            logger.error(f"Error reading {symbol} from database: {e}", exc_info=True)
            return None, None

        finally:
            db.close()

    def _enrich_with_quality_metadata(
        self,
        symbol: str,
        data: Dict,
        market: Optional[str],
    ) -> Optional[str]:
        """Compute and attach T2 quality metadata to ``data`` in place.

        Must be called *before* writes to Redis/DB so both stores see the
        same snapshot — otherwise a warm Redis read would omit the
        derived fields for up to the cache TTL.

        Returns the resolved market string so the caller can pass it on
        to ``_store_in_database`` without a second DB lookup.
        """
        if market is None:
            market = self._resolve_market(symbol)
        data['field_completeness_score'] = compute_completeness_score(data, market)
        data['field_provenance'] = derive_field_provenance(data, market)
        data['field_availability'] = (
            field_capability_registry.derive_ownership_sentiment_availability(
                data, market
            )
        )
        data['provider_data_plan'] = provider_data_plan_registry.plan_for(
            market, DATASET_FUNDAMENTALS
        ).provenance_metadata()
        return market

    def _ensure_field_availability_metadata(
        self,
        symbol: str,
        data: Dict,
        market: Optional[str],
        currency: Optional[str] = None,
        *,
        convert_fx: bool = True,
    ) -> bool:
        """Backfill derived metadata fields for older cache rows.

        ``convert_fx=False`` leaves the USD fields alone: a missing FX rate
        would otherwise be fetched from a provider.

        Returns ``True`` when any field was backfilled.
        """
        changed = False
        row_facts = self._row_facts_resolver.resolve_for_payload(
            symbol,
            data,
            market=market,
            currency=currency,
        )
        resolved_market = row_facts.market
        if data.get("field_completeness_score") is None:
            data["field_completeness_score"] = compute_completeness_score(
                data, resolved_market
            )
            changed = True
        if not isinstance(data.get("field_provenance"), dict):
            data["field_provenance"] = derive_field_provenance(data, resolved_market)
            changed = True
        if not isinstance(data.get("field_availability"), dict):
            data["field_availability"] = (
                field_capability_registry.derive_ownership_sentiment_availability(
                    data, resolved_market
                )
            )
            changed = True
        expected_plan = provider_data_plan_registry.plan_for(
            resolved_market, DATASET_FUNDAMENTALS
        ).provenance_metadata()
        if data.get("provider_data_plan") != expected_plan:
            data["provider_data_plan"] = expected_plan
            changed = True

        previous_fx_values = (
            data.get("market_cap_usd"),
            data.get("adv_usd"),
            data.get("fx_metadata"),
        )
        missing_fx_metadata = (
            data.get("market_cap_usd") is None
            or data.get("adv_usd") is None
            or not isinstance(data.get("fx_metadata"), dict)
        )
        if missing_fx_metadata and convert_fx:
            self._enrich_with_fx_normalization(
                data,
                row_facts.currency,
                market=resolved_market,
            )
            changed = changed or (
                previous_fx_values
                != (
                    data.get("market_cap_usd"),
                    data.get("adv_usd"),
                    data.get("fx_metadata"),
                )
            )

        return changed

    def _get_fx_service(self) -> FXService:
        """Return the FX service, lazily resolving the process-wide default.

        Lazy so that constructing a cache service without Redis/DB (e.g.
        in unit tests) does not trigger FX service initialisation.
        """
        if self._fx_service is None:
            self._fx_service = get_fx_service()
        return self._fx_service

    def _enrich_with_fx_normalization(
        self,
        data: Dict,
        currency: Optional[str],
        *,
        market: Optional[str] = None,
    ) -> None:
        """Compute USD normalisation (``market_cap_usd``, ``adv_usd``) and
        attach an ``fx_metadata`` snapshot to ``data`` in place.

        Called *alongside* ``_enrich_with_quality_metadata`` before Redis/
        DB writes so derived values survive a warm-cache read. Missing
        rate or missing source amounts → NULL USD columns (honest
        "unknown" rather than a fabricated zero).
        """
        row_currency = normalize_universe_text(currency)
        market_cap = data.get("market_cap")
        shares_out = data.get("shares_outstanding")
        avg_volume = data.get("avg_volume")

        if row_currency is None:
            data["market_cap_usd"] = None
            data["adv_usd"] = None
            data["fx_metadata"] = self._missing_currency_metadata(market)
            return

        quote = self._get_fx_service().get_usd_rate(row_currency)
        if quote is None:
            # Rate unavailable — leave USD columns NULL but record the
            # intent so consumers can see we tried.
            data["market_cap_usd"] = None
            data["adv_usd"] = None
            data["fx_metadata"] = FXQuote.unavailable_metadata(row_currency)
            return

        rate = quote.rate
        data["market_cap_usd"] = (
            int(market_cap * rate) if market_cap is not None else None
        )
        # ADV-USD = avg_volume × price-per-share (local) × rate.
        # Price-per-share is derived from market_cap / shares_outstanding
        # to avoid coupling to a separate price cache.
        if (
            avg_volume is not None
            and market_cap is not None
            and shares_out
        ):
            price_local = market_cap / shares_out
            data["adv_usd"] = int(avg_volume * price_local * rate)
        else:
            data["adv_usd"] = None
        data["fx_metadata"] = quote.to_metadata()

    @staticmethod
    def _missing_currency_metadata(market: Optional[str]) -> Dict[str, Any]:
        metadata = {
            "from_currency": None,
            "to_currency": "USD",
            "rate": None,
            "as_of_date": None,
            "source": "missing_currency",
        }
        if market:
            metadata["market"] = str(market).strip().upper()
        return metadata

    def _resolve_market(self, symbol: str) -> Optional[str]:
        """Look up the stock universe market for ``symbol``.

        Returns ``None`` if the symbol is not in the universe (or the lookup
        fails) — the routing policy treats ``None`` as the legacy US default
        so this never crashes on-demand fetches.
        """
        return self._row_facts_resolver.resolve(symbol).market

    def _fetch_and_cache(
        self,
        symbol: str,
        market: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Fetch fundamental data from finvizfinance (primary) or yfinance (fallback) and cache it.

        Graceful fallback: If fetch fails, return None (caller will handle).

        ``market`` is consulted by the provider routing policy so US-only
        providers (finviz) are skipped for HK/JP/TW/CN symbols.
        When omitted, falls back to a single DB lookup.
        """
        try:
            if not settings.provider_snapshot_on_demand_fallback_enabled:
                logger.warning("On-demand provider fallback is disabled for %s", symbol)
                return None

            from ..wiring.bootstrap import get_data_source_service

            logger.info(f"Fetching fresh fundamental data for {symbol} from data sources")
            self.record_on_demand_fallback()

            row_facts = None
            if market is None:
                row_facts = self._row_facts_resolver.resolve(symbol)
                market = row_facts.market

            # Use DataSourceService which handles finviz → yfinance fallback
            fundamentals = get_data_source_service().get_fundamentals(
                symbol, market=market
            )

            if not fundamentals:
                logger.warning(f"All data sources failed for {symbol} fundamentals")
                return None

            # Remove metadata fields before caching
            data_source = fundamentals.pop('data_source', 'unknown')
            data_source_timestamp = fundamentals.pop('data_source_timestamp', None)

            logger.info(
                f"Fetched {symbol} from {data_source}: "
                f"{len([v for v in fundamentals.values() if v is not None])} fields populated"
            )

            fundamentals = self._normalize_payload_for_storage(fundamentals)

            # Enrich with completeness/provenance so both Redis and DB writes
            # include the derived metadata (avoids warm-read staleness).
            row_facts = self._row_facts_resolver.resolve_for_payload(
                symbol,
                fundamentals,
                market=market,
                fallback=row_facts,
            )
            market = self._enrich_with_quality_metadata(
                symbol,
                fundamentals,
                row_facts.market,
            )
            self._enrich_with_fx_normalization(
                fundamentals,
                row_facts.currency,
                market=market,
            )

            # Cache in Redis (7-day TTL)
            self._store_in_redis_for_market(symbol, fundamentals, market=market)

            # Persist to database (permanent storage) with data source metadata.
            self._store_in_database(
                symbol, fundamentals, data_source=data_source, market=market
            )

            # Re-add metadata for return value
            fundamentals['data_source'] = data_source
            if data_source_timestamp:
                fundamentals['data_source_timestamp'] = data_source_timestamp

            return fundamentals

        except Exception as exc:
            error = CacheRefreshError(
                f"Fundamentals refresh failed for {symbol}: {exc}",
                error_code="fundamentals_cache_refresh_failed",
            )
            logger.error(
                "%s",
                error,
                extra={
                    "event": "cache_refresh_failed",
                    "path": "fundamentals_cache_service._fetch_and_cache",
                    "pipeline": None,
                    "run_id": None,
                    "symbol": symbol,
                    "error_code": error.error_code,
                },
                exc_info=exc,
            )
            return None

    def _redis_data_key(self, symbol: str, market: Optional[str] = None) -> str:
        return self._cache_policy.key("fundamentals", symbol, market=market)

    def _store_in_redis_for_market(
        self,
        symbol: str,
        data: Dict,
        market: Optional[str] = None,
    ) -> None:
        store_signature = inspect.signature(self._store_in_redis)
        if "market" in store_signature.parameters:
            self._store_in_redis(symbol, data, market=market)
        else:
            self._store_in_redis(symbol, data)

    def _store_in_redis(self, symbol: str, data: Dict, market: Optional[str] = None) -> None:
        """
        Store fundamental data in Redis with 7-day TTL.
        """
        if not self._redis_client:
            return

        try:
            redis_key = self._redis_data_key(symbol, market)

            self._redis_client.setex(
                redis_key,
                self._cache_policy.ttl_seconds("fundamentals", market=market),
                encode_dict(data),
            )

            logger.debug(f"Cached {symbol} fundamental data in Redis (TTL: 7 days)")

        except Exception as e:
            logger.error(f"Error storing {symbol} in Redis: {e}", exc_info=True)

    def _store_in_database(
        self,
        symbol: str,
        data: Dict,
        data_source: str = 'unknown',
        market: Optional[str] = None,
    ) -> bool:
        """
        Store fundamental data in database (StockFundamental table).

        Args:
            symbol: Stock ticker symbol
            data: Fundamental data dict
            data_source: Data source ('finviz' or 'yfinance')
            market: Market code used for completeness/provenance scoring.
                When omitted, falls back to a DB lookup via _resolve_market.

        Also persists T2 quality metadata: ``field_completeness_score``
        and ``field_provenance``.

        Uses upsert to handle existing records gracefully.
        """
        db = self._session_factory()

        try:
            if market is None:
                market = self._resolve_market(symbol)

            # Prefer pre-computed values from the dict (attached by
            # ``_enrich_with_quality_metadata`` on the write path); only
            # recompute if the caller reached this method without going
            # through the standard cache-enrichment path.
            completeness_score = data.get('field_completeness_score')
            if completeness_score is None:
                completeness_score = compute_completeness_score(data, market)
            provenance = data.get('field_provenance')
            if provenance is None:
                provenance = derive_field_provenance(data, market)

            record = db.query(StockFundamental).filter(
                StockFundamental.symbol == symbol
            ).first()
            is_new = record is None
            if is_new:
                record = StockFundamental(symbol=symbol)
                db.add(record)
            else:
                record.updated_at = datetime.now(UTC)

            for column in _PAYLOAD_COLUMNS:
                if column in _UNCOPIED_COLUMNS:
                    continue
                key = _WRITE_KEY_BY_COLUMN.get(column, column)
                if column in _KEEP_ON_NONE_COLUMNS and not is_new:
                    _assign_if_present(record, column, data, key)
                elif column in _TIMESTAMP_COLUMNS:
                    setattr(record, column, self._coerce_datetime(data.get(key)))
                else:
                    setattr(record, column, data.get(key))
            # A refresh without a calendar observation keeps the stored one.
            if is_new or "event_calendar_as_of_date" in data:
                record.event_calendar_as_of_date = self._coerce_date(
                    data.get("event_calendar_as_of_date")
                )
                record.next_earnings_date = self._coerce_date(data.get("next_earnings_date"))
                # Reported quarters never un-happen; a response without any keeps them.
                if data.get("recent_earnings"):
                    record.recent_earnings = data["recent_earnings"]
            record.ipo_date = (
                self._coerce_date(data.get("ipo_date"))
                or self._parse_ipo_date(data.get("first_trade_date_ms"))
            )
            record.data_source = data_source
            record.field_completeness_score = completeness_score
            record.field_provenance = provenance
            logger.info(
                f"{'Inserted' if is_new else 'Updated'} fundamental data for {symbol} in database"
            )

            # Propagate sector/industry into the universe row when it's missing
            # there (foreign-market ingest lacks it); never clobbers a real value.
            backfill_universe_classification(
                db, symbol, sector=data.get("sector"), industry=data.get("industry")
            )

            db.commit()

            # Update institutional ownership history (SCD2)
            try:
                ownership_service = InstitutionalOwnershipService(db)
                ownership_service.update_ownership(
                    symbol=symbol,
                    institutional_pct=data.get("institutional_ownership"),
                    insider_pct=data.get("insider_ownership"),
                    institutional_transactions=data.get("institutional_transactions"),
                    data_source=data_source
                )
                db.commit()
            except Exception as e:
                logger.warning(f"Error updating ownership history for {symbol}: {e}")
                db.rollback()

            return True
        except Exception as e:
            logger.error(f"Error storing {symbol} in database: {e}", exc_info=True)
            db.rollback()
            return False

        finally:
            db.close()

    def _is_data_fresh(self, last_update: datetime, max_age_days: int = 7) -> bool:
        """
        Check if cached fundamental data is fresh enough.

        Data is considered fresh if the last update is within max_age_days (default 7 days).

        Args:
            last_update: Timestamp of last update
            max_age_days: Maximum age in days before data is stale

        Returns:
            True if data is fresh (age <= max_age_days)
        """
        if last_update is None:
            return False

        # Compare as naive UTC, the zone updated_at is stored in.
        if last_update.tzinfo is not None:
            last_update = last_update.astimezone(UTC).replace(tzinfo=None)

        now = datetime.now(UTC).replace(tzinfo=None)
        age_days = (now - last_update).days

        if max_age_days == self.MAX_AGE_DAYS:
            is_fresh = self._cache_policy.is_datetime_fresh("fundamentals", last_update)
        else:
            is_fresh = age_days <= max_age_days

        if not is_fresh:
            logger.debug(f"Data is stale (age: {age_days} days, last update: {last_update})")

        return is_fresh

    def _needs_db_enrichment(self, fundamentals: Optional[Dict]) -> bool:
        """True when a cached payload is missing keys needed by scan enrichment."""
        if not isinstance(fundamentals, dict):
            return True
        return any(
            key not in fundamentals or fundamentals.get(key) in (None, "")
            for key in self.REQUIRED_SCAN_KEYS
        )

    def _needs_on_demand_enrichment(self, fundamentals: Optional[Dict]) -> bool:
        """True when a single-symbol lookup should hydrate missing required fields."""
        return (
            settings.provider_snapshot_on_demand_fallback_enabled
            and self._needs_db_enrichment(fundamentals)
        )

    @staticmethod
    def _merge_fundamentals(primary: Dict, fallback: Dict) -> Dict:
        """Merge two fundamentals payloads, preferring non-null primary values."""
        merged = dict(primary)
        for key, value in fallback.items():
            if key not in merged or merged[key] is None:
                merged[key] = value
        return merged

    def _get_many_from_database(
        self,
        symbols: list[str]
    ) -> Dict[str, tuple[Optional[Dict], Optional[datetime]]]:
        """
        Bulk fetch fundamentals from database for multiple symbols.

        More efficient than calling _get_from_database() repeatedly
        because it uses a single DB session for all queries.

        Args:
            symbols: List of stock symbols to fetch

        Returns:
            Dict mapping symbol to (fundamentals_dict, last_update) or (None, None)
        """
        if not symbols:
            return {}

        db = self._session_factory()
        results = {}

        try:
            # Query all symbols at once using IN clause
            records = db.query(StockFundamental).filter(
                StockFundamental.symbol.in_(symbols)
            ).all()

            # Build lookup dict from records
            record_map = {r.symbol: r for r in records}

            for symbol in symbols:
                record = record_map.get(symbol)

                if not record:
                    results[symbol] = (None, None)
                    continue

                results[symbol] = (_fundamentals_from_record(record), record.updated_at)

            db_hits = sum(1 for v in results.values() if v[0] is not None)
            logger.debug(f"Bulk DB query for fundamentals: {db_hits} hits, {len(symbols) - db_hits} misses")

            return results

        except Exception as e:
            logger.error(f"Error in bulk database query for fundamentals: {e}", exc_info=True)
            return {symbol: (None, None) for symbol in symbols}

        finally:
            db.close()

    @staticmethod
    def _market_for_symbol(
        symbol: str,
        *,
        market: Optional[str] = None,
        market_by_symbol: Optional[Dict[str, Optional[str]]] = None,
    ) -> Optional[str]:
        if market_by_symbol is not None and symbol in market_by_symbol:
            return market_by_symbol[symbol]
        return market

    def get_many(
        self,
        symbols: list[str],
        market: Optional[str] = None,
        market_by_symbol: Optional[Dict[str, Optional[str]]] = None,
    ) -> Dict[str, Optional[Dict]]:
        """
        Get cached fundamental data for multiple symbols using Redis pipeline.

        Uses a single Redis pipeline operation to fetch all symbols at once,
        dramatically reducing network round-trip overhead (10-20x faster than
        individual get calls).

        Falls back to database for Redis misses, and warms Redis with DB results.

        Args:
            symbols: List of stock ticker symbols
            market: Optional market for homogeneous batches.
            market_by_symbol: Optional per-symbol market map for mixed batches.

        Returns:
            Dict mapping symbols to their cached fundamental dicts (or None if not cached)
        """
        if not symbols:
            return {}

        if not self._redis_client:
            logger.warning("Redis not available for bulk get - using database fallback")
            # Fallback: query database directly
            db_results = self._get_many_from_database(symbols)
            cached_data = {}
            for symbol, (data, _) in db_results.items():
                if data is not None:
                    symbol_market = self._market_for_symbol(
                        symbol,
                        market=market,
                        market_by_symbol=market_by_symbol,
                    )
                    self._ensure_field_availability_metadata(
                        symbol,
                        data,
                        symbol_market,
                    )
                cached_data[symbol] = data
            return cached_data

        try:
            # Build pipeline for bulk fetch from Redis
            pipeline = self._redis_client.pipeline()

            # Queue all get operations
            for symbol in symbols:
                symbol_market = self._market_for_symbol(
                    symbol,
                    market=market,
                    market_by_symbol=market_by_symbol,
                )
                redis_key = self._redis_data_key(symbol, symbol_market)
                pipeline.get(redis_key)

            # Execute all gets in single network call
            results = pipeline.execute()

            # Parse results - track hits and misses
            cached_data = {}
            redis_hits = []
            redis_misses = []
            redis_needs_enrichment = []

            for symbol, raw_data in zip(symbols, results):
                if raw_data:
                    try:
                        fundamentals = decode_dict(raw_data)
                        if fundamentals is None:  # legacy (pickled) payload: a miss
                            cached_data[symbol] = None
                            redis_misses.append(symbol)
                            continue
                        cached_data[symbol] = fundamentals
                        if self._needs_db_enrichment(fundamentals):
                            redis_needs_enrichment.append(symbol)
                            logger.debug(
                                f"Bulk cache HIT for {symbol} fundamentals (Redis, stale shape)"
                            )
                        else:
                            redis_hits.append(symbol)
                            logger.debug(f"Bulk cache HIT for {symbol} fundamentals (Redis)")
                    except Exception as e:
                        logger.warning(f"Error deserializing {symbol}: {e}")
                        cached_data[symbol] = None
                        redis_misses.append(symbol)
                else:
                    cached_data[symbol] = None
                    redis_misses.append(symbol)
                    logger.debug(f"Bulk cache MISS for {symbol} fundamentals (Redis)")

            # DATABASE FALLBACK: Fetch misses + stale-shape payloads from database.
            db_lookup_symbols = redis_misses + redis_needs_enrichment
            if db_lookup_symbols:
                logger.info(
                    "Fetching %d fundamentals from database (%d misses, %d stale shape)",
                    len(db_lookup_symbols),
                    len(redis_misses),
                    len(redis_needs_enrichment),
                )

                db_results = self._get_many_from_database(db_lookup_symbols)

                db_hits = []
                for symbol in db_lookup_symbols:
                    fundamentals, last_update = db_results.get(symbol, (None, None))

                    if fundamentals is not None and self._is_data_fresh(last_update):
                        symbol_market = self._market_for_symbol(
                            symbol,
                            market=market,
                            market_by_symbol=market_by_symbol,
                        )
                        if symbol in redis_needs_enrichment and cached_data.get(symbol):
                            cached_data[symbol] = self._merge_fundamentals(
                                cached_data[symbol], fundamentals
                            )
                        else:
                            cached_data[symbol] = fundamentals
                        self._ensure_field_availability_metadata(
                            symbol,
                            cached_data[symbol],
                            symbol_market,
                        )
                        db_hits.append(symbol)

                        # Warm Redis for next time
                        self._store_in_redis_for_market(
                            symbol,
                            cached_data[symbol],
                            market=symbol_market,
                        )

                logger.info(
                    "Database fallback: %d hits, %d still missing",
                    len(db_hits),
                    len(db_lookup_symbols) - len(db_hits),
                )

            total_hits = sum(1 for s in symbols if cached_data.get(s) is not None)
            total_misses = len(symbols) - total_hits

            logger.info(f"Bulk fetched {len(symbols)} fundamentals: {len(redis_hits)} Redis hits, {total_hits - len(redis_hits)} DB hits, {total_misses} misses")

            return cached_data

        except Exception as e:
            logger.error(f"Error in bulk get fundamentals: {e}", exc_info=True)
            return {symbol: None for symbol in symbols}

    def get_many_cached_only(self, symbols: list[str]) -> Dict[str, Optional[Dict]]:
        """
        Get fundamentals from the persistent cache only.

        This bypasses both Redis and any live provider fallback so offline/static
        export paths can reuse previously hydrated fundamentals without triggering
        fresh network work.
        """
        if not symbols:
            return {}

        db_results = self._get_many_from_database(symbols)
        hits = sum(1 for data, _ in db_results.values() if data is not None)
        logger.info(
            "Bulk cache-only fetched %d fundamentals rows from database (%d misses)",
            hits,
            len(symbols) - hits,
        )
        return {symbol: data for symbol, (data, _) in db_results.items()}

    def store(
        self,
        symbol: str,
        data: Dict,
        data_source: str = 'hybrid',
        market: Optional[str] = None,
    ) -> bool:
        """
        Store fundamental data in cache (Redis + Database).

        Public method for external services (like HybridFundamentalsService)
        to store pre-fetched fundamental data.

        Args:
            symbol: Stock ticker symbol
            data: Fundamental data dict
            data_source: Source identifier (default 'hybrid')
            market: Market code; used to compute T2 completeness and
                provenance without an extra DB lookup. When omitted,
                ``_store_in_database`` falls back to DB resolve.
        """
        if not data:
            logger.warning(f"Cannot store empty data for {symbol}")
            return False

        normalized_data = self._normalize_payload_for_storage(data)

        # Enrich before Redis/DB writes so both see the same snapshot.
        row_facts = self._row_facts_resolver.resolve_for_payload(
            symbol,
            normalized_data,
            market=market,
        )
        market = self._enrich_with_quality_metadata(
            symbol,
            normalized_data,
            row_facts.market,
        )
        self._enrich_with_fx_normalization(
            normalized_data,
            row_facts.currency,
            market=market,
        )

        # Store in Redis
        self._store_in_redis_for_market(symbol, normalized_data, market=market)

        # Store in database
        persisted = self._store_in_database(
            symbol, normalized_data, data_source=data_source, market=market
        )

        if persisted:
            logger.debug(f"Stored fundamentals for {symbol} from {data_source}")
        return persisted

    def record_on_demand_fallback(self) -> None:
        """Track explicit single-symbol provider fallbacks for observability."""
        if not self._redis_client:
            return
        try:
            self._redis_client.incr(self.REDIS_KEY_FALLBACK_COUNT)
            self._redis_client.expire(
                self.REDIS_KEY_FALLBACK_COUNT,
                self._cache_policy.ttl_seconds("fundamentals"),
            )
        except Exception as e:
            logger.debug(f"Error recording on-demand fallback: {e}")

    def get_on_demand_fallback_count(self) -> int:
        """Return the recent single-symbol fallback count."""
        if not self._redis_client:
            return 0
        try:
            raw = self._redis_client.get(self.REDIS_KEY_FALLBACK_COUNT)
            return int(raw) if raw is not None else 0
        except Exception as e:
            logger.debug(f"Error reading on-demand fallback count: {e}")
            return 0

    def invalidate_cache(self, symbol: str, market: str | None = None) -> None:
        """
        Invalidate cached fundamental data for a specific symbol.

        Args:
            symbol: Stock symbol to invalidate
            market: Market for the market-scoped cache key. Defaults to US for legacy callers.
        """
        if not self._redis_client:
            logger.warning("Redis not available for cache invalidation")
            return

        try:
            redis_key = self._redis_data_key(symbol, market=market)
            self._redis_client.delete(redis_key)

            logger.info(f"Invalidated fundamental cache for {symbol}")

        except Exception as e:
            logger.error(f"Error invalidating cache for {symbol}: {e}", exc_info=True)

    def get_cache_stats(self, symbol: str, market: str | None = None) -> Dict:
        """
        Get cache statistics for a symbol.

        Returns:
            Dict with cache info (last_update, source, etc.)
        """
        stats = {
            'symbol': symbol,
            'market': self._cache_policy.normalize_market(market),
            'redis_cached': False,
            'db_cached': False,
            'last_update': None,
            'age_days': None
        }

        # Check Redis
        if self._redis_client:
            try:
                redis_key = self._redis_data_key(symbol, market=market)
                cached_data = self._redis_client.get(redis_key)

                if cached_data:
                    stats['redis_cached'] = True
            except Exception as e:
                logger.debug(f"Error checking Redis stats for {symbol}: {e}")

        # Check Database
        db = self._session_factory()
        try:
            record = db.query(StockFundamental).filter(
                StockFundamental.symbol == symbol
            ).first()

            if record:
                stats['db_cached'] = True
                stats['last_update'] = str(record.updated_at)

                # Calculate age
                if record.updated_at:
                    last_update = record.updated_at
                    if last_update.tzinfo is not None:
                        last_update = last_update.astimezone(UTC).replace(tzinfo=None)
                    age_days = (datetime.now(UTC).replace(tzinfo=None) - last_update).days
                    stats['age_days'] = age_days

        except Exception as e:
            logger.debug(f"Error checking DB stats for {symbol}: {e}")
        finally:
            db.close()

        return stats

    def _parse_ipo_date(self, first_trade_timestamp: Optional[int], is_milliseconds: bool = True) -> Optional[date]:
        """
        Parse IPO date from yfinance firstTradeDateMilliseconds timestamp.

        Args:
            first_trade_timestamp: Timestamp from yfinance (milliseconds or seconds)
            is_milliseconds: If True, timestamp is in milliseconds (default for yfinance)

        Returns:
            date object or None if invalid
        """
        if first_trade_timestamp is None:
            return None

        try:
            # Convert timestamp to seconds if in milliseconds
            if is_milliseconds:
                timestamp_seconds = first_trade_timestamp / 1000
            else:
                timestamp_seconds = first_trade_timestamp

            # Convert Unix timestamp to date
            dt = datetime.fromtimestamp(timestamp_seconds)
            ipo_date = dt.date()

            # Validate date is within reasonable range
            min_valid_date = date(1980, 1, 1)  # Earliest reasonable IPO
            max_valid_date = date.today()  # Can't IPO in the future

            if ipo_date < min_valid_date:
                logger.warning(f"Suspicious IPO date {ipo_date} is before 1980 - rejecting")
                return None

            if ipo_date > max_valid_date:
                logger.warning(f"Invalid IPO date {ipo_date} is in the future - rejecting")
                return None

            return ipo_date
        except Exception as e:
            logger.warning(f"Error parsing IPO date from timestamp {first_trade_timestamp}: {e}")
            return None
