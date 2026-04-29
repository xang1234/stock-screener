"""
Fundamentals Cache Service for stock fundamental data caching.

Provides intelligent caching of stock fundamental data (PE ratios, market cap,
institutional ownership, margins, etc.) with weekly refresh to minimize API calls.
Uses Redis for hot cache and database for persistence.
"""
import logging
import math
import pickle
from typing import Any, Optional, Dict, Callable
from datetime import datetime, date
from sqlalchemy.orm import Session

try:
    import redis  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in desktop packaging
    redis = Any  # type: ignore

from ..database import SessionLocal
from ..models.stock import StockFundamental
from ..config import settings
from .errors import CacheRefreshError
from .fundamentals_completeness import (
    compute_completeness_score,
    derive_field_provenance,
)
from .field_capability_registry import field_capability_registry
from .fx_service import FXQuote, FXService, currency_for_market, get_fx_service
from .institutional_ownership_service import InstitutionalOwnershipService
from .redis_pool import get_redis_client, is_redis_enabled

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
    ):
        """Initialize fundamentals cache service.

        ``fx_service`` is injectable for tests; in production callers
        typically rely on the process-wide default via ``get_fx_service()``.
        """
        self._session_factory = session_factory or SessionLocal
        self._fx_service = fx_service
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

        Returns:
            Dict with fundamental metrics or None if unavailable

        Logic:
        1. Check Redis for cached data (if not force_refresh)
        2. Check database for cached data (if Redis miss)
        3. If data is stale or missing, fetch from yfinance
        4. Update cache with fresh data
        5. Return fundamentals dict
        """
        if force_refresh:
            logger.info(f"Force refresh requested for {symbol}")
            return self._fetch_and_cache(symbol, market=market)

        # Try Redis first (fast path)
        if self._redis_client:
            try:
                redis_key = self.REDIS_KEY_FORMAT.format(symbol=symbol)
                cached_data = self._redis_client.get(redis_key)

                if cached_data:
                    fundamentals = pickle.loads(cached_data)
                    if self._needs_db_enrichment(fundamentals):
                        db_data, last_update = self._get_from_database(symbol)
                        if db_data is not None and self._is_data_fresh(last_update):
                            fundamentals = self._merge_fundamentals(fundamentals, db_data)
                            self._store_in_redis(symbol, fundamentals)
                    if self._needs_on_demand_enrichment(fundamentals):
                        logger.info(
                            "Cache HIT but incomplete for %s fundamentals - fetching on-demand enrichment",
                            symbol,
                        )
                        enriched = self._fetch_and_cache(symbol, market=market)
                        if enriched is not None:
                            return enriched
                    if self._ensure_field_availability_metadata(symbol, fundamentals, market):
                        self._store_in_redis(symbol, fundamentals)
                    logger.debug(f"Cache HIT for {symbol} (Redis)")
                    return fundamentals
            except (pickle.PickleError, TypeError, ValueError, RuntimeError, OSError) as exc:
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
            if self._is_data_fresh(last_update):
                if self._needs_on_demand_enrichment(cached_data):
                    logger.info(
                        "Cache HIT but incomplete for %s fundamentals in database - fetching on-demand enrichment",
                        symbol,
                    )
                    enriched = self._fetch_and_cache(symbol, market=market)
                    if enriched is not None:
                        return enriched
                self._ensure_field_availability_metadata(symbol, cached_data, market)
                logger.info(f"Cache HIT for {symbol} (Database, updated: {last_update})")

                # Also store in Redis for faster next access
                self._store_in_redis(symbol, cached_data)

                return cached_data
            else:
                # Data is stale - fetch fresh data
                logger.info(f"Cache HIT but STALE for {symbol} (last update: {last_update}) - fetching fresh data")
                return self._fetch_and_cache(symbol, market=market)

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

            # Convert database record to dict with ALL fields
            fundamentals = {
                # Market data
                "market_cap": record.market_cap,
                "shares_outstanding": record.shares_outstanding,
                "avg_volume": record.avg_volume,
                "relative_volume": record.relative_volume,
                # Valuation metrics
                "pe_ratio": record.pe_ratio,
                "forward_pe": record.forward_pe,
                "peg_ratio": record.peg_ratio,
                "price_to_book": record.price_to_book,
                "price_to_sales": record.price_to_sales,
                "price_to_cash": record.price_to_cash,
                "price_to_fcf": record.price_to_fcf,
                "ev_ebitda": record.ev_ebitda,
                "ev_sales": record.ev_sales,
                "target_price": record.target_price,
                # Growth metrics
                "eps_current": record.eps_current,
                "eps_next_y": record.eps_next_y,
                "eps_next_5y": record.eps_next_5y,
                "eps_next_q": record.eps_next_q,
                "eps_growth_quarterly": record.eps_growth_quarterly,
                "eps_growth_annual": record.eps_growth_annual,
                "eps_growth_yy": record.eps_growth_yy,
                "revenue_current": record.revenue_current,
                "revenue_growth": record.revenue_growth,
                "sales_past_5y": record.sales_past_5y,
                "sales_growth_yy": record.sales_growth_yy,
                "sales_growth_qq": record.sales_growth_qq,
                # Quarter metadata (consolidated from QuarterlyData)
                "recent_quarter_date": record.recent_quarter_date,
                "previous_quarter_date": record.previous_quarter_date,
                "growth_reporting_cadence": record.growth_reporting_cadence,
                "growth_metric_basis": record.growth_metric_basis,
                "growth_comparable_period_date": record.growth_comparable_period_date,
                "growth_reference_gap_days": record.growth_reference_gap_days,
                # Alias for CANSLIM compatibility
                "eps_growth_qq": record.eps_growth_quarterly,
                # Profitability metrics
                "profit_margin": record.profit_margin,
                "operating_margin": record.operating_margin,
                "gross_margin": record.gross_margin,
                "roe": record.roe,
                "roa": record.roa,
                "roic": record.roic,
                # Financial health
                "current_ratio": record.current_ratio,
                "quick_ratio": record.quick_ratio,
                "debt_to_equity": record.debt_to_equity,
                "lt_debt_to_equity": record.lt_debt_to_equity,
                # Ownership & sentiment
                "insider_ownership": record.insider_ownership,
                "insider_transactions": record.insider_transactions,
                "institutional_ownership": record.institutional_ownership,
                "institutional_transactions": record.institutional_transactions,
                "institutional_change": record.institutional_change,
                "short_float": record.short_float,
                "short_ratio": record.short_ratio,
                "short_interest": record.short_interest,
                # Technical indicators
                "beta": record.beta,
                "rsi_14": record.rsi_14,
                "atr_14": record.atr_14,
                "sma_20": record.sma_20,
                "sma_50": record.sma_50,
                "sma_200": record.sma_200,
                "volatility_week": record.volatility_week,
                "volatility_month": record.volatility_month,
                # Performance metrics
                "perf_week": record.perf_week,
                "perf_month": record.perf_month,
                "perf_quarter": record.perf_quarter,
                "perf_half_year": record.perf_half_year,
                "perf_year": record.perf_year,
                "perf_ytd": record.perf_ytd,
                # Dividend metrics
                "dividend_ttm": record.dividend_ttm,
                "dividend_yield": record.dividend_yield,
                "payout_ratio": record.payout_ratio,
                # 52-week range
                "week_52_high": record.week_52_high,
                "week_52_high_distance": record.week_52_high_distance,
                "week_52_low": record.week_52_low,
                "week_52_low_distance": record.week_52_low_distance,
                # Company info
                "sector": record.sector,
                "industry": record.industry,
                "country": record.country,
                "employees": record.employees,
                # IPO date
                "ipo_date": record.ipo_date,
                "first_trade_date": int(datetime.combine(record.ipo_date, datetime.min.time()).timestamp()) if record.ipo_date else None,  # Epoch format for compatibility
                # Company descriptions
                "description_yfinance": record.description_yfinance,
                "description_finviz": record.description_finviz,
                # Analyst recommendations
                "recommendation": record.recommendation,
                # EPS Rating components
                "eps_5yr_cagr": record.eps_5yr_cagr,
                "eps_q1_yoy": record.eps_q1_yoy,
                "eps_q2_yoy": record.eps_q2_yoy,
                "eps_raw_score": record.eps_raw_score,
                "eps_rating": record.eps_rating,
                "eps_years_available": record.eps_years_available,
                # Data source tracking
                "data_source": record.data_source,
                "finviz_snapshot_revision": record.finviz_snapshot_revision,
                "finviz_snapshot_at": record.finviz_snapshot_at.isoformat() if record.finviz_snapshot_at else None,
                "yahoo_profile_refreshed_at": (
                    record.yahoo_profile_refreshed_at.isoformat() if record.yahoo_profile_refreshed_at else None
                ),
                "yahoo_statements_refreshed_at": (
                    record.yahoo_statements_refreshed_at.isoformat() if record.yahoo_statements_refreshed_at else None
                ),
                "technicals_refreshed_at": (
                    record.technicals_refreshed_at.isoformat() if record.technicals_refreshed_at else None
                ),
                # T2 quality metadata
                "field_completeness_score": record.field_completeness_score,
                "field_provenance": record.field_provenance,
                # T3 FX normalisation
                "market_cap_usd": record.market_cap_usd,
                "adv_usd": record.adv_usd,
                "fx_metadata": record.fx_metadata,
            }

            # Compute fallback description (finviz preferred, yfinance as fallback)
            fundamentals["description"] = fundamentals.get("description_finviz") or fundamentals.get("description_yfinance")
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
        return market

    def _ensure_field_availability_metadata(
        self,
        symbol: str,
        data: Dict,
        market: Optional[str],
    ) -> bool:
        """Backfill derived metadata fields for older cache rows.

        Returns ``True`` when any field was backfilled.
        """
        changed = False
        resolved_market = market if market is not None else self._resolve_market(symbol)
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
        if missing_fx_metadata:
            self._enrich_with_fx_normalization(data, resolved_market)
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
        market: Optional[str],
    ) -> None:
        """Compute USD normalisation (``market_cap_usd``, ``adv_usd``) and
        attach an ``fx_metadata`` snapshot to ``data`` in place.

        Called *alongside* ``_enrich_with_quality_metadata`` before Redis/
        DB writes so derived values survive a warm-cache read. Missing
        rate or missing source amounts → NULL USD columns (honest
        "unknown" rather than a fabricated zero).
        """
        currency = currency_for_market(market)
        market_cap = data.get("market_cap")
        shares_out = data.get("shares_outstanding")
        avg_volume = data.get("avg_volume")

        quote = self._get_fx_service().get_usd_rate(currency)
        if quote is None:
            # Rate unavailable — leave USD columns NULL but record the
            # intent so consumers can see we tried.
            data["market_cap_usd"] = None
            data["adv_usd"] = None
            data["fx_metadata"] = FXQuote.unavailable_metadata(currency)
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

    def _resolve_market(self, symbol: str) -> Optional[str]:
        """Look up the stock universe market for ``symbol``.

        Returns ``None`` if the symbol is not in the universe (or the lookup
        fails) — the routing policy treats ``None`` as the legacy US default
        so this never crashes on-demand fetches.
        """
        from ..models.stock_universe import StockUniverse

        db = None
        try:
            db = self._session_factory()
            row = (
                db.query(StockUniverse.market)
                .filter(StockUniverse.symbol == symbol)
                .first()
            )
            return row[0] if row else None
        except Exception as exc:  # pragma: no cover - DB hiccup shouldn't block fetch
            logger.debug(
                "Market lookup failed for %s (%s) - defaulting to US policy",
                symbol, exc,
            )
            return None
        finally:
            if db is not None:
                db.close()

    def _fetch_and_cache(
        self,
        symbol: str,
        market: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Fetch fundamental data from finvizfinance (primary) or yfinance (fallback) and cache it.

        Graceful fallback: If fetch fails, return None (caller will handle).

        ``market`` is consulted by the provider routing policy so US-only
        providers (finviz, alphavantage) are skipped for HK/JP/TW/CN symbols.
        When omitted, falls back to a single DB lookup.
        """
        try:
            if not settings.provider_snapshot_on_demand_fallback_enabled:
                logger.warning("On-demand provider fallback is disabled for %s", symbol)
                return None

            from ..wiring.bootstrap import get_data_source_service

            logger.info(f"Fetching fresh fundamental data for {symbol} from data sources")
            self.record_on_demand_fallback()

            if market is None:
                market = self._resolve_market(symbol)

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
            market = self._enrich_with_quality_metadata(symbol, fundamentals, market)
            self._enrich_with_fx_normalization(fundamentals, market)

            # Cache in Redis (7-day TTL)
            self._store_in_redis(symbol, fundamentals)

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

    def _store_in_redis(self, symbol: str, data: Dict) -> None:
        """
        Store fundamental data in Redis with 7-day TTL.
        """
        if not self._redis_client:
            return

        try:
            redis_key = self.REDIS_KEY_FORMAT.format(symbol=symbol)
            pickled_data = pickle.dumps(data)

            self._redis_client.setex(
                redis_key,
                self.CACHE_TTL_SECONDS,
                pickled_data
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

            # Check if record exists
            existing_record = db.query(StockFundamental).filter(
                StockFundamental.symbol == symbol
            ).first()

            if existing_record:
                # Update existing record - organized by category
                # Market data
                _assign_if_present(existing_record, "market_cap", data, "market_cap")
                _assign_if_present(existing_record, "shares_outstanding", data, "shares_outstanding")
                _assign_if_present(existing_record, "avg_volume", data, "avg_volume")
                existing_record.relative_volume = data.get("relative_volume")

                # Valuation metrics
                existing_record.pe_ratio = data.get("pe_ratio")
                existing_record.forward_pe = data.get("forward_pe")
                existing_record.peg_ratio = data.get("peg_ratio")
                existing_record.price_to_book = data.get("price_to_book")
                existing_record.price_to_sales = data.get("price_to_sales")
                existing_record.price_to_cash = data.get("price_to_cash")
                existing_record.price_to_fcf = data.get("price_to_fcf")
                existing_record.ev_ebitda = data.get("ev_ebitda")
                existing_record.ev_sales = data.get("ev_sales")
                existing_record.target_price = data.get("target_price")

                # Growth metrics
                existing_record.eps_current = data.get("eps_current")
                existing_record.eps_next_y = data.get("eps_next_y")
                existing_record.eps_next_5y = data.get("eps_next_5y")
                existing_record.eps_next_q = data.get("eps_next_q")
                existing_record.eps_growth_quarterly = data.get("eps_growth_qq")
                existing_record.eps_growth_annual = data.get("eps_growth_yy")
                existing_record.eps_growth_yy = data.get("eps_growth_yy")
                existing_record.revenue_current = data.get("revenue_current")
                existing_record.revenue_growth = data.get("revenue_growth")
                existing_record.sales_past_5y = data.get("sales_past_5y")
                existing_record.sales_growth_yy = data.get("sales_growth_yy")
                existing_record.sales_growth_qq = data.get("sales_growth_qq")
                existing_record.recent_quarter_date = data.get("recent_quarter_date")
                existing_record.previous_quarter_date = data.get("previous_quarter_date")
                existing_record.growth_reporting_cadence = data.get("growth_reporting_cadence")
                existing_record.growth_metric_basis = data.get("growth_metric_basis")
                existing_record.growth_comparable_period_date = data.get("growth_comparable_period_date")
                existing_record.growth_reference_gap_days = data.get("growth_reference_gap_days")

                # Profitability metrics
                existing_record.profit_margin = data.get("profit_margin")
                existing_record.operating_margin = data.get("operating_margin")
                existing_record.gross_margin = data.get("gross_margin")
                existing_record.roe = data.get("roe")
                existing_record.roa = data.get("roa")
                existing_record.roic = data.get("roic")

                # Financial health
                existing_record.current_ratio = data.get("current_ratio")
                existing_record.quick_ratio = data.get("quick_ratio")
                existing_record.debt_to_equity = data.get("debt_to_equity")
                existing_record.lt_debt_to_equity = data.get("lt_debt_to_equity")

                # Ownership & sentiment
                existing_record.insider_ownership = data.get("insider_ownership")
                existing_record.insider_transactions = data.get("insider_transactions")
                existing_record.institutional_ownership = data.get("institutional_ownership")
                existing_record.institutional_transactions = data.get("institutional_transactions")
                existing_record.short_float = data.get("short_float")
                existing_record.short_ratio = data.get("short_ratio")
                existing_record.short_interest = data.get("short_interest")

                # Technical indicators
                existing_record.beta = data.get("beta")
                existing_record.rsi_14 = data.get("rsi_14")
                existing_record.atr_14 = data.get("atr_14")
                existing_record.sma_20 = data.get("sma_20")
                existing_record.sma_50 = data.get("sma_50")
                existing_record.sma_200 = data.get("sma_200")
                existing_record.volatility_week = data.get("volatility_week")
                existing_record.volatility_month = data.get("volatility_month")

                # Performance metrics
                existing_record.perf_week = data.get("perf_week")
                existing_record.perf_month = data.get("perf_month")
                existing_record.perf_quarter = data.get("perf_quarter")
                existing_record.perf_half_year = data.get("perf_half_year")
                existing_record.perf_year = data.get("perf_year")
                existing_record.perf_ytd = data.get("perf_ytd")

                # Dividend metrics
                existing_record.dividend_ttm = data.get("dividend_ttm")
                existing_record.dividend_yield = data.get("dividend_yield")
                existing_record.payout_ratio = data.get("payout_ratio")

                # 52-week range
                existing_record.week_52_high = data.get("week_52_high")
                existing_record.week_52_high_distance = data.get("week_52_high_distance")
                existing_record.week_52_low = data.get("week_52_low")
                existing_record.week_52_low_distance = data.get("week_52_low_distance")

                # Company info
                existing_record.sector = data.get("sector")
                existing_record.industry = data.get("industry")
                existing_record.country = data.get("country")
                existing_record.employees = data.get("employees")

                # IPO date (prefer explicit date, fall back to yfinance timestamp)
                existing_record.ipo_date = (
                    self._coerce_date(data.get("ipo_date"))
                    or self._parse_ipo_date(data.get("first_trade_date_ms"))
                )

                # Company descriptions
                existing_record.description_yfinance = data.get("description_yfinance")
                existing_record.description_finviz = data.get("description_finviz")

                # Analyst recommendations
                existing_record.recommendation = data.get("recommendation")

                # EPS Rating components
                existing_record.eps_5yr_cagr = data.get("eps_5yr_cagr")
                existing_record.eps_q1_yoy = data.get("eps_q1_yoy")
                existing_record.eps_q2_yoy = data.get("eps_q2_yoy")
                existing_record.eps_raw_score = data.get("eps_raw_score")
                existing_record.eps_rating = data.get("eps_rating")
                existing_record.eps_years_available = data.get("eps_years_available")

                # Metadata
                existing_record.data_source = data_source
                existing_record.finviz_snapshot_revision = data.get("finviz_snapshot_revision")
                existing_record.finviz_snapshot_at = self._coerce_datetime(data.get("finviz_snapshot_at"))
                existing_record.yahoo_profile_refreshed_at = self._coerce_datetime(
                    data.get("yahoo_profile_refreshed_at")
                )
                existing_record.yahoo_statements_refreshed_at = self._coerce_datetime(
                    data.get("yahoo_statements_refreshed_at")
                )
                existing_record.technicals_refreshed_at = self._coerce_datetime(
                    data.get("technicals_refreshed_at")
                )
                # T2 quality metadata
                existing_record.field_completeness_score = completeness_score
                existing_record.field_provenance = provenance
                # T3 FX normalisation
                _assign_if_present(existing_record, "market_cap_usd", data, "market_cap_usd")
                _assign_if_present(existing_record, "adv_usd", data, "adv_usd")
                _assign_if_present(existing_record, "fx_metadata", data, "fx_metadata")
                existing_record.updated_at = datetime.now()

                logger.info(f"Updated fundamental data for {symbol} in database")
            else:
                # Insert new record - organized by category
                new_record = StockFundamental(
                    symbol=symbol,
                    # Market data
                    market_cap=data.get("market_cap"),
                    shares_outstanding=data.get("shares_outstanding"),
                    avg_volume=data.get("avg_volume"),
                    relative_volume=data.get("relative_volume"),
                    # Valuation metrics
                    pe_ratio=data.get("pe_ratio"),
                    forward_pe=data.get("forward_pe"),
                    peg_ratio=data.get("peg_ratio"),
                    price_to_book=data.get("price_to_book"),
                    price_to_sales=data.get("price_to_sales"),
                    price_to_cash=data.get("price_to_cash"),
                    price_to_fcf=data.get("price_to_fcf"),
                    ev_ebitda=data.get("ev_ebitda"),
                    ev_sales=data.get("ev_sales"),
                    target_price=data.get("target_price"),
                    # Growth metrics
                    eps_current=data.get("eps_current"),
                    eps_next_y=data.get("eps_next_y"),
                    eps_next_5y=data.get("eps_next_5y"),
                    eps_next_q=data.get("eps_next_q"),
                    eps_growth_quarterly=data.get("eps_growth_qq"),
                    eps_growth_annual=data.get("eps_growth_yy"),
                    eps_growth_yy=data.get("eps_growth_yy"),
                    revenue_current=data.get("revenue_current"),
                    revenue_growth=data.get("revenue_growth"),
                    sales_past_5y=data.get("sales_past_5y"),
                    sales_growth_yy=data.get("sales_growth_yy"),
                    sales_growth_qq=data.get("sales_growth_qq"),
                    recent_quarter_date=data.get("recent_quarter_date"),
                    previous_quarter_date=data.get("previous_quarter_date"),
                    growth_reporting_cadence=data.get("growth_reporting_cadence"),
                    growth_metric_basis=data.get("growth_metric_basis"),
                    growth_comparable_period_date=data.get("growth_comparable_period_date"),
                    growth_reference_gap_days=data.get("growth_reference_gap_days"),
                    # Profitability metrics
                    profit_margin=data.get("profit_margin"),
                    operating_margin=data.get("operating_margin"),
                    gross_margin=data.get("gross_margin"),
                    roe=data.get("roe"),
                    roa=data.get("roa"),
                    roic=data.get("roic"),
                    # Financial health
                    current_ratio=data.get("current_ratio"),
                    quick_ratio=data.get("quick_ratio"),
                    debt_to_equity=data.get("debt_to_equity"),
                    lt_debt_to_equity=data.get("lt_debt_to_equity"),
                    # Ownership & sentiment
                    insider_ownership=data.get("insider_ownership"),
                    insider_transactions=data.get("insider_transactions"),
                    institutional_ownership=data.get("institutional_ownership"),
                    institutional_transactions=data.get("institutional_transactions"),
                    short_float=data.get("short_float"),
                    short_ratio=data.get("short_ratio"),
                    short_interest=data.get("short_interest"),
                    # Technical indicators
                    beta=data.get("beta"),
                    rsi_14=data.get("rsi_14"),
                    atr_14=data.get("atr_14"),
                    sma_20=data.get("sma_20"),
                    sma_50=data.get("sma_50"),
                    sma_200=data.get("sma_200"),
                    volatility_week=data.get("volatility_week"),
                    volatility_month=data.get("volatility_month"),
                    # Performance metrics
                    perf_week=data.get("perf_week"),
                    perf_month=data.get("perf_month"),
                    perf_quarter=data.get("perf_quarter"),
                    perf_half_year=data.get("perf_half_year"),
                    perf_year=data.get("perf_year"),
                    perf_ytd=data.get("perf_ytd"),
                    # Dividend metrics
                    dividend_ttm=data.get("dividend_ttm"),
                    dividend_yield=data.get("dividend_yield"),
                    payout_ratio=data.get("payout_ratio"),
                    # 52-week range
                    week_52_high=data.get("week_52_high"),
                    week_52_high_distance=data.get("week_52_high_distance"),
                    week_52_low=data.get("week_52_low"),
                    week_52_low_distance=data.get("week_52_low_distance"),
                    # Company info
                    sector=data.get("sector"),
                    industry=data.get("industry"),
                    country=data.get("country"),
                    employees=data.get("employees"),
                    # IPO date (prefer explicit date, fall back to yfinance timestamp)
                    ipo_date=(
                        self._coerce_date(data.get("ipo_date"))
                        or self._parse_ipo_date(data.get("first_trade_date_ms"))
                    ),
                    # Company descriptions
                    description_yfinance=data.get("description_yfinance"),
                    description_finviz=data.get("description_finviz"),
                    # Analyst recommendations
                    recommendation=data.get("recommendation"),
                    # EPS Rating components
                    eps_5yr_cagr=data.get("eps_5yr_cagr"),
                    eps_q1_yoy=data.get("eps_q1_yoy"),
                    eps_q2_yoy=data.get("eps_q2_yoy"),
                    eps_raw_score=data.get("eps_raw_score"),
                    eps_rating=data.get("eps_rating"),
                    eps_years_available=data.get("eps_years_available"),
                    # Metadata
                    data_source=data_source,
                    finviz_snapshot_revision=data.get("finviz_snapshot_revision"),
                    finviz_snapshot_at=self._coerce_datetime(data.get("finviz_snapshot_at")),
                    yahoo_profile_refreshed_at=self._coerce_datetime(
                        data.get("yahoo_profile_refreshed_at")
                    ),
                    yahoo_statements_refreshed_at=self._coerce_datetime(
                        data.get("yahoo_statements_refreshed_at")
                    ),
                    technicals_refreshed_at=self._coerce_datetime(
                        data.get("technicals_refreshed_at")
                    ),
                    # T2 quality metadata
                    field_completeness_score=completeness_score,
                    field_provenance=provenance,
                    # T3 FX normalisation
                    market_cap_usd=data.get("market_cap_usd"),
                    adv_usd=data.get("adv_usd"),
                    fx_metadata=data.get("fx_metadata"),
                )
                db.add(new_record)
                logger.info(f"Inserted fundamental data for {symbol} in database")

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

        # Ensure timezone-naive comparison
        if last_update.tzinfo is not None:
            last_update = last_update.replace(tzinfo=None)

        now = datetime.now()
        age_days = (now - last_update).days

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

                # Convert database record to dict (same as _get_from_database)
                fundamentals = {
                    # Market data
                    "market_cap": record.market_cap,
                    "shares_outstanding": record.shares_outstanding,
                    "avg_volume": record.avg_volume,
                    "relative_volume": record.relative_volume,
                    # Valuation metrics
                    "pe_ratio": record.pe_ratio,
                    "forward_pe": record.forward_pe,
                    "peg_ratio": record.peg_ratio,
                    "price_to_book": record.price_to_book,
                    "price_to_sales": record.price_to_sales,
                    "price_to_cash": record.price_to_cash,
                    "price_to_fcf": record.price_to_fcf,
                    "ev_ebitda": record.ev_ebitda,
                    "ev_sales": record.ev_sales,
                    "target_price": record.target_price,
                    # Growth metrics
                    "eps_current": record.eps_current,
                    "eps_next_y": record.eps_next_y,
                    "eps_next_5y": record.eps_next_5y,
                    "eps_next_q": record.eps_next_q,
                    "eps_growth_quarterly": record.eps_growth_quarterly,
                    "eps_growth_annual": record.eps_growth_annual,
                    "eps_growth_yy": record.eps_growth_yy,
                    "revenue_current": record.revenue_current,
                    "revenue_growth": record.revenue_growth,
                    "sales_past_5y": record.sales_past_5y,
                    "sales_growth_yy": record.sales_growth_yy,
                    "sales_growth_qq": record.sales_growth_qq,
                    # Quarter metadata
                    "recent_quarter_date": record.recent_quarter_date,
                    "previous_quarter_date": record.previous_quarter_date,
                    "growth_reporting_cadence": record.growth_reporting_cadence,
                    "growth_metric_basis": record.growth_metric_basis,
                    "growth_comparable_period_date": record.growth_comparable_period_date,
                    "growth_reference_gap_days": record.growth_reference_gap_days,
                    # Alias for CANSLIM compatibility
                    "eps_growth_qq": record.eps_growth_quarterly,
                    # Profitability metrics
                    "profit_margin": record.profit_margin,
                    "operating_margin": record.operating_margin,
                    "gross_margin": record.gross_margin,
                    "roe": record.roe,
                    "roa": record.roa,
                    "roic": record.roic,
                    # Financial health
                    "current_ratio": record.current_ratio,
                    "quick_ratio": record.quick_ratio,
                    "debt_to_equity": record.debt_to_equity,
                    "lt_debt_to_equity": record.lt_debt_to_equity,
                    # Ownership & sentiment
                    "insider_ownership": record.insider_ownership,
                    "insider_transactions": record.insider_transactions,
                    "institutional_ownership": record.institutional_ownership,
                    "institutional_transactions": record.institutional_transactions,
                    "institutional_change": record.institutional_change,
                    "short_float": record.short_float,
                    "short_ratio": record.short_ratio,
                    "short_interest": record.short_interest,
                    # Technical indicators
                    "beta": record.beta,
                    "rsi_14": record.rsi_14,
                    "atr_14": record.atr_14,
                    "sma_20": record.sma_20,
                    "sma_50": record.sma_50,
                    "sma_200": record.sma_200,
                    "volatility_week": record.volatility_week,
                    "volatility_month": record.volatility_month,
                    # Performance metrics
                    "perf_week": record.perf_week,
                    "perf_month": record.perf_month,
                    "perf_quarter": record.perf_quarter,
                    "perf_half_year": record.perf_half_year,
                    "perf_year": record.perf_year,
                    "perf_ytd": record.perf_ytd,
                    # Dividend metrics
                    "dividend_ttm": record.dividend_ttm,
                    "dividend_yield": record.dividend_yield,
                    "payout_ratio": record.payout_ratio,
                    # 52-week range
                    "week_52_high": record.week_52_high,
                    "week_52_high_distance": record.week_52_high_distance,
                    "week_52_low": record.week_52_low,
                    "week_52_low_distance": record.week_52_low_distance,
                    # Company info
                    "sector": record.sector,
                    "industry": record.industry,
                    "country": record.country,
                    "employees": record.employees,
                    # IPO date
                    "ipo_date": record.ipo_date,
                    "first_trade_date": int(datetime.combine(record.ipo_date, datetime.min.time()).timestamp()) if record.ipo_date else None,
                    # Company descriptions
                    "description_yfinance": record.description_yfinance,
                    "description_finviz": record.description_finviz,
                    # Analyst recommendations
                    "recommendation": record.recommendation,
                    # EPS Rating components
                    "eps_5yr_cagr": record.eps_5yr_cagr,
                    "eps_q1_yoy": record.eps_q1_yoy,
                    "eps_q2_yoy": record.eps_q2_yoy,
                    "eps_raw_score": record.eps_raw_score,
                    "eps_rating": record.eps_rating,
                    "eps_years_available": record.eps_years_available,
                    # Data source tracking
                    "data_source": record.data_source,
                    "finviz_snapshot_revision": record.finviz_snapshot_revision,
                    "finviz_snapshot_at": record.finviz_snapshot_at.isoformat() if record.finviz_snapshot_at else None,
                    "yahoo_profile_refreshed_at": (
                        record.yahoo_profile_refreshed_at.isoformat() if record.yahoo_profile_refreshed_at else None
                    ),
                    "yahoo_statements_refreshed_at": (
                        record.yahoo_statements_refreshed_at.isoformat() if record.yahoo_statements_refreshed_at else None
                    ),
                    "technicals_refreshed_at": (
                        record.technicals_refreshed_at.isoformat() if record.technicals_refreshed_at else None
                    ),
                }

                # Compute fallback description
                fundamentals["description"] = fundamentals.get("description_finviz") or fundamentals.get("description_yfinance")

                results[symbol] = (fundamentals, record.updated_at)

            db_hits = sum(1 for v in results.values() if v[0] is not None)
            logger.debug(f"Bulk DB query for fundamentals: {db_hits} hits, {len(symbols) - db_hits} misses")

            return results

        except Exception as e:
            logger.error(f"Error in bulk database query for fundamentals: {e}", exc_info=True)
            return {symbol: (None, None) for symbol in symbols}

        finally:
            db.close()

    def get_many(self, symbols: list[str]) -> Dict[str, Optional[Dict]]:
        """
        Get cached fundamental data for multiple symbols using Redis pipeline.

        Uses a single Redis pipeline operation to fetch all symbols at once,
        dramatically reducing network round-trip overhead (10-20x faster than
        individual get calls).

        Falls back to database for Redis misses, and warms Redis with DB results.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dict mapping symbols to their cached fundamental dicts (or None if not cached)
        """
        if not symbols:
            return {}

        if not self._redis_client:
            logger.warning("Redis not available for bulk get - using database fallback")
            # Fallback: query database directly
            db_results = self._get_many_from_database(symbols)
            return {symbol: data for symbol, (data, _) in db_results.items()}

        try:
            # Build pipeline for bulk fetch from Redis
            pipeline = self._redis_client.pipeline()

            # Queue all get operations
            for symbol in symbols:
                redis_key = self.REDIS_KEY_FORMAT.format(symbol=symbol)
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
                        fundamentals = pickle.loads(raw_data)
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
                        if symbol in redis_needs_enrichment and cached_data.get(symbol):
                            cached_data[symbol] = self._merge_fundamentals(
                                cached_data[symbol], fundamentals
                            )
                        else:
                            cached_data[symbol] = fundamentals
                        db_hits.append(symbol)

                        # Warm Redis for next time
                        self._store_in_redis(symbol, cached_data[symbol])

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
        market = self._enrich_with_quality_metadata(symbol, normalized_data, market)
        self._enrich_with_fx_normalization(normalized_data, market)

        # Store in Redis
        self._store_in_redis(symbol, normalized_data)

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
            self._redis_client.expire(self.REDIS_KEY_FALLBACK_COUNT, self.CACHE_TTL_SECONDS)
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

    def invalidate_cache(self, symbol: str) -> None:
        """
        Invalidate cached fundamental data for a specific symbol.

        Args:
            symbol: Stock symbol to invalidate
        """
        if not self._redis_client:
            logger.warning("Redis not available for cache invalidation")
            return

        try:
            redis_key = self.REDIS_KEY_FORMAT.format(symbol=symbol)
            self._redis_client.delete(redis_key)

            logger.info(f"Invalidated fundamental cache for {symbol}")

        except Exception as e:
            logger.error(f"Error invalidating cache for {symbol}: {e}", exc_info=True)

    def get_cache_stats(self, symbol: str) -> Dict:
        """
        Get cache statistics for a symbol.

        Returns:
            Dict with cache info (last_update, source, etc.)
        """
        stats = {
            'symbol': symbol,
            'redis_cached': False,
            'db_cached': False,
            'last_update': None,
            'age_days': None
        }

        # Check Redis
        if self._redis_client:
            try:
                redis_key = self.REDIS_KEY_FORMAT.format(symbol=symbol)
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
                        last_update = last_update.replace(tzinfo=None)
                    age_days = (datetime.now() - last_update).days
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
