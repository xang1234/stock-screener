"""
Data preparation layer for multi-screener architecture.

Centralizes data fetching so that data is fetched once and shared
across all screeners. This eliminates redundant API calls and improves
performance when running multiple screeners on the same stock.
"""
import logging
import random
import time
from typing import List, Dict, Optional
import pandas as pd

from .base_screener import DataRequirements, StockData
from .criteria.relative_strength import RelativeStrengthCalculator
from ..domain.common.errors import DataFetchError
from ..domain.scanning.mixed_market_policy import is_mixed_market
from ..services.benchmark_cache_service import BenchmarkCacheService, BenchmarkDataBundle
from ..services.fundamentals_cache_service import FundamentalsCacheService
from ..services.price_cache_service import PriceCacheService
from ..services.rate_limiter import RateLimitTimeoutError
from ..services.security_master_service import SecurityMasterResolver, security_master_resolver
from ..wiring.bootstrap import get_rate_limiter, get_yfinance_service

logger = logging.getLogger(__name__)

_TRANSIENT_TYPES = (ConnectionError, TimeoutError, RateLimitTimeoutError)
_RS_PERCENTILE_PERIODS = (21, 63, 252)


class DataPreparationLayer:
    """
    Centralized data fetching for all screeners.

    Fetches all required data once based on merged requirements from
    multiple screeners, then shares that data across all screeners.
    """

    def __init__(
        self,
        *,
        price_cache: PriceCacheService,
        benchmark_cache: BenchmarkCacheService,
        fundamentals_cache: FundamentalsCacheService,
        max_retries: int = 0,
        retry_base_delay: float = 1.0,
    ):
        """Initialize data preparation layer."""
        self.price_cache = price_cache
        self.benchmark_cache = benchmark_cache
        self.fundamentals_cache = fundamentals_cache
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._yfinance_service = get_yfinance_service()
        self._rate_limiter = get_rate_limiter()
        self._security_master = security_master_resolver
        self._rs_calc = RelativeStrengthCalculator()

    def _resolve_identity(self, symbol: str):
        resolver = getattr(self, "_security_master", None) or SecurityMasterResolver()
        return resolver.resolve_identity(symbol=symbol)

    def _get_rs_calc(self) -> RelativeStrengthCalculator:
        rs_calc = getattr(self, "_rs_calc", None)
        if rs_calc is None:
            rs_calc = RelativeStrengthCalculator()
            self._rs_calc = rs_calc
        return rs_calc

    def _fetch_benchmark_bundle(
        self,
        *,
        market: str,
        period: str,
    ) -> BenchmarkDataBundle | None:
        """Fetch benchmark data and preserve resolved benchmark metadata."""
        bundle_fn = getattr(self.benchmark_cache, "get_benchmark_bundle", None)
        if callable(bundle_fn):
            bundle = self._fetch_with_retry(
                bundle_fn,
                market=market,
                period=period,
            )
            if isinstance(bundle, BenchmarkDataBundle):
                return bundle
            if bundle is None:
                return None
            bundle_data = getattr(bundle, "data", None)
            if isinstance(bundle_data, pd.DataFrame):
                return BenchmarkDataBundle(
                    market=getattr(bundle, "market", market),
                    period=getattr(bundle, "period", period),
                    benchmark_symbol=getattr(bundle, "benchmark_symbol", "UNKNOWN"),
                    benchmark_role=getattr(bundle, "benchmark_role", "primary"),
                    benchmark_kind=getattr(bundle, "benchmark_kind", None),
                    candidate_symbols=tuple(getattr(bundle, "candidate_symbols", ()) or ()),
                    data=bundle_data,
                )

        # Backward-compatible fallback for test doubles that only expose get_benchmark_data.
        benchmark_data = self._fetch_with_retry(
            self.benchmark_cache.get_benchmark_data,
            market=market,
            period=period,
        )
        if benchmark_data is None or benchmark_data.empty:
            return None

        benchmark_symbol = None
        benchmark_candidates: tuple[str, ...] = ()
        symbol_fn = getattr(self.benchmark_cache, "get_benchmark_symbol", None)
        candidates_fn = getattr(self.benchmark_cache, "get_benchmark_candidates", None)
        if callable(symbol_fn):
            try:
                benchmark_symbol = symbol_fn(market)
            except Exception:
                benchmark_symbol = None
        if callable(candidates_fn):
            try:
                benchmark_candidates = tuple(candidates_fn(market))
            except Exception:
                benchmark_candidates = ()

        if benchmark_symbol is None and benchmark_candidates:
            benchmark_symbol = benchmark_candidates[0]

        return BenchmarkDataBundle(
            market=market,
            period=period,
            benchmark_symbol=benchmark_symbol or "UNKNOWN",
            benchmark_role="primary",
            benchmark_kind=None,
            candidate_symbols=benchmark_candidates,
            data=benchmark_data,
        )

    def _compute_market_rs_universe_performances(
        self,
        stock_data_items: list[StockData],
    ) -> dict[str, dict[int | str, list[float]]]:
        """Build market-partitioned RS universe performance lists for percentile ranking."""
        rs_calc = self._get_rs_calc()
        by_market: dict[str, dict[int | str, list[float]]] = {}
        seen_symbols: set[tuple[str, str]] = set()

        for item in stock_data_items:
            market = item.market or "US"
            dedupe_key = (market, item.symbol)
            if dedupe_key in seen_symbols:
                continue
            seen_symbols.add(dedupe_key)
            if (
                item.price_data is None
                or item.price_data.empty
                or item.benchmark_data is None
                or item.benchmark_data.empty
                or "Close" not in item.price_data.columns
                or "Close" not in item.benchmark_data.columns
            ):
                continue

            prices = item.price_data["Close"].reset_index(drop=True)[::-1].reset_index(drop=True)
            benchmark = item.benchmark_data["Close"].reset_index(drop=True)[::-1].reset_index(drop=True)

            market_universe = by_market.setdefault(
                market,
                {"weighted": [], 21: [], 63: [], 252: []},
            )

            weighted_perf = rs_calc.calculate_weighted_performance(prices, benchmark)
            if weighted_perf is not None:
                market_universe["weighted"].append(weighted_perf)

            for period in _RS_PERCENTILE_PERIODS:
                stock_return = rs_calc.calculate_return(prices, period)
                benchmark_return = rs_calc.calculate_return(benchmark, period)
                if stock_return is None or benchmark_return is None:
                    continue
                market_universe[period].append(stock_return - benchmark_return)

        normalized: dict[str, dict[int | str, list[float]]] = {}
        for market, values in by_market.items():
            market_values = {
                key: series
                for key, series in values.items()
                if len(series) >= 2
            }
            if market_values:
                normalized[market] = market_values

        return normalized

    def _attach_market_rs_universe_performances(
        self,
        results: dict[str, StockData],
    ) -> None:
        market_universe = self._compute_market_rs_universe_performances(list(results.values()))
        # Compute the mixed-market flag once per scan so the filter policy is
        # stable across symbols. See app.domain.scanning.mixed_market_policy.
        mixed = is_mixed_market(item.market for item in results.values())
        for item in results.values():
            item.rs_universe_performances = market_universe.get(item.market or "US", {})
            item.is_mixed_market = mixed

    def _is_transient(self, exc: Exception) -> bool:
        """Classify whether an exception is transient (worth retrying)."""
        if isinstance(exc, _TRANSIENT_TYPES):
            return True
        msg = str(exc).lower()
        return any(k in msg for k in ("rate limit", "429", "too many", "timeout", "connection reset", "connection refused"))

    def _fetch_with_retry(self, fn, *args, **kwargs):
        """Call *fn* with exponential-backoff retry on transient errors."""
        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if attempt < self._max_retries and self._is_transient(e):
                    delay = self._retry_base_delay * (2 ** attempt)
                    jitter = random.uniform(0, delay * 0.1)
                    time.sleep(delay + jitter)
                else:
                    raise
        raise last_exc  # pragma: no cover — defensive, loop always re-raises

    def merge_requirements(self, requirements_list: List[DataRequirements]) -> DataRequirements:
        """
        Merge multiple data requirements into a single requirement.

        Takes the union of all needs - if any screener needs a particular
        type of data, it will be fetched.

        Args:
            requirements_list: List of DataRequirements from screeners

        Returns:
            Merged DataRequirements
        """
        if not requirements_list:
            return DataRequirements()

        # Start with first requirement
        merged = requirements_list[0]

        # Merge with remaining requirements
        for req in requirements_list[1:]:
            merged = merged.merge(req)

        return merged

    def prepare_data(
        self,
        symbol: str,
        requirements: DataRequirements,
        *,
        allow_partial: bool = True,
    ) -> StockData:
        """
        Fetch all required data for a symbol.

        Args:
            symbol: Stock symbol
            requirements: Merged data requirements
            allow_partial: If False, raise DataFetchError when any component fails.

        Returns:
            StockData object with all fetched data
        """
        identity = self._resolve_identity(symbol)
        normalized_symbol = identity.normalized_symbol
        canonical_symbol = identity.canonical_symbol
        fetch_errors = {}

        # 1. Fetch price data (always needed)
        # OPTIMIZATION: Check cache first to avoid rate limiting on cache hits
        price_data = None
        try:
            # Check cache first (no rate limiting)
            # Note: get_historical_data() handles Redis → DB fallback internally and ensures sufficient data
            price_data = self.price_cache.get_historical_data(
                canonical_symbol,
                period=requirements.price_period,
            )

            if price_data is None or price_data.empty:
                # Cache miss or insufficient - fetch directly
                # (yfinance_service has its own rate limiter)
                logger.debug(f"Cache MISS for {normalized_symbol} - fetching from yfinance")
                price_data = self._fetch_with_retry(
                    self._yfinance_service.get_historical_data,
                    canonical_symbol,
                    period=requirements.price_period,
                    use_cache=False,  # Already checked cache
                )
            else:
                logger.debug(
                    "Cache HIT for %s (%d days) - no rate limiting",
                    normalized_symbol,
                    len(price_data),
                )

            if price_data is None or price_data.empty:
                fetch_errors["price_data"] = "No price data returned"
        except Exception as e:
            logger.error(f"Error fetching price data for {normalized_symbol}: {e}")
            fetch_errors["price_data"] = str(e)

        # Normalize None market to "US" so single-symbol and bulk paths are
        # consistent (bulk path normalizes at line ~535 for the same reason).
        normalized_market = identity.market or "US"

        # 2. Fetch benchmark data (if needed)
        benchmark_bundle = None
        benchmark_data = None
        if requirements.needs_benchmark:
            try:
                benchmark_bundle = self._fetch_benchmark_bundle(
                    market=normalized_market,
                    period=requirements.price_period,
                )
                benchmark_data = benchmark_bundle.data if benchmark_bundle is not None else None
                if benchmark_data is None or benchmark_data.empty:
                    fetch_errors["benchmark_data"] = "No benchmark data returned"
            except Exception as e:
                logger.error(f"Error fetching benchmark data: {e}")
                fetch_errors["benchmark_data"] = str(e)

        # 3. Fetch fundamentals (if needed, using cache)
        fundamentals = None
        if requirements.needs_fundamentals:
            try:
                # Use cache service instead of direct API call (no rate limiting needed - cache handles it)
                fundamentals = self._fetch_with_retry(
                    self.fundamentals_cache.get_fundamentals,
                    canonical_symbol,
                    force_refresh=False,  # Use cache by default
                )
                if fundamentals is None:
                    fetch_errors["fundamentals"] = "No fundamental data returned"
            except Exception as e:
                logger.warning(f"Error fetching fundamentals for {normalized_symbol}: {e}")
                fetch_errors["fundamentals"] = str(e)

        # 4. Extract quarterly growth from fundamentals (consolidated cache)
        quarterly_growth = None
        if requirements.needs_quarterly_growth and fundamentals:
            quarterly_growth = {
                'eps_growth_qq': fundamentals.get('eps_growth_qq') or fundamentals.get('eps_growth_quarterly'),
                'sales_growth_qq': fundamentals.get('sales_growth_qq'),
                'eps_growth_yy': fundamentals.get('eps_growth_yy'),
                'sales_growth_yy': fundamentals.get('sales_growth_yy'),
                'recent_quarter_date': fundamentals.get('recent_quarter_date'),
                'previous_quarter_date': fundamentals.get('previous_quarter_date'),
            }

        # 5. Earnings history - no longer used (consolidated into fundamentals)
        # CANSLIM now uses eps_growth_yy from fundamentals instead
        earnings_history = None

        # Create StockData container
        stock_data = StockData(
            symbol=canonical_symbol,
            price_data=price_data if price_data is not None else pd.DataFrame(),
            benchmark_data=benchmark_data if benchmark_data is not None else pd.DataFrame(),
            fundamentals=fundamentals,
            quarterly_growth=quarterly_growth,
            earnings_history=earnings_history,
            benchmark_symbol=(
                benchmark_bundle.benchmark_symbol
                if benchmark_bundle is not None
                else None
            ),
            benchmark_role=(
                benchmark_bundle.benchmark_role
                if benchmark_bundle is not None
                else None
            ),
            benchmark_kind=(
                benchmark_bundle.benchmark_kind
                if benchmark_bundle is not None
                else None
            ),
            benchmark_candidates=(
                benchmark_bundle.candidate_symbols
                if benchmark_bundle is not None
                else ()
            ),
            market=normalized_market,
            exchange=identity.exchange,
            currency=identity.currency,
            timezone=identity.timezone,
            local_code=identity.local_code,
            fetch_errors=fetch_errors
        )

        if not allow_partial and fetch_errors:
            raise DataFetchError(canonical_symbol, fetch_errors, partial_data=stock_data)

        return stock_data

    def prepare_data_bulk(
        self,
        symbols: List[str],
        requirements: DataRequirements,
        *,
        allow_partial: bool = True,
        batch_only_prices: bool = False,
        batch_only_fundamentals: bool = False,
    ) -> Dict[str, StockData]:
        """
        Fetch data for multiple stocks efficiently using bulk cache operations.

        Uses Redis pipelines to fetch cache data for all symbols in a single
        network call (10-20x faster than individual lookups).

        This is Phase 3 optimization: for a batch of 50 stocks, instead of
        making 150 individual Redis calls (50 × 3 caches), we make just
        3 pipeline calls (1 per cache type).

        Args:
            symbols: List of stock symbols
            requirements: Merged data requirements
            allow_partial: If False, raise DataFetchError when any component fails.

        Returns:
            Dict mapping symbols to their StockData objects
        """
        if not symbols:
            return {}

        logger.info(f"Preparing data for {len(symbols)} symbols using bulk cache operations")
        resolver = getattr(self, "_security_master", None) or SecurityMasterResolver()
        requested_keys = [resolver.normalize_symbol(symbol) for symbol in symbols]
        identities = [self._resolve_identity(symbol) for symbol in symbols]
        canonical_symbols = [identity.canonical_symbol for identity in identities]
        unique_canonical_symbols = list(dict.fromkeys(canonical_symbols))

        # PHASE 3 OPTIMIZATION: Bulk cache lookups using Redis pipelines
        # Instead of N individual Redis calls, make 2 pipeline calls (1 per cache type)
        # Get cached data for all symbols in parallel
        # Price data is always needed (no needs_price_data flag)
        # IMPORTANT: Pass period so get_many() can fall back to database if Redis only has 30 days
        cached_prices = self.price_cache.get_many(
            unique_canonical_symbols,
            period=requirements.price_period,
        )
        # Fundamentals cache now includes quarterly growth data (consolidated)
        cached_fundamentals = (
            self.fundamentals_cache.get_many(unique_canonical_symbols)
            if requirements.needs_fundamentals
            else {}
        )

        # Get benchmark data once per market (shared by symbols in that market)
        benchmark_bundle_by_market: dict[str, BenchmarkDataBundle | None] = {}
        all_errors: dict[str, str] = {}
        if requirements.needs_benchmark:
            unique_markets = sorted({identity.market or "US" for identity in identities})
            for market in unique_markets:
                try:
                    benchmark_bundle = self._fetch_benchmark_bundle(
                        market=market,
                        period=requirements.price_period,
                    )
                    benchmark_data = benchmark_bundle.data if benchmark_bundle is not None else None
                    if benchmark_data is None or benchmark_data.empty:
                        logger.warning("No benchmark data returned for market %s", market)
                        all_errors[f"<bulk>/benchmark_data:{market}"] = "No benchmark data returned"
                        benchmark_bundle_by_market[market] = None
                    else:
                        benchmark_bundle_by_market[market] = benchmark_bundle
                except Exception as e:
                    logger.warning("Error fetching benchmark data for market %s: %s", market, e)
                    all_errors[f"<bulk>/benchmark_data:{market}"] = str(e)
                    benchmark_bundle_by_market[market] = None

        # Build StockData for each symbol
        results = {}
        for requested_key, identity in zip(requested_keys, identities):
            symbol = identity.canonical_symbol
            fetch_errors = {}

            # Get price data (from cache or fetch)
            # Note: get_many() already handles sufficiency checks and database fallback
            price_data = cached_prices.get(symbol)

            if price_data is None or price_data.empty:
                if batch_only_prices:
                    fetch_errors["price_data"] = "No price data returned from batch-only price path"
                    price_data = pd.DataFrame()
                else:
                    # Cache miss or insufficient - fetch directly
                    # (yfinance_service has its own rate limiter)
                    try:
                        price_data = self._fetch_with_retry(
                            self._yfinance_service.get_historical_data,
                            symbol,
                            period=requirements.price_period,
                        )
                        if price_data is None or price_data.empty:
                            fetch_errors["price_data"] = "No price data returned"
                    except Exception as e:
                        logger.error(f"Error fetching price data for {symbol}: {e}")
                        fetch_errors["price_data"] = str(e)
                        price_data = pd.DataFrame()

            # Get fundamentals (from cache or fetch)
            fundamentals = cached_fundamentals.get(symbol)
            if fundamentals is None and requirements.needs_fundamentals:
                if batch_only_fundamentals:
                    fetch_errors["fundamentals"] = "No fundamental data returned from batch-only fundamentals path"
                else:
                    # Cache miss - fetch
                    try:
                        fundamentals = self._fetch_with_retry(
                            self.fundamentals_cache.get_fundamentals,
                            symbol,
                            force_refresh=False,
                        )
                        if fundamentals is None:
                            fetch_errors["fundamentals"] = "No fundamental data returned"
                    except Exception as e:
                        logger.warning(f"Error fetching fundamentals for {symbol}: {e}")
                        fetch_errors["fundamentals"] = str(e)

            # Extract quarterly growth from fundamentals (consolidated cache)
            quarterly_growth = None
            if requirements.needs_quarterly_growth and fundamentals:
                quarterly_growth = {
                    'eps_growth_qq': fundamentals.get('eps_growth_qq') or fundamentals.get('eps_growth_quarterly'),
                    'sales_growth_qq': fundamentals.get('sales_growth_qq'),
                    'eps_growth_yy': fundamentals.get('eps_growth_yy'),
                    'sales_growth_yy': fundamentals.get('sales_growth_yy'),
                    'recent_quarter_date': fundamentals.get('recent_quarter_date'),
                    'previous_quarter_date': fundamentals.get('previous_quarter_date'),
                }

            # Earnings history - no longer used (consolidated into fundamentals)
            # CANSLIM now uses eps_growth_yy from fundamentals instead
            earnings_history = None

            # Normalize None market to "US" so None and "US" map to the same
            # benchmark bucket (consistent with _attach_market_rs_universe_performances).
            normalized_market = identity.market or "US"

            # Create StockData
            market_benchmark_bundle = (
                benchmark_bundle_by_market.get(normalized_market)
                if requirements.needs_benchmark
                else None
            )
            market_benchmark_data = (
                market_benchmark_bundle.data
                if market_benchmark_bundle is not None
                else None
            )
            if requirements.needs_benchmark and (
                market_benchmark_data is None or market_benchmark_data.empty
            ):
                fetch_errors["benchmark_data"] = (
                    f"No benchmark data returned for market {normalized_market}"
                )
            results[requested_key] = StockData(
                symbol=symbol,
                price_data=price_data if price_data is not None else pd.DataFrame(),
                benchmark_data=(
                    market_benchmark_data
                    if market_benchmark_data is not None
                    else pd.DataFrame()
                ),
                fundamentals=fundamentals,
                quarterly_growth=quarterly_growth,
                earnings_history=earnings_history,
                benchmark_symbol=(
                    market_benchmark_bundle.benchmark_symbol
                    if market_benchmark_bundle is not None
                    else None
                ),
                benchmark_role=(
                    market_benchmark_bundle.benchmark_role
                    if market_benchmark_bundle is not None
                    else None
                ),
                benchmark_kind=(
                    market_benchmark_bundle.benchmark_kind
                    if market_benchmark_bundle is not None
                    else None
                ),
                benchmark_candidates=(
                    market_benchmark_bundle.candidate_symbols
                    if market_benchmark_bundle is not None
                    else ()
                ),
                market=normalized_market,
                exchange=identity.exchange,
                currency=identity.currency,
                timezone=identity.timezone,
                local_code=identity.local_code,
                fetch_errors=fetch_errors
            )

            # Collect per-symbol errors into namespaced dict for bulk error reporting
            for key, msg in fetch_errors.items():
                all_errors[f"{requested_key}/{key}"] = msg

        logger.info(f"Bulk data preparation completed for {len(results)} symbols")

        if requirements.needs_benchmark and results:
            self._attach_market_rs_universe_performances(results)

        if not allow_partial and all_errors:
            raise DataFetchError("<bulk>", all_errors, partial_data=results)

        return results

    def _fetch_fundamentals(self, symbol: str) -> Optional[Dict]:
        """
        Fetch fundamental data for a symbol.

        Returns:
            Dict with fundamental metrics or None
        """
        try:
            import yfinance as yf
            from ..config import settings
            self._rate_limiter.wait("yfinance", min_interval_s=1.0 / settings.yfinance_rate_limit)

            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract key fundamentals
            fundamentals = {
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "dividend_yield": info.get("dividendYield"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "institutional_ownership": info.get("heldPercentInstitutions"),
                "short_ratio": info.get("shortRatio"),
                "short_percent": info.get("shortPercentOfFloat"),
                "beta": info.get("beta"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                # IPO date: yfinance provides firstTradeDateMilliseconds (in ms), convert to seconds
                "first_trade_date": info.get("firstTradeDateMilliseconds") / 1000 if info.get("firstTradeDateMilliseconds") else None,
            }

            return fundamentals

        except Exception as e:
            logger.error(f"Error extracting fundamentals for {symbol}: {e}")
            return None

    def validate_data(self, data: StockData, min_days: int = 100) -> bool:
        """
        Validate that we have sufficient data for screening.

        Args:
            data: StockData to validate
            min_days: Minimum number of price data points required

        Returns:
            True if data is valid for screening
        """
        # Must have price data
        if data.price_data is None or data.price_data.empty:
            logger.warning(f"{data.symbol}: No price data")
            return False

        # Must have minimum number of days
        if len(data.price_data) < min_days:
            logger.warning(
                f"{data.symbol}: Insufficient price data "
                f"({len(data.price_data)} days, need {min_days})"
            )
            return False

        return True
