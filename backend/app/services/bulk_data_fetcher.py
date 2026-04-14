"""
Bulk data fetcher for Yahoo-first batch price ingestion.

Scheduled/background price ingestion must use batched ``yf.download()`` only.
Per-symbol Yahoo metadata/history fetches remain legacy fallback tools and are
not used by the hot-path price refresh jobs.

Canonical price contract ADR:
docs/learning_loop/adr_ll2_e1_canonical_price_contract_v1.md
"""
import logging
from typing import TYPE_CHECKING, List, Dict, Optional, Any
from threading import RLock
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ..config import settings
from .growth_cadence_service import compute_cadence_aware_growth

if TYPE_CHECKING:
    from .eps_rating_service import EPSRatingService
    from .rate_limiter import RedisRateLimiter

logger = logging.getLogger(__name__)


class BulkDataFetcher:
    """
    Fetch data for multiple stocks efficiently using ``yf.download()``.
    """

    DEFAULT_PRICE_BATCH_SIZE = 100
    MAX_PRICE_BATCH_SIZE = 200
    MIN_PRICE_BATCH_SIZE = 25
    PRICE_BATCH_GROWTH_STEP = 25
    PRICE_BATCH_SUCCESS_STREAK_TO_GROW = 5
    PRICE_BATCH_GROWTH_COOLDOWN_BATCHES = 3
    PRICE_BATCH_RETRY_BACKOFF_SECONDS = (30, 60, 120)
    _fallback_lock = RLock()
    _fallback_rate_limiter: "RedisRateLimiter | None" = None

    def __init__(
        self,
        *,
        rate_limiter: "RedisRateLimiter | None" = None,
        eps_rating_service: "EPSRatingService | None" = None,
    ) -> None:
        # Keep standalone construction safe for tests/scripts while allowing
        # process-scoped injection from runtime wiring where needed.
        self._rate_limiter = rate_limiter or self._get_fallback_rate_limiter()
        self._eps_rating_service = (
            eps_rating_service or self._build_default_eps_rating_service()
        )

    @classmethod
    def _get_fallback_rate_limiter(cls) -> "RedisRateLimiter":
        with cls._fallback_lock:
            if cls._fallback_rate_limiter is None:
                from .rate_limiter import RedisRateLimiter

                cls._fallback_rate_limiter = RedisRateLimiter()
            return cls._fallback_rate_limiter

    @staticmethod
    def _build_default_eps_rating_service() -> "EPSRatingService":
        from .eps_rating_service import EPSRatingService

        return EPSRatingService()

    @staticmethod
    def _build_error_result(symbol: str, error: str) -> Dict[str, Any]:
        return {
            'symbol': symbol,
            'price_data': None,
            'info': None,
            'fundamentals': None,
            'has_error': True,
            'error': error,
        }

    @staticmethod
    def _is_rate_limit_error(error: str) -> bool:
        lower = (error or "").lower()
        return any(indicator in lower for indicator in ("rate", "429", "too many", "limit", "throttl"))

    def _transient_failure_rate(self, results: Dict[str, Dict[str, Any]]) -> float:
        if not results:
            return 1.0

        transient_failures = 0
        for data in results.values():
            if not data.get("has_error"):
                continue
            error = data.get("error", "")
            if self._is_rate_limit_error(error) or "empty" in error.lower():
                transient_failures += 1
        return transient_failures / len(results)

    def fetch_batch_data(
        self,
        symbols: List[str],
        period: str = '2y',
        include_fundamentals: bool = True,
        delay_per_ticker: float = 0.0
    ) -> Dict[str, Dict]:
        """
        Deprecated compatibility wrapper for legacy bulk price callers.

        Scheduled/background jobs must not request fundamentals through this path.
        """
        if not symbols:
            return {}

        if include_fundamentals:
            raise RuntimeError(
                "Bulk fundamentals fetching through fetch_batch_data() is disabled. "
                "Use the provider snapshot pipeline or explicit single-symbol fallback."
            )

        logger.warning(
            "fetch_batch_data() is deprecated for scheduled price ingestion. "
            "Delegating to fetch_prices_in_batches() for %d symbols.",
            len(symbols),
        )
        return self.fetch_prices_in_batches(symbols, period=period)

    def _extract_fundamentals(self, info: Dict) -> Dict:
        """
        Extract fundamental data from ticker info.

        Args:
            info: Ticker info dictionary from yfinance

        Returns:
            Dict with cleaned fundamental metrics
        """
        if not info:
            return {}

        fundamentals = {
            # Market data
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            'shares_outstanding': info.get('sharesOutstanding'),
            'shares_float': info.get('floatShares'),

            # Valuation metrics
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'ev_sales': info.get('enterpriseToRevenue'),
            'ev_ebitda': info.get('enterpriseToEbitda'),

            # Profitability metrics
            'eps_current': info.get('trailingEps'),
            'forward_eps': info.get('forwardEps'),
            'roe': self._convert_to_percentage(info.get('returnOnEquity')),
            'roa': self._convert_to_percentage(info.get('returnOnAssets')),
            'profit_margin': self._convert_to_percentage(info.get('profitMargins')),
            'operating_margin': self._convert_to_percentage(info.get('operatingMargins')),
            'gross_margin': self._convert_to_percentage(info.get('grossMargins')),

            # Growth metrics (as percentages)
            'revenue_growth': self._convert_to_percentage(info.get('revenueGrowth')),
            'earnings_growth': self._convert_to_percentage(info.get('earningsGrowth')),
            'earnings_quarterly_growth': self._convert_to_percentage(info.get('earningsQuarterlyGrowth')),

            # Revenue
            'revenue_current': info.get('totalRevenue'),
            'net_income': info.get('netIncomeToCommon'),

            # Dividend metrics
            'dividend_ttm': info.get('dividendRate'),
            'dividend_yield': self._convert_to_percentage(info.get('dividendYield')),
            'payout_ratio': self._convert_to_percentage(info.get('payoutRatio')),

            # Financial health
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'debt_to_equity': info.get('debtToEquity'),
            'total_cash': info.get('totalCash'),
            'total_debt': info.get('totalDebt'),
            'free_cashflow': info.get('freeCashflow'),
            'operating_cashflow': info.get('operatingCashflow'),

            # Ownership (convert to percentage)
            'institutional_ownership': self._convert_to_percentage(info.get('heldPercentInstitutions')),
            'insider_ownership': self._convert_to_percentage(info.get('heldPercentInsiders')),

            # Stock info
            'beta': info.get('beta'),
            'week_52_high': info.get('fiftyTwoWeekHigh'),
            'week_52_low': info.get('fiftyTwoWeekLow'),
            'avg_volume': info.get('averageVolume'),
            'avg_volume_10d': info.get('averageVolume10days'),
            'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),

            # Company info
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'employees': info.get('fullTimeEmployees'),
            'country': info.get('country'),

            # Analyst info
            'target_price': info.get('targetMeanPrice'),
            'recommendation': info.get('recommendationKey'),

            # Company description
            'description_yfinance': info.get('longBusinessSummary'),

            # Mark data source
            'data_source': 'yfinance',
            'data_source_timestamp': datetime.utcnow().isoformat(),
        }

        # Remove None values to save space
        return {k: v for k, v in fundamentals.items() if v is not None}

    def _convert_to_percentage(self, value: Optional[float]) -> Optional[float]:
        """Convert decimal to percentage (0.15 -> 15.0)."""
        if value is None:
            return None
        return round(value * 100, 2)

    def _extract_quarterly_growth(
        self,
        ticker,
        *,
        market: str | None = None,
    ) -> Dict[str, Optional[float]]:
        """
        Extract cadence-aware growth metrics from ticker income statement.

        Args:
            ticker: yfinance Ticker object
            market: Optional market code for cadence-aware basis selection.

        Returns:
            Dict with quarterly growth metrics
        """
        try:
            quarterly_income = ticker.quarterly_income_stmt
            return compute_cadence_aware_growth(quarterly_income, market=market)

        except Exception as e:
            logger.debug(f"Error extracting quarterly growth: {e}")
            return compute_cadence_aware_growth(None, market=market)

    def _extract_eps_rating_data(self, ticker) -> Dict[str, Optional[float]]:
        """
        Extract EPS rating data from ticker's annual and quarterly income statements.

        Uses EPSRatingService to calculate:
        - 5-year CAGR from annual income statement
        - Q1 and Q2 YoY growth from quarterly income statement
        - Raw composite score

        Args:
            ticker: yfinance Ticker object

        Returns:
            Dict with EPS rating components:
            {
                'eps_5yr_cagr': float or None,
                'eps_q1_yoy': float or None,
                'eps_q2_yoy': float or None,
                'eps_raw_score': float or None,
                'eps_years_available': int
            }
        """
        result = {
            'eps_5yr_cagr': None,
            'eps_q1_yoy': None,
            'eps_q2_yoy': None,
            'eps_raw_score': None,
            'eps_years_available': 0
        }

        try:
            # Get annual income statement (for 5-year CAGR)
            annual_income = ticker.income_stmt

            # Get quarterly income statement (for YoY comparisons)
            quarterly_income = ticker.quarterly_income_stmt

            # Use EPS rating service to calculate all components
            eps_data = self._eps_rating_service.calculate_eps_rating_data(
                annual_income,
                quarterly_income
            )

            result.update(eps_data)
            logger.debug(f"Extracted EPS rating data: CAGR={result['eps_5yr_cagr']}, Q1={result['eps_q1_yoy']}, Q2={result['eps_q2_yoy']}")

        except Exception as e:
            logger.debug(f"Error extracting EPS rating data: {e}")

        return result

    def fetch_batch_prices(
        self,
        symbols: List[str],
        period: str = '2y',
    ) -> Dict[str, Dict]:
        """
        Fetch price data for multiple symbols using yf.download() (single HTTP request).

        This is significantly faster than fetch_batch_data() which calls ticker.history()
        per symbol with per-ticker delays. yf.download() batches all symbols into one
        request and returns data for all tickers at once.

        Args:
            symbols: List of ticker symbols to fetch
            period: Time period for historical data (default: 2y)

        Returns:
            Dict mapping symbols to their data:
            {
                'AAPL': {
                    'symbol': 'AAPL',
                    'price_data': pd.DataFrame,
                    'info': None,
                    'fundamentals': None,
                    'has_error': False
                },
                ...
            }
        """
        if not symbols:
            return {}

        logger.info(f"Batch fetching prices for {len(symbols)} symbols using yf.download()")

        results = {}

        try:
            # yf.download with group_by='ticker' returns MultiIndex columns: (ticker, OHLCV)
            # threads=False avoids threading issues inside Celery workers
            # progress=False suppresses the tqdm progress bar
            raw = yf.download(
                tickers=symbols,
                period=period,
                group_by='ticker',
                threads=False,
                progress=False,
                auto_adjust=False,
                actions=True,
            )

            if raw is None or raw.empty:
                logger.warning("yf.download returned empty DataFrame")
                for symbol in symbols:
                    results[symbol] = self._build_error_result(symbol, 'yf.download returned empty')
                return results

            # Single symbol: yf.download returns flat columns (Open, High, etc.)
            # Multi-symbol: returns MultiIndex columns (AAPL/Open, AAPL/High, etc.)
            if len(symbols) == 1:
                symbol = symbols[0]
                df = raw.copy()
                if df is not None and not df.empty:
                    results[symbol] = {
                        'symbol': symbol, 'price_data': df, 'info': None,
                        'fundamentals': None, 'has_error': False, 'error': None
                    }
                else:
                    results[symbol] = self._build_error_result(symbol, 'No data returned')
            else:
                # Multi-symbol: split by ticker
                for symbol in symbols:
                    try:
                        if symbol in raw.columns.get_level_values(0):
                            df = raw[symbol].copy()
                            # Drop rows where all values are NaN (symbol had no data for that date)
                            df = df.dropna(how='all')
                            if not df.empty:
                                results[symbol] = {
                                    'symbol': symbol, 'price_data': df, 'info': None,
                                    'fundamentals': None, 'has_error': False, 'error': None
                                }
                            else:
                                results[symbol] = self._build_error_result(symbol, 'No data after filtering NaN rows')
                        else:
                            results[symbol] = self._build_error_result(symbol, 'Symbol not in download results')
                    except Exception as e:
                        logger.warning(f"Error extracting {symbol} from download: {e}")
                        results[symbol] = self._build_error_result(symbol, str(e))

            success = len([r for r in results.values() if not r['has_error']])
            failed = len(results) - success
            logger.info(f"Batch price download: {success} successful, {failed} failed")

            # Add missing symbols as errors
            for symbol in symbols:
                if symbol not in results:
                    results[symbol] = self._build_error_result(symbol, 'Missing from results')

            return results

        except Exception as e:
            logger.error(f"yf.download batch failed: {e}", exc_info=True)
            return {
                symbol: self._build_error_result(symbol, f"Batch download error: {str(e)}")
                for symbol in symbols
            }

    def fetch_prices_in_batches(
        self,
        symbols: List[str],
        period: str = '2y',
        start_batch_size: Optional[int] = None,
        market: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """Fetch prices for many symbols using adaptive ``yf.download()`` batches.

        Background jobs should call this method instead of any per-symbol Yahoo path.

        When ``market`` is supplied, per-market rate budget keys
        (``yfinance:hk`` / ``yfinance:batch:hk``) and per-market batch sizes
        from RateBudgetPolicy take effect, so cross-market refreshes don't
        starve each other of provider tokens.
        """
        if not symbols:
            return {}

        # Per-market initial batch size from RateBudgetPolicy when caller
        # didn't pass an explicit ``start_batch_size``.
        if start_batch_size is None and market is not None:
            from .rate_budget_policy import get_rate_budget_policy
            start_batch_size = get_rate_budget_policy().get_batch_size("yfinance", market)

        batch_size = max(
            self.MIN_PRICE_BATCH_SIZE,
            min(start_batch_size or self.DEFAULT_PRICE_BATCH_SIZE, self.MAX_PRICE_BATCH_SIZE),
        )
        success_streak = 0
        growth_cooldown_remaining = 0
        combined_results: Dict[str, Dict] = {}
        batch_start = 0
        while batch_start < len(symbols):
            batch_symbols = symbols[batch_start:batch_start + batch_size]
            batch_results = self._fetch_price_batch_with_retries(
                batch_symbols,
                period=period,
                initial_batch_size=batch_size,
            )
            combined_results.update(batch_results)

            failure_rate = self._transient_failure_rate(batch_results)
            if failure_rate > 0.20:
                success_streak = 0
                batch_size = max(self.MIN_PRICE_BATCH_SIZE, batch_size // 2)
                growth_cooldown_remaining = self.PRICE_BATCH_GROWTH_COOLDOWN_BATCHES
            else:
                if growth_cooldown_remaining > 0:
                    growth_cooldown_remaining -= 1
                if failure_rate < 0.02:
                    success_streak += 1
                    if growth_cooldown_remaining == 0 and success_streak >= self.PRICE_BATCH_SUCCESS_STREAK_TO_GROW:
                        batch_size = min(
                            self.MAX_PRICE_BATCH_SIZE,
                            batch_size + self.PRICE_BATCH_GROWTH_STEP,
                        )
                        success_streak = 0
                else:
                    success_streak = 0

            if batch_start + len(batch_symbols) < len(symbols):
                # Per-market batch interval when market is supplied; falls
                # back to the legacy global key for shared/unmarked calls.
                if market is not None:
                    self._rate_limiter.wait_for_market("yfinance:batch", market)
                else:
                    self._rate_limiter.wait(
                        "yfinance:batch",
                        min_interval_s=settings.yfinance_batch_rate_limit_interval,
                    )
            batch_start += len(batch_symbols)

        return combined_results

    def _fetch_price_batch_with_retries(
        self,
        symbols: List[str],
        *,
        period: str,
        initial_batch_size: int,
    ) -> Dict[str, Dict]:
        """Retry transient batch failures using degraded sub-batches."""
        current_batch_size = max(
            self.MIN_PRICE_BATCH_SIZE,
            min(initial_batch_size, self.MAX_PRICE_BATCH_SIZE),
        )
        last_results: Dict[str, Dict] = {
            symbol: self._build_error_result(symbol, "Batch not attempted")
            for symbol in symbols
        }

        for attempt in range(len(self.PRICE_BATCH_RETRY_BACKOFF_SECONDS) + 1):
            attempt_results: Dict[str, Dict] = {}
            for chunk_start in range(0, len(symbols), current_batch_size):
                chunk_symbols = symbols[chunk_start:chunk_start + current_batch_size]
                attempt_results.update(self.fetch_batch_prices(chunk_symbols, period=period))

            last_results = attempt_results
            failure_rate = self._transient_failure_rate(attempt_results)
            if failure_rate <= 0.20 or attempt == len(self.PRICE_BATCH_RETRY_BACKOFF_SECONDS):
                return attempt_results

            current_batch_size = max(self.MIN_PRICE_BATCH_SIZE, current_batch_size // 2)
            wait_seconds = self.PRICE_BATCH_RETRY_BACKOFF_SECONDS[attempt]
            logger.warning(
                "Transient Yahoo batch failure rate %.1f%% for %d symbols; "
                "retrying with batch size %d after %ds (attempt %d/%d)",
                failure_rate * 100,
                len(symbols),
                current_batch_size,
                wait_seconds,
                attempt + 1,
                len(self.PRICE_BATCH_RETRY_BACKOFF_SECONDS),
            )
            time.sleep(wait_seconds)

        return last_results

    def fetch_batch_with_cache_check(
        self,
        symbols: List[str],
        cached_data: Dict[str, any],
        period: str = '2y'
    ) -> Dict[str, Dict]:
        """
        Fetch data for symbols, using cached data where available.

        Args:
            symbols: List of symbols to fetch
            cached_data: Dict of already cached symbol data
            period: Time period for historical data

        Returns:
            Combined dict of cached + freshly fetched data
        """
        # Identify which symbols need fetching
        cache_misses = [s for s in symbols if s not in cached_data or cached_data[s] is None]

        if not cache_misses:
            logger.info(f"All {len(symbols)} symbols served from cache")
            return cached_data

        logger.info(
            f"Cache: {len(symbols) - len(cache_misses)} hits, "
            f"{len(cache_misses)} misses - fetching {len(cache_misses)} symbols"
        )

        # Fetch missing symbols in bulk
        fresh_data = self.fetch_prices_in_batches(cache_misses, period=period)

        # Combine cached and fresh data
        combined = {**cached_data}
        combined.update(fresh_data)

        return combined

    def fetch_batch_fundamentals(
        self,
        symbols: List[str],
        batch_size: int = 50,
        include_quarterly: bool = True,
        delay_between_batches: float = 2.0,
        delay_per_ticker: float = 1.5,
        market_by_symbol: Optional[Dict[str, str]] = None,
        market: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        Fetch fundamentals for multiple symbols efficiently.

        This method is optimized for fundamental data fetching only (no price history),
        making it much faster for bulk fundamental updates.

        Args:
            symbols: List of ticker symbols to fetch
            batch_size: Number of symbols per batch (default 50)
            include_quarterly: Whether to fetch quarterly growth data (slower but more complete)
            delay_between_batches: Seconds to wait between batches (default 2.0)
            delay_per_ticker: Seconds to wait between individual ticker info fetches (default 0.2)
            market_by_symbol: Optional mapping ``{symbol: market}`` for
                cadence-aware growth extraction.

        Returns:
            Dict mapping symbols to their fundamental data
        """
        if not symbols:
            return {}

        # Per-market batch sizing + backoff. Without market
        # the legacy global behavior preserves; with market, RateBudgetPolicy
        # takes over.
        backoff_base_s = 60
        backoff_max_s = 480
        backoff_factor = 2.0
        if market is not None:
            from .rate_budget_policy import get_rate_budget_policy
            _policy = get_rate_budget_policy()
            batch_size = _policy.get_batch_size("yfinance", market)
            _bp = _policy.get_backoff_params("yfinance", market)
            backoff_base_s = _bp["base_s"]
            backoff_max_s = _bp["max_s"]
            backoff_factor = _bp["factor"]

        logger.info(
            f"Fetching fundamentals for {len(symbols)} symbols "
            f"(market={market or 'shared'}, batch_size={batch_size}, "
            f"delay={delay_per_ticker}s/ticker)"
        )

        all_results = {}
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        consecutive_backoffs = 0  # Track consecutive rate-limited batches

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(symbols))
            batch_symbols = symbols[start_idx:end_idx]

            logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_symbols)} symbols)")

            batch_rate_limit_failures = 0

            try:
                # Use yf.Tickers for efficient batch fetching
                tickers = yf.Tickers(' '.join(batch_symbols))

                for i, symbol in enumerate(batch_symbols):
                    try:
                        ticker = tickers.tickers[symbol]

                        # Get info (fundamentals)
                        info = ticker.info
                        fundamentals = self._extract_fundamentals(info)

                        # Optionally get quarterly growth
                        if include_quarterly:
                            quarterly = self._extract_quarterly_growth(
                                ticker,
                                market=(market_by_symbol or {}).get(symbol),
                            )
                            fundamentals.update(quarterly)

                            # Also extract EPS rating data
                            eps_rating_data = self._extract_eps_rating_data(ticker)
                            fundamentals.update(eps_rating_data)

                        all_results[symbol] = fundamentals

                    except Exception as e:
                        error_str = str(e).lower()
                        logger.warning(f"Error fetching fundamentals for {symbol}: {e}")
                        all_results[symbol] = {'has_error': True, 'error': str(e)}
                        if any(ind in error_str for ind in ("rate", "429", "too many", "limit", "throttl")):
                            batch_rate_limit_failures += 1
                            if market is not None:
                                from .rate_budget_policy import get_rate_budget_policy
                                get_rate_budget_policy().record_429("yfinance", market)

                    # Rate limit between individual ticker fetches within batch
                    if i < len(batch_symbols) - 1 and delay_per_ticker > 0:
                        if market is not None:
                            self._rate_limiter.wait_for_market("yfinance", market)
                        else:
                            self._rate_limiter.wait("yfinance", min_interval_s=delay_per_ticker)

            except Exception as e:
                logger.error(f"Batch error: {e}")
                for symbol in batch_symbols:
                    all_results[symbol] = {'has_error': True, 'error': str(e)}

            # Batch-level backoff: if >50% of batch hit rate limits, back off
            batch_failure_rate = batch_rate_limit_failures / len(batch_symbols) if batch_symbols else 0
            if batch_failure_rate > 0.5:
                consecutive_backoffs += 1
                backoff_time = min(
                    backoff_base_s * (backoff_factor ** (consecutive_backoffs - 1)),
                    backoff_max_s,
                )
                logger.warning(
                    f"Rate limited: {batch_rate_limit_failures}/{len(batch_symbols)} symbols in batch {batch_num + 1}. "
                    f"Backing off {backoff_time}s before next batch (market={market or 'shared'})."
                )
                time.sleep(backoff_time)
            else:
                consecutive_backoffs = 0  # Reset on successful batch
                # Normal rate limit between batches
                if batch_num < total_batches - 1:
                    if market is not None:
                        self._rate_limiter.wait_for_market("yfinance:batch", market)
                    else:
                        self._rate_limiter.wait("yfinance:batch", min_interval_s=delay_between_batches)

        success_count = len([r for r in all_results.values() if not r.get('has_error', False)])
        logger.info(f"Batch fundamentals complete: {success_count}/{len(symbols)} successful")

        return all_results

    def fetch_fundamentals_parallel(
        self,
        symbols: List[str],
        batch_size: int = 50,
        max_workers: int = 3,
        include_quarterly: bool = True,
        delay_per_ticker: float = 1.5,
        market_by_symbol: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict]:
        """
        Fetch fundamentals using parallel batch processing.

        Splits symbols into batches and processes batches in parallel
        using ThreadPoolExecutor for faster throughput.

        Args:
            symbols: List of ticker symbols
            batch_size: Symbols per batch
            max_workers: Number of parallel workers
            include_quarterly: Whether to include quarterly growth data
            delay_per_ticker: Seconds between individual ticker fetches (default 0.2)
            market_by_symbol: Optional mapping ``{symbol: market}`` for
                cadence-aware growth extraction.

        Returns:
            Dict mapping symbols to fundamental data
        """
        if not symbols:
            return {}

        # Split into batches
        batches = [
            symbols[i:i + batch_size]
            for i in range(0, len(symbols), batch_size)
        ]

        logger.info(f"Parallel fetch: {len(symbols)} symbols in {len(batches)} batches with {max_workers} workers")

        all_results = {}

        def fetch_batch(batch_symbols: List[str]) -> Dict[str, Dict]:
            """Fetch a single batch."""
            batch_results = {}
            try:
                tickers = yf.Tickers(' '.join(batch_symbols))

                for i, symbol in enumerate(batch_symbols):
                    try:
                        ticker = tickers.tickers[symbol]
                        info = ticker.info
                        fundamentals = self._extract_fundamentals(info)

                        if include_quarterly:
                            quarterly = self._extract_quarterly_growth(
                                ticker,
                                market=(market_by_symbol or {}).get(symbol),
                            )
                            fundamentals.update(quarterly)

                            # Also extract EPS rating data
                            eps_rating_data = self._extract_eps_rating_data(ticker)
                            fundamentals.update(eps_rating_data)

                        batch_results[symbol] = fundamentals

                    except Exception as e:
                        logger.debug(f"Error for {symbol}: {e}")
                        batch_results[symbol] = {'has_error': True}

                    # Rate limit between individual ticker fetches
                    if i < len(batch_symbols) - 1 and delay_per_ticker > 0:
                        self._rate_limiter.wait("yfinance", min_interval_s=delay_per_ticker)

            except Exception as e:
                logger.warning(f"Batch error: {e}")
                for s in batch_symbols:
                    batch_results[s] = {'has_error': True}

            return batch_results

        # Execute batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_batch, batch): batch for batch in batches}

            for future in as_completed(futures):
                batch_result = future.result()
                all_results.update(batch_result)

        success_count = len([r for r in all_results.values() if not r.get('has_error', False)])
        logger.info(f"Parallel fetch complete: {success_count}/{len(symbols)} successful")

        return all_results
