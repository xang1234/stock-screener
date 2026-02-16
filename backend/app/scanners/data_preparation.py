"""
Data preparation layer for multi-screener architecture.

Centralizes data fetching so that data is fetched once and shared
across all screeners. This eliminates redundant API calls and improves
performance when running multiple screeners on the same stock.
"""
import logging
from typing import List, Dict, Optional
import pandas as pd

from .base_screener import DataRequirements, StockData
from ..services.yfinance_service import yfinance_service
from ..services.benchmark_cache_service import BenchmarkCacheService
from ..services.fundamentals_cache_service import FundamentalsCacheService

logger = logging.getLogger(__name__)


class DataPreparationLayer:
    """
    Centralized data fetching for all screeners.

    Fetches all required data once based on merged requirements from
    multiple screeners, then shares that data across all screeners.
    """

    def __init__(self):
        """Initialize data preparation layer."""
        self.benchmark_cache = BenchmarkCacheService.get_instance()
        self.fundamentals_cache = FundamentalsCacheService.get_instance()

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
        requirements: DataRequirements
    ) -> StockData:
        """
        Fetch all required data for a symbol.

        Args:
            symbol: Stock symbol
            requirements: Merged data requirements

        Returns:
            StockData object with all fetched data
        """
        fetch_errors = {}

        # 1. Fetch price data (always needed)
        # OPTIMIZATION: Check cache first to avoid rate limiting on cache hits
        price_data = None
        try:
            from ..services.price_cache_service import PriceCacheService
            price_cache = PriceCacheService.get_instance()

            # Check cache first (no rate limiting)
            # Note: get_historical_data() handles Redis → DB fallback internally and ensures sufficient data
            price_data = price_cache.get_historical_data(symbol, period=requirements.price_period)

            if price_data is None or price_data.empty:
                # Cache miss or insufficient - fetch directly
                # (yfinance_service has its own rate limiter)
                logger.debug(f"Cache MISS for {symbol} - fetching from yfinance")
                price_data = yfinance_service.get_historical_data(
                    symbol,
                    period=requirements.price_period,
                    use_cache=False  # Already checked cache
                )
            else:
                logger.debug(f"Cache HIT for {symbol} ({len(price_data)} days) - no rate limiting")

            if price_data is None or price_data.empty:
                fetch_errors["price_data"] = "No price data returned"
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            fetch_errors["price_data"] = str(e)

        # 2. Fetch benchmark data (if needed)
        benchmark_data = None
        if requirements.needs_benchmark:
            try:
                benchmark_data = self.benchmark_cache.get_spy_data(
                    period=requirements.price_period
                )
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
                fundamentals = self.fundamentals_cache.get_fundamentals(
                    symbol=symbol,
                    force_refresh=False  # Use cache by default
                )
                if fundamentals is None:
                    fetch_errors["fundamentals"] = "No fundamental data returned"
            except Exception as e:
                logger.warning(f"Error fetching fundamentals for {symbol}: {e}")
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
            symbol=symbol,
            price_data=price_data if price_data is not None else pd.DataFrame(),
            benchmark_data=benchmark_data if benchmark_data is not None else pd.DataFrame(),
            fundamentals=fundamentals,
            quarterly_growth=quarterly_growth,
            earnings_history=earnings_history,
            fetch_errors=fetch_errors
        )

        return stock_data

    def prepare_data_bulk(
        self,
        symbols: List[str],
        requirements: DataRequirements
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

        Returns:
            Dict mapping symbols to their StockData objects
        """
        if not symbols:
            return {}

        logger.info(f"Preparing data for {len(symbols)} symbols using bulk cache operations")

        # PHASE 3 OPTIMIZATION: Bulk cache lookups using Redis pipelines
        # Instead of N individual Redis calls, make 2 pipeline calls (1 per cache type)
        from ..services.price_cache_service import PriceCacheService
        price_cache = PriceCacheService.get_instance()

        # Get cached data for all symbols in parallel
        # Price data is always needed (no needs_price_data flag)
        # IMPORTANT: Pass period so get_many() can fall back to database if Redis only has 30 days
        cached_prices = price_cache.get_many(symbols, period=requirements.price_period)
        # Fundamentals cache now includes quarterly growth data (consolidated)
        cached_fundamentals = self.fundamentals_cache.get_many(symbols) if requirements.needs_fundamentals else {}

        # Get benchmark data once (shared by all stocks)
        benchmark_data = None
        if requirements.needs_benchmark:
            try:
                benchmark_data = self.benchmark_cache.get_spy_data(period=requirements.price_period)
            except Exception as e:
                logger.warning(f"Error fetching benchmark data: {e}")

        # Build StockData for each symbol
        results = {}
        for symbol in symbols:
            fetch_errors = {}

            # Get price data (from cache or fetch)
            # Note: get_many() already handles sufficiency checks and database fallback
            price_data = cached_prices.get(symbol)

            if price_data is None or price_data.empty:
                # Cache miss or insufficient - fetch directly
                # (yfinance_service has its own rate limiter)
                try:
                    price_data = yfinance_service.get_historical_data(
                        symbol,
                        period=requirements.price_period
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
                # Cache miss - fetch
                try:
                    fundamentals = self.fundamentals_cache.get_fundamentals(symbol, force_refresh=False)
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

            # Create StockData
            results[symbol] = StockData(
                symbol=symbol,
                price_data=price_data if price_data is not None else pd.DataFrame(),
                benchmark_data=benchmark_data if benchmark_data is not None else pd.DataFrame(),
                fundamentals=fundamentals,
                quarterly_growth=quarterly_growth,
                earnings_history=earnings_history,
                fetch_errors=fetch_errors
            )

        logger.info(f"Bulk data preparation completed for {len(results)} symbols")
        return results

    def _fetch_fundamentals(self, symbol: str) -> Optional[Dict]:
        """
        Fetch fundamental data for a symbol.

        Returns:
            Dict with fundamental metrics or None
        """
        try:
            import yfinance as yf
            from ..services.rate_limiter import rate_limiter
            from ..config import settings
            rate_limiter.wait("yfinance", min_interval_s=1.0 / settings.yfinance_rate_limit)

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
