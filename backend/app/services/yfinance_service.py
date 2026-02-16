"""
yfinance service wrapper for fetching stock data.
"""
import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class YFinanceService:
    """Service for fetching stock data using yfinance."""

    def __init__(self):
        """Initialize yfinance service."""
        pass

    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get basic stock information.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with stock info or None if error
        """
        try:
            from .rate_limiter import rate_limiter
            from ..config import settings
            rate_limiter.wait("yfinance", min_interval_s=1.0 / settings.yfinance_rate_limit)

            logger.info(f"Fetching stock info for {symbol}")
            ticker = yf.Ticker(symbol)
            info = ticker.info

            logger.info(f"Received info response for {symbol}, keys count: {len(info) if info else 0}")

            # Validate response has meaningful data
            if not info or len(info) < 3:
                logger.error(f"Empty or invalid info response for {symbol}. Info keys: {list(info.keys()) if info else 'None'}")
                return None

            # Check for essential fields
            if not info.get("longName") and not info.get("shortName") and not info.get("symbol"):
                logger.error(f"Missing essential fields for {symbol}. Available fields: {list(info.keys())[:10]}")
                return None

            logger.info(f"Successfully fetched stock info for {symbol}: {info.get('longName', 'N/A')}")

            return {
                "symbol": symbol,
                "name": info.get("longName") or info.get("shortName", ""),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
                "pe_ratio": info.get("trailingPE"),
                "price_to_book": info.get("priceToBook"),
                "shares_outstanding": info.get("sharesOutstanding", 0),
                "institutional_ownership": info.get("heldPercentInstitutions", 0) * 100 if info.get("heldPercentInstitutions") else None,
            }
        except Exception as e:
            logger.error(f"Exception fetching info for {symbol}: {type(e).__name__}: {str(e)}", exc_info=True)
            return None

    def get_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get historical price data with optional caching.

        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            use_cache: Whether to use cache (default True for 1d interval and common periods)

        Returns:
            DataFrame with OHLCV data or None if error
        """
        # Only use cache for daily data and common periods (1y, 2y, 5y, max)
        should_cache = (
            use_cache and
            interval == "1d" and
            period in ["1y", "2y", "5y", "max"]
        )

        if should_cache:
            try:
                # Import here to avoid circular dependency at module level
                from .price_cache_service import PriceCacheService

                cache_service = PriceCacheService.get_instance()
                cached_data = cache_service.get_historical_data(symbol, period=period)

                if cached_data is not None:
                    return cached_data

                # If cache returns None, fall through to direct fetch
                logger.debug(f"Cache returned None for {symbol}, fetching directly")

            except Exception as e:
                logger.warning(f"Cache error for {symbol}: {e}, falling back to direct fetch")
                # Fall through to direct fetch on cache error

        # Direct fetch from yfinance (either cache disabled or cache failed)
        try:
            from .rate_limiter import rate_limiter
            from ..config import settings
            rate_limiter.wait("yfinance", min_interval_s=1.0 / settings.yfinance_rate_limit)

            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                logger.warning(f"No historical data for {symbol}")
                return None

            return data

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    def get_price_range(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get 52-week high/low.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with 52-week range data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "high_52w": info.get("fiftyTwoWeekHigh"),
                "low_52w": info.get("fiftyTwoWeekLow"),
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            }

        except Exception as e:
            logger.error(f"Error fetching price range for {symbol}: {e}")
            return None

    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get fundamental data including EPS rating components.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with fundamental data including EPS rating fields
        """
        try:
            from .rate_limiter import rate_limiter
            from ..config import settings
            rate_limiter.wait("yfinance", min_interval_s=1.0 / settings.yfinance_rate_limit)

            ticker = yf.Ticker(symbol)
            info = ticker.info

            result = {
                "symbol": symbol,
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "eps_current": info.get("trailingEps"),
                "revenue_current": info.get("totalRevenue"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "roe": info.get("returnOnEquity"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "institutional_ownership": info.get("heldPercentInstitutions", 0) * 100 if info.get("heldPercentInstitutions") else None,
                "description_yfinance": info.get("longBusinessSummary"),
                # IPO date - yfinance provides this as milliseconds timestamp
                "first_trade_date_ms": info.get("firstTradeDateMilliseconds"),
            }

            # Calculate EPS Rating components from income statements
            eps_data = self._extract_eps_rating_data(ticker)
            result.update(eps_data)

            return result

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return None

    def _extract_eps_rating_data(self, ticker) -> Dict[str, Optional[float]]:
        """
        Extract EPS rating components from income statement data.

        Uses annual and quarterly income statements to calculate:
        - 5-year EPS CAGR
        - Recent quarterly YoY growth
        - Raw composite score for percentile ranking

        Args:
            ticker: yfinance Ticker object

        Returns:
            Dict with EPS rating fields:
            {
                'eps_5yr_cagr': float or None,
                'eps_q1_yoy': float or None,
                'eps_q2_yoy': float or None,
                'eps_raw_score': float or None,
                'eps_years_available': int
            }
        """
        from .eps_rating_service import eps_rating_service

        result = {
            'eps_5yr_cagr': None,
            'eps_q1_yoy': None,
            'eps_q2_yoy': None,
            'eps_raw_score': None,
            'eps_years_available': 0
        }

        try:
            # Get income statements
            annual_income = ticker.income_stmt
            quarterly_income = ticker.quarterly_income_stmt

            # Calculate EPS rating data using the service
            eps_data = eps_rating_service.calculate_eps_rating_data(
                annual_income,
                quarterly_income
            )

            result.update(eps_data)
            logger.debug(f"Extracted EPS rating data: CAGR={result['eps_5yr_cagr']}, Q1={result['eps_q1_yoy']}, Q2={result['eps_q2_yoy']}")

        except Exception as e:
            logger.debug(f"Could not extract EPS rating data: {e}")

        return result

    def get_earnings_history(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get earnings history.

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with earnings history
        """
        try:
            ticker = yf.Ticker(symbol)
            earnings = ticker.earnings_history

            if earnings is None or earnings.empty:
                logger.warning(f"No earnings history for {symbol}")
                return None

            return earnings

        except Exception as e:
            logger.error(f"Error fetching earnings history for {symbol}: {e}")
            return None

    def get_quarterly_growth(self, symbol: str) -> Optional[Dict]:
        """
        Get quarter-over-quarter growth metrics.

        Fetches most recent 2 quarters and calculates Q/Q growth for:
        - EPS (Earnings Per Share)
        - Revenue (Total Revenue)

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict containing:
            {
                'eps_growth_qq': float or None,  # EPS growth % Q/Q
                'sales_growth_qq': float or None,  # Revenue growth % Q/Q
                'recent_quarter_date': str or None,
                'previous_quarter_date': str or None
            }

            Returns None if ticker cannot be fetched.
            Individual metrics are None if data missing.

        Example:
            {
                'eps_growth_qq': 15.5,  # 15.5% growth Q/Q
                'sales_growth_qq': 8.2,  # 8.2% growth Q/Q
                'recent_quarter_date': '2024-Q4',
                'previous_quarter_date': '2024-Q3'
            }
        """
        try:
            ticker = yf.Ticker(symbol)

            result = {
                'eps_growth_qq': None,
                'sales_growth_qq': None,
                'recent_quarter_date': None,
                'previous_quarter_date': None
            }

            # Calculate EPS Growth Q/Q using quarterly income statement
            try:
                quarterly_income = ticker.quarterly_income_stmt

                if quarterly_income is not None and quarterly_income.shape[1] >= 2:
                    # quarterly_income_stmt: line items as index (rows), dates as columns
                    recent_quarter_col = quarterly_income.columns[0]
                    previous_quarter_col = quarterly_income.columns[1]

                    # Store quarter dates
                    result['recent_quarter_date'] = str(recent_quarter_col)
                    result['previous_quarter_date'] = str(previous_quarter_col)

                    # Look for Diluted EPS or Basic EPS row
                    eps_row = None
                    for idx in quarterly_income.index:
                        idx_str = str(idx).lower()
                        if 'diluted eps' in idx_str or 'dilutedeps' in idx_str:
                            eps_row = idx
                            break

                    # If not found, try basic EPS
                    if eps_row is None:
                        for idx in quarterly_income.index:
                            idx_str = str(idx).lower()
                            if 'basic eps' in idx_str or 'basiceps' in idx_str:
                                eps_row = idx
                                break

                    if eps_row is not None:
                        recent_eps = quarterly_income.loc[eps_row, recent_quarter_col]
                        previous_eps = quarterly_income.loc[eps_row, previous_quarter_col]

                        # Calculate growth percentage
                        if previous_eps != 0 and not pd.isna(previous_eps) and not pd.isna(recent_eps):
                            eps_growth = ((recent_eps - previous_eps) / abs(previous_eps)) * 100

                            # Cap at reasonable range to avoid huge percentages from tiny numbers
                            if abs(previous_eps) > 0.05:  # Only if denominator is meaningful
                                result['eps_growth_qq'] = round(float(eps_growth), 2)

            except (KeyError, IndexError, AttributeError) as e:
                logger.debug(f"Could not calculate EPS growth for {symbol}: {e}")

            # Calculate Sales/Revenue Growth Q/Q
            try:
                quarterly_income = ticker.quarterly_income_stmt

                if quarterly_income is not None and quarterly_income.shape[1] >= 2:
                    # quarterly_income_stmt: line items as index (rows), dates as columns
                    # Columns are most recent first
                    recent_quarter_col = quarterly_income.columns[0]
                    previous_quarter_col = quarterly_income.columns[1]

                    # Find revenue row (case-insensitive search)
                    revenue_row = None
                    for idx in quarterly_income.index:
                        idx_str = str(idx).lower()
                        if 'revenue' in idx_str and 'total' in idx_str:
                            revenue_row = idx
                            break

                    # Alternative search if not found
                    if revenue_row is None:
                        for idx in quarterly_income.index:
                            idx_str = str(idx).lower()
                            if 'total revenue' in idx_str or 'totalrevenue' in idx_str:
                                revenue_row = idx
                                break

                    if revenue_row is not None:
                        recent_revenue = quarterly_income.loc[revenue_row, recent_quarter_col]
                        previous_revenue = quarterly_income.loc[revenue_row, previous_quarter_col]

                        # Calculate growth percentage
                        if previous_revenue != 0 and not pd.isna(previous_revenue) and not pd.isna(recent_revenue):
                            sales_growth = ((recent_revenue - previous_revenue) / abs(previous_revenue)) * 100
                            result['sales_growth_qq'] = round(float(sales_growth), 2)

            except (KeyError, IndexError, AttributeError) as e:
                logger.debug(f"Could not calculate sales growth for {symbol}: {e}")

            return result

        except Exception as e:
            logger.warning(f"Error fetching quarterly growth for {symbol}: {e}")
            return None

    def calculate_moving_averages(
        self,
        symbol: str,
        periods: List[int] = [50, 150, 200]
    ) -> Optional[Dict[str, float]]:
        """
        Calculate moving averages.

        Args:
            symbol: Stock ticker symbol
            periods: List of MA periods to calculate

        Returns:
            Dict with MA values
        """
        try:
            # Get enough historical data
            max_period = max(periods) + 30  # Extra buffer
            data = self.get_historical_data(symbol, period=f"{max_period}d")

            if data is None or data.empty or len(data) < max_period:
                logger.warning(f"Insufficient data for MA calculation: {symbol}")
                return None

            result = {
                "current_price": data["Close"].iloc[-1],
            }

            for period in periods:
                ma_value = data["Close"].rolling(window=period).mean().iloc[-1]
                result[f"ma_{period}"] = ma_value

            # Also get MA 200 from 1 month ago for trend
            if 200 in periods and len(data) >= 220:
                ma_200_month_ago = data["Close"].rolling(window=200).mean().iloc[-20]
                result["ma_200_month_ago"] = ma_200_month_ago

            return result

        except Exception as e:
            logger.error(f"Error calculating moving averages for {symbol}: {e}")
            return None

    def get_volume_data(self, symbol: str, period: str = "3mo") -> Optional[Dict[str, Any]]:
        """
        Get volume statistics.

        Args:
            symbol: Stock ticker symbol
            period: Period for volume calculation

        Returns:
            Dict with volume data
        """
        try:
            data = self.get_historical_data(symbol, period=period)

            if data is None or data.empty:
                return None

            return {
                "current_volume": int(data["Volume"].iloc[-1]),
                "avg_volume_50d": int(data["Volume"].tail(50).mean()),
                "avg_volume_20d": int(data["Volume"].tail(20).mean()),
            }

        except Exception as e:
            logger.error(f"Error fetching volume data for {symbol}: {e}")
            return None


# Global service instance
yfinance_service = YFinanceService()
