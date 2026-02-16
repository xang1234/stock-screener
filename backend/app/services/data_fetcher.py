"""
Unified data fetching service that coordinates between yfinance and Alpha Vantage.
Handles caching and rate limiting.
"""
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session

from .yfinance_service import yfinance_service
from .alphavantage_service import alphavantage_service
from .rate_limiter import rate_limiter
from ..utils.rate_limiter import alphavantage_limiter, alphavantage_quota
from ..models.stock import StockFundamental, StockTechnical, StockIndustry, StockPrice
from ..config import settings

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Unified data fetcher that coordinates between data sources.
    Implements caching and respects rate limits.
    """

    def __init__(self, db: Optional[Session] = None):
        """
        Initialize data fetcher.

        Args:
            db: Database session for caching (optional)
        """
        self.db = db

    def get_stock_fundamentals(
        self,
        symbol: str,
        use_alpha_vantage: bool = False,
        force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get stock fundamental data with caching.

        Args:
            symbol: Stock ticker symbol
            use_alpha_vantage: Whether to use Alpha Vantage (saves quota)
            force_refresh: Force data refresh (ignore cache)

        Returns:
            Dict with fundamental data or None if error
        """
        # Check cache first if database available
        if self.db and not force_refresh:
            cached = self._get_cached_fundamentals(symbol)
            if cached:
                logger.debug(f"Using cached fundamentals for {symbol}")
                return cached

        # Fetch from appropriate source
        if use_alpha_vantage and alphavantage_quota.can_make_request():
            logger.info(f"Fetching fundamentals from Alpha Vantage for {symbol}")
            alphavantage_limiter.wait_if_needed()

            # Get overview data
            overview = alphavantage_service.get_company_overview(symbol)
            if overview:
                alphavantage_quota.record_request()

            # Get earnings data
            earnings = None
            if alphavantage_quota.can_make_request():
                alphavantage_limiter.wait_if_needed()
                earnings = alphavantage_service.get_earnings(symbol)
                if earnings:
                    alphavantage_quota.record_request()

            # Combine data
            if overview:
                data = {**overview}
                if earnings:
                    data.update({
                        "eps_current": earnings.get("eps_current"),
                        "eps_growth_quarterly": earnings.get("eps_growth_quarterly"),
                        "eps_growth_annual": earnings.get("eps_growth_annual"),
                    })

                # Cache the result
                if self.db:
                    self._cache_fundamentals(symbol, data)

                return data

        # Fallback to yfinance
        logger.info(f"Fetching fundamentals from yfinance for {symbol}")
        rate_limiter.wait("yfinance", min_interval_s=1.0 / settings.yfinance_rate_limit)

        data = yfinance_service.get_fundamentals(symbol)

        if data and self.db:
            self._cache_fundamentals(symbol, data)

        return data

    def get_stock_technicals(
        self,
        symbol: str,
        force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get stock technical indicators.

        Args:
            symbol: Stock ticker symbol
            force_refresh: Force data refresh

        Returns:
            Dict with technical data or None if error
        """
        # Check cache
        if self.db and not force_refresh:
            cached = self._get_cached_technicals(symbol)
            if cached:
                logger.debug(f"Using cached technicals for {symbol}")
                return cached

        # Fetch from yfinance
        logger.info(f"Calculating technicals for {symbol}")
        rate_limiter.wait("yfinance", min_interval_s=1.0 / settings.yfinance_rate_limit)

        # Get price range
        price_range = yfinance_service.get_price_range(symbol)
        if not price_range:
            return None

        # Get moving averages
        rate_limiter.wait("yfinance", min_interval_s=1.0 / settings.yfinance_rate_limit)
        mas = yfinance_service.calculate_moving_averages(symbol)
        if not mas:
            return None

        # Get volume data
        rate_limiter.wait("yfinance", min_interval_s=1.0 / settings.yfinance_rate_limit)
        volume = yfinance_service.get_volume_data(symbol)

        # Combine data
        data = {
            "symbol": symbol,
            "current_price": mas.get("current_price"),
            "ma_50": mas.get("ma_50"),
            "ma_150": mas.get("ma_150"),
            "ma_200": mas.get("ma_200"),
            "ma_200_month_ago": mas.get("ma_200_month_ago"),
            "high_52w": price_range.get("high_52w"),
            "low_52w": price_range.get("low_52w"),
            "avg_volume_50d": volume.get("avg_volume_50d") if volume else None,
            "current_volume": volume.get("current_volume") if volume else None,
        }

        # Cache the result
        if self.db:
            self._cache_technicals(symbol, data)

        return data

    def get_industry_classification(self, symbol: str) -> Optional[Dict[str, str]]:
        """
        Get industry classification for a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with sector and industry info
        """
        # Check cache
        if self.db:
            cached = self._get_cached_industry(symbol)
            if cached:
                return cached

        # Fetch from yfinance
        rate_limiter.wait("yfinance", min_interval_s=1.0 / settings.yfinance_rate_limit)
        info = yfinance_service.get_stock_info(symbol)

        if not info:
            return None

        classification = {
            "symbol": symbol,
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
        }

        # Cache the result
        if self.db:
            self._cache_industry(symbol, classification)

        return classification

    # Cache helper methods

    def _get_cached_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached fundamental data."""
        try:
            cached = self.db.query(StockFundamental).filter(
                StockFundamental.symbol == symbol
            ).first()

            if not cached:
                return None

            # Check if cache is stale (7 days for fundamentals)
            cache_age = datetime.now() - cached.updated_at.replace(tzinfo=None)
            if cache_age > timedelta(days=7):
                logger.debug(f"Cache stale for {symbol} ({cache_age.days} days old)")
                return None

            return {
                "symbol": cached.symbol,
                "market_cap": cached.market_cap,
                "pe_ratio": cached.pe_ratio,
                "peg_ratio": cached.peg_ratio,
                "eps_current": cached.eps_current,
                "eps_growth_quarterly": cached.eps_growth_quarterly,
                "eps_growth_annual": cached.eps_growth_annual,
                # Add other fields as needed
            }

        except Exception as e:
            logger.error(f"Error reading cache for {symbol}: {e}")
            return None

    def _cache_fundamentals(self, symbol: str, data: Dict[str, Any]):
        """Cache fundamental data."""
        try:
            # Upsert fundamental data
            fundamental = self.db.query(StockFundamental).filter(
                StockFundamental.symbol == symbol
            ).first()

            if fundamental:
                # Update existing
                for key, value in data.items():
                    if hasattr(fundamental, key):
                        setattr(fundamental, key, value)
            else:
                # Create new
                fundamental = StockFundamental(**data)
                self.db.add(fundamental)

            self.db.commit()
            logger.debug(f"Cached fundamentals for {symbol}")

        except Exception as e:
            logger.error(f"Error caching fundamentals for {symbol}: {e}")
            self.db.rollback()

    def _get_cached_technicals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached technical data."""
        try:
            cached = self.db.query(StockTechnical).filter(
                StockTechnical.symbol == symbol
            ).first()

            if not cached:
                return None

            # Check if cache is stale (1 day for technicals)
            cache_age = datetime.now() - cached.updated_at.replace(tzinfo=None)
            if cache_age > timedelta(hours=settings.cache_ttl_hours):
                return None

            return {
                "symbol": cached.symbol,
                "current_price": cached.current_price,
                "ma_50": cached.ma_50,
                "ma_150": cached.ma_150,
                "ma_200": cached.ma_200,
                "ma_200_month_ago": cached.ma_200_month_ago,
                "high_52w": cached.high_52w,
                "low_52w": cached.low_52w,
                "avg_volume_50d": cached.avg_volume_50d,
                "current_volume": cached.current_volume,
                "rs_rating": cached.rs_rating,
                "stage": cached.stage,
                "vcp_score": cached.vcp_score,
            }

        except Exception as e:
            logger.error(f"Error reading technical cache for {symbol}: {e}")
            return None

    def _cache_technicals(self, symbol: str, data: Dict[str, Any]):
        """Cache technical data."""
        try:
            technical = self.db.query(StockTechnical).filter(
                StockTechnical.symbol == symbol
            ).first()

            if technical:
                for key, value in data.items():
                    if hasattr(technical, key):
                        setattr(technical, key, value)
            else:
                technical = StockTechnical(**data)
                self.db.add(technical)

            self.db.commit()
            logger.debug(f"Cached technicals for {symbol}")

        except Exception as e:
            logger.error(f"Error caching technicals for {symbol}: {e}")
            self.db.rollback()

    def _get_cached_industry(self, symbol: str) -> Optional[Dict[str, str]]:
        """Get cached industry classification."""
        try:
            cached = self.db.query(StockIndustry).filter(
                StockIndustry.symbol == symbol
            ).first()

            if not cached:
                return None

            return {
                "symbol": cached.symbol,
                "sector": cached.sector,
                "industry": cached.industry,
            }

        except Exception as e:
            logger.error(f"Error reading industry cache for {symbol}: {e}")
            return None

    def _cache_industry(self, symbol: str, data: Dict[str, str]):
        """Cache industry classification."""
        try:
            industry = self.db.query(StockIndustry).filter(
                StockIndustry.symbol == symbol
            ).first()

            if industry:
                for key, value in data.items():
                    if hasattr(industry, key):
                        setattr(industry, key, value)
            else:
                industry = StockIndustry(**data)
                self.db.add(industry)

            self.db.commit()
            logger.debug(f"Cached industry for {symbol}")

        except Exception as e:
            logger.error(f"Error caching industry for {symbol}: {e}")
            self.db.rollback()
