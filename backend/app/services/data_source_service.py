"""
Data Source Service

Coordinates between finvizfinance (primary) and yfinance (fallback) data sources.
"""
import logging
from typing import Dict, Optional
from datetime import datetime

from .finviz_service import finviz_service
from .yfinance_service import yfinance_service
from .finviz_validator import FinvizValidator

logger = logging.getLogger(__name__)


class DataSourceService:
    """
    Coordinates data fetching from multiple sources with intelligent fallback.

    Strategy:
    1. Try finvizfinance first (faster, more complete)
    2. Validate finvizfinance data
    3. Fall back to yfinance if finvizfinance fails or validation fails
    4. Track metrics for monitoring
    """

    def __init__(
        self,
        prefer_finviz: bool = True,
        enable_fallback: bool = True,
        strict_validation: bool = True,
    ):
        """
        Initialize DataSourceService.

        Args:
            prefer_finviz: If True, try finviz first (default: True)
            enable_fallback: If True, fall back to yfinance on errors (default: True)
            strict_validation: If True, use strict validation rules (default: True)
        """
        self.prefer_finviz = prefer_finviz
        self.enable_fallback = enable_fallback
        self.strict_validation = strict_validation
        self.validator = FinvizValidator()

        # Metrics tracking
        self.metrics = {
            'finviz_success': 0,
            'finviz_failed': 0,
            'yfinance_fallback': 0,
            'yfinance_primary': 0,
            'total_calls': 0,
        }

    def get_fundamentals(self, symbol: str) -> Optional[Dict]:
        """
        Fetch fundamental data with intelligent source selection and fallback.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with fundamental metrics and metadata, or None if all sources fail
        """
        self.metrics['total_calls'] += 1

        # Try finvizfinance first if preferred
        if self.prefer_finviz:
            logger.debug(f"Attempting to fetch {symbol} fundamentals from finvizfinance")

            finviz_data = finviz_service.get_fundamentals(symbol)

            if finviz_data:
                # Validate data (range checks produce warnings, not blocking errors)
                is_valid, errors = self.validator.validate_fundamentals(finviz_data)

                if not is_valid:
                    # Log warning but still use the data
                    logger.warning(f"Range validation warnings for {symbol} fundamentals: {errors}")

                self.metrics['finviz_success'] += 1
                logger.info(f"Using finvizfinance data for {symbol} fundamentals")
                finviz_data['data_source'] = 'finviz'
                finviz_data['data_source_timestamp'] = datetime.utcnow()

                # Supplement with EPS rating data from yfinance (finviz doesn't have income statements)
                eps_data = self._get_eps_rating_data(symbol)
                if eps_data:
                    finviz_data.update(eps_data)
                    logger.debug(f"Supplemented finviz data with EPS rating data for {symbol}")

                return finviz_data
            else:
                self.metrics['finviz_failed'] += 1
                logger.warning(f"finvizfinance failed to fetch {symbol}")

                if not self.enable_fallback:
                    return None

            # Fall back to yfinance
            logger.info(f"Falling back to yfinance for {symbol} fundamentals")
            self.metrics['yfinance_fallback'] += 1

        else:
            # Use yfinance as primary source
            logger.debug(f"Using yfinance as primary source for {symbol}")
            self.metrics['yfinance_primary'] += 1

        # Fetch from yfinance (now includes EPS rating data)
        yf_data = yfinance_service.get_fundamentals(symbol)

        if yf_data:
            yf_data['data_source'] = 'yfinance'
            yf_data['data_source_timestamp'] = datetime.utcnow()
            logger.info(f"Using yfinance data for {symbol} fundamentals")
            return yf_data

        logger.error(f"All data sources failed for {symbol} fundamentals")
        return None

    def _get_eps_rating_data(self, symbol: str) -> Optional[Dict]:
        """
        Get EPS rating data and IPO date from yfinance for a symbol.

        Used to supplement finviz data which doesn't have income statement history
        or IPO date information.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with EPS rating fields and first_trade_date, or None if error
        """
        try:
            import yfinance as yf
            from .eps_rating_service import eps_rating_service
            from .rate_limiter import rate_limiter
            from ..config import settings
            rate_limiter.wait("yfinance", min_interval_s=1.0 / settings.yfinance_rate_limit)

            ticker = yf.Ticker(symbol)

            # Get income statements
            annual_income = ticker.income_stmt
            quarterly_income = ticker.quarterly_income_stmt

            # Calculate EPS rating data using the service
            eps_data = eps_rating_service.calculate_eps_rating_data(
                annual_income,
                quarterly_income
            )

            # Also fetch IPO date from ticker info (finviz doesn't have this)
            try:
                info = ticker.info
                if info and info.get("firstTradeDateMilliseconds"):
                    eps_data["first_trade_date_ms"] = info.get("firstTradeDateMilliseconds")
                else:
                    logger.warning(f"Missing IPO date (firstTradeDateMilliseconds) for {symbol} - field not available from yfinance")
            except Exception as e:
                logger.warning(f"Failed to fetch IPO date for {symbol}: {e}")

            return eps_data

        except Exception as e:
            logger.debug(f"Could not get EPS rating data for {symbol}: {e}")
            return None

    def get_quarterly_growth(self, symbol: str) -> Optional[Dict]:
        """
        Fetch quarterly growth metrics with intelligent source selection and fallback.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with growth metrics and metadata, or None if all sources fail
        """
        self.metrics['total_calls'] += 1

        # Try finvizfinance first if preferred
        if self.prefer_finviz:
            logger.debug(f"Attempting to fetch {symbol} quarterly growth from finvizfinance")

            finviz_data = finviz_service.get_quarterly_growth(symbol)

            if finviz_data:
                # Validate growth metrics (range checks produce warnings, not blocking errors)
                is_valid, errors = self.validator.validate_growth_metrics(finviz_data)

                if not is_valid:
                    # Log warning but still use the data
                    logger.warning(f"Range validation warnings for {symbol} growth: {errors}")

                self.metrics['finviz_success'] += 1
                logger.info(f"Using finvizfinance data for {symbol} quarterly growth")
                finviz_data['data_source'] = 'finviz'
                finviz_data['data_source_timestamp'] = datetime.utcnow()
                return finviz_data
            else:
                self.metrics['finviz_failed'] += 1
                logger.warning(f"finvizfinance failed to fetch {symbol} growth")

                if not self.enable_fallback:
                    return None

            # Fall back to yfinance (only when finviz fetch failed, not on validation warnings)
            logger.info(f"Falling back to yfinance for {symbol} quarterly growth")
            self.metrics['yfinance_fallback'] += 1

        else:
            # Use yfinance as primary source
            logger.debug(f"Using yfinance as primary source for {symbol}")
            self.metrics['yfinance_primary'] += 1

        # Fetch from yfinance
        yf_data = yfinance_service.get_quarterly_growth(symbol)

        if yf_data:
            yf_data['data_source'] = 'yfinance'
            yf_data['data_source_timestamp'] = datetime.utcnow()
            logger.info(f"Using yfinance data for {symbol} quarterly growth")
            return yf_data

        logger.error(f"All data sources failed for {symbol} quarterly growth")
        return None

    def get_combined_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch both fundamentals and quarterly growth in an optimized way.

        For finvizfinance, this is a single API call.
        For yfinance fallback, it makes two separate calls.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with keys 'fundamentals' and 'growth', both containing data + metadata
        """
        self.metrics['total_calls'] += 1

        # Try finvizfinance first if preferred
        if self.prefer_finviz:
            logger.debug(f"Attempting to fetch {symbol} combined data from finvizfinance")

            combined_data = finviz_service.get_combined_data(symbol, validate=self.strict_validation)

            if combined_data:
                self.metrics['finviz_success'] += 1
                logger.info(f"Using finvizfinance for {symbol} combined data")

                # Add metadata
                timestamp = datetime.utcnow()
                combined_data['fundamentals']['data_source'] = 'finviz'
                combined_data['fundamentals']['data_source_timestamp'] = timestamp
                combined_data['growth']['data_source'] = 'finviz'
                combined_data['growth']['data_source_timestamp'] = timestamp

                return combined_data
            else:
                self.metrics['finviz_failed'] += 1
                logger.warning(f"finvizfinance failed for {symbol} combined data")

                if not self.enable_fallback:
                    return None

            # Fall back to yfinance
            logger.info(f"Falling back to yfinance for {symbol} combined data")
            self.metrics['yfinance_fallback'] += 1

        else:
            logger.debug(f"Using yfinance as primary source for {symbol}")
            self.metrics['yfinance_primary'] += 1

        # Fetch from yfinance (requires two API calls)
        fundamentals = yfinance_service.get_fundamentals(symbol)
        growth = yfinance_service.get_quarterly_growth(symbol)

        if fundamentals and growth:
            timestamp = datetime.utcnow()
            fundamentals['data_source'] = 'yfinance'
            fundamentals['data_source_timestamp'] = timestamp
            growth['data_source'] = 'yfinance'
            growth['data_source_timestamp'] = timestamp

            logger.info(f"Using yfinance for {symbol} combined data")

            return {
                'fundamentals': fundamentals,
                'growth': growth,
                'data_source': 'yfinance',
            }

        logger.error(f"All data sources failed for {symbol} combined data")
        return None

    def get_metrics(self) -> Dict:
        """
        Get service usage metrics.

        Returns:
            Dict with metrics
        """
        total = self.metrics['total_calls']

        if total == 0:
            return {
                **self.metrics,
                'finviz_success_rate': 0.0,
                'fallback_rate': 0.0,
            }

        return {
            **self.metrics,
            'finviz_success_rate': (self.metrics['finviz_success'] / total) * 100,
            'fallback_rate': (self.metrics['yfinance_fallback'] / total) * 100,
        }

    def reset_metrics(self):
        """Reset metrics counters"""
        for key in self.metrics:
            self.metrics[key] = 0


# Global instance
data_source_service = DataSourceService(
    prefer_finviz=True,
    enable_fallback=True,
    strict_validation=True,
)
