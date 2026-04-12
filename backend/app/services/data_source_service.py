"""
Data Source Service

Coordinates between finvizfinance (primary) and yfinance (fallback) data sources.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional
from datetime import datetime

from .finviz_validator import FinvizValidator
from . import provider_routing_policy as routing_policy

if TYPE_CHECKING:
    from app.services.eps_rating_service import EPSRatingService
    from app.services.finviz_service import FinvizService
    from app.services.rate_limiter import RedisRateLimiter
    from app.services.yfinance_service import YFinanceService

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
        *,
        finviz_service: FinvizService | None = None,
        yfinance_service: YFinanceService | None = None,
        eps_rating_service: EPSRatingService | None = None,
        rate_limiter: RedisRateLimiter | None = None,
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
        if rate_limiter is None:
            from .rate_limiter import RedisRateLimiter

            rate_limiter = RedisRateLimiter()
        if eps_rating_service is None:
            from .eps_rating_service import EPSRatingService

            eps_rating_service = EPSRatingService()
        if yfinance_service is None:
            from .yfinance_service import YFinanceService

            yfinance_service = YFinanceService(
                rate_limiter=rate_limiter,
                eps_rating_service=eps_rating_service,
            )
        if finviz_service is None:
            from .finviz_service import FinvizService

            finviz_service = FinvizService(rate_limiter=rate_limiter)
        self.finviz_service = finviz_service
        self.yfinance_service = yfinance_service

        # Metrics tracking
        self.metrics = {
            'finviz_success': 0,
            'finviz_failed': 0,
            'yfinance_fallback': 0,
            'yfinance_primary': 0,
            'total_calls': 0,
            # Count of calls where prefer_finviz=True but routing policy
            # excluded finviz (e.g. HK/JP/TW). Useful for measuring how many
            # wasted US-only provider calls the policy is preventing.
            'finviz_skipped_by_policy': 0,
        }

    def _finviz_allowed(self, market: Optional[str]) -> bool:
        """Return True iff finviz may be attempted for ``market`` per policy.

        The ``prefer_finviz`` instance flag is still honoured — policy only
        *narrows* the set of attempted providers; it never forces finviz on
        a caller that opted out. Increments a telemetry counter when policy
        (not ``prefer_finviz=False``) is the reason finviz is skipped.
        """
        if not self.prefer_finviz:
            return False
        allowed = routing_policy.is_supported(
            market, routing_policy.PROVIDER_FINVIZ
        )
        if not allowed:
            self.metrics['finviz_skipped_by_policy'] += 1
            logger.info(
                "Routing policy %s excluded finviz for market=%r; "
                "using yfinance as primary.",
                routing_policy.policy_version(),
                market,
            )
        return allowed

    def get_fundamentals(
        self,
        symbol: str,
        market: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Fetch fundamental data with intelligent source selection and fallback.

        Args:
            symbol: Stock ticker symbol
            market: Optional market code (US/HK/JP/TW). When provided, the
                provider routing policy filters out providers that do not
                cover this market (e.g. finviz is skipped for HK/JP/TW).
                ``None`` preserves legacy US-equivalent behaviour.

        Returns:
            Dict with fundamental metrics and metadata, or None if all sources fail
        """
        self.metrics['total_calls'] += 1

        # Try finvizfinance first if preferred AND allowed by policy
        if self._finviz_allowed(market):
            logger.debug(f"Attempting to fetch {symbol} fundamentals from finvizfinance")

            finviz_data = self.finviz_service.get_fundamentals(symbol)

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
        yf_data = self.yfinance_service.get_fundamentals(symbol)

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
            yf_data = self.yfinance_service.get_fundamentals(symbol)
        except Exception as e:  # pragma: no cover - defensive path
            logger.debug(f"Could not get EPS rating data for {symbol}: {e}")
            return None

        if not yf_data:
            return None

        keys = (
            "first_trade_date_ms",
            "eps_5yr_cagr",
            "eps_q1_yoy",
            "eps_q2_yoy",
            "eps_raw_score",
            "eps_years_available",
        )
        eps_data = {key: yf_data.get(key) for key in keys if yf_data.get(key) is not None}
        return eps_data or None

    def get_quarterly_growth(
        self,
        symbol: str,
        market: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Fetch quarterly growth metrics with intelligent source selection and fallback.

        Args:
            symbol: Stock ticker symbol
            market: Optional market code (US/HK/JP/TW); see
                ``get_fundamentals`` for semantics.

        Returns:
            Dict with growth metrics and metadata, or None if all sources fail
        """
        self.metrics['total_calls'] += 1

        # Try finvizfinance first if preferred AND allowed by policy
        if self._finviz_allowed(market):
            logger.debug(f"Attempting to fetch {symbol} quarterly growth from finvizfinance")

            finviz_data = self.finviz_service.get_quarterly_growth(symbol)

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
        yf_data = self.yfinance_service.get_quarterly_growth(symbol)

        if yf_data:
            yf_data['data_source'] = 'yfinance'
            yf_data['data_source_timestamp'] = datetime.utcnow()
            logger.info(f"Using yfinance data for {symbol} quarterly growth")
            return yf_data

        logger.error(f"All data sources failed for {symbol} quarterly growth")
        return None

    def get_combined_data(
        self,
        symbol: str,
        market: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Fetch both fundamentals and quarterly growth in an optimized way.

        For finvizfinance, this is a single API call.
        For yfinance fallback, it makes two separate calls.

        Args:
            symbol: Stock ticker symbol
            market: Optional market code (US/HK/JP/TW); see
                ``get_fundamentals`` for semantics.

        Returns:
            Dict with keys 'fundamentals' and 'growth', both containing data + metadata
        """
        self.metrics['total_calls'] += 1

        # Try finvizfinance first if preferred AND allowed by policy
        if self._finviz_allowed(market):
            logger.debug(f"Attempting to fetch {symbol} combined data from finvizfinance")

            combined_data = self.finviz_service.get_combined_data(symbol, validate=self.strict_validation)

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
        fundamentals = self.yfinance_service.get_fundamentals(symbol)
        growth = self.yfinance_service.get_quarterly_growth(symbol)

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
