"""
Custom/Flexible Stock Screener.

A highly configurable screener that allows users to define their own
criteria combinations from a set of common technical and fundamental filters.

Supported Filters:
- Price range (min/max)
- RS Rating (minimum)
- Volume (minimum)
- Market cap (min/max)
- EPS growth (minimum)
- Sales growth (minimum)
- Moving average alignment (Price > 50 > 150 > 200)
- Sector inclusion/exclusion
- Industry inclusion/exclusion
- 52-week high proximity
- Debt-to-Equity ratio (maximum)

Each enabled filter contributes to the final score. Filters can be weighted.
"""
import logging
from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .base_screener import BaseStockScreener, DataRequirements, ScreenerResult, StockData
from .screener_registry import register_screener
from .criteria.relative_strength import RelativeStrengthCalculator

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result from evaluating a single filter."""
    name: str
    passes: bool
    points: float  # Points earned (0 to max_points)
    max_points: float  # Maximum possible points for this filter
    details: Dict[str, Any]


@register_screener
class CustomScanner(BaseStockScreener):
    """
    Custom/Flexible stock screener with user-configurable filters.

    Configuration is passed via the criteria dict with the following structure:
    {
        "custom_filters": {
            "price_min": 20,
            "price_max": 500,
            "rs_rating_min": 75,
            "volume_min": 1000000,
            "market_cap_min": 1000000000,
            "market_cap_max": 50000000000,
            "eps_growth_min": 20,
            "sales_growth_min": 15,
            "ma_alignment": True,
            "near_52w_high": 15,  # Within 15% of 52-week high
            "debt_to_equity_max": 0.5,
            "sectors": ["Technology", "Healthcare"],  # Include only these
            "exclude_industries": ["Tobacco", "Gambling"]
        },
        "min_score": 70  # Minimum score to pass
    }
    """

    def __init__(self):
        """Initialize custom scanner."""
        self.rs_calc = RelativeStrengthCalculator()

    @property
    def screener_name(self) -> str:
        """Return screener name."""
        return "custom"

    def get_data_requirements(self, criteria: Optional[Dict] = None) -> DataRequirements:
        """
        Get data requirements based on enabled filters.

        Args:
            criteria: Screening criteria with custom_filters

        Returns:
            DataRequirements specifying needed data
        """
        # Parse criteria to determine what data we need
        filters = self._get_filters_config(criteria)

        # Determine requirements based on filters
        needs_fundamentals = any([
            filters.get("market_cap_min") is not None,
            filters.get("market_cap_max") is not None,
            filters.get("debt_to_equity_max") is not None,
            filters.get("sectors") is not None,
            filters.get("exclude_industries") is not None,
        ])

        needs_quarterly_growth = any([
            filters.get("eps_growth_min") is not None,
            filters.get("sales_growth_min") is not None,
        ])

        needs_benchmark = filters.get("rs_rating_min") is not None

        return DataRequirements(
            price_period="2y",  # Need enough data for MAs and RS
            needs_fundamentals=needs_fundamentals,
            needs_quarterly_growth=needs_quarterly_growth,
            needs_benchmark=needs_benchmark,
            needs_earnings_history=False
        )

    def scan_stock(
        self,
        symbol: str,
        data: StockData,
        criteria: Optional[Dict] = None
    ) -> ScreenerResult:
        """
        Scan a stock using custom filters.

        Args:
            symbol: Stock symbol
            data: Pre-fetched stock data
            criteria: Screening criteria with custom_filters

        Returns:
            ScreenerResult with score, pass status, and details
        """
        # Get filter configuration
        filters = self._get_filters_config(criteria)
        min_score = criteria.get("min_score", 70) if criteria else 70

        # Extract data
        price_data = data.price_data
        fundamentals = data.fundamentals
        quarterly_growth = data.quarterly_growth
        benchmark_data = data.benchmark_data

        # Validate data
        if price_data.empty or len(price_data) < 200:
            return self._create_error_result(
                symbol,
                "Insufficient price data (need 200+ days)"
            )

        # Run all enabled filters
        filter_results: List[FilterResult] = []

        # Price filters
        if filters.get("price_min") is not None or filters.get("price_max") is not None:
            try:
                filter_results.append(self._check_price_range(price_data, filters))
            except Exception as e:
                logger.error(f"{symbol}: Error in price_range filter: {e}")
                raise

        # Volume filter
        if filters.get("volume_min") is not None:
            try:
                filter_results.append(self._check_volume(price_data, filters))
            except Exception as e:
                logger.error(f"{symbol}: Error in volume filter: {e}")
                raise

        # RS Rating filter
        if filters.get("rs_rating_min") is not None:
            if not benchmark_data.empty:
                try:
                    filter_results.append(self._check_rs_rating(
                        symbol,
                        price_data,
                        benchmark_data,
                        filters,
                        (
                            data.rs_universe_performances.get("weighted")
                            if data.rs_universe_performances
                            else None
                        ),
                    ))
                except Exception as e:
                    logger.error(f"{symbol}: Error in rs_rating filter: {e}")
                    raise
            else:
                logger.warning(f"{symbol}: RS rating filter enabled but no benchmark data")

        # Market cap filter
        if filters.get("market_cap_min") is not None or filters.get("market_cap_max") is not None:
            if fundamentals:
                filter_results.append(self._check_market_cap(fundamentals, filters))
            else:
                logger.warning(f"{symbol}: Market cap filter enabled but no fundamentals")

        # EPS growth filter
        if filters.get("eps_growth_min") is not None:
            if quarterly_growth:
                filter_results.append(self._check_eps_growth(quarterly_growth, filters))
            else:
                logger.warning(f"{symbol}: EPS growth filter enabled but no growth data")

        # Sales growth filter
        if filters.get("sales_growth_min") is not None:
            if quarterly_growth:
                filter_results.append(self._check_sales_growth(quarterly_growth, filters))
            else:
                logger.warning(f"{symbol}: Sales growth filter enabled but no growth data")

        # MA alignment filter
        if filters.get("ma_alignment") is True:
            filter_results.append(self._check_ma_alignment(price_data))

        # 52-week high proximity filter
        if filters.get("near_52w_high") is not None:
            filter_results.append(self._check_near_high(price_data, filters))

        # Debt-to-equity filter
        if filters.get("debt_to_equity_max") is not None:
            if fundamentals:
                filter_results.append(self._check_debt_to_equity(fundamentals, filters))
            else:
                logger.warning(f"{symbol}: D/E filter enabled but no fundamentals")

        # Sector filter
        if filters.get("sectors") is not None:
            if fundamentals:
                filter_results.append(self._check_sector(fundamentals, filters))
            else:
                logger.warning(f"{symbol}: Sector filter enabled but no fundamentals")

        # Industry exclusion filter
        if filters.get("exclude_industries") is not None:
            if fundamentals:
                filter_results.append(self._check_industry_exclusion(fundamentals, filters))
            else:
                logger.warning(f"{symbol}: Industry filter enabled but no fundamentals")

        # Calculate total score
        if not filter_results:
            return self._create_error_result(
                symbol,
                "No filters enabled in custom screener"
            )

        total_points = sum(r.points for r in filter_results)
        total_possible = sum(r.max_points for r in filter_results)
        score = (total_points / total_possible) * 100 if total_possible > 0 else 0

        # Determine pass status
        passes = score >= min_score

        # Build breakdown
        breakdown = {r.name: r.points for r in filter_results}

        # Build details
        details = {
            "filters_enabled": len(filter_results),
            "filters_passed": sum(1 for r in filter_results if r.passes),
            "total_points": total_points,
            "total_possible": total_possible,
            "min_score_required": min_score,
            "filter_results": {
                r.name: {
                    "passes": r.passes,
                    "points": r.points,
                    "max_points": r.max_points,
                    **r.details
                }
                for r in filter_results
            }
        }

        # Calculate rating
        rating = self.calculate_rating(score, details)

        return ScreenerResult(
            score=score,
            passes=passes,
            rating=rating,
            breakdown=breakdown,
            details=details,
            screener_name="custom"
        )

    def calculate_rating(self, score: float, details: Dict) -> str:
        """
        Calculate rating based on score.

        Args:
            score: Composite score (0-100)
            details: Screening details

        Returns:
            Rating string
        """
        if score >= 80:
            return "Strong Buy"
        elif score >= 70:
            return "Buy"
        elif score >= 50:
            return "Watch"
        else:
            return "Pass"

    # Filter implementation methods

    def _check_price_range(self, price_data: pd.DataFrame, filters: Dict) -> FilterResult:
        """Check if price is within specified range."""
        current_price = float(price_data['Close'].iloc[-1])
        price_min = filters.get("price_min", 0)
        price_max = filters.get("price_max", float('inf'))

        passes = price_min <= current_price <= price_max
        points = 10.0 if passes else 0.0

        return FilterResult(
            name="price_range",
            passes=passes,
            points=points,
            max_points=10.0,
            details={
                "current_price": current_price,
                "price_min": price_min,
                "price_max": price_max
            }
        )

    def _check_volume(self, price_data: pd.DataFrame, filters: Dict) -> FilterResult:
        """Check if average volume meets minimum."""
        avg_volume = float(price_data['Volume'].tail(20).mean())
        volume_min = filters.get("volume_min", 0)

        passes = bool(avg_volume >= volume_min)

        # Scaled scoring: meeting minimum = 5pts, 2x minimum = 10pts
        if avg_volume >= volume_min * 2:
            points = 10.0
        elif avg_volume >= volume_min:
            ratio = avg_volume / volume_min
            points = 5.0 + (ratio - 1.0) * 5.0  # Linear scale from 5 to 10
        else:
            points = 0.0

        return FilterResult(
            name="volume",
            passes=passes,
            points=points,
            max_points=10.0,
            details={
                "avg_volume": int(avg_volume),
                "volume_min": volume_min,
                "ratio": avg_volume / volume_min if volume_min > 0 else 0
            }
        )

    def _check_rs_rating(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        filters: Dict,
        universe_performances: Optional[list[float]] = None,
    ) -> FilterResult:
        """Check if RS rating meets minimum."""
        # calculate_rs_rating returns a dict, extract the rs_rating value
        rs_result = self.rs_calc.calculate_rs_rating(
            symbol,
            price_data['Close'],
            benchmark_data['Close'],
            universe_performances=universe_performances,
        )
        rs_rating = float(rs_result["rs_rating"])
        rs_min = float(filters.get("rs_rating_min", 0))

        passes = bool(rs_rating >= rs_min)

        # Scaled scoring: meeting minimum = 10pts, RS 90+ = 15pts
        if rs_rating >= 90:
            points = 15.0
        elif rs_rating >= rs_min:
            # Linear scale from 10 to 15
            points = 10.0 + ((rs_rating - rs_min) / (90 - rs_min)) * 5.0
        else:
            points = 0.0

        return FilterResult(
            name="rs_rating",
            passes=passes,
            points=points,
            max_points=15.0,
            details={
                "rs_rating": rs_rating,
                "rs_min": rs_min,
                "relative_performance": rs_result.get("relative_performance")
            }
        )

    def _check_market_cap(self, fundamentals: Dict, filters: Dict) -> FilterResult:
        """Check if market cap is within range."""
        market_cap = fundamentals.get("market_cap", 0)
        cap_min = filters.get("market_cap_min", 0)
        cap_max = filters.get("market_cap_max", float('inf'))

        passes = cap_min <= market_cap <= cap_max if market_cap else False
        points = 10.0 if passes else 0.0

        return FilterResult(
            name="market_cap",
            passes=passes,
            points=points,
            max_points=10.0,
            details={
                "market_cap": market_cap,
                "market_cap_min": cap_min,
                "market_cap_max": cap_max
            }
        )

    def _check_eps_growth(self, quarterly_growth: Dict, filters: Dict) -> FilterResult:
        """Check if EPS growth meets minimum."""
        eps_growth = quarterly_growth.get("eps_growth_qq", 0)
        eps_min = filters.get("eps_growth_min", 0)

        # Handle None values
        if eps_growth is None:
            eps_growth = 0

        passes = eps_growth >= eps_min

        # Scaled scoring: meeting minimum = 10pts, 2x minimum = 15pts
        if eps_growth >= eps_min * 2:
            points = 15.0
        elif eps_growth >= eps_min:
            ratio = eps_growth / eps_min if eps_min > 0 else 1
            points = 10.0 + (ratio - 1.0) * 5.0
        else:
            points = 0.0

        return FilterResult(
            name="eps_growth",
            passes=passes,
            points=points,
            max_points=15.0,
            details={
                "eps_growth_qq": eps_growth,
                "eps_growth_min": eps_min
            }
        )

    def _check_sales_growth(self, quarterly_growth: Dict, filters: Dict) -> FilterResult:
        """Check if sales growth meets minimum."""
        sales_growth = quarterly_growth.get("sales_growth_qq", 0)
        sales_min = filters.get("sales_growth_min", 0)

        # Handle None values
        if sales_growth is None:
            sales_growth = 0

        passes = sales_growth >= sales_min

        # Scaled scoring: meeting minimum = 10pts, 2x minimum = 15pts
        if sales_growth >= sales_min * 2:
            points = 15.0
        elif sales_growth >= sales_min:
            ratio = sales_growth / sales_min if sales_min > 0 else 1
            points = 10.0 + (ratio - 1.0) * 5.0
        else:
            points = 0.0

        return FilterResult(
            name="sales_growth",
            passes=passes,
            points=points,
            max_points=15.0,
            details={
                "sales_growth_qq": sales_growth,
                "sales_growth_min": sales_min
            }
        )

    def _check_ma_alignment(self, price_data: pd.DataFrame) -> FilterResult:
        """Check if moving averages are properly aligned (Price > 50 > 150 > 200)."""
        current_price = float(price_data['Close'].iloc[-1])

        # Calculate moving averages - explicitly convert to float
        ma_50 = float(price_data['Close'].rolling(window=50).mean().iloc[-1])
        ma_150 = float(price_data['Close'].rolling(window=150).mean().iloc[-1])
        ma_200 = float(price_data['Close'].rolling(window=200).mean().iloc[-1])

        # Check alignment
        aligned = bool(current_price > ma_50 > ma_150 > ma_200)

        # Partial credit for partial alignment
        if aligned:
            points = 15.0
        elif current_price > ma_50 > ma_150:
            points = 10.0
        elif current_price > ma_50:
            points = 5.0
        else:
            points = 0.0

        passes = aligned

        return FilterResult(
            name="ma_alignment",
            passes=passes,
            points=points,
            max_points=15.0,
            details={
                "current_price": current_price,
                "ma_50": float(ma_50),
                "ma_150": float(ma_150),
                "ma_200": float(ma_200),
                "aligned": aligned
            }
        )

    def _check_near_high(self, price_data: pd.DataFrame, filters: Dict) -> FilterResult:
        """Check if price is near 52-week high."""
        current_price = float(price_data['Close'].iloc[-1])
        high_52w = price_data['High'].tail(252).max()

        pct_from_high = ((high_52w - current_price) / high_52w) * 100
        max_distance = filters.get("near_52w_high", 15)

        passes = pct_from_high <= max_distance

        # Scaled scoring: at high = 10pts, at max distance = 5pts
        if pct_from_high <= 5:
            points = 10.0
        elif pct_from_high <= max_distance:
            # Linear scale from 10 to 5
            points = 10.0 - ((pct_from_high - 5) / (max_distance - 5)) * 5.0
        else:
            points = 0.0

        return FilterResult(
            name="near_52w_high",
            passes=passes,
            points=points,
            max_points=10.0,
            details={
                "current_price": current_price,
                "high_52w": float(high_52w),
                "pct_from_high": pct_from_high,
                "max_distance": max_distance
            }
        )

    def _check_debt_to_equity(self, fundamentals: Dict, filters: Dict) -> FilterResult:
        """Check if debt-to-equity ratio is below maximum."""
        de_ratio = fundamentals.get("debt_to_equity", 0)
        de_max = filters.get("debt_to_equity_max", float('inf'))

        # Handle None or missing data
        if de_ratio is None:
            passes = False
            points = 0.0
        else:
            passes = de_ratio <= de_max

            # Scaled scoring: 0 debt = 10pts, at max = 5pts
            if de_ratio == 0:
                points = 10.0
            elif de_ratio <= de_max / 2:
                points = 10.0 - (de_ratio / (de_max / 2)) * 2.5
            elif de_ratio <= de_max:
                points = 7.5 - ((de_ratio - de_max / 2) / (de_max / 2)) * 2.5
            else:
                points = 0.0

        return FilterResult(
            name="debt_to_equity",
            passes=passes,
            points=points,
            max_points=10.0,
            details={
                "debt_to_equity": de_ratio,
                "debt_to_equity_max": de_max
            }
        )

    def _check_sector(self, fundamentals: Dict, filters: Dict) -> FilterResult:
        """Check if stock is in allowed sectors."""
        sector = fundamentals.get("sector")
        allowed_sectors = filters.get("sectors", [])

        passes = sector in allowed_sectors if sector else False
        points = 10.0 if passes else 0.0

        return FilterResult(
            name="sector",
            passes=passes,
            points=points,
            max_points=10.0,
            details={
                "sector": sector,
                "allowed_sectors": allowed_sectors
            }
        )

    def _check_industry_exclusion(self, fundamentals: Dict, filters: Dict) -> FilterResult:
        """Check if stock is NOT in excluded industries."""
        industry = fundamentals.get("industry")
        excluded = filters.get("exclude_industries", [])

        passes = industry not in excluded if industry else True
        points = 10.0 if passes else 0.0

        return FilterResult(
            name="industry_exclusion",
            passes=passes,
            points=points,
            max_points=10.0,
            details={
                "industry": industry,
                "excluded_industries": excluded,
                "is_excluded": not passes
            }
        )

    # Helper methods

    def _get_filters_config(self, criteria: Optional[Dict]) -> Dict:
        """Extract filter configuration from criteria."""
        if not criteria:
            return {}

        # Support both top-level criteria keys and nested custom_filters
        custom_filters = criteria.get("custom_filters", {})

        # Merge with top-level keys (for backward compatibility)
        filters = {**criteria, **custom_filters}

        # Remove non-filter keys
        filters.pop("custom_filters", None)
        filters.pop("min_score", None)

        return filters

    def _create_error_result(self, symbol: str, error_msg: str) -> ScreenerResult:
        """Create error result."""
        return ScreenerResult(
            score=0.0,
            passes=False,
            rating="Pass",
            breakdown={},
            details={"error": error_msg},
            screener_name="custom"
        )
