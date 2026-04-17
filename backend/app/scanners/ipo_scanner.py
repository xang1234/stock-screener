"""
IPO Scanner - Recent IPO stock screening methodology.

Focuses on stocks that have recently gone public (6 months to 2 years)
and exhibit strong post-IPO performance characteristics.
"""
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .base_screener import (
    BaseStockScreener,
    DataRequirements,
    ScreenerResult,
    StockData
)
from .screener_registry import register_screener

logger = logging.getLogger(__name__)


@register_screener
class IPOScanner(BaseStockScreener):
    """
    IPO stock screening methodology.

    Focuses on recent IPOs with strong growth and stability patterns.
    Ideal for finding emerging growth stocks in their early public trading phase.
    """

    def __init__(self):
        """Initialize IPO scanner."""
        pass

    @property
    def screener_name(self) -> str:
        """Unique identifier for this screener."""
        return "ipo"

    def get_data_requirements(self, criteria: Optional[Dict] = None) -> DataRequirements:
        """
        Specify data requirements for IPO screening.

        Args:
            criteria: Optional criteria

        Returns:
            DataRequirements
        """
        return DataRequirements(
            price_period="2y",  # Need full history to determine IPO date
            needs_fundamentals=True,  # Need firstTradeDateEpochUtc and market data
            needs_quarterly_growth=True,  # For revenue growth
            needs_benchmark=False,  # Don't need SPY comparison
            needs_earnings_history=False
        )

    def scan_stock(
        self,
        symbol: str,
        data: StockData,
        criteria: Optional[Dict] = None
    ) -> ScreenerResult:
        """
        Scan a stock using IPO methodology.

        Args:
            symbol: Stock symbol
            data: Pre-fetched stock data
            criteria: Optional criteria

        Returns:
            ScreenerResult with score, rating, and details
        """
        try:
            # Validate sufficient data
            if not data.has_sufficient_data(min_days=30):
                return self._insufficient_data_result(symbol, "Insufficient price data")

            # Extract data
            price_data = data.price_data
            fundamentals = data.fundamentals
            quarterly_growth = data.quarterly_growth

            # Prepare price series (chronological order)
            prices_chrono = price_data["Close"].reset_index(drop=True)
            volumes_chrono = price_data["Volume"].reset_index(drop=True)

            # Current price
            current_price = float(prices_chrono.iloc[-1])

            # Calculate all IPO criteria
            age_result = self._check_ipo_age(price_data, fundamentals)
            performance_result = self._check_performance_since_ipo(prices_chrono, age_result)
            stability_result = self._check_price_stability(prices_chrono)
            volume_result = self._check_volume_patterns(volumes_chrono, age_result)
            growth_result = self._check_revenue_growth(quarterly_growth)

            # Calculate overall score
            score_result = self._calculate_ipo_score(
                age_result,
                performance_result,
                stability_result,
                volume_result,
                growth_result
            )

            # Build breakdown
            breakdown = {
                "ipo_age": age_result["points"],
                "performance_since_ipo": performance_result["points"],
                "price_stability": stability_result["points"],
                "volume_patterns": volume_result["points"],
                "revenue_growth": growth_result["points"]
            }

            # Build details
            details = {
                "current_price": current_price,
                "ipo_date": age_result.get("ipo_date"),
                "days_since_ipo": age_result.get("days_since_ipo"),
                "ipo_age_category": age_result.get("category"),
                "ipo_price": performance_result.get("ipo_price"),
                "gain_since_ipo_pct": performance_result.get("gain_pct"),
                "volatility_pct": stability_result.get("volatility_pct"),
                "volume_trend": volume_result.get("volume_trend"),
                "revenue_growth_qq": growth_result.get("revenue_growth_qq"),
                "full_analysis": {
                    "ipo_age": age_result,
                    "performance": performance_result,
                    "stability": stability_result,
                    "volume": volume_result,
                    "growth": growth_result
                }
            }

            # Calculate rating
            rating = self.calculate_rating(score_result["score"], details)

            return ScreenerResult(
                score=score_result["score"],
                passes=score_result["passes"],
                rating=rating,
                breakdown=breakdown,
                details=details,
                screener_name=self.screener_name
            )

        except Exception as e:
            logger.error(f"Error scanning {symbol} with IPO: {e}")
            return self._error_result(symbol, str(e))

    def _check_ipo_age(self, price_data: pd.DataFrame, fundamentals: Optional[Dict]) -> Dict:
        """
        Check IPO age (25 points).

        Ideal: 6 months to 2 years since IPO
        This is the "sweet spot" for IPO growth stocks.

        Scoring:
        - 6mo-2yr: 25 points (ideal)
        - 3-6mo: 20 points (very early)
        - 2-3yr: 15 points (maturing)
        - <3mo or >3yr: proportional points
        """
        try:
            # Try to get IPO date from fundamentals
            # Priority: ipo_date (Date object from cache) > first_trade_date (epoch seconds)
            ipo_date = None
            if fundamentals:
                # First try ipo_date (Date object, more reliable from cache)
                cached_ipo_date = fundamentals.get('ipo_date')
                if cached_ipo_date:
                    if isinstance(cached_ipo_date, datetime):
                        ipo_date = cached_ipo_date
                    elif hasattr(cached_ipo_date, 'year'):  # date object
                        ipo_date = datetime.combine(cached_ipo_date, datetime.min.time())
                # Fallback to first_trade_date (epoch seconds)
                elif fundamentals.get('first_trade_date'):
                    first_trade_timestamp = fundamentals['first_trade_date']
                    if first_trade_timestamp:
                        ipo_date = datetime.fromtimestamp(first_trade_timestamp)

            # Fallback: use first price data point
            if not ipo_date:
                first_date = price_data.index[0]
                if hasattr(first_date, 'to_pydatetime'):
                    ipo_date = first_date.to_pydatetime()
                elif isinstance(first_date, pd.Timestamp):
                    ipo_date = first_date.to_pydatetime()
                else:
                    # Convert date to datetime for consistent handling
                    ipo_date = datetime.combine(first_date, datetime.min.time())

            # Ensure ipo_date is datetime
            if not isinstance(ipo_date, datetime):
                ipo_date = datetime.combine(ipo_date, datetime.min.time())

            # Remove timezone info to ensure naive datetime for comparison
            if ipo_date.tzinfo is not None:
                ipo_date = ipo_date.replace(tzinfo=None)

            # Calculate days since IPO
            now = datetime.now()
            days_since_ipo = (now - ipo_date).days
            months_since_ipo = days_since_ipo / 30.0
            years_since_ipo = days_since_ipo / 365.0

            # Determine category
            if months_since_ipo < 3:
                category = "Very Early (<3mo)"
            elif months_since_ipo < 6:
                category = "Early (3-6mo)"
            elif months_since_ipo < 24:
                category = "Sweet Spot (6mo-2yr)"
            elif months_since_ipo < 36:
                category = "Maturing (2-3yr)"
            else:
                category = "Mature (>3yr)"

            # Calculate points
            if 6 <= months_since_ipo <= 24:
                # Sweet spot: 6 months to 2 years
                points = 25
            elif 3 <= months_since_ipo < 6:
                # Very early but promising
                points = 20
            elif 24 < months_since_ipo <= 36:
                # Maturing but still relevant
                points = 15
            elif months_since_ipo < 3:
                # Too early, proportional points
                points = (months_since_ipo / 3) * 15
            else:
                # Too old, declining points
                points = max(0, 15 - (years_since_ipo - 3) * 3)

            return {
                "points": round(points, 2),
                "max_points": 25,
                "ipo_date": ipo_date.strftime("%Y-%m-%d") if ipo_date else None,
                "days_since_ipo": days_since_ipo,
                "months_since_ipo": round(months_since_ipo, 1),
                "years_since_ipo": round(years_since_ipo, 2),
                "category": category,
                "passes": 6 <= months_since_ipo <= 24,
                "reason": f"IPO {months_since_ipo:.1f} months ago ({category})"
            }

        except Exception as e:
            logger.warning(f"Error calculating IPO age: {e}")
            return {
                "points": 0,
                "max_points": 25,
                "ipo_date": None,
                "days_since_ipo": None,
                "category": "Unknown",
                "passes": False,
                "reason": f"Error: {str(e)}"
            }

    def _check_performance_since_ipo(self, prices: pd.Series, age_result: Dict) -> Dict:
        """
        Performance Since IPO (25 points).

        Strong post-IPO performance indicates market acceptance.

        Scoring:
        - 100%+ gain: 25 points
        - 50-100% gain: 20 points
        - 25-50% gain: 15 points
        - 0-25% gain: proportional
        - Negative: 0 points
        """
        try:
            # IPO price is the first price in our data
            ipo_price = float(prices.iloc[0])
            current_price = float(prices.iloc[-1])

            # Calculate gain
            gain_pct = ((current_price - ipo_price) / ipo_price) * 100

            # Calculate points
            if gain_pct >= 100:
                points = 25
            elif gain_pct >= 50:
                points = 20
            elif gain_pct >= 25:
                points = 15
            elif gain_pct > 0:
                # Proportional for 0-25%
                points = (gain_pct / 25) * 15
            else:
                points = 0

            return {
                "points": round(points, 2),
                "max_points": 25,
                "ipo_price": round(ipo_price, 2),
                "current_price": round(current_price, 2),
                "gain_pct": round(gain_pct, 2),
                "passes": gain_pct >= 50,
                "reason": f"Gained {gain_pct:.1f}% since IPO"
            }

        except Exception as e:
            logger.warning(f"Error calculating IPO performance: {e}")
            return {
                "points": 0,
                "max_points": 25,
                "gain_pct": None,
                "passes": False,
                "reason": f"Error: {str(e)}"
            }

    def _check_price_stability(self, prices: pd.Series) -> Dict:
        """
        Price Stability (20 points).

        Low volatility indicates institutional quality and sustainable growth.

        Scoring:
        - <3% volatility: 20 points (very stable)
        - 3-5% volatility: 15 points (stable)
        - 5-7% volatility: 10 points (moderate)
        - >7% volatility: proportional (0-5 points)
        """
        try:
            # Calculate daily returns
            returns = prices.pct_change(fill_method=None).dropna()

            # Calculate volatility (annualized standard deviation)
            daily_volatility = returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252) * 100  # Convert to percentage

            # For scoring, use recent 90-day volatility
            recent_returns = returns.tail(90)
            recent_volatility = recent_returns.std() * np.sqrt(252) * 100

            # Calculate points
            if recent_volatility < 3:
                points = 20
            elif recent_volatility < 5:
                points = 15
            elif recent_volatility < 7:
                points = 10
            elif recent_volatility < 10:
                # Proportional for 7-10%
                points = 10 - ((recent_volatility - 7) / 3) * 5
            else:
                # High volatility, minimal points
                points = max(0, 5 - (recent_volatility - 10) * 0.5)

            return {
                "points": round(points, 2),
                "max_points": 20,
                "volatility_pct": round(recent_volatility, 2),
                "annualized_volatility": round(annualized_volatility, 2),
                "passes": recent_volatility < 5,
                "reason": f"Volatility: {recent_volatility:.1f}% (annualized)"
            }

        except Exception as e:
            logger.warning(f"Error calculating price stability: {e}")
            return {
                "points": 0,
                "max_points": 20,
                "volatility_pct": None,
                "passes": False,
                "reason": f"Error: {str(e)}"
            }

    def _check_volume_patterns(self, volumes: pd.Series, age_result: Dict) -> Dict:
        """
        Volume Patterns (15 points).

        Increasing volume over time indicates growing institutional interest.

        Scoring:
        - Ratio >= 1.5: 15 points (strong increase)
        - Ratio >= 1.2: 12 points (moderate increase)
        - Ratio >= 1.0: 8 points (stable)
        - Ratio < 1.0: proportional (declining)
        """
        try:
            # Compare recent 30 days vs first 30 days
            if len(volumes) < 60:
                # Not enough data
                return {
                    "points": 0,
                    "max_points": 15,
                    "volume_trend": None,
                    "passes": False,
                    "reason": "Insufficient volume data"
                }

            # First 30 days of trading
            early_volume = volumes.iloc[:30].mean()

            # Recent 30 days
            recent_volume = volumes.iloc[-30:].mean()

            # Calculate ratio
            volume_ratio = recent_volume / early_volume if early_volume > 0 else 0

            # Calculate points
            if volume_ratio >= 1.5:
                points = 15
            elif volume_ratio >= 1.2:
                points = 12
            elif volume_ratio >= 1.0:
                points = 8
            elif volume_ratio >= 0.8:
                # Proportional for declining volume
                points = (volume_ratio - 0.8) / 0.2 * 8
            else:
                points = 0

            return {
                "points": round(points, 2),
                "max_points": 15,
                "volume_ratio": round(volume_ratio, 2),
                "early_volume": int(early_volume),
                "recent_volume": int(recent_volume),
                "volume_trend": "Increasing" if volume_ratio >= 1.2 else "Stable" if volume_ratio >= 1.0 else "Declining",
                "passes": volume_ratio >= 1.2,
                "reason": f"Volume ratio: {volume_ratio:.2f}x (recent vs early)"
            }

        except Exception as e:
            logger.warning(f"Error calculating volume patterns: {e}")
            return {
                "points": 0,
                "max_points": 15,
                "volume_trend": None,
                "passes": False,
                "reason": f"Error: {str(e)}"
            }

    def _check_revenue_growth(self, quarterly_growth: Optional[Dict]) -> Dict:
        """
        Revenue Growth (15 points).

        Strong revenue growth is critical for recent IPOs.

        Scoring:
        - 30%+ growth: 15 points
        - 20-30% growth: 12 points
        - 10-20% growth: 8 points
        - 0-10% growth: proportional
        - Negative: 0 points
        """
        if not quarterly_growth or quarterly_growth.get('sales_growth_qq') is None:
            return {
                "points": 0,
                "max_points": 15,
                "revenue_growth_qq": None,
                "passes": False,
                "reason": "No revenue growth data"
            }

        revenue_growth = quarterly_growth['sales_growth_qq']

        # Calculate points
        if revenue_growth >= 30:
            points = 15
        elif revenue_growth >= 20:
            points = 12
        elif revenue_growth >= 10:
            points = 8
        elif revenue_growth > 0:
            # Proportional for 0-10%
            points = (revenue_growth / 10) * 8
        else:
            points = 0

        return {
            "points": round(points, 2),
            "max_points": 15,
            "revenue_growth_qq": revenue_growth,
            "passes": revenue_growth >= 20,
            "reason": f"Revenue growth Q/Q: {revenue_growth:.1f}%"
        }

    def _calculate_ipo_score(
        self,
        age_result: Dict,
        performance_result: Dict,
        stability_result: Dict,
        volume_result: Dict,
        growth_result: Dict
    ) -> Dict:
        """
        Calculate overall IPO score (0-100).

        Total points available: 100
        Pass threshold: 65 points
        """
        score = (
            age_result["points"] +          # 25
            performance_result["points"] +  # 25
            stability_result["points"] +    # 20
            volume_result["points"] +       # 15
            growth_result["points"]         # 15
        )

        # Determine if passes (score >= 65 AND in ideal age range)
        passes = (
            score >= 65 and
            age_result["passes"]  # Must be in 6mo-2yr range
        )

        return {
            "score": round(score, 2),
            "passes": passes
        }

    def calculate_rating(self, score: float, details: Dict) -> str:
        """
        Calculate human-readable rating from score.

        Args:
            score: Numeric score (0-100)
            details: Analysis details

        Returns:
            Rating string
        """
        # Check key IPO criteria
        ipo_age_category = details.get("ipo_age_category", "")
        gain_since_ipo = details.get("gain_since_ipo_pct", 0) or 0

        if score >= 80 and "Sweet Spot" in ipo_age_category and gain_since_ipo >= 50:
            return "Strong Buy"
        elif score >= 65:
            return "Buy"
        elif score >= 50:
            return "Watch"
        else:
            return "Pass"

    def _insufficient_data_result(self, symbol: str, reason: str) -> ScreenerResult:
        """Return result for insufficient data."""
        return ScreenerResult(
            score=0.0,
            passes=False,
            rating="Insufficient Data",
            breakdown={},
            details={"error": reason},
            screener_name=self.screener_name
        )

    def _error_result(self, symbol: str, error: str) -> ScreenerResult:
        """Return result for errors."""
        return ScreenerResult(
            score=0.0,
            passes=False,
            rating="Error",
            breakdown={},
            details={"error": f"Scan error: {error}"},
            screener_name=self.screener_name
        )
