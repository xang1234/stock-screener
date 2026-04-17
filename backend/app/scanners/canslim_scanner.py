"""
CANSLIM Scanner - William O'Neil's stock selection methodology.

Implements the CANSLIM screening strategy:
- C: Current Quarterly Earnings
- A: Annual Earnings Growth
- N: New Highs (price near 52-week high)
- S: Supply and Demand (volume patterns)
- L: Leader (Relative Strength)
- I: Institutional Sponsorship
- M: Market Direction (skip - scan-level)
"""
import logging
from typing import Dict, Optional
import pandas as pd

from .base_screener import (
    BaseStockScreener,
    DataRequirements,
    ScreenerResult,
    StockData
)
from .screener_registry import register_screener
from .criteria.relative_strength import RelativeStrengthCalculator

logger = logging.getLogger(__name__)


@register_screener
class CANSLIMScanner(BaseStockScreener):
    """
    CANSLIM stock screening methodology by William O'Neil.

    Focuses on growth stocks with strong fundamentals, price momentum,
    and institutional support.
    """

    def __init__(self):
        """Initialize CANSLIM scanner."""
        self.rs_calc = RelativeStrengthCalculator()

    @property
    def screener_name(self) -> str:
        """Unique identifier for this screener."""
        return "canslim"

    def get_data_requirements(self, criteria: Optional[Dict] = None) -> DataRequirements:
        """
        Specify data requirements for CANSLIM screening.

        Args:
            criteria: Optional criteria

        Returns:
            DataRequirements
        """
        return DataRequirements(
            price_period="2y",  # Need for 52-week high/low
            needs_fundamentals=True,  # Need institutional ownership
            needs_quarterly_growth=True,  # C: Current quarterly earnings
            needs_benchmark=True,  # L: Need SPY for RS rating
            needs_earnings_history=False  # A: Now uses eps_growth_yy from quarterly_growth
        )

    def scan_stock(
        self,
        symbol: str,
        data: StockData,
        criteria: Optional[Dict] = None
    ) -> ScreenerResult:
        """
        Scan a stock using CANSLIM methodology.

        Args:
            symbol: Stock symbol
            data: Pre-fetched stock data
            criteria: Optional criteria

        Returns:
            ScreenerResult with score, rating, and details
        """
        try:
            # Validate sufficient data
            if not data.has_sufficient_data(min_days=240):
                return self._insufficient_data_result(symbol, "Insufficient price data")

            # Extract data
            price_data = data.price_data
            benchmark_data = data.benchmark_data
            quarterly_growth = data.quarterly_growth
            fundamentals = data.fundamentals
            precomputed = data.precomputed_scan_context

            # Prepare price series (chronological order)
            prices_chrono = (
                precomputed.close_chrono
                if precomputed is not None and precomputed.close_chrono is not None
                else price_data["Close"].reset_index(drop=True)
            )
            volumes_chrono = (
                precomputed.volume_chrono
                if precomputed is not None and precomputed.volume_chrono is not None
                else price_data["Volume"].reset_index(drop=True)
            )
            spy_prices_chrono = (
                precomputed.benchmark_close_chrono
                if precomputed is not None and precomputed.benchmark_close_chrono is not None
                else benchmark_data["Close"].reset_index(drop=True)
            )

            # Current price
            current_price = (
                float(precomputed.current_price)
                if precomputed is not None and precomputed.current_price is not None
                else float(prices_chrono.iloc[-1])
            )

            # Reverse for calculations expecting most recent first
            prices = (
                precomputed.close_rev
                if precomputed is not None and precomputed.close_rev is not None
                else prices_chrono[::-1].reset_index(drop=True)
            )
            volumes = (
                precomputed.volume_rev
                if precomputed is not None and precomputed.volume_rev is not None
                else volumes_chrono[::-1].reset_index(drop=True)
            )
            spy_prices = (
                precomputed.benchmark_close_rev
                if precomputed is not None and precomputed.benchmark_close_rev is not None
                else spy_prices_chrono[::-1].reset_index(drop=True)
            )
            rs_ratings = (
                precomputed.rs_ratings
                if precomputed is not None and precomputed.rs_ratings is not None
                else None
            )

            # Calculate all CANSLIM criteria
            c_result = self._check_current_earnings(quarterly_growth)
            a_result = self._check_annual_earnings(quarterly_growth)
            n_result = self._check_new_highs(current_price, prices)
            s_result = self._check_supply_demand(prices, volumes)
            l_result = self._check_leader(
                symbol,
                prices,
                spy_prices,
                (
                    data.rs_universe_performances.get("weighted")
                    if data.rs_universe_performances
                    else None
                ),
                precomputed_rs_rating=(
                    rs_ratings.get("rs_rating")
                    if rs_ratings is not None
                    else None
                ),
            )
            i_result = self._check_institutional(fundamentals)

            # Calculate overall score
            score_result = self._calculate_canslim_score(
                c_result, a_result, n_result, s_result, l_result, i_result
            )

            # Build breakdown
            breakdown = {
                "current_earnings": c_result["points"],
                "annual_earnings": a_result["points"],
                "new_highs": n_result["points"],
                "supply_demand": s_result["points"],
                "leader": l_result["points"],
                "institutional": i_result["points"]
            }

            # Calculate all RS ratings for comprehensive analysis
            if rs_ratings is None:
                rs_ratings = self.rs_calc.calculate_all_rs_ratings(
                    symbol,
                    prices,
                    spy_prices,
                    data.rs_universe_performances,
                )

            # Build details
            details = {
                "current_price": current_price,
                "eps_growth_qq": c_result.get("eps_growth_qq"),
                "annual_growth_consistent": a_result.get("consistent_growth", False),
                "from_52w_high_pct": n_result.get("from_high_pct"),
                "volume_ratio": s_result.get("volume_ratio"),
                "rs_rating": l_result.get("rs_rating"),
                "rs_rating_1m": rs_ratings.get("rs_rating_1m"),
                "rs_rating_3m": rs_ratings.get("rs_rating_3m"),
                "rs_rating_12m": rs_ratings.get("rs_rating_12m"),
                "institutional_ownership": i_result.get("ownership_pct"),
                "full_analysis": {
                    "C_current_earnings": c_result,
                    "A_annual_earnings": a_result,
                    "N_new_highs": n_result,
                    "S_supply_demand": s_result,
                    "L_leader": l_result,
                    "I_institutional": i_result,
                    "rs_all_periods": rs_ratings
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
            logger.error(f"Error scanning {symbol} with CANSLIM: {e}")
            return self._error_result(symbol, str(e))

    def _check_current_earnings(self, quarterly_growth: Optional[Dict]) -> Dict:
        """
        C - Current Quarterly Earnings (20 points).

        EPS growth Q/Q should be >= 25% ideally.

        Scoring:
        - 40%+ growth: 20 points
        - 25-40% growth: 15 points
        - 15-25% growth: 10 points
        - <15% growth: proportional
        """
        if not quarterly_growth or quarterly_growth.get('eps_growth_qq') is None:
            return {
                "points": 0,
                "max_points": 20,
                "eps_growth_qq": None,
                "passes": False,
                "reason": "No quarterly EPS data"
            }

        eps_growth = quarterly_growth['eps_growth_qq']

        # Calculate points
        if eps_growth >= 40:
            points = 20
        elif eps_growth >= 25:
            points = 15
        elif eps_growth >= 15:
            points = 10
        elif eps_growth > 0:
            points = (eps_growth / 15) * 10  # Proportional up to 10 points
        else:
            points = 0

        return {
            "points": round(points, 2),
            "max_points": 20,
            "eps_growth_qq": eps_growth,
            "passes": eps_growth >= 25,
            "reason": f"EPS growth Q/Q: {eps_growth:.1f}%"
        }

    def _check_annual_earnings(self, quarterly_growth: Optional[Dict]) -> Dict:
        """
        A - Annual Earnings Growth (15 points).

        Uses year-over-year EPS growth as proxy for annual growth trend.
        Strong Y/Y growth indicates consistent annual performance.

        Scoring:
        - 25%+ Y/Y growth: 15 points (strong growth)
        - 15-25% Y/Y growth: 12 points (good growth)
        - 5-15% Y/Y growth: 8 points (moderate growth)
        - 0-5% Y/Y growth: 4 points (minimal growth)
        - Negative growth: 0 points
        """
        if not quarterly_growth or quarterly_growth.get('eps_growth_yy') is None:
            return {
                "points": 0,
                "max_points": 15,
                "eps_growth_yy": None,
                "consistent_growth": False,
                "passes": False,
                "reason": "No Y/Y EPS growth data available"
            }

        eps_yy = quarterly_growth.get('eps_growth_yy', 0) or 0

        # Calculate points based on Y/Y growth
        if eps_yy >= 25:
            points = 15
            consistent = True
        elif eps_yy >= 15:
            points = 12
            consistent = True
        elif eps_yy >= 5:
            points = 8
            consistent = False
        elif eps_yy >= 0:
            points = 4
            consistent = False
        else:
            points = 0
            consistent = False

        return {
            "points": round(points, 2),
            "max_points": 15,
            "eps_growth_yy": round(eps_yy, 2) if eps_yy else 0,
            "consistent_growth": consistent,
            "passes": consistent,
            "reason": f"EPS Y/Y growth: {eps_yy:.1f}%"
        }

    def _check_new_highs(self, current_price: float, prices: pd.Series) -> Dict:
        """
        N - New Highs (15 points).

        Stock should be within 15% of 52-week high.

        Scoring:
        - <5% from high: 15 points
        - 5-10% from high: 12 points
        - 10-15% from high: 8 points
        - >15% from high: proportional (0-5 points)
        """
        high_52w = float(prices.max())
        from_high_pct = ((high_52w - current_price) / high_52w) * 100

        # Calculate points
        if from_high_pct < 5:
            points = 15
        elif from_high_pct < 10:
            points = 12
        elif from_high_pct < 15:
            points = 8
        elif from_high_pct < 25:
            # Proportional points for 15-25% range
            points = max(0, (25 - from_high_pct) / 10 * 5)
        else:
            points = 0

        return {
            "points": round(points, 2),
            "max_points": 15,
            "from_high_pct": round(from_high_pct, 2),
            "high_52w": high_52w,
            "passes": from_high_pct <= 15,
            "reason": f"{from_high_pct:.1f}% from 52-week high"
        }

    def _check_supply_demand(self, prices: pd.Series, volumes: pd.Series) -> Dict:
        """
        S - Supply and Demand (15 points).

        Volume on up days should exceed volume on down days.

        Scoring:
        - Ratio >= 1.5: 15 points
        - Ratio >= 1.3: 12 points
        - Ratio >= 1.1: 8 points
        - Ratio < 1.1: proportional
        """
        try:
            # Calculate daily price changes
            price_changes = prices.diff()

            # Last 50 days for recent pattern
            recent_changes = price_changes.head(50)
            recent_volumes = volumes.head(50)

            # Volume on up days vs down days
            up_days = recent_changes > 0
            down_days = recent_changes < 0

            up_volume = recent_volumes[up_days].sum() if up_days.any() else 0
            down_volume = recent_volumes[down_days].sum() if down_days.any() else 1  # Avoid division by zero

            volume_ratio = up_volume / down_volume if down_volume > 0 else 0

            # Calculate points
            if volume_ratio >= 1.5:
                points = 15
            elif volume_ratio >= 1.3:
                points = 12
            elif volume_ratio >= 1.1:
                points = 8
            elif volume_ratio >= 0.9:
                points = (volume_ratio - 0.9) / 0.2 * 8  # Proportional
            else:
                points = 0

            return {
                "points": round(points, 2),
                "max_points": 15,
                "volume_ratio": round(volume_ratio, 2),
                "up_volume": int(up_volume),
                "down_volume": int(down_volume),
                "passes": volume_ratio >= 1.3,
                "reason": f"Volume ratio (up/down): {volume_ratio:.2f}"
            }

        except Exception as e:
            logger.warning(f"Error calculating supply/demand: {e}")
            return {
                "points": 0,
                "max_points": 15,
                "volume_ratio": None,
                "passes": False,
                "reason": f"Error: {str(e)}"
            }

    def _check_leader(
        self,
        symbol: str,
        prices: pd.Series,
        spy_prices: pd.Series,
        universe_performances: Optional[list[float]] = None,
        precomputed_rs_rating: Optional[float] = None,
    ) -> Dict:
        """
        L - Leader (20 points).

        Stock should be a market leader with RS Rating >= 80.

        Scoring:
        - RS >= 90: 20 points
        - RS >= 80: 15 points
        - RS >= 70: 10 points
        - RS < 70: proportional
        """
        try:
            if precomputed_rs_rating is not None:
                rs_rating = precomputed_rs_rating
            else:
                rs_result = self.rs_calc.calculate_rs_rating(
                    symbol,
                    prices,
                    spy_prices,
                    universe_performances=universe_performances,
                )
                rs_rating = rs_result["rs_rating"]

            # Calculate points
            if rs_rating >= 90:
                points = 20
            elif rs_rating >= 80:
                points = 15
            elif rs_rating >= 70:
                points = 10
            else:
                points = (rs_rating / 70) * 10  # Proportional

            return {
                "points": round(points, 2),
                "max_points": 20,
                "rs_rating": rs_rating,
                "passes": rs_rating >= 80,
                "reason": f"RS Rating: {rs_rating:.1f}"
            }

        except Exception as e:
            logger.warning(f"Error calculating RS rating: {e}")
            return {
                "points": 0,
                "max_points": 20,
                "rs_rating": None,
                "passes": False,
                "reason": f"Error: {str(e)}"
            }

    def _check_institutional(self, fundamentals: Optional[Dict]) -> Dict:
        """
        I - Institutional Sponsorship (15 points).

        Institutional ownership should be 50-70% (ideal sweet spot).

        Scoring:
        - 50-70% ownership: 15 points
        - 40-80% ownership: 12 points
        - 30-90% ownership: 8 points
        - Outside range: proportional
        """
        if not fundamentals or fundamentals.get('institutional_ownership') is None:
            return {
                "points": 0,
                "max_points": 15,
                "ownership_pct": None,
                "passes": False,
                "reason": "No institutional ownership data"
            }

        # Convert to percentage if needed
        ownership = fundamentals['institutional_ownership']
        if ownership > 1:  # Already in percentage
            ownership_pct = ownership
        else:  # Convert from decimal
            ownership_pct = ownership * 100

        # Calculate points
        if 50 <= ownership_pct <= 70:
            points = 15
        elif 40 <= ownership_pct <= 80:
            points = 12
        elif 30 <= ownership_pct <= 90:
            points = 8
        elif 20 <= ownership_pct < 30 or 90 < ownership_pct <= 95:
            points = 4
        else:
            points = 0

        return {
            "points": round(points, 2),
            "max_points": 15,
            "ownership_pct": round(ownership_pct, 2),
            "passes": 40 <= ownership_pct <= 80,
            "reason": f"Institutional ownership: {ownership_pct:.1f}%"
        }

    def _calculate_canslim_score(
        self,
        c_result: Dict,
        a_result: Dict,
        n_result: Dict,
        s_result: Dict,
        l_result: Dict,
        i_result: Dict
    ) -> Dict:
        """
        Calculate overall CANSLIM score (0-100).

        Total points available: 100
        Pass threshold: 70 points
        """
        score = (
            c_result["points"] +  # 20
            a_result["points"] +  # 15
            n_result["points"] +  # 15
            s_result["points"] +  # 15
            l_result["points"] +  # 20
            i_result["points"]    # 15
        )

        # Determine if passes (score >= 70 AND key criteria met)
        passes = (
            score >= 70 and
            c_result["passes"] and  # Must have strong current earnings
            l_result["passes"]       # Must be a leader (RS >= 80)
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
        # Check key CANSLIM criteria
        eps_growth = details.get("eps_growth_qq", 0) or 0
        rs_rating = details.get("rs_rating", 0) or 0

        if score >= 80 and eps_growth >= 25 and rs_rating >= 80:
            return "Strong Buy"
        elif score >= 70:
            return "Buy"
        elif score >= 60:
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
