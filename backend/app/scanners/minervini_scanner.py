"""
Minervini Scanner V2 - Multi-screener architecture adaptation.

Wraps the existing MinerviniScanner logic to conform to the BaseStockScreener
interface, enabling it to work with the multi-screener orchestrator.

All original criteria and scoring logic is preserved for backward compatibility.
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
from .criteria.stage_analysis import WeinsteinstageAnalyzer
from .criteria.moving_averages import MovingAverageAnalyzer
from .criteria.vcp_detection import VCPDetector
from .criteria.adr_calculator import ADRCalculator
from .criteria.rs_sparkline import RSSparklineCalculator
from .criteria.price_sparkline import PriceSparklineCalculator
from .criteria.beta_calculator import BetaCalculator

logger = logging.getLogger(__name__)


@register_screener
class MinerviniScanner(BaseStockScreener):
    """
    Minervini Template Scanner.

    Implements Mark Minervini's stock selection criteria:
    1. RS Rating > 70 (preferably > 80)
    2. Price > 50-day > 150-day > 200-day MA
    3. 200-day MA trending up for at least 1 month
    4. 50-day MA above both 150-day and 200-day MA
    5. Price at least 30% above 52-week low
    6. Price within 25% of 52-week high
    7. Stage 2 uptrend (Weinstein)
    8. VCP pattern (optional but ideal)
    """

    def __init__(self):
        """Initialize Minervini scanner with criteria calculators."""
        self.rs_calc = RelativeStrengthCalculator()
        self.stage_analyzer = WeinsteinstageAnalyzer()
        self.ma_analyzer = MovingAverageAnalyzer()
        self.vcp_detector = VCPDetector()
        self.adr_calc = ADRCalculator()
        self.rs_sparkline_calc = RSSparklineCalculator()
        self.price_sparkline_calc = PriceSparklineCalculator()
        self.beta_calc = BetaCalculator()

    @property
    def screener_name(self) -> str:
        """Unique identifier for this screener."""
        return "minervini"

    def get_data_requirements(self, criteria: Optional[Dict] = None) -> DataRequirements:
        """
        Specify data requirements for Minervini screening.

        Args:
            criteria: Optional criteria (e.g., include_vcp)

        Returns:
            DataRequirements
        """
        return DataRequirements(
            price_period="2y",  # Need 2 years for 200-day MA and RS calc
            needs_fundamentals=False,  # Minervini doesn't use fundamentals
            needs_quarterly_growth=False,  # NOT used in Minervini scoring (only informational)
            needs_benchmark=True,  # Need SPY for RS rating
            needs_earnings_history=False
        )

    def scan_stock(
        self,
        symbol: str,
        data: StockData,
        criteria: Optional[Dict] = None
    ) -> ScreenerResult:
        """
        Scan a stock using Minervini template.

        Args:
            symbol: Stock symbol
            data: Pre-fetched stock data
            criteria: Optional criteria (e.g., include_vcp)

        Returns:
            ScreenerResult with score, rating, and details
        """
        try:
            # Extract criteria
            include_vcp = criteria.get("include_vcp", True) if criteria else True

            # Validate we have sufficient data
            if not data.has_sufficient_data(min_days=240):
                return self._insufficient_data_result(symbol, "Insufficient price data")

            price_data = data.price_data
            spy_data = data.benchmark_data

            # Calculate ADR (Average Daily Range)
            adr_percent = None
            try:
                adr_percent = self.adr_calc.calculate_adr_percent(price_data, period=20)
            except Exception as e:
                logger.warning(f"ADR calculation failed for {symbol}: {e}")

            # Extract quarterly growth
            quarterly_growth = data.quarterly_growth or {
                'eps_growth_qq': None,
                'sales_growth_qq': None,
                'eps_growth_yy': None,
                'sales_growth_yy': None,
                'recent_quarter_date': None,
                'previous_quarter_date': None
            }

            # Extract price and volume series in chronological order
            prices_chrono = price_data["Close"].reset_index(drop=True)
            volumes_chrono = price_data["Volume"].reset_index(drop=True)
            spy_prices_chrono = spy_data["Close"].reset_index(drop=True)

            # Calculate RS Sparkline data (30-day stock/SPY ratio trend)
            rs_sparkline_result = self.rs_sparkline_calc.calculate_rs_sparkline(
                prices_chrono,
                spy_prices_chrono
            )

            # Calculate Price Sparkline data (30-day normalized price trend)
            price_sparkline_result = self.price_sparkline_calc.calculate_price_sparkline(
                prices_chrono
            )

            # Current values (most recent = last in chronological series)
            current_price = prices_chrono.iloc[-1]

            # Calculate moving averages on chronological data
            ma_50 = prices_chrono.rolling(window=50, min_periods=50).mean().iloc[-1]
            ma_150 = prices_chrono.rolling(window=150, min_periods=150).mean().iloc[-1]
            ma_200 = prices_chrono.rolling(window=200, min_periods=200).mean().iloc[-1]
            ma_200_month_ago = (
                prices_chrono.rolling(window=200, min_periods=200).mean().iloc[-21]
                if len(prices_chrono) > 220
                else ma_200
            )

            # Calculate EMAs (for filtering)
            ema_10 = prices_chrono.ewm(span=10, adjust=False).mean().iloc[-1]
            ema_20 = prices_chrono.ewm(span=20, adjust=False).mean().iloc[-1]
            ema_50 = prices_chrono.ewm(span=50, adjust=False).mean().iloc[-1]

            # EMA distances (% above/below)
            ema_10_distance = ((current_price - ema_10) / ema_10) * 100 if ema_10 > 0 else None
            ema_20_distance = ((current_price - ema_20) / ema_20) * 100 if ema_20 > 0 else None
            ema_50_distance = ((current_price - ema_50) / ema_50) * 100 if ema_50 > 0 else None

            # Performance metrics (week/month price changes)
            perf_week = None
            perf_month = None
            if len(prices_chrono) >= 6:
                price_5d_ago = prices_chrono.iloc[-6]
                if price_5d_ago > 0:
                    perf_week = ((current_price - price_5d_ago) / price_5d_ago) * 100
            if len(prices_chrono) >= 22:
                price_21d_ago = prices_chrono.iloc[-22]
                if price_21d_ago > 0:
                    perf_month = ((current_price - price_21d_ago) / price_21d_ago) * 100

            # Qullamaggie screening metrics
            # 3-month performance (67 trading days) - Qullamaggie requires >=50%
            perf_3m = None
            if len(prices_chrono) >= 68:
                price_67d_ago = prices_chrono.iloc[-68]
                if price_67d_ago > 0:
                    perf_3m = ((current_price - price_67d_ago) / price_67d_ago) * 100

            # 6-month performance (126 trading days) - Qullamaggie requires >=150%
            perf_6m = None
            if len(prices_chrono) >= 127:
                price_126d_ago = prices_chrono.iloc[-127]
                if price_126d_ago > 0:
                    perf_6m = ((current_price - price_126d_ago) / price_126d_ago) * 100

            # Gap percentage (Episodic Pivot) - requires >=10%
            gap_percent = None
            if len(price_data) >= 2:
                today_open = price_data["Open"].iloc[-1]
                yesterday_close = price_data["Close"].iloc[-2]
                if yesterday_close > 0:
                    gap_percent = ((today_open - yesterday_close) / yesterday_close) * 100

            # Volume surge ratio (Episodic Pivot) - requires >=2.0
            volume_surge = None
            if len(volumes_chrono) >= 51:
                today_volume = volumes_chrono.iloc[-1]
                avg_volume_50d = volumes_chrono.iloc[-51:-1].mean()
                if avg_volume_50d > 0:
                    volume_surge = today_volume / avg_volume_50d

            # Reverse for calculations that expect most recent first
            prices = prices_chrono[::-1].reset_index(drop=True)
            volumes = volumes_chrono[::-1].reset_index(drop=True)
            spy_prices = spy_prices_chrono[::-1].reset_index(drop=True)

            # 52-week range
            high_52w = prices.max()
            low_52w = prices.min()

            # 1. Calculate RS Ratings (weighted + individual periods)
            rs_ratings = self.rs_calc.calculate_all_rs_ratings(
                symbol,
                prices,
                spy_prices,
                data.rs_universe_performances,
            )

            # For backward compatibility, build rs_result dict
            weighted_universe = (
                data.rs_universe_performances.get("weighted")
                if data.rs_universe_performances
                else None
            )
            weighted_rs_detail = self.rs_calc.calculate_rs_rating(
                symbol,
                prices,
                spy_prices,
                weighted_universe,
            )
            rs_result = {
                'rs_rating': rs_ratings['rs_rating'],
                'relative_performance': weighted_rs_detail.get("relative_performance"),
                'percentile_rank': weighted_rs_detail.get("percentile_rank"),
            }

            # Calculate Beta and Beta-Adjusted RS metrics
            beta_metrics = self.beta_calc.calculate_all_beta_metrics(
                prices_chrono,
                spy_prices_chrono,
                rs_rating=rs_ratings['rs_rating'],
                rs_rating_1m=rs_ratings['rs_rating_1m'],
                rs_rating_3m=rs_ratings['rs_rating_3m'],
                rs_rating_12m=rs_ratings['rs_rating_12m']
            )

            # 2. Check MA Alignment
            ma_analysis = self.ma_analyzer.comprehensive_ma_analysis(
                current_price,
                ma_50,
                ma_150,
                ma_200,
                ma_200_month_ago
            )

            # 3. Check Stage
            ma_200_series_chrono = prices_chrono.rolling(window=200, min_periods=200).mean()
            ma_200_series = ma_200_series_chrono[::-1].reset_index(drop=True)

            stage_result = self.stage_analyzer.determine_stage(
                current_price,
                ma_200,
                ma_200_series,
                prices,
                volumes
            )

            # 4. Check 52-week positioning
            position_52w = self._check_52w_position(
                current_price,
                high_52w,
                low_52w
            )

            # 5. VCP Detection (optional)
            vcp_result = None
            if include_vcp:
                try:
                    vcp_result = self.vcp_detector.detect_vcp(prices, volumes)
                except Exception as e:
                    logger.warning(f"VCP detection failed for {symbol}: {e}")

            # Calculate Minervini score
            score_result = self._calculate_minervini_score(
                rs_result,
                ma_analysis,
                stage_result,
                position_52w,
                vcp_result
            )

            # Build breakdown dict
            breakdown = {
                component: details["points"]
                for component, details in score_result["breakdown"].items()
            }

            # Build details dict
            details = {
                "rs_rating": rs_ratings["rs_rating"],
                "rs_rating_1m": rs_ratings["rs_rating_1m"],
                "rs_rating_3m": rs_ratings["rs_rating_3m"],
                "rs_rating_12m": rs_ratings["rs_rating_12m"],
                "rs_sparkline_data": rs_sparkline_result["rs_data"],
                "rs_trend": rs_sparkline_result["rs_trend"],
                "price_sparkline_data": price_sparkline_result["price_data"],
                "price_change_1d": price_sparkline_result["price_change_1d"],
                "price_trend": price_sparkline_result["price_trend"],
                "stage": stage_result["stage"],
                "stage_name": stage_result["stage_name"],
                "ma_alignment": ma_analysis["meets_all_criteria"],
                "ma_50": float(ma_50),
                "ma_150": float(ma_150),
                "ma_200": float(ma_200),
                "high_52w": float(high_52w),
                "low_52w": float(low_52w),
                "above_52w_low_pct": position_52w["above_low_pct"],
                "from_52w_high_pct": position_52w["from_high_pct"],
                "vcp_detected": vcp_result["vcp_detected"] if vcp_result else None,
                "vcp_score": vcp_result["vcp_score"] if vcp_result else None,
                "vcp_pivot": vcp_result.get("pivot_info", {}).get("pivot") if vcp_result else None,
                "vcp_ready_for_breakout": vcp_result.get("pivot_info", {}).get("ready_for_breakout") if vcp_result else None,
                "vcp_contraction_ratio": vcp_result.get("contraction_ratio") if vcp_result else None,
                "vcp_atr_score": vcp_result.get("atr_score") if vcp_result else None,
                "adr_percent": adr_percent,
                "eps_growth_qq": quarterly_growth.get('eps_growth_qq'),
                "sales_growth_qq": quarterly_growth.get('sales_growth_qq'),
                "eps_growth_yy": quarterly_growth.get('eps_growth_yy'),
                "sales_growth_yy": quarterly_growth.get('sales_growth_yy'),
                # EMA distances for filtering
                "ema_10_distance": round(ema_10_distance, 2) if ema_10_distance is not None else None,
                "ema_20_distance": round(ema_20_distance, 2) if ema_20_distance is not None else None,
                "ema_50_distance": round(ema_50_distance, 2) if ema_50_distance is not None else None,
                # Performance metrics for filtering
                "perf_week": round(perf_week, 2) if perf_week is not None else None,
                "perf_month": round(perf_month, 2) if perf_month is not None else None,
                # Qullamaggie extended performance metrics
                "perf_3m": round(perf_3m, 2) if perf_3m is not None else None,
                "perf_6m": round(perf_6m, 2) if perf_6m is not None else None,
                # Episodic Pivot metrics
                "gap_percent": round(gap_percent, 2) if gap_percent is not None else None,
                "volume_surge": round(volume_surge, 2) if volume_surge is not None else None,
                # Beta and Beta-Adjusted RS metrics
                "beta": beta_metrics.get("beta"),
                "beta_adj_rs": beta_metrics.get("beta_adj_rs"),
                "beta_adj_rs_1m": beta_metrics.get("beta_adj_rs_1m"),
                "beta_adj_rs_3m": beta_metrics.get("beta_adj_rs_3m"),
                "beta_adj_rs_12m": beta_metrics.get("beta_adj_rs_12m"),
                "full_analysis": {
                    "rs": rs_result,
                    "ma_analysis": ma_analysis,
                    "stage": stage_result,
                    "position_52w": position_52w,
                    "vcp": vcp_result,
                    "quarterly_growth": quarterly_growth,
                    "adr": {
                        "adr_percent": adr_percent,
                        "period_days": 20
                    }
                }
            }

            # Calculate rating
            rating = self.calculate_rating(score_result["score"], details)

            return ScreenerResult(
                score=score_result["score"],
                passes=score_result["passes_template"],
                rating=rating,
                breakdown=breakdown,
                details=details,
                screener_name=self.screener_name
            )

        except Exception as e:
            logger.error(f"Error scanning {symbol} with Minervini: {e}")
            return self._error_result(symbol, str(e))

    def calculate_rating(self, score: float, details: Dict) -> str:
        """
        Calculate human-readable rating from score.

        Args:
            score: Numeric score (0-100)
            details: Analysis details

        Returns:
            Rating string
        """
        # Check if passes template (strict criteria)
        rs_rating = details.get("rs_rating", 0)
        stage = details.get("stage", 0)
        passes_template = score >= 70 and rs_rating >= 70 and stage == 2

        if passes_template:
            if score >= 85:
                return "Strong Buy"
            else:
                return "Buy"
        elif score >= 60:
            return "Watch"
        else:
            return "Pass"

    def _check_52w_position(
        self,
        current_price: float,
        high_52w: float,
        low_52w: float
    ) -> Dict:
        """
        Check position relative to 52-week range.

        Minervini requires:
        - At least 30% above 52-week low
        - Within 25% of 52-week high (ideal is <10%)
        """
        if low_52w == 0 or high_52w == 0:
            return {
                "above_low_pct": None,
                "from_high_pct": None,
                "meets_low_criteria": False,
                "meets_high_criteria": False,
            }

        above_low_pct = ((current_price - low_52w) / low_52w) * 100
        from_high_pct = ((high_52w - current_price) / high_52w) * 100

        return {
            "above_low_pct": round(above_low_pct, 2),
            "from_high_pct": round(from_high_pct, 2),
            "meets_low_criteria": above_low_pct >= 30,
            "meets_high_criteria": from_high_pct <= 25,
        }

    def _calculate_minervini_score(
        self,
        rs_result: Dict,
        ma_analysis: Dict,
        stage_result: Dict,
        position_52w: Dict,
        vcp_result: Optional[Dict]
    ) -> Dict:
        """
        Calculate overall Minervini template score (0-100).

        Scoring weights:
        - RS Rating: 20 points
        - Stage 2: 20 points
        - MA Alignment: 15 points
        - 200-day MA Trend: 10 points (included in MA score)
        - 52-week Position: 15 points
        - VCP Pattern: 20 points (optional)
        """
        score = 0
        breakdown = {}

        # 1. RS Rating (20 points)
        rs_rating = rs_result["rs_rating"]
        if rs_rating >= 80:
            rs_points = 20
        elif rs_rating >= 70:
            rs_points = 15
        elif rs_rating >= 60:
            rs_points = 10
        else:
            rs_points = (rs_rating / 60) * 10  # Partial credit

        score += rs_points
        breakdown["rs_rating"] = {
            "points": round(rs_points, 2),
            "max_points": 20,
            "value": rs_rating,
            "passes": rs_rating >= 70,
        }

        # 2. Stage 2 (20 points)
        stage = stage_result["stage"]
        if stage == 2:
            stage_points = 20
        elif stage == 1:
            stage_points = 10  # Basing, potential setup
        else:
            stage_points = 0  # Stage 3 or 4

        score += stage_points
        breakdown["stage"] = {
            "points": stage_points,
            "max_points": 20,
            "value": stage,
            "passes": stage == 2,
        }

        # 3. MA Alignment (15 points)
        ma_score = ma_analysis["minervini_ma_score"]
        ma_points = (ma_score / 100) * 15

        score += ma_points
        breakdown["ma_alignment"] = {
            "points": round(ma_points, 2),
            "max_points": 15,
            "value": ma_score,
            "passes": ma_analysis["meets_all_criteria"],
        }

        # 4. 52-week Position (15 points)
        position_points = 0
        if position_52w["meets_low_criteria"]:
            position_points += 7
        if position_52w["meets_high_criteria"]:
            position_points += 8

        score += position_points
        breakdown["position_52w"] = {
            "points": position_points,
            "max_points": 15,
            "above_low": position_52w["above_low_pct"],
            "from_high": position_52w["from_high_pct"],
            "passes": position_52w["meets_low_criteria"] and position_52w["meets_high_criteria"],
        }

        # 5. VCP Pattern (20 points, optional)
        if vcp_result:
            vcp_points = (vcp_result["vcp_score"] / 100) * 20
            score += vcp_points
            breakdown["vcp"] = {
                "points": round(vcp_points, 2),
                "max_points": 20,
                "value": vcp_result["vcp_score"],
                "passes": vcp_result["vcp_detected"],
            }
        else:
            # If VCP not calculated, redistribute points proportionally
            score = (score / 80) * 100  # Scale up to 100

        # Determine if passes template (score >= 70)
        passes_template = score >= 70 and rs_rating >= 70 and stage == 2

        return {
            "score": round(score, 2),
            "passes_template": passes_template,
            "breakdown": breakdown,
        }

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
