"""
Volume Breakthrough Screener

Detects exceptional volume spikes by checking if any day in the last 5 trading
days had record-level volume across different timeframes.

Detection criteria:
1. Five-Year High Volume: Volume is highest in the past 5 years (1260 days)
2. One-Year High Volume: Volume is highest in the past year (252 days)
3. Since-IPO High Volume: Volume is highest since IPO (excluding early period)

Lookback window:
- Checks the last 5 trading days for breakthroughs (not just today)
- Reports which day the breakthrough occurred (e.g., "Yesterday", "2 days ago")
- Applies score decay for older breakthroughs (today=100%, yesterday=95%, etc.)

Scoring:
- Base: 33 points per breakthrough
- Bonus: +10 points if volume is 20%+ above previous high
- Decay: Score reduced by 5% per day for older breakthroughs
- Max possible: 129 points (3 breakthroughs with bonuses, today)

Pass threshold: At least one breakthrough (33+ points after decay)

Note: Changed from "all-time" to "5-year" for long-term check to avoid
split-unadjusted historical data issues.
"""
import logging
from typing import Dict, Optional
import pandas as pd

from app.domain.scanning.scoring import CRITERION_MAX_POINTS_BY_SCREENER

from .base_screener import BaseStockScreener, ScreenerResult, StockData, DataRequirements
from .screener_registry import register_screener

logger = logging.getLogger(__name__)


@register_screener
class VolumeBreakthroughScanner(BaseStockScreener):
    """
    Screener for detecting record-breaking volume activity.

    Identifies stocks experiencing exceptional volume spikes by comparing
    current day's volume against historical highs across multiple timeframes.
    """

    CRITERION_MAX_POINTS = CRITERION_MAX_POINTS_BY_SCREENER["volume_breakthrough"]

    @property
    def screener_name(self) -> str:
        """Return screener identifier."""
        return "volume_breakthrough"

    def needs_overall_rs(self, criteria: Optional[Dict] = None) -> bool:
        return False

    def get_data_requirements(self, criteria: Optional[Dict] = None) -> DataRequirements:
        """
        Specify data requirements for volume breakthrough detection.

        Returns:
            DataRequirements with maximum price history needed
        """
        return DataRequirements(
            price_period="5y",            # Need 5 years for 5-year high volume check
            needs_fundamentals=True,      # Need IPO date for since-IPO check
            needs_quarterly_growth=False,
            needs_benchmark=False,
            needs_earnings_history=False
        )

    # Number of recent trading days to check for breakthroughs
    LOOKBACK_RECENT_DAYS = 5

    # Score decay multipliers for older breakthroughs (most recent gets full score)
    DECAY_MULTIPLIERS = [1.0, 0.95, 0.90, 0.85, 0.80]

    def scan_stock(
        self,
        symbol: str,
        data: StockData,
        criteria: Optional[Dict] = None
    ) -> ScreenerResult:
        """
        Scan a stock for volume breakthroughs in the last 5 trading days.

        Checks each of the last 5 days for volume breakthroughs against
        5-year high, 1-year high, and since-IPO high. Returns the best
        breakthrough found (highest score), with a slight decay for older
        breakthroughs to prioritize recent signals.

        Args:
            symbol: Stock ticker symbol
            data: Stock data container with price history
            criteria: Optional criteria (not used for this screener)

        Returns:
            ScreenerResult with score, passes flag, rating, and details
        """
        try:
            # Validate data availability
            if not data.has_sufficient_data(min_days=252):  # Need at least 1 year
                return self._insufficient_data_result(
                    symbol,
                    "Insufficient volume history (need at least 1 year of data)"
                )

            price_data = data.price_data

            if price_data is None or price_data.empty:
                return self._insufficient_data_result(symbol, "No price data available")

            if 'Volume' not in price_data.columns:
                return self._insufficient_data_result(symbol, "Volume data not available")

            # Warn if less than 5 years available
            if len(price_data) < 1260:
                logger.warning(f"{symbol}: Only {len(price_data)} days of data available (< 5 years). "
                              f"5-year high detection may be inaccurate for young stocks.")

            # Search through recent days for breakthroughs
            best_result = None
            best_score = 0

            for day_offset in range(min(self.LOOKBACK_RECENT_DAYS, len(price_data) - 252)):
                # Run detection methods for this day
                all_time_result = self._check_all_time_high_volume(price_data, day_offset)
                one_year_result = self._check_one_year_high_volume(price_data, day_offset)
                since_ipo_result = self._check_since_ipo_high_volume(price_data, data.fundamentals, day_offset)

                # Check if any breakthrough found for this day
                has_breakthrough = any([
                    all_time_result["is_breakthrough"],
                    one_year_result["is_breakthrough"],
                    since_ipo_result["is_breakthrough"]
                ])

                if has_breakthrough:
                    # Calculate base score
                    score_result = self._calculate_score(
                        all_time_result,
                        one_year_result,
                        since_ipo_result
                    )

                    # Apply time decay to score (more recent = higher priority)
                    decay_multiplier = self.DECAY_MULTIPLIERS[day_offset] if day_offset < len(self.DECAY_MULTIPLIERS) else 0.75
                    decayed_score = score_result["score"] * decay_multiplier

                    # Track best breakthrough
                    if decayed_score > best_score:
                        best_score = decayed_score
                        best_result = {
                            "all_time_result": all_time_result,
                            "one_year_result": one_year_result,
                            "since_ipo_result": since_ipo_result,
                            "score_result": score_result,
                            "day_offset": day_offset,
                            "decayed_score": decayed_score,
                            "decay_multiplier": decay_multiplier
                        }

            # If no breakthrough found in any of the recent days, use today's data for "no breakthrough" result
            if best_result is None:
                all_time_result = self._check_all_time_high_volume(price_data, 0)
                one_year_result = self._check_one_year_high_volume(price_data, 0)
                since_ipo_result = self._check_since_ipo_high_volume(price_data, data.fundamentals, 0)
                score_result = self._calculate_score(all_time_result, one_year_result, since_ipo_result)

                return self._build_result(
                    all_time_result, one_year_result, since_ipo_result,
                    score_result, day_offset=0, decay_multiplier=1.0
                )

            # Build result from best breakthrough
            return self._build_result(
                best_result["all_time_result"],
                best_result["one_year_result"],
                best_result["since_ipo_result"],
                best_result["score_result"],
                best_result["day_offset"],
                best_result["decay_multiplier"]
            )

        except Exception as e:
            logger.error(f"Error scanning {symbol} for volume breakthroughs: {e}", exc_info=True)
            return self._error_result(symbol, str(e))

    MAX_RAW_SCORE = float(sum(CRITERION_MAX_POINTS.values()))

    def _build_result(
        self,
        all_time_result: Dict,
        one_year_result: Dict,
        since_ipo_result: Dict,
        score_result: Dict,
        day_offset: int,
        decay_multiplier: float
    ) -> ScreenerResult:
        """Build the ScreenerResult from component results."""
        # Apply decay and normalize score to 0-100
        raw_score = score_result["score"] * decay_multiplier
        normalized_score = round((raw_score / self.MAX_RAW_SCORE) * 100, 2)

        # Build breakdown for scoring transparency
        breakdown = {
            "five_year_high": all_time_result["points"],
            "one_year_high": one_year_result["points"],
            "since_ipo_high": since_ipo_result["points"],
            "bonus_points": score_result.get("bonus_points", 0),
            "decay_multiplier": decay_multiplier
        }

        # Generate breakthrough label
        if day_offset == 0:
            breakthrough_label = "Today"
        elif day_offset == 1:
            breakthrough_label = "Yesterday"
        else:
            breakthrough_label = f"{day_offset} days ago"

        # Get breakthrough date from one of the results
        breakthrough_date = all_time_result.get("breakthrough_date")
        breakthrough_date_str = breakthrough_date.strftime('%Y-%m-%d') if breakthrough_date is not None else None

        # Build detailed results
        details = {
            "current_volume": all_time_result["current_volume"],
            "five_year_high_volume": all_time_result["all_time_high"],
            "one_year_high_volume": one_year_result["one_year_high"],
            "since_ipo_high_volume": since_ipo_result["since_ipo_high"],
            "is_five_year_high": all_time_result["is_breakthrough"],
            "is_one_year_high": one_year_result["is_breakthrough"],
            "is_since_ipo_high": since_ipo_result["is_breakthrough"],
            "magnitude_vs_five_year": all_time_result["magnitude_pct"],
            "magnitude_vs_one_year": one_year_result["magnitude_pct"],
            "magnitude_vs_ipo": since_ipo_result["magnitude_pct"],
            "breakthrough_count": score_result["breakthrough_count"],
            "days_ago": day_offset,
            "breakthrough_label": breakthrough_label,
            "breakthrough_date": breakthrough_date_str,
        }

        # Calculate rating based on breakthrough count and magnitude (not normalized score)
        rating = self.calculate_rating(raw_score, details)

        return ScreenerResult(
            score=normalized_score,
            passes=score_result["passes"],
            rating=rating,
            breakdown=breakdown,
            details=details,
            screener_name=self.screener_name
        )

    def _check_all_time_high_volume(self, price_data: pd.DataFrame, day_offset: int = 0) -> Dict:
        """
        Check if a specific day's volume is highest in the last 5 years.

        Note: Changed from "all-time" to "5-year" to avoid issues with
        split-unadjusted historical data that can make old volumes appear
        artificially inflated.

        Args:
            price_data: DataFrame with Volume column
            day_offset: Days back from most recent (0=today, 1=yesterday, etc.)

        Returns:
            Dict with points, is_breakthrough, current_volume, all_time_high, magnitude_pct
        """
        volumes = price_data["Volume"]

        # Get the day we're checking (offset from end)
        check_index = -(day_offset + 1)
        check_volume = float(volumes.iloc[check_index])
        check_date = price_data.index[check_index]

        # Handle NaN volume data
        if pd.isna(check_volume):
            return {
                "points": 0,
                "max_points": self.CRITERION_MAX_POINTS["five_year_high"],
                "is_breakthrough": False,
                "current_volume": 0,
                "all_time_high": 0,
                "magnitude_pct": 0,
                "breakthrough_date": check_date,
                "days_ago": day_offset,
                "reason": "Volume data unavailable (NaN)"
            }

        # Get last 5 years of data BEFORE the day we're checking
        # 5 years * 252 trading days = 1260 days
        available_before_check = len(volumes) - 1 - day_offset
        lookback_days = min(1260, available_before_check)

        if day_offset == 0:
            historical_volumes = volumes.iloc[-(lookback_days+1):-1]
        else:
            # Slice from lookback start to just before the check day
            start_idx = -(lookback_days + day_offset + 1)
            end_idx = check_index
            historical_volumes = volumes.iloc[start_idx:end_idx]

        if len(historical_volumes) < 100:  # Need reasonable history
            # Fallback to all available data before the check day
            if day_offset == 0:
                historical_volumes = volumes.iloc[:-1]
            else:
                historical_volumes = volumes.iloc[:check_index]

        all_time_high = float(historical_volumes.max())

        # Handle NaN historical volume
        if pd.isna(all_time_high):
            return {
                "points": 0,
                "max_points": self.CRITERION_MAX_POINTS["five_year_high"],
                "is_breakthrough": False,
                "current_volume": int(check_volume),
                "all_time_high": 0,
                "magnitude_pct": 0,
                "breakthrough_date": check_date,
                "days_ago": day_offset,
                "reason": "Historical volume data unavailable"
            }

        all_time_high_date = historical_volumes.idxmax()

        # Log diagnostic info
        logger.debug(
            f"Volume breakthrough check (5-year) - Check: {check_volume:,.0f} on {check_date} | "
            f"5-year high: {all_time_high:,.0f} on {all_time_high_date} | "
            f"Lookback: {len(historical_volumes)} days | Day offset: {day_offset}"
        )

        # Calculate magnitude (percentage above previous high)
        if all_time_high > 0:
            magnitude_pct = ((check_volume - all_time_high) / all_time_high) * 100
        else:
            magnitude_pct = 0.0

        # Check if breakthrough
        is_breakthrough = check_volume > all_time_high

        # Calculate points
        base_points = 33.0 if is_breakthrough else 0.0
        bonus_points = 10.0 if (is_breakthrough and magnitude_pct >= 20) else 0.0

        return {
            "points": base_points + bonus_points,
            "max_points": self.CRITERION_MAX_POINTS["five_year_high"],
            "is_breakthrough": is_breakthrough,
            "current_volume": int(check_volume),
            "all_time_high": int(all_time_high),
            "magnitude_pct": round(magnitude_pct, 2),
            "breakthrough_date": check_date,
            "days_ago": day_offset,
            "reason": f"Volume {check_volume:,.0f} on {check_date.strftime('%Y-%m-%d')} vs 5-year high {all_time_high:,.0f}"
        }

    def _check_one_year_high_volume(self, price_data: pd.DataFrame, day_offset: int = 0) -> Dict:
        """
        Check if a specific day's volume is highest in the last year (252 trading days).

        Args:
            price_data: DataFrame with Volume column
            day_offset: Days back from most recent (0=today, 1=yesterday, etc.)

        Returns:
            Dict with points, is_breakthrough, current_volume, one_year_high, magnitude_pct
        """
        volumes = price_data["Volume"]

        # Get the day we're checking (offset from end)
        check_index = -(day_offset + 1)
        check_volume = float(volumes.iloc[check_index])
        check_date = price_data.index[check_index]

        # Handle NaN volume data
        if pd.isna(check_volume):
            return {
                "points": 0,
                "max_points": self.CRITERION_MAX_POINTS["one_year_high"],
                "is_breakthrough": False,
                "current_volume": 0,
                "one_year_high": 0,
                "magnitude_pct": 0,
                "breakthrough_date": check_date,
                "days_ago": day_offset,
                "reason": "Volume data unavailable (NaN)"
            }

        # Get last year's data BEFORE the day we're checking
        # 252 trading days = ~1 year
        available_before_check = len(volumes) - 1 - day_offset
        lookback_days = min(252, available_before_check)

        if day_offset == 0:
            year_volumes = volumes.iloc[-(lookback_days+1):-1]
        else:
            start_idx = -(lookback_days + day_offset + 1)
            end_idx = check_index
            year_volumes = volumes.iloc[start_idx:end_idx]

        if len(year_volumes) < 50:  # Need reasonable history
            return {
                "points": 0,
                "max_points": self.CRITERION_MAX_POINTS["one_year_high"],
                "is_breakthrough": False,
                "current_volume": int(check_volume),
                "one_year_high": 0,
                "magnitude_pct": 0,
                "breakthrough_date": check_date,
                "days_ago": day_offset,
                "reason": "Insufficient 1-year history"
            }

        one_year_high = float(year_volumes.max())

        # Handle NaN historical volume
        if pd.isna(one_year_high):
            return {
                "points": 0,
                "max_points": self.CRITERION_MAX_POINTS["one_year_high"],
                "is_breakthrough": False,
                "current_volume": int(check_volume),
                "one_year_high": 0,
                "magnitude_pct": 0,
                "breakthrough_date": check_date,
                "days_ago": day_offset,
                "reason": "Historical volume data unavailable"
            }

        one_year_high_date = year_volumes.idxmax()

        # Log diagnostic info
        logger.debug(
            f"1-year volume check - Check: {check_volume:,.0f} on {check_date} | "
            f"1-year high: {one_year_high:,.0f} on {one_year_high_date} | Day offset: {day_offset}"
        )

        if one_year_high > 0:
            magnitude_pct = ((check_volume - one_year_high) / one_year_high) * 100
        else:
            magnitude_pct = 0.0

        is_breakthrough = check_volume > one_year_high

        base_points = 33.0 if is_breakthrough else 0.0
        bonus_points = 10.0 if (is_breakthrough and magnitude_pct >= 20) else 0.0

        return {
            "points": base_points + bonus_points,
            "max_points": self.CRITERION_MAX_POINTS["one_year_high"],
            "is_breakthrough": is_breakthrough,
            "current_volume": int(check_volume),
            "one_year_high": int(one_year_high),
            "magnitude_pct": round(magnitude_pct, 2),
            "breakthrough_date": check_date,
            "days_ago": day_offset,
            "reason": f"Volume {check_volume:,.0f} on {check_date.strftime('%Y-%m-%d')} vs 1-year high {one_year_high:,.0f}"
        }

    def _check_since_ipo_high_volume(self, price_data: pd.DataFrame, fundamentals: Optional[Dict] = None, day_offset: int = 0) -> Dict:
        """
        Check if a specific day's volume is highest since IPO using real IPO date.

        Uses real IPO date from fundamentals if available, otherwise falls back
        to heuristic (skip first 10% of data).

        Args:
            price_data: DataFrame with Volume column
            fundamentals: Optional fundamentals dict with ipo_date or first_trade_date
            day_offset: Days back from most recent (0=today, 1=yesterday, etc.)

        Returns:
            Dict with points, is_breakthrough, current_volume, since_ipo_high, magnitude_pct
        """
        volumes = price_data["Volume"]

        # Get the day we're checking (offset from end)
        check_index = -(day_offset + 1)
        check_volume = float(volumes.iloc[check_index])
        check_date = price_data.index[check_index]

        # Handle NaN volume data
        if pd.isna(check_volume):
            return {
                "points": 0,
                "max_points": self.CRITERION_MAX_POINTS["since_ipo_high"],
                "is_breakthrough": False,
                "current_volume": 0,
                "since_ipo_high": 0,
                "magnitude_pct": 0,
                "used_real_ipo_date": False,
                "breakthrough_date": check_date,
                "days_ago": day_offset,
                "reason": "Volume data unavailable (NaN)"
            }

        # Try to use real IPO date from fundamentals
        ipo_skip_idx = 0
        used_real_ipo = False

        if fundamentals:
            ipo_date = None

            # Try ipo_date (date object) or first_trade_date (epoch)
            if "ipo_date" in fundamentals and fundamentals["ipo_date"]:
                ipo_date = pd.Timestamp(fundamentals["ipo_date"])
            elif "first_trade_date" in fundamentals and fundamentals["first_trade_date"]:
                ipo_date = pd.Timestamp.fromtimestamp(fundamentals["first_trade_date"])

            if ipo_date:
                # Skip first 90 days after IPO (stabilization period)
                ipo_stabilization_date = ipo_date + pd.Timedelta(days=90)

                # Handle timezone mismatch: localize to match price_data index
                if price_data.index.tz is not None and ipo_stabilization_date.tz is None:
                    ipo_stabilization_date = ipo_stabilization_date.tz_localize(price_data.index.tz)
                elif price_data.index.tz is None and ipo_stabilization_date.tz is not None:
                    ipo_stabilization_date = ipo_stabilization_date.tz_convert(None)

                post_ipo_mask = price_data.index >= ipo_stabilization_date

                if post_ipo_mask.any():
                    # Get integer index of first post-IPO date (argmax returns first True in boolean array)
                    ipo_skip_idx = int(post_ipo_mask.argmax())
                    used_real_ipo = True
                    logger.debug(f"Using real IPO date: {ipo_date.strftime('%Y-%m-%d')}")

        # Fallback to heuristic if no IPO date
        if ipo_skip_idx == 0:
            ipo_skip_idx = max(1, len(volumes) // 10)
            logger.debug(f"No IPO date, using 10% heuristic: skipping {ipo_skip_idx} days")

        # Get post-IPO volumes BEFORE the day we're checking
        if day_offset == 0:
            post_ipo_volumes = volumes.iloc[ipo_skip_idx:-1]
        else:
            post_ipo_volumes = volumes.iloc[ipo_skip_idx:check_index]

        if len(post_ipo_volumes) < 50:
            if day_offset == 0:
                post_ipo_volumes = volumes.iloc[:-1]
            else:
                post_ipo_volumes = volumes.iloc[:check_index]
            logger.debug(f"Insufficient post-IPO data, using all available")

        since_ipo_high = float(post_ipo_volumes.max())

        # Handle NaN historical volume
        if pd.isna(since_ipo_high):
            return {
                "points": 0,
                "max_points": self.CRITERION_MAX_POINTS["since_ipo_high"],
                "is_breakthrough": False,
                "current_volume": int(check_volume),
                "since_ipo_high": 0,
                "magnitude_pct": 0,
                "used_real_ipo_date": used_real_ipo,
                "breakthrough_date": check_date,
                "days_ago": day_offset,
                "reason": "Historical volume data unavailable"
            }

        if since_ipo_high > 0:
            magnitude_pct = ((check_volume - since_ipo_high) / since_ipo_high) * 100
        else:
            magnitude_pct = 0.0

        is_breakthrough = check_volume > since_ipo_high
        base_points = 33.0 if is_breakthrough else 0.0
        bonus_points = 10.0 if (is_breakthrough and magnitude_pct >= 20) else 0.0

        return {
            "points": base_points + bonus_points,
            "max_points": self.CRITERION_MAX_POINTS["since_ipo_high"],
            "is_breakthrough": is_breakthrough,
            "current_volume": int(check_volume),
            "since_ipo_high": int(since_ipo_high),
            "magnitude_pct": round(magnitude_pct, 2),
            "used_real_ipo_date": used_real_ipo,
            "breakthrough_date": check_date,
            "days_ago": day_offset,
            "reason": f"Volume {check_volume:,.0f} on {check_date.strftime('%Y-%m-%d')} vs post-IPO high {since_ipo_high:,.0f}"
        }

    def _calculate_score(
        self,
        all_time_result: Dict,
        one_year_result: Dict,
        since_ipo_result: Dict
    ) -> Dict:
        """
        Combine all breakthrough checks into overall score.

        Args:
            all_time_result: Results from all-time high check
            one_year_result: Results from 1-year high check
            since_ipo_result: Results from since-IPO high check

        Returns:
            Dict with score, passes, breakthrough_count, bonus_points
        """
        total_points = (
            all_time_result["points"] +
            one_year_result["points"] +
            since_ipo_result["points"]
        )

        breakthrough_count = sum([
            all_time_result["is_breakthrough"],
            one_year_result["is_breakthrough"],
            since_ipo_result["is_breakthrough"]
        ])

        # Calculate total bonus points
        bonus_points = sum([
            10.0 if (all_time_result["is_breakthrough"] and all_time_result["magnitude_pct"] >= 20) else 0.0,
            10.0 if (one_year_result["is_breakthrough"] and one_year_result["magnitude_pct"] >= 20) else 0.0,
            10.0 if (since_ipo_result["is_breakthrough"] and since_ipo_result["magnitude_pct"] >= 20) else 0.0,
        ])

        # Passes if at least one breakthrough
        passes = breakthrough_count >= 1

        return {
            "score": round(total_points, 2),
            "passes": passes,
            "breakthrough_count": breakthrough_count,
            "bonus_points": bonus_points
        }

    def calculate_rating(self, score: float, details: Dict) -> str:
        """
        Convert score to human-readable rating.

        Score breakdown:
        - 0-32: No breakthroughs (Pass)
        - 33-65: One breakthrough (Watch)
        - 66-98: Two breakthroughs (Buy)
        - 99+: All three breakthroughs (Strong Buy)

        Bonus: If magnitude is very high (50%+), upgrade rating

        Args:
            score: Composite score (0-129)
            details: Detail dict with breakthrough_count and magnitude values

        Returns:
            Rating string
        """
        breakthrough_count = details.get("breakthrough_count", 0)
        max_magnitude = max(
            details.get("magnitude_vs_five_year", 0),
            details.get("magnitude_vs_one_year", 0),
            details.get("magnitude_vs_ipo", 0)
        )

        if breakthrough_count == 0:
            return "Pass"
        elif breakthrough_count == 1:
            # Upgrade to Buy if magnitude is exceptional
            if max_magnitude >= 50:
                return "Buy"
            return "Watch"
        elif breakthrough_count == 2:
            # Upgrade to Strong Buy if magnitude is exceptional
            if max_magnitude >= 50:
                return "Strong Buy"
            return "Buy"
        else:  # All 3 breakthroughs
            return "Strong Buy"

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
