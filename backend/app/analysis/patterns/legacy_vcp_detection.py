"""
Volatility Contraction Pattern (VCP) detection.

Identifies VCP patterns as described by Mark Minervini:
- Series of 3-4 pullbacks with progressively tighter ranges
- Each pullback is shallower than the previous
- Volume decreases on pullbacks (drying up)
- Price consolidates near highs before breakout
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VCPDetector:
    """
    Detect Volatility Contraction Patterns in stock price data.

    VCP Characteristics:
    1. Base formation with 3-4 distinct pullbacks
    2. Each pullback progressively tighter (contracting volatility)
    3. Volume dries up on each successive pullback
    4. Price remains relatively close to recent highs
    5. Breakout typically occurs on expanding volume
    """

    def __init__(
        self,
        min_bases: int = 3,
        max_bases: int = 5,
        lookback_days: int = 150
    ):
        """
        Initialize VCP detector.

        Args:
            min_bases: Minimum number of consolidation bases required
            max_bases: Maximum number of bases to analyze
            lookback_days: Days of price history to analyze
        """
        self.min_bases = min_bases
        self.max_bases = max_bases
        self.lookback_days = lookback_days

    def find_consolidation_bases(
        self,
        prices: pd.Series,
        min_duration: int = 10
    ) -> List[Dict]:
        """
        Identify consolidation bases in price series.

        A base is a period where price trades in a relatively tight range
        after a pullback from a high.

        Args:
            prices: Price series (most recent first)
            min_duration: Minimum number of days for a base

        Returns:
            List of base dicts with start, end, depth, duration
        """
        if len(prices) < self.lookback_days:
            return []

        recent_prices = prices.iloc[:self.lookback_days]
        bases = []

        # Find local highs (peaks)
        peaks = self._find_peaks(recent_prices)

        if len(peaks) < 2:
            return []

        # Analyze pullbacks between peaks
        # Note: Data is most-recent-first, so lower indices are more recent
        # peaks are sorted ascending, so peaks[i] < peaks[i+1]
        for i in range(len(peaks) - 1):
            peak_idx = peaks[i]  # More recent peak (lower index)
            next_peak_idx = peaks[i + 1]  # Older peak (higher index)

            # Get price range between peaks (from more recent to older)
            segment = recent_prices.iloc[peak_idx:next_peak_idx + 1]

            if len(segment) < min_duration:
                continue

            # Find lowest point in this segment (the pullback low)
            low_idx = segment.idxmin()
            low_price = segment.min()
            high_price = recent_prices.iloc[peak_idx]

            # Calculate pullback depth
            if high_price > 0:
                depth_pct = ((high_price - low_price) / high_price) * 100
            else:
                continue

            # First contraction can be deeper (5-45%), subsequent ones 5-35%
            # This better matches classic VCP patterns where initial pullback is larger
            is_first_base = len(bases) == 0
            if is_first_base:
                valid_depth = 5 <= depth_pct <= 45
            else:
                valid_depth = 5 <= depth_pct <= 35

            if valid_depth:
                bases.append({
                    "start_idx": peak_idx,
                    "end_idx": next_peak_idx,
                    "duration": peak_idx - next_peak_idx,
                    "high_price": high_price,
                    "low_price": low_price,
                    "depth_pct": depth_pct,
                })

        return bases[:self.max_bases]

    def _find_peaks(
        self,
        prices: pd.Series,
        order: int = 4
    ) -> List[int]:
        """
        Find local peaks in price series.

        Args:
            prices: Price series
            order: Number of points on each side to use for comparison
                   (reduced from 5 to 4 for better VCP detection)

        Returns:
            List of indices where peaks occur
        """
        peaks = []
        prices_array = prices.values

        for i in range(order, len(prices_array) - order):
            # Check if this point is higher than surrounding points
            left_window = prices_array[i - order:i]
            right_window = prices_array[i + 1:i + order + 1]

            if (prices_array[i] > np.max(left_window) and
                prices_array[i] > np.max(right_window)):
                peaks.append(i)

        return peaks

    def check_contracting_volatility(
        self,
        bases: List[Dict]
    ) -> Tuple[bool, float, float]:
        """
        Check if pullback depths are contracting.

        Args:
            bases: List of consolidation bases

        Returns:
            (contracting: bool, contraction_score: float, contraction_ratio: float)
        """
        if len(bases) < self.min_bases:
            return False, 0.0, 0.0

        # Bases are ordered most-recent-first, but VCP contraction means
        # oldest pullback is deepest, newest is shallowest.
        # Reverse to get oldest-first for contraction check.
        depths = [base["depth_pct"] for base in reversed(bases)]

        # Count how many successive pullbacks are shallower (oldest to newest)
        # Require 75% to be contracting (allows one exception in 4-base pattern)
        decreasing_count = sum(
            1 for i in range(len(depths) - 1)
            if depths[i] > depths[i + 1]
        )
        total_pairs = len(depths) - 1
        contraction_ratio = decreasing_count / total_pairs if total_pairs > 0 else 0

        # Consider contracting if at least 75% of successive pullbacks are shallower
        contracting = contraction_ratio >= 0.75

        # Calculate contraction score (0-100)
        if contracting:
            # Perfect contraction: each pullback ~50% shallower than previous
            # Score based on how well depths decrease
            ratios = [
                depths[i + 1] / depths[i]
                for i in range(len(depths) - 1)
                if depths[i] > 0
            ]
            avg_ratio = np.mean(ratios) if ratios else 1.0

            # Ideal ratio is around 0.5-0.7 (each pullback 30-50% shallower)
            if 0.4 <= avg_ratio <= 0.7:
                score = 100
            elif 0.3 <= avg_ratio <= 0.8:
                score = 80
            else:
                score = 60

            # Bonus for perfect contraction (all decreasing)
            if contraction_ratio == 1.0:
                score = min(100, score + 10)
        else:
            # Partial credit for near-contracting patterns
            score = contraction_ratio * 50  # 0-50 points based on ratio

        return contracting, score, contraction_ratio

    def check_volume_contraction(
        self,
        bases: List[Dict],
        volumes: pd.Series
    ) -> Tuple[bool, float]:
        """
        Check if volume is decreasing on successive pullbacks.

        Args:
            bases: List of consolidation bases
            volumes: Volume series

        Returns:
            (volume_contracting: bool, volume_score: float)
        """
        if len(bases) < self.min_bases or volumes is None:
            return False, 0.0

        # Calculate average volume for each base
        base_volumes = []
        for base in bases:
            start = base["end_idx"]
            end = base["start_idx"] + 1
            segment_vol = volumes.iloc[start:end]

            if len(segment_vol) > 0:
                avg_vol = segment_vol.mean()
                base_volumes.append(avg_vol)

        if len(base_volumes) < self.min_bases:
            return False, 0.0

        # Check if volume is decreasing
        volume_decreasing = all(
            base_volumes[i] > base_volumes[i + 1]
            for i in range(len(base_volumes) - 1)
        )

        # Calculate volume contraction score
        if volume_decreasing:
            ratios = [
                base_volumes[i + 1] / base_volumes[i]
                for i in range(len(base_volumes) - 1)
            ]
            avg_ratio = np.mean(ratios)

            # Ideal: each base has 20-40% less volume than previous
            if 0.5 <= avg_ratio <= 0.8:
                score = 100
            elif 0.4 <= avg_ratio <= 0.9:
                score = 70
            else:
                score = 50
        else:
            score = 0

        return volume_decreasing, score

    def check_tightness_near_highs(
        self,
        current_price: float,
        recent_high: float,
        max_distance_pct: float = 5.0
    ) -> Tuple[bool, float]:
        """
        Check if current price is tight near recent highs.

        Args:
            current_price: Current stock price
            recent_high: Recent high price
            max_distance_pct: Maximum % from high to be considered "tight"

        Returns:
            (is_tight: bool, tightness_score: float)
        """
        if recent_high == 0:
            return False, 0.0

        distance_pct = ((recent_high - current_price) / recent_high) * 100

        is_tight = distance_pct <= max_distance_pct

        # Score based on how close to highs
        if distance_pct <= 2:
            score = 100
        elif distance_pct <= 5:
            score = 80
        elif distance_pct <= 10:
            score = 50
        else:
            score = 0

        return is_tight, score

    def check_atr_contraction(
        self,
        prices: pd.Series,
        lookback: int = 50
    ) -> Tuple[float, float]:
        """
        Check if Average True Range (ATR) is contracting.

        ATR contraction indicates decreasing volatility, which is a key
        component of VCP patterns.

        Args:
            prices: Price series (most recent first)
            lookback: Number of days to analyze

        Returns:
            (atr_score: float 0-100, contraction_ratio: float)
        """
        if len(prices) < lookback:
            return 0.0, 1.0

        # Calculate simple daily ranges (high - low approximation using close)
        # For a more accurate ATR, we'd need high/low data
        price_changes = prices.diff().abs()

        # Use 14-day rolling average of absolute price changes as ATR proxy
        atr_proxy = price_changes.rolling(window=14, min_periods=14).mean()

        # Drop NaN values and check we have enough data
        atr_valid = atr_proxy.dropna()
        if len(atr_valid) < 40:
            return 0.0, 1.0

        # Compare recent ATR to past ATR
        # Use valid data only (skip initial NaN values from rolling window)
        recent_atr = atr_valid.iloc[:10].mean()
        past_atr = atr_valid.iloc[30:40].mean()

        if past_atr <= 0 or pd.isna(past_atr) or pd.isna(recent_atr):
            return 0.0, 1.0

        contraction_ratio = recent_atr / past_atr

        # Score: lower ratio (more contraction) = higher score
        # Ideal contraction is 0.5-0.7 (ATR decreased by 30-50%)
        if contraction_ratio <= 0.5:
            score = 100
        elif contraction_ratio <= 0.7:
            score = 80
        elif contraction_ratio <= 0.85:
            score = 60
        elif contraction_ratio <= 1.0:
            score = 40
        else:
            score = 0  # ATR expanding, not contracting

        return score, contraction_ratio

    def find_pivot_point(
        self,
        bases: List[Dict],
        current_price: float
    ) -> Dict:
        """
        Identify the VCP pivot point for breakout entry.

        The pivot is typically the highest high of recent consolidation,
        representing the resistance level to break through.

        Args:
            bases: List of consolidation bases
            current_price: Current stock price

        Returns:
            Dict with pivot information
        """
        if not bases:
            return {
                "pivot": None,
                "distance_pct": None,
                "ready_for_breakout": False
            }

        # Pivot is the highest high of the last 2 bases (most recent consolidation)
        recent_bases = bases[:2] if len(bases) >= 2 else bases
        pivot = max(base["high_price"] for base in recent_bases)

        # Calculate distance from current price to pivot
        if pivot > 0:
            distance_pct = ((pivot - current_price) / current_price) * 100
        else:
            distance_pct = None

        # Ready for breakout if within 3% of pivot
        ready_for_breakout = distance_pct is not None and distance_pct <= 3

        return {
            "pivot": round(pivot, 2) if pivot else None,
            "distance_pct": round(distance_pct, 2) if distance_pct is not None else None,
            "ready_for_breakout": ready_for_breakout
        }

    def detect_vcp(
        self,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None
    ) -> Dict:
        """
        Detect VCP pattern in price/volume data.

        Args:
            prices: Price series (most recent first)
            volumes: Volume series (optional)

        Returns:
            Dict with VCP detection results and score
        """
        # Find consolidation bases
        bases = self.find_consolidation_bases(prices)

        current_price = prices.iloc[0]

        if len(bases) < self.min_bases:
            return {
                "vcp_detected": False,
                "vcp_score": 0,
                "num_bases": len(bases),
                "contracting_depth": False,
                "contraction_ratio": 0.0,
                "contracting_volume": False,
                "tight_near_highs": False,
                "atr_score": 0,
                "pivot_info": self.find_pivot_point(bases, current_price),
                "details": "Insufficient consolidation bases",
            }

        # Check for contracting volatility (depth of pullbacks)
        contracting_depth, depth_score, contraction_ratio = self.check_contracting_volatility(bases)

        # Check for contracting volume
        contracting_volume, volume_score = self.check_volume_contraction(
            bases, volumes
        ) if volumes is not None else (False, 0.0)

        # Check if price is tight near recent highs
        recent_high = max([base["high_price"] for base in bases])
        tight_near_highs, tightness_score = self.check_tightness_near_highs(
            current_price, recent_high
        )

        # Check ATR contraction (volatility tightening)
        atr_score, atr_ratio = self.check_atr_contraction(prices)

        # Find pivot point for potential breakout entry
        pivot_info = self.find_pivot_point(bases, current_price)

        # Calculate overall VCP score (0-100)
        # Updated weights: Depth (35%), Volume (25%), Tightness (25%), ATR (15%)
        vcp_score = (
            depth_score * 0.35 +
            volume_score * 0.25 +
            tightness_score * 0.25 +
            atr_score * 0.15
        )

        # VCP detected if score >= 65 and key criteria met
        # Relaxed from 70 to 65 to catch more valid patterns
        vcp_detected = (
            vcp_score >= 65 and
            contracting_depth and
            tight_near_highs
        )

        return {
            "vcp_detected": vcp_detected,
            "vcp_score": round(vcp_score, 2),
            "num_bases": len(bases),
            "contracting_depth": contracting_depth,
            "contraction_ratio": round(contraction_ratio, 2),
            "depth_score": round(depth_score, 2),
            "contracting_volume": contracting_volume,
            "volume_score": round(volume_score, 2) if volumes is not None else None,
            "tight_near_highs": tight_near_highs,
            "tightness_score": round(tightness_score, 2),
            "atr_score": round(atr_score, 2),
            "atr_contraction_ratio": round(atr_ratio, 2),
            "pivot_info": pivot_info,
            "bases_depth": [round(b["depth_pct"], 2) for b in bases],
            "current_price": current_price,
            "recent_high": recent_high,
            "distance_from_high_pct": round(
                ((recent_high - current_price) / recent_high) * 100, 2
            ) if recent_high > 0 else None,
        }


def quick_vcp_score(
    prices: pd.Series,
    volumes: Optional[pd.Series] = None
) -> float:
    """
    Quick VCP score calculation.

    Args:
        prices: Price series
        volumes: Volume series (optional)

    Returns:
        VCP score (0-100)
    """
    detector = VCPDetector()
    result = detector.detect_vcp(prices, volumes)
    return result["vcp_score"]
