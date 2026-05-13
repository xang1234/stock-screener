"""
Price Sparkline Calculator.

Calculates normalized price series for the last 30 trading days
for sparkline visualization in the bulk screener results.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PriceSparklineCalculator:
    """
    Calculate normalized price series for sparkline visualization.

    Similar to RSSparklineCalculator but shows pure price movement
    normalized to start at 1.0 for visual comparison.
    """

    SPARKLINE_DAYS = 30  # Number of trading days for sparkline

    def calculate_price_sparkline(
        self,
        stock_prices: pd.Series,
        normalize: bool = True
    ) -> Dict:
        """
        Calculate normalized price series for the last 30 trading days.

        Args:
            stock_prices: Stock closing prices (chronological order, oldest first)
            normalize: If True, normalize to start at 1.0 for better visual comparison

        Returns:
            Dict with:
            - price_data: List of 30 normalized price values (or None if insufficient data)
            - price_trend: -1 (down overall), 0 (flat), 1 (up overall)
            - price_change_1d: 1-day percentage change (most recent)
        """
        if len(stock_prices) < self.SPARKLINE_DAYS:
            logger.debug(
                f"Insufficient data for price sparkline: stock={len(stock_prices)}"
            )
            return {
                "price_data": None,
                "price_trend": 0,
                "price_change_1d": self._calculate_price_change_1d(stock_prices),
            }

        try:
            # Get last 30 trading days (most recent data) as float to make
            # NaN/Inf checks consistent regardless of source dtype.
            stock_last_30 = np.asarray(
                stock_prices.iloc[-self.SPARKLINE_DAYS:].values, dtype=float
            )

            price_change_1d = self._calculate_price_change_1d(stock_prices)

            # If the window contains no finite samples (all NaN/Inf), bail out
            # to None instead of emitting [nan, ...] — downstream Pydantic
            # serialization converts NaN to null and rejects List[None].
            finite_mask = np.isfinite(stock_last_30)
            if not finite_mask.any():
                logger.debug("All-non-finite price series; returning None sparkline")
                return {
                    "price_data": None,
                    "price_trend": 0,
                    "price_change_1d": price_change_1d,
                }

            # Replace any NaN/Inf with the first finite value so the series is
            # fully finite even when the leading bar is missing.
            first_finite = float(stock_last_30[np.argmax(finite_mask)])
            price_series = np.where(finite_mask, stock_last_30, first_finite)

            # Normalize to start at 1.0 if requested (better for visual comparison)
            if normalize and price_series[0] != 0:
                price_series = price_series / price_series[0]

            # Calculate overall trend (start to end)
            if price_series[0] != 0:
                overall_change_pct = ((price_series[-1] - price_series[0]) / price_series[0]) * 100
            else:
                overall_change_pct = 0

            # Determine trend direction
            # Use a small threshold to avoid noise (0.5% change)
            if overall_change_pct > 0.5:
                trend = 1  # Up
            elif overall_change_pct < -0.5:
                trend = -1  # Down
            else:
                trend = 0  # Flat

            # Round values for JSON storage efficiency (4 decimal places)
            price_data = [round(float(v), 4) for v in price_series]

            # Final safety net: if rounding/normalization produced any non-finite
            # value, collapse the whole payload to None.
            if not all(np.isfinite(v) for v in price_data):
                logger.debug("Non-finite values after normalization; returning None sparkline")
                return {
                    "price_data": None,
                    "price_trend": 0,
                    "price_change_1d": price_change_1d,
                }

            return {
                "price_data": price_data,
                "price_trend": trend,
                "price_change_1d": price_change_1d,
            }

        except Exception as e:
            logger.warning(f"Error calculating price sparkline: {e}")
            return {
                "price_data": None,
                "price_trend": 0,
                "price_change_1d": self._calculate_price_change_1d(stock_prices),
            }

    @staticmethod
    def _calculate_price_change_1d(stock_prices: pd.Series) -> Optional[float]:
        if len(stock_prices) < 2:
            return None
        yesterday_price = stock_prices.iloc[-2]
        today_price = stock_prices.iloc[-1]
        if pd.isna(yesterday_price) or pd.isna(today_price) or yesterday_price == 0:
            return None
        try:
            if not np.isfinite(yesterday_price) or not np.isfinite(today_price):
                return None
        except TypeError:
            return None
        price_change_1d = ((today_price - yesterday_price) / yesterday_price) * 100
        return round(float(price_change_1d), 2)
