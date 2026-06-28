"""RS line (stock / benchmark ratio) and the "blue dot" leadership signal.

The RS line is ``stock_close / benchmark_close``. A "blue dot" (DeepVue/O'Neil
leadership signal) fires when the RS line makes a new trailing-``lookback`` high
**before** price does — the RS line is at a new high while price is not.

These helpers produce *series* for chart overlay and single-point snapshots for
scanner/filter rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from app.analysis.patterns.technicals import rolling_at_new_high

DEFAULT_LOOKBACK = 252
DEFAULT_BLUE_DOT_RECENT_DAYS = 5


@dataclass(frozen=True)
class RsLineLeadershipSnapshot:
    """Latest RS leadership fields promoted onto scan rows.

    Missing or unaligned benchmark data serializes as a false signal; the date
    stays ``None`` because there is no concrete RS high to annotate.
    """

    rs_line_new_high: bool = False
    rs_line_new_high_before_price: bool = False
    rs_line_blue_dot_recent: bool = False
    rs_line_new_high_date: str | None = None

    @classmethod
    def empty(cls) -> "RsLineLeadershipSnapshot":
        return cls()

    def as_scan_fields(self) -> dict[str, bool | str | None]:
        return {
            "rs_line_new_high": self.rs_line_new_high,
            "rs_line_new_high_before_price": self.rs_line_new_high_before_price,
            "rs_line_blue_dot_recent": self.rs_line_blue_dot_recent,
            "rs_line_new_high_date": self.rs_line_new_high_date,
        }


@dataclass(frozen=True)
class RsLineLeadershipSeries:
    """Benchmark-aligned RS leadership predicates for each date."""

    rs: pd.Series
    price: pd.Series
    rs_new_high: pd.Series
    price_new_high: pd.Series
    blue_dot: pd.Series
    latest_is_aligned: bool = True

    @classmethod
    def empty(cls) -> "RsLineLeadershipSeries":
        empty_float = pd.Series([], dtype=float)
        empty_bool = pd.Series([], dtype=bool)
        return cls(
            rs=empty_float,
            price=empty_float,
            rs_new_high=empty_bool,
            price_new_high=empty_bool,
            blue_dot=empty_bool,
            latest_is_aligned=False,
        )

    def to_snapshot(
        self,
        *,
        recent_days: int = DEFAULT_BLUE_DOT_RECENT_DAYS,
    ) -> RsLineLeadershipSnapshot:
        """Collapse the per-date predicates into latest scan-row fields."""
        if recent_days < 1:
            raise ValueError("recent_days must be >= 1")
        if self.rs.empty or not self.latest_is_aligned:
            return RsLineLeadershipSnapshot.empty()

        new_high_dates = self.rs_new_high.index[self.rs_new_high]
        latest_new_high_date = (
            _date_string(new_high_dates[-1]) if len(new_high_dates) > 0 else None
        )

        return RsLineLeadershipSnapshot(
            rs_line_new_high=bool(self.rs_new_high.iloc[-1]),
            rs_line_new_high_before_price=bool(self.blue_dot.iloc[-1]),
            rs_line_blue_dot_recent=bool(self.blue_dot.tail(recent_days).any()),
            rs_line_new_high_date=latest_new_high_date,
        )


def _aligned_ratio(stock_close: pd.Series, benchmark_close: pd.Series) -> pd.Series:
    stock_close = stock_close.astype(float)
    aligned_benchmark = benchmark_close.astype(float).reindex(stock_close.index)
    return stock_close / aligned_benchmark.replace(0.0, np.nan)


def compute_rs_line(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    normalize: bool = True,
) -> pd.Series:
    """RS line aligned to the stock's index.

    When ``normalize`` is True the series is scaled to start at 1.0 for display
    (a positive monotonic transform; it does not affect new-high detection).
    """
    rs = _aligned_ratio(stock_close, benchmark_close)
    if normalize:
        valid = rs.dropna()
        if not valid.empty and valid.iloc[0] != 0:
            rs = rs / valid.iloc[0]
    return rs


def blue_dot_series(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    lookback: int = DEFAULT_LOOKBACK,
) -> pd.Series:
    """Per-date boolean: RS line at a new high while price is not."""
    return rs_line_leadership_series(
        stock_close,
        benchmark_close,
        lookback=lookback,
    ).blue_dot


def _date_string(value: object) -> str | None:
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.date().isoformat()
    if isinstance(value, np.datetime64):
        timestamp = pd.Timestamp(value)
        return None if pd.isna(timestamp) else timestamp.date().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and len(value) >= 10 and value[4] == "-" and value[7] == "-":
        try:
            return pd.Timestamp(value).date().isoformat()
        except (TypeError, ValueError):
            return None
    return None


def rs_line_leadership_series(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    *,
    lookback: int = DEFAULT_LOOKBACK,
) -> RsLineLeadershipSeries:
    """Per-date RS leadership predicates from one benchmark-aligned frame."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    price = stock_close.astype(float)
    rs = _aligned_ratio(stock_close, benchmark_close)
    frame = pd.DataFrame({"rs": rs}).dropna()
    if frame.empty:
        return RsLineLeadershipSeries.empty()
    latest_is_aligned = bool(len(price.index) > 0 and frame.index[-1] == price.index[-1])
    rs_new_high = rolling_at_new_high(frame["rs"], window=lookback)
    price_new_high = (
        rolling_at_new_high(price, window=lookback)
        .reindex(frame.index)
        .fillna(False)
        .astype(bool)
    )
    return RsLineLeadershipSeries(
        rs=frame["rs"],
        price=price.reindex(frame.index),
        rs_new_high=rs_new_high,
        price_new_high=price_new_high,
        blue_dot=rs_new_high & (~price_new_high),
        latest_is_aligned=latest_is_aligned,
    )


def rs_line_leadership_snapshot(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    recent_days: int = DEFAULT_BLUE_DOT_RECENT_DAYS,
) -> RsLineLeadershipSnapshot:
    """Latest RS leadership flags for scanner/filter rows."""
    return rs_line_leadership_series(
        stock_close,
        benchmark_close,
        lookback=lookback,
    ).to_snapshot(recent_days=recent_days)
