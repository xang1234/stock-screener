"""Tests for the RS-line series helpers (chart overlay + blue-dot signal)."""

from __future__ import annotations

import pandas as pd
import pytest

from app.analysis.patterns.rs_line import (
    RsLineLeadershipSnapshot,
    blue_dot_series,
    compute_rs_line,
    rs_line_leadership_series,
    rs_line_leadership_snapshot,
)


def _series(values, start="2025-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series([float(v) for v in values], index=idx)


def test_compute_rs_line_returns_normalized_ratio():
    rs = compute_rs_line(_series([10.0, 20.0, 30.0]), _series([2.0, 4.0, 5.0]), normalize=True)

    # raw ratio = [5.0, 5.0, 6.0]; normalized to start 1.0 = [1.0, 1.0, 1.2]
    assert rs.tolist() == pytest.approx([1.0, 1.0, 1.2])


def test_compute_rs_line_aligns_benchmark_by_index():
    stock = _series([10.0, 20.0, 30.0])
    # benchmark carries an extra leading day; reindex must align by date.
    extra = pd.bdate_range("2025-01-01", periods=4)
    benchmark = pd.Series([99.0, 10.0, 10.0, 10.0], index=extra)

    rs = compute_rs_line(stock, benchmark, normalize=False)

    assert list(rs.index) == list(stock.index)
    assert rs.tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_blue_dot_series_marks_only_leading_dates():
    # rs = [1.0, 1.0, 1.1667, 1.1368]; running price high is 110 at idx1.
    #  idx2: rs new high (1.1667) & price 105 < 110 -> blue dot
    #  idx3: rs 1.1368 < 1.1667 -> not an rs new high -> no blue dot
    series = blue_dot_series(_series([100.0, 110.0, 105.0, 108.0]), _series([100.0, 110.0, 90.0, 95.0]))

    assert series.tolist() == [False, False, True, False]


def test_blue_dot_series_false_when_price_also_at_new_high():
    # Both rs and price rise to new highs together -> not a blue dot.
    series = blue_dot_series(_series([100.0, 105.0, 110.0]), _series([100.0, 100.0, 100.0]))

    assert series.iloc[-1] is False or bool(series.iloc[-1]) is False


def test_blue_dot_series_empty_on_insufficient_data():
    assert blue_dot_series(_series([]), _series([])).empty


def test_rs_line_leadership_snapshot_detects_current_blue_dot():
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    stock = pd.Series([10, 11, 12, 13, 12.5, 12.8], index=dates)
    benchmark = pd.Series([10, 10, 10, 10, 9, 8], index=dates)

    snapshot = rs_line_leadership_snapshot(stock, benchmark, lookback=6, recent_days=5)

    assert snapshot == RsLineLeadershipSnapshot(
        rs_line_new_high=True,
        rs_line_new_high_before_price=True,
        rs_line_blue_dot_recent=True,
        rs_line_new_high_date="2026-01-06",
    )
    assert snapshot.as_scan_fields() == {
        "rs_line_new_high": True,
        "rs_line_new_high_before_price": True,
        "rs_line_blue_dot_recent": True,
        "rs_line_new_high_date": "2026-01-06",
    }


def test_rs_line_leadership_snapshot_distinguishes_price_new_high():
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    stock = pd.Series([10, 11, 12, 13, 14, 15], index=dates)
    benchmark = pd.Series([10, 10, 10, 10, 10, 9], index=dates)

    snapshot = rs_line_leadership_snapshot(stock, benchmark, lookback=6, recent_days=5)

    assert snapshot.rs_line_new_high is True
    assert snapshot.rs_line_new_high_before_price is False
    assert snapshot.rs_line_blue_dot_recent is False
    assert snapshot.rs_line_new_high_date == "2026-01-06"


def test_rs_line_leadership_price_high_uses_full_price_window_when_benchmark_has_gap():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    stock = pd.Series([100, 120, 105, 106, 110], index=dates)
    benchmark = pd.Series([100, None, 100, 90, 80], index=dates, dtype=float)

    leadership = rs_line_leadership_series(stock, benchmark, lookback=5)

    assert list(leadership.rs.index) == [dates[0], dates[2], dates[3], dates[4]]
    assert bool(leadership.rs_new_high.iloc[-1]) is True
    assert bool(leadership.price_new_high.iloc[-1]) is False
    assert bool(leadership.blue_dot.iloc[-1]) is True


def test_rs_line_leadership_snapshot_keeps_recent_blue_dot_after_current_flag_fades():
    dates = pd.date_range("2026-01-01", periods=8, freq="D")
    stock = pd.Series([10, 11, 12, 13, 12.5, 12.3, 12.2, 12.1], index=dates)
    benchmark = pd.Series([10, 10, 10, 10, 8, 8.2, 8.4, 8.6], index=dates)

    snapshot = rs_line_leadership_snapshot(stock, benchmark, lookback=8, recent_days=5)

    assert snapshot.rs_line_new_high is False
    assert snapshot.rs_line_new_high_before_price is False
    assert snapshot.rs_line_blue_dot_recent is True
    assert snapshot.rs_line_new_high_date == "2026-01-05"


def test_rs_line_leadership_snapshot_empty_when_benchmark_missing():
    stock = pd.Series([10, 11, 12])
    benchmark = pd.Series([], dtype=float)

    snapshot = rs_line_leadership_snapshot(stock, benchmark)

    assert snapshot == RsLineLeadershipSnapshot.empty()
    assert snapshot.as_scan_fields() == {
        "rs_line_new_high": False,
        "rs_line_new_high_before_price": False,
        "rs_line_blue_dot_recent": False,
        "rs_line_new_high_date": None,
    }


def test_rs_line_leadership_snapshot_empty_when_latest_benchmark_bar_is_missing():
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    stock = pd.Series([10, 11, 12, 13, 12.5, 12.8], index=dates)
    benchmark = pd.Series([10, 10, 10, 10, 8], index=dates[:-1])

    leadership = rs_line_leadership_series(stock, benchmark, lookback=6)
    snapshot = leadership.to_snapshot(recent_days=5)

    assert leadership.latest_is_aligned is False
    assert bool(leadership.blue_dot.iloc[-1]) is True
    assert snapshot == RsLineLeadershipSnapshot.empty()


def test_rs_line_leadership_snapshot_omits_non_date_index_label():
    stock = pd.Series([10, 11, 12, 11], index=pd.RangeIndex(4))
    benchmark = pd.Series([10, 10, 9, 7], index=pd.RangeIndex(4))

    snapshot = rs_line_leadership_snapshot(stock, benchmark, lookback=4)

    assert snapshot.rs_line_new_high is True
    assert snapshot.rs_line_new_high_date is None


def test_rs_line_leadership_series_is_canonical_for_blue_dot_series():
    stock = _series([100.0, 110.0, 105.0, 108.0])
    benchmark = _series([100.0, 110.0, 90.0, 95.0])

    leadership = rs_line_leadership_series(stock, benchmark)

    assert leadership.blue_dot.tolist() == blue_dot_series(stock, benchmark).tolist()
