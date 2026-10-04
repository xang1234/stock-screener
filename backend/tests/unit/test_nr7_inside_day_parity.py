"""Vectorized NR7/inside-day selection matches the scalar loop exactly (#495).

``_reference_top_signals`` is the detector's original per-bar loop and sort,
kept here as a test-only oracle. Selected candidates, their order and every
field must be identical; floats are compared exactly (NaN equal to NaN).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from app.analysis.patterns.nr7_inside_day import (
    _MAX_TRIGGER_CANDIDATES,
    _NR7_LOOKBACK_BARS,
    _signal_score,
    _top_trigger_signals,
    _trigger_subtype,
    _TriggerSignal,
)


def _make_price_data(num_days: int) -> pd.DataFrame:
    """Same shape as the Setup Engine perf fixture, with a private RNG (no global reseed)."""
    rng = np.random.RandomState(42)
    close = np.maximum(50.0 + np.cumsum(rng.randn(num_days) * 0.5 + 0.05), 1.0)
    return pd.DataFrame(
        {
            "Open": close * (1 - rng.uniform(0, 0.02, num_days)),
            "High": close * (1 + rng.uniform(0, 0.03, num_days)),
            "Low": close * (1 - rng.uniform(0, 0.03, num_days)),
            "Close": close,
            "Volume": rng.randint(100_000, 5_000_000, size=num_days).astype(float),
        },
        index=pd.bdate_range(end=pd.Timestamp("2025-12-19"), periods=num_days),
    )


def _reference_top_signals(frame: pd.DataFrame) -> list[_TriggerSignal]:
    high = frame["High"]
    low = frame["Low"]
    close = frame["Close"]
    volume = frame["Volume"]
    ranges = high - low
    ema21 = close.ewm(span=21, adjust=False).mean()

    signals: list[_TriggerSignal] = []
    for idx in range(_NR7_LOOKBACK_BARS - 1, len(frame)):
        trigger_range_points = float(ranges.iat[idx])
        if pd.isna(trigger_range_points) or trigger_range_points < 0.0:
            continue
        range_window = ranges.iloc[idx - _NR7_LOOKBACK_BARS + 1 : idx + 1]
        if range_window.empty:
            continue
        range_min_7d_points = float(range_window.min())
        trigger_is_nr7 = bool(trigger_range_points <= range_min_7d_points + 1e-9)
        prev_idx = idx - 1
        trigger_is_inside_day = bool(
            prev_idx >= 0
            and float(high.iat[idx]) < float(high.iat[prev_idx])
            and float(low.iat[idx]) > float(low.iat[prev_idx])
        )
        if not (trigger_is_nr7 or trigger_is_inside_day):
            continue
        trigger_subtype = _trigger_subtype(
            trigger_is_nr7=trigger_is_nr7, trigger_is_inside_day=trigger_is_inside_day
        )
        trigger_high = float(high.iat[idx])
        trigger_low = float(low.iat[idx])
        trigger_range_pct = (trigger_range_points / max(abs(trigger_high), 1e-9)) * 100.0
        range_rank_7d = int((range_window <= (trigger_range_points + 1e-9)).sum())
        trigger_volume = float(volume.iat[idx])
        prior_volume_window = volume.iloc[max(0, idx - 20) : idx]
        volume_mean_20d = (
            float(prior_volume_window.mean()) if not prior_volume_window.empty else float("nan")
        )
        if volume_mean_20d <= 0.0 or pd.isna(volume_mean_20d):
            volume_ratio_20d = 1.0
        else:
            volume_ratio_20d = trigger_volume / volume_mean_20d
        ema21_trigger_raw = float(ema21.iat[idx])
        if pd.isna(ema21_trigger_raw):
            ema21_trigger = None
            close_above_ema21 = False
        else:
            ema21_trigger = ema21_trigger_raw
            close_above_ema21 = bool(float(close.iat[idx]) >= ema21_trigger)
        recency_bars = (len(frame) - 1) - idx
        score = _signal_score(
            trigger_subtype=trigger_subtype,
            trigger_range_pct=trigger_range_pct,
            volume_ratio_20d=volume_ratio_20d,
            close_above_ema21=close_above_ema21,
            recency_bars=recency_bars,
        )
        signals.append(
            _TriggerSignal(
                idx=idx,
                trigger_subtype=trigger_subtype,
                trigger_is_nr7=trigger_is_nr7,
                trigger_is_inside_day=trigger_is_inside_day,
                trigger_high=trigger_high,
                trigger_low=trigger_low,
                trigger_range_points=trigger_range_points,
                trigger_range_pct=trigger_range_pct,
                range_min_7d_points=range_min_7d_points,
                range_rank_7d=range_rank_7d,
                trigger_volume=trigger_volume,
                volume_mean_20d=volume_mean_20d,
                volume_ratio_20d=volume_ratio_20d,
                ema21_trigger=ema21_trigger,
                close_above_ema21=close_above_ema21,
                recency_bars=recency_bars,
                score=score,
            )
        )
    signals.sort(
        key=lambda s: (-s.score, s.recency_bars, s.trigger_subtype != "nr7_inside_day", -s.idx)
    )
    return signals[:_MAX_TRIGGER_CANDIDATES]


def _comparable(signals):
    """Field tuples with exact float repr; NaN compares equal to NaN."""

    def norm(value):
        if isinstance(value, float) and math.isnan(value):
            return "nan"
        return (type(value).__name__, repr(value))

    return [tuple(norm(getattr(s, f)) for f in _TriggerSignal.__dataclass_fields__) for s in signals]


def _assert_parity(frame: pd.DataFrame) -> list[_TriggerSignal]:
    expected = _reference_top_signals(frame)
    actual = _top_trigger_signals(frame)
    assert _comparable(actual) == _comparable(expected)
    return actual


def _frame(high, low, close=None, volume=None, start="2025-01-01") -> pd.DataFrame:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = (high + low) / 2 if close is None else np.asarray(close, dtype=float)
    volume = np.full(len(high), 1_000_000.0) if volume is None else np.asarray(volume, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=pd.bdate_range(start=start, periods=len(high)),
    )


@pytest.mark.parametrize("bars", [7, 8, 30, 350, 504, 1260])
def test_parity_on_setup_engine_fixture(bars):
    _assert_parity(_make_price_data(bars))


@pytest.mark.parametrize("seed", range(40))
def test_parity_on_random_frames(seed):
    rng = np.random.default_rng(seed)
    bars = int(rng.integers(7, 400))
    close = 50 + np.cumsum(rng.normal(0, 1, bars))
    spread = np.abs(rng.normal(0, 1, bars))
    # Coarse rounding forces equal ranges, equal highs/lows and tied scores.
    high = np.round(close + spread, int(rng.integers(0, 3)))
    low = np.round(close - spread, int(rng.integers(0, 3)))
    volume = rng.integers(0, 5, bars).astype(float) * 1e6  # includes zero volume
    _assert_parity(_frame(high, low, close, volume))


@pytest.mark.parametrize("seed", range(20))
def test_parity_with_fractional_volumes(seed):
    # Whole-number volumes sum exactly in any order; fractional ones pin the
    # summation order (an ulp in the mean can reorder near-tied scores).
    rng = np.random.default_rng(1000 + seed)
    bars = int(rng.integers(30, 300))
    close = 20 + np.cumsum(rng.normal(0, 0.5, bars))
    spread = np.abs(rng.normal(0, 0.3, bars)) + 0.01
    volume = rng.lognormal(12, 1, bars) * 1.000123
    volume[rng.random(bars) < 0.05] = np.nan
    _assert_parity(_frame(close + spread, close - spread, close, volume))


def test_equal_range_minima_tie_and_rank():
    # Constant bars: every bar ties the 7-bar minimum, rank counts all ties.
    signals = _assert_parity(_frame([10.0] * 30, [9.0] * 30))
    assert {s.range_rank_7d for s in signals} == {7}
    assert len(signals) == _MAX_TRIGGER_CANDIDATES  # more than five tied candidates


def test_epsilon_near_threshold():
    base = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5]
    for delta in (0.0, 0.5e-9, 1e-9, 2e-9, -1e-12):
        high = np.array([10 + r for r in base] + [10 + 1.5 + delta])
        low = np.full(7, 10.0)
        _assert_parity(_frame(high, low))


def test_inside_day_is_strict_on_equal_bounds():
    high = [12, 11, 11, 12, 11.5, 13, 12.9, 12.9]
    low = [8, 9, 9, 8, 8.5, 7, 7.1, 7.0]
    _assert_parity(_frame(high, low))


def test_missing_and_nonpositive_volume_and_partial_windows():
    high = 10 + np.abs(np.sin(np.arange(40)))
    low = high - 1 - 0.1 * np.cos(np.arange(40))
    volume = np.where(np.arange(40) % 3 == 0, np.nan, 1e6)
    volume[:12] = np.nan  # all-missing initial window -> NaN mean, ratio 1.0
    volume[20:24] = 0.0
    _assert_parity(_frame(high, low, volume=volume))


def test_missing_and_inverted_ranges_are_skipped():
    high = 10 + np.abs(np.sin(np.arange(30)))
    low = high - 0.5
    high[10] = np.nan
    low[15] = high[15] + 0.1  # negative range
    _assert_parity(_frame(high, low))


def test_infinite_bounds_do_not_warn():
    # inf - inf is NaN (never a trigger); the old pandas subtraction was silent.
    high = 10 + np.abs(np.sin(np.arange(20)))
    low = high - 0.5
    high[8] = low[8] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _assert_parity(_frame(high, low))


def test_float32_volume_is_averaged_in_float64():
    # Documented divergence: pandas would sum a float32 column in float32; the
    # vectorized path upcasts first, so it equals the float64 result instead.
    # The price pipeline stores volume as float64/int, where parity is exact.
    frame = _make_price_data(120)
    as_float32 = frame.assign(Volume=frame["Volume"].astype("float32") * np.float32(1.0001))
    as_float64 = as_float32.assign(Volume=as_float32["Volume"].astype("float64"))
    assert _comparable(_top_trigger_signals(as_float32)) == _comparable(
        _reference_top_signals(as_float64)
    )


def test_no_signal_frame():
    # Strictly expanding ranges: no NR7 and no inside days.
    high = 10 + np.arange(20) * 0.5
    low = 10 - np.arange(20) * 0.5
    assert _assert_parity(_frame(high, low)) == []
