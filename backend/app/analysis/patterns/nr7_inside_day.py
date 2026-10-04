"""NR7/Inside-Day trigger detector entrypoint.

Expected input orientation:
- Daily bars in chronological order.
- Trigger bars are evaluated on completed bars only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from app.analysis.patterns.config import SetupEngineParameters
from app.analysis.patterns.detectors.base import (
    PatternDetector,
    PatternDetectorInput,
    PatternDetectorResult,
)
from app.analysis.patterns.models import PatternCandidateModel
from app.analysis.patterns.normalization import normalize_detector_input_ohlcv

_NR7_LOOKBACK_BARS = 7
_VOLUME_LOOKBACK_BARS = 20
_MAX_TRIGGER_CANDIDATES = 5
_RECENT_TRIGGER_BARS = 20
_TRIGGER_RANGE_REFERENCE_PCT = 4.0
_VOLUME_NEUTRAL_RATIO = 1.05


@dataclass(frozen=True)
class _TriggerSignal:
    idx: int
    trigger_subtype: str
    trigger_is_nr7: bool
    trigger_is_inside_day: bool
    trigger_high: float
    trigger_low: float
    trigger_range_points: float
    trigger_range_pct: float
    range_min_7d_points: float
    range_rank_7d: int
    trigger_volume: float
    volume_mean_20d: float
    volume_ratio_20d: float
    ema21_trigger: float | None
    close_above_ema21: bool
    recency_bars: int
    score: float


class _TriggerRow(NamedTuple):
    """Per-trigger values gathered from the vectorized window pass."""

    idx: int
    is_nr7: bool
    is_inside: bool
    high: float
    low: float
    close: float
    range_points: float
    range_min: float
    range_rank: int
    volume: float
    volume_mean: float
    ema21: float


class NR7InsideDayDetector(PatternDetector):
    """Compile-safe entrypoint for trigger-family detection."""

    name = "nr7_inside_day"

    def detect(
        self,
        detector_input: PatternDetectorInput,
        parameters: SetupEngineParameters,
    ) -> PatternDetectorResult:
        del parameters
        normalized = normalize_detector_input_ohlcv(
            features=detector_input.features,
            timeframe="daily",
            min_bars=_NR7_LOOKBACK_BARS,
            feature_key="daily_ohlcv",
            fallback_bar_count=detector_input.daily_bars,
        )
        if not normalized.prerequisites_ok:
            return PatternDetectorResult.insufficient_data(
                self.name, normalized=normalized
            )

        if normalized.frame is None:
            return PatternDetectorResult.insufficient_data(
                self.name,
                failed_checks=("missing_daily_ohlcv_for_nr7_inside_day",),
                warnings=normalized.warnings,
            )

        frame = normalized.frame
        signals = _top_trigger_signals(frame)
        if not signals:
            return PatternDetectorResult.no_detection(
                self.name,
                failed_checks=("nr7_inside_day_trigger_not_found",),
                warnings=normalized.warnings,
            )

        last_close = float(frame["Close"].iat[-1])
        candidates: list[PatternCandidateModel] = []
        for rank, signal in enumerate(signals, start=1):
            recency_component = max(
                0.0, 1.0 - (signal.recency_bars / _RECENT_TRIGGER_BARS)
            )
            confidence = min(0.78, max(0.05, 0.20 + signal.score * 0.55))
            quality_score = min(65.0, max(0.0, 20.0 + signal.score * 55.0))
            readiness_score = min(
                70.0,
                max(
                    0.0,
                    24.0
                    + (recency_component * 22.0)
                    + (8.0 if signal.trigger_subtype == "nr7_inside_day" else 4.0)
                    + (4.0 if signal.close_above_ema21 else 0.0),
                ),
            )

            candidates.append(
                PatternCandidateModel(
                    pattern=self.name,
                    timeframe="daily",
                    source_detector=self.name,
                    pivot_price=signal.trigger_high,
                    pivot_type=f"{signal.trigger_subtype}_trigger_high",
                    pivot_date=frame.index[signal.idx].date().isoformat(),
                    distance_to_pivot_pct=(
                        ((signal.trigger_high - last_close) / max(abs(last_close), 1e-9))
                        * 100.0
                    ),
                    confidence=confidence,
                    quality_score=quality_score,
                    readiness_score=readiness_score,
                    metrics={
                        "trigger_rank": rank,
                        "trigger_subtype": signal.trigger_subtype,
                        "trigger_is_nr7": signal.trigger_is_nr7,
                        "trigger_is_inside_day": signal.trigger_is_inside_day,
                        "trigger_high": round(signal.trigger_high, 4),
                        "trigger_low": round(signal.trigger_low, 4),
                        "trigger_range_points": round(
                            signal.trigger_range_points, 6
                        ),
                        "trigger_range_pct": round(signal.trigger_range_pct, 6),
                        "range_min_7d_points": round(
                            signal.range_min_7d_points, 6
                        ),
                        "range_rank_7d": signal.range_rank_7d,
                        "trigger_volume": round(signal.trigger_volume, 4),
                        "volume_mean_20d": round(signal.volume_mean_20d, 4),
                        "volume_ratio_20d": round(signal.volume_ratio_20d, 6),
                        "ema21_trigger": (
                            round(signal.ema21_trigger, 6)
                            if signal.ema21_trigger is not None
                            else None
                        ),
                        "close_above_ema21": signal.close_above_ema21,
                        "trigger_recency_bars": signal.recency_bars,
                        "trigger_score": round(signal.score, 6),
                    },
                    checks={
                        "trigger_detected": True,
                        "trigger_is_nr7": bool(signal.trigger_is_nr7),
                        "trigger_is_inside_day": bool(signal.trigger_is_inside_day),
                        "trigger_is_combined": bool(
                            signal.trigger_subtype == "nr7_inside_day"
                        ),
                        "range_is_7d_min": bool(signal.trigger_is_nr7),
                        "inside_day_structure_valid": bool(
                            signal.trigger_is_inside_day
                        ),
                        "volume_not_expanded": bool(
                            signal.volume_ratio_20d <= _VOLUME_NEUTRAL_RATIO
                        ),
                        "context_close_above_ema21": bool(
                            signal.close_above_ema21
                        ),
                    },
                    notes=(
                        "trigger_detector_lightweight_scoring",
                        f"subtype_{signal.trigger_subtype}",
                    ),
                )
            )

        return PatternDetectorResult.detected(
            self.name,
            tuple(candidates),
            passed_checks=(
                "nr7_inside_day_trigger_found",
                "trigger_subtypes_labeled",
            ),
            warnings=normalized.warnings,
        )


def _top_trigger_signals(frame: pd.DataFrame) -> list[_TriggerSignal]:
    """Best-ranked NR7/inside-day triggers, at most ``_MAX_TRIGGER_CANDIDATES``.

    Window statistics (7-bar range minimum and rank, prior-20-bar volume mean)
    are computed for all bars at once; scoring and ordering then run only over
    the bars that are triggers, and signal objects are built only for the
    selected ones. Results match the original per-bar loop exactly for the
    float64/integer columns the price pipeline produces; a float32 Volume
    column is averaged in float64 here (the loop summed it in float32).
    """
    bar_count = len(frame)
    if bar_count < _NR7_LOOKBACK_BARS:
        return []
    high = frame["High"].to_numpy(dtype=float)
    low = frame["Low"].to_numpy(dtype=float)
    close = frame["Close"].to_numpy(dtype=float)
    volume = frame["Volume"].to_numpy(dtype=float)
    ema21 = frame["Close"].ewm(span=21, adjust=False).mean().to_numpy(dtype=float)

    first = _NR7_LOOKBACK_BARS - 1
    with np.errstate(invalid="ignore"):
        ranges = high - low  # inf - inf -> NaN, silently, as pandas did
        # Missing (NaN compares False) and inverted ranges are never triggers.
        idx = np.flatnonzero(ranges[first:] >= 0.0) + first
    windows = sliding_window_view(ranges, _NR7_LOOKBACK_BARS)[idx - first]
    trigger_ranges = ranges[idx]
    # nanmin like pandas' min(); each window holds its own non-NaN trigger.
    range_min = np.nanmin(windows, axis=1) if idx.size else trigger_ranges
    with np.errstate(invalid="ignore"):
        is_nr7 = trigger_ranges <= range_min + 1e-9
        is_inside = (high[idx] < high[idx - 1]) & (low[idx] > low[idx - 1])
    hit = is_nr7 | is_inside
    if not hit.any():
        return []
    idx, windows, trigger_ranges, range_min = idx[hit], windows[hit], trigger_ranges[hit], range_min[hit]
    is_nr7, is_inside = is_nr7[hit], is_inside[hit]
    range_rank = (windows <= (trigger_ranges + 1e-9)[:, None]).sum(axis=1)
    volume_mean = _prior_volume_means(volume, idx)

    # .tolist() yields native Python bool/int/float, as the loop's float()/int() did.
    rows = map(
        _TriggerRow._make,
        zip(
            idx.tolist(),
            is_nr7.tolist(),
            is_inside.tolist(),
            high[idx].tolist(),
            low[idx].tolist(),
            close[idx].tolist(),
            trigger_ranges.tolist(),
            range_min.tolist(),
            range_rank.tolist(),
            volume[idx].tolist(),
            volume_mean.tolist(),
            ema21[idx].tolist(),
        ),
    )
    ranked = []
    for row in rows:
        subtype = _trigger_subtype(trigger_is_nr7=row.is_nr7, trigger_is_inside_day=row.is_inside)
        range_pct = (row.range_points / max(abs(row.high), 1e-9)) * 100.0
        mean = row.volume_mean
        ratio = 1.0 if mean <= 0.0 or math.isnan(mean) else row.volume / mean
        above = False if math.isnan(row.ema21) else row.close >= row.ema21
        recency = (bar_count - 1) - row.idx
        score = _signal_score(
            trigger_subtype=subtype,
            trigger_range_pct=range_pct,
            volume_ratio_20d=ratio,
            close_above_ema21=above,
            recency_bars=recency,
        )
        sort_key = (-score, recency, subtype != "nr7_inside_day", -row.idx)
        ranked.append((sort_key, row, subtype, range_pct, ratio, above, recency, score))
    ranked.sort(key=lambda entry: entry[0])

    return [
        _TriggerSignal(
            idx=row.idx,
            trigger_subtype=subtype,
            trigger_is_nr7=row.is_nr7,
            trigger_is_inside_day=row.is_inside,
            trigger_high=row.high,
            trigger_low=row.low,
            trigger_range_points=row.range_points,
            trigger_range_pct=range_pct,
            range_min_7d_points=row.range_min,
            range_rank_7d=row.range_rank,
            trigger_volume=row.volume,
            volume_mean_20d=row.volume_mean,
            volume_ratio_20d=ratio,
            ema21_trigger=None if math.isnan(row.ema21) else row.ema21,
            close_above_ema21=above,
            recency_bars=recency,
            score=score,
        )
        for _, row, subtype, range_pct, ratio, above, recency, score in ranked[:_MAX_TRIGGER_CANDIDATES]
    ]


def _prior_volume_means(volume: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Mean of up to 20 volumes before each index, NaN skipped, NaN if none.

    Mirrors ``Series.iloc[max(0, i - 20):i].mean()`` bit for bit on float64
    input: NaN filled with 0, summed per contiguous row (numpy's pairwise sum,
    as pandas uses), divided by the non-NaN count.
    """
    missing = np.isnan(volume)
    filled = np.where(missing, 0.0, volume)
    sums = np.empty(idx.size)
    counts = np.empty(idx.size)
    full = idx >= _VOLUME_LOOKBACK_BARS
    if full.any():
        starts = idx[full] - _VOLUME_LOOKBACK_BARS
        # Fancy indexing copies each window into a C-contiguous row.
        sums[full] = sliding_window_view(filled, _VOLUME_LOOKBACK_BARS)[starts].sum(axis=1)
        counts[full] = sliding_window_view(~missing, _VOLUME_LOOKBACK_BARS)[starts].sum(axis=1)
    for position in np.flatnonzero(~full):  # at most 14 early bars
        end = idx[position]
        sums[position] = filled[:end].sum()
        counts[position] = (~missing[:end]).sum()
    with np.errstate(invalid="ignore"):
        return sums / counts  # 0 / 0 -> NaN, as pandas returns for all-NaN


def _trigger_subtype(*, trigger_is_nr7: bool, trigger_is_inside_day: bool) -> str:
    if trigger_is_nr7 and trigger_is_inside_day:
        return "nr7_inside_day"
    if trigger_is_nr7:
        return "nr7"
    return "inside_day"


def _signal_score(
    *,
    trigger_subtype: str,
    trigger_range_pct: float,
    volume_ratio_20d: float,
    close_above_ema21: bool,
    recency_bars: int,
) -> float:
    subtype_bonus = {
        "nr7_inside_day": 0.18,
        "nr7": 0.10,
        "inside_day": 0.08,
    }.get(trigger_subtype, 0.05)
    range_tight_component = max(
        0.0, 1.0 - (trigger_range_pct / _TRIGGER_RANGE_REFERENCE_PCT)
    )
    volume_dry_component = max(
        0.0, 1.0 - min(max(volume_ratio_20d, 0.0), 2.0)
    )
    recency_component = max(0.0, 1.0 - (recency_bars / _RECENT_TRIGGER_BARS))

    return (
        0.20
        + subtype_bonus
        + (range_tight_component * 0.28)
        + (volume_dry_component * 0.12)
        + (0.08 if close_above_ema21 else 0.0)
        + (recency_component * 0.14)
    )
