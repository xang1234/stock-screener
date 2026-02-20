"""Three-Weeks-Tight / Multi-Weeks-Tight detector entrypoint.

Expected input orientation:
- Weekly bars derived from chronological daily bars.
- Current incomplete week excluded unless policy explicitly permits.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.analysis.patterns.config import SetupEngineParameters
from app.analysis.patterns.detectors.base import (
    PatternDetector,
    PatternDetectorInput,
    PatternDetectorResult,
)
from app.analysis.patterns.models import PatternCandidateModel
from app.analysis.patterns.normalization import normalize_detector_input_ohlcv
from app.analysis.patterns.technicals import resample_ohlcv

_MIN_WEEKS_TIGHT = 3
_MAX_WEEKS_TIGHT = 8
_MAX_CANDIDATES = 5


@dataclass(frozen=True)
class _TightRun:
    start_idx: int
    end_idx: int
    weeks_tight: int
    mode: str
    max_contraction_pct: float
    tight_band_pct: float
    tight_range_pct: float
    vol_vs_10w: float | None
    pivot_idx: int
    pivot_price: float
    recency_weeks: int
    score: float


class ThreeWeeksTightDetector(PatternDetector):
    """Compile-safe entrypoint for 3WT/MWT detection."""

    name = "three_weeks_tight"

    def detect(
        self,
        detector_input: PatternDetectorInput,
        parameters: SetupEngineParameters,
    ) -> PatternDetectorResult:
        weekly_frame, warnings = _resolve_weekly_frame(detector_input)
        if weekly_frame is None:
            return PatternDetectorResult.insufficient_data(
                self.name,
                failed_checks=("missing_weekly_ohlcv_for_three_weeks_tight",),
                warnings=warnings,
            )

        normalized_weekly = normalize_detector_input_ohlcv(
            features={"weekly_ohlcv": weekly_frame},
            timeframe="weekly",
            min_bars=_MIN_WEEKS_TIGHT + 2,
            feature_key="weekly_ohlcv",
            fallback_bar_count=len(weekly_frame),
        )
        if not normalized_weekly.prerequisites_ok:
            return PatternDetectorResult.insufficient_data(
                self.name,
                normalized=normalized_weekly,
                warnings=warnings,
            )

        if normalized_weekly.frame is None:
            return PatternDetectorResult.insufficient_data(
                self.name,
                failed_checks=("missing_weekly_ohlcv_for_three_weeks_tight",),
                warnings=warnings,
            )

        warnings = tuple(warnings) + normalized_weekly.warnings
        weekly = normalized_weekly.frame
        runs = _find_tight_runs(weekly, parameters)
        if not runs:
            return PatternDetectorResult.no_detection(
                self.name,
                failed_checks=("tight_run_not_found",),
                warnings=warnings,
            )

        candidates: list[PatternCandidateModel] = []
        for run_rank, run in enumerate(runs[:_MAX_CANDIDATES], start=1):
            pivot_date = weekly.index[run.pivot_idx].date().isoformat()
            run_slice = weekly.iloc[run.start_idx : run.end_idx + 1]
            median_close = float(run_slice["Close"].median())
            confidence = min(
                0.95,
                max(
                    0.05,
                    0.35
                    + min(run.weeks_tight, _MAX_WEEKS_TIGHT) * 0.06
                    + max(0.0, 1.0 - (run.tight_band_pct / 3.0)) * 0.20,
                ),
            )
            quality_score = min(
                100.0,
                max(
                    0.0,
                    30.0
                    + min(run.weeks_tight, _MAX_WEEKS_TIGHT) * 6.0
                    + max(0.0, 1.0 - (run.tight_band_pct / 3.0)) * 30.0,
                ),
            )
            readiness_score = min(
                100.0,
                max(
                    0.0,
                    55.0
                    + max(0.0, 1.0 - run.recency_weeks / 8.0) * 20.0
                    + max(0.0, 1.0 - (run.tight_band_pct / 2.0)) * 15.0,
                ),
            )

            candidates.append(
                PatternCandidateModel(
                    pattern=self.name,
                    timeframe="weekly",
                    source_detector=self.name,
                    pivot_price=run.pivot_price,
                    pivot_type="tight_area_high",
                    pivot_date=pivot_date,
                    distance_to_pivot_pct=(
                        ((run.pivot_price - float(weekly["Close"].iat[-1]))
                         / max(float(weekly["Close"].iat[-1]), 1e-9))
                        * 100.0
                    ),
                    quality_score=quality_score,
                    readiness_score=readiness_score,
                    confidence=confidence,
                    metrics={
                        "run_rank": run_rank,
                        "weeks_tight": run.weeks_tight,
                        "tight_mode": run.mode,
                        "tight_mode_threshold_pct": round(
                            run.max_contraction_pct, 4
                        ),
                        "tight_band_pct": round(run.tight_band_pct, 6),
                        "tight_range_pct": round(run.tight_range_pct, 6),
                        "vol_vs_10w": (
                            round(run.vol_vs_10w, 6)
                            if run.vol_vs_10w is not None
                            else None
                        ),
                        "run_start_date": weekly.index[run.start_idx]
                        .date()
                        .isoformat(),
                        "run_end_date": weekly.index[run.end_idx]
                        .date()
                        .isoformat(),
                        "median_close": round(median_close, 6),
                        "recency_weeks": run.recency_weeks,
                    },
                    checks={
                        "weeks_tight_min_met": run.weeks_tight >= _MIN_WEEKS_TIGHT,
                        "tight_band_ok": run.tight_band_pct
                        <= run.max_contraction_pct,
                        "tight_mode_strict": run.mode == "strict",
                        "tight_mode_relaxed": run.mode == "relaxed",
                    },
                    notes=("strict_relaxed_modes_evaluated",),
                )
            )

        return PatternDetectorResult.detected(
            self.name,
            tuple(candidates),
            passed_checks=("tight_run_detected",),
            warnings=warnings,
        )


def _resolve_weekly_frame(
    detector_input: PatternDetectorInput,
) -> tuple[pd.DataFrame | None, tuple[str, ...]]:
    warnings: list[str] = []
    weekly_raw = detector_input.features.get("weekly_ohlcv")
    if isinstance(weekly_raw, pd.DataFrame):
        return weekly_raw, ()

    daily_raw = detector_input.features.get("daily_ohlcv")
    if isinstance(daily_raw, pd.DataFrame):
        normalized_daily = normalize_detector_input_ohlcv(
            features={"daily_ohlcv": daily_raw},
            timeframe="daily",
            min_bars=30,
            feature_key="daily_ohlcv",
            fallback_bar_count=detector_input.daily_bars,
        )
        warnings.extend(normalized_daily.warnings)
        if not normalized_daily.prerequisites_ok or normalized_daily.frame is None:
            return None, tuple(warnings)
        resampled = resample_ohlcv(normalized_daily.frame, rule="W-FRI")
        warnings.append("weekly_ohlcv_resampled_from_daily")
        return resampled, tuple(warnings)

    return None, tuple(warnings)


def _find_tight_runs(
    weekly: pd.DataFrame,
    parameters: SetupEngineParameters,
) -> list[_TightRun]:
    closes = weekly["Close"]
    highs = weekly["High"]
    lows = weekly["Low"]
    volumes = weekly["Volume"]
    n = len(weekly)
    runs: list[_TightRun] = []

    strict_threshold = (
        parameters.three_weeks_tight_max_contraction_pct_strict
    )
    relaxed_threshold = (
        parameters.three_weeks_tight_max_contraction_pct_relaxed
    )

    for weeks_tight in range(_MIN_WEEKS_TIGHT, _MAX_WEEKS_TIGHT + 1):
        for end_idx in range(weeks_tight - 1, n):
            start_idx = end_idx - weeks_tight + 1
            close_window = closes.iloc[start_idx : end_idx + 1]
            high_window = highs.iloc[start_idx : end_idx + 1]
            low_window = lows.iloc[start_idx : end_idx + 1]

            median_close = float(close_window.median())
            if median_close <= 0.0:
                continue

            tight_band_pct = float(
                (
                    (close_window - median_close).abs() / median_close
                ).max()
                * 100.0
            )
            tight_range_pct = float(
                ((high_window.max() - low_window.min()) / median_close) * 100.0
            )

            if start_idx >= 10:
                prior_10w = float(volumes.iloc[start_idx - 10 : start_idx].mean())
                run_vol = float(volumes.iloc[start_idx : end_idx + 1].mean())
                vol_vs_10w = (run_vol / prior_10w) if prior_10w > 0 else None
            else:
                vol_vs_10w = None

            pivot_offset = int(high_window.to_numpy(dtype=float).argmax())
            pivot_idx = start_idx + pivot_offset
            pivot_price = float(highs.iat[pivot_idx])
            recency_weeks = n - 1 - end_idx

            if tight_band_pct <= strict_threshold:
                mode = "strict"
                threshold = strict_threshold
                mode_bias = 0.10
            elif tight_band_pct <= relaxed_threshold:
                mode = "relaxed"
                threshold = relaxed_threshold
                mode_bias = 0.0
            else:
                continue

            score = (
                min(weeks_tight, _MAX_WEEKS_TIGHT) * 0.16
                + max(0.0, 1.0 - (tight_band_pct / max(threshold, 1e-9)))
                * 0.55
                + max(0.0, 1.0 - recency_weeks / 10.0) * 0.19
                + mode_bias
            )
            runs.append(
                _TightRun(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    weeks_tight=weeks_tight,
                    mode=mode,
                    max_contraction_pct=threshold,
                    tight_band_pct=tight_band_pct,
                    tight_range_pct=tight_range_pct,
                    vol_vs_10w=vol_vs_10w,
                    pivot_idx=pivot_idx,
                    pivot_price=pivot_price,
                    recency_weeks=recency_weeks,
                    score=score,
                )
            )

    runs.sort(
        key=lambda run: (
            -run.score,
            run.recency_weeks,
            -run.weeks_tight,
            run.tight_band_pct,
            run.mode != "strict",
            -run.pivot_idx,
        )
    )
    return runs
