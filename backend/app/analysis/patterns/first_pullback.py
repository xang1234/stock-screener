"""First-pullback / trend-resumption detector entrypoint.

Expected input orientation:
- Chronological bars with MA features.
- Distinct touch counting must avoid clustered double-counts.
"""

from __future__ import annotations

import pandas as pd

from app.analysis.patterns.config import SetupEngineParameters
from app.analysis.patterns.detectors.base import (
    PatternDetector,
    PatternDetectorInput,
    PatternDetectorResult,
)
from app.analysis.patterns.models import PatternCandidateModel
from app.analysis.patterns.normalization import normalize_detector_input_ohlcv

_MA_TOUCH_BAND_PCT = 1.5
_TOUCH_SEPARATION_BARS = 5
_MA_PERIOD_BARS = 21
_ORDERLINESS_LOOKBACK_BARS = 35
_PULLBACK_HIGH_LOOKBACK_BARS = 12
_RESUMPTION_EMA_SPAN = 10
_RESUMPTION_LOOKAHEAD_BARS = 15
_RESUMPTION_MIN_VOLUME_RATIO = 0.90


class FirstPullbackDetector(PatternDetector):
    """Compile-safe entrypoint for first-pullback detection."""

    name = "first_pullback"

    def detect(
        self,
        detector_input: PatternDetectorInput,
        parameters: SetupEngineParameters,
    ) -> PatternDetectorResult:
        del parameters
        normalized = normalize_detector_input_ohlcv(
            features=detector_input.features,
            timeframe="daily",
            min_bars=60,
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
                failed_checks=("missing_daily_ohlcv_for_pullback_detection",),
                warnings=normalized.warnings,
            )

        frame = normalized.frame
        ma_reference = _resolve_ma_reference(frame)
        touch_mask = _ma_touch_mask(
            low=frame["Low"],
            ma_reference=ma_reference,
            band_pct=_MA_TOUCH_BAND_PCT,
        )
        test_positions = _distinct_touch_positions(
            touch_mask=touch_mask,
            separation_bars=_TOUCH_SEPARATION_BARS,
        )
        tests_count = len(test_positions)

        if tests_count == 0:
            return PatternDetectorResult.no_detection(
                self.name,
                failed_checks=("no_ma_tests_detected",),
                warnings=normalized.warnings,
            )

        latest_test_idx = test_positions[-1]
        latest_ma = float(ma_reference.iat[latest_test_idx])
        latest_close = float(frame["Close"].iat[latest_test_idx])
        latest_low = float(frame["Low"].iat[latest_test_idx])
        ma_touch_distance_pct = (
            abs(latest_low - latest_ma) / max(abs(latest_ma), 1e-9)
        ) * 100.0

        orderliness = _pullback_orderliness(frame, end_idx=latest_test_idx)
        is_first_test = tests_count == 1
        is_second_test = tests_count == 2
        first_test_idx = test_positions[0]
        pullback_high_start_idx = max(
            0, first_test_idx - _PULLBACK_HIGH_LOOKBACK_BARS
        )
        pullback_high_idx = _argmax_index(
            frame["High"],
            start_idx=pullback_high_start_idx,
            end_idx=latest_test_idx,
        )
        pullback_high_price = float(frame["High"].iat[pullback_high_idx])
        pullback_high_date = frame.index[pullback_high_idx].date().isoformat()
        resumption = _find_resumption_candidate(
            frame,
            start_idx=latest_test_idx + 1,
            end_idx=min(
                len(frame) - 1,
                latest_test_idx + _RESUMPTION_LOOKAHEAD_BARS,
            ),
        )

        if resumption["index"] is None:
            chosen_mode = "pullback_high"
            alternate_mode = "resumption_high"
            chosen_reason = "fallback_no_resumption_trigger"
            pivot_price = pullback_high_price
            pivot_date = pullback_high_date
        else:
            resumption_idx = int(resumption["index"])
            chosen_mode = "resumption_high"
            alternate_mode = "pullback_high"
            chosen_reason = "resumption_trigger_confirmed"
            pivot_price = float(frame["High"].iat[resumption_idx])
            pivot_date = frame.index[resumption_idx].date().isoformat()

        resumption_bonus = 0.08 if chosen_mode == "resumption_high" else -0.02
        confidence = _confidence_from_tests(
            tests_count=tests_count,
            orderliness_score=orderliness["pullback_orderliness_score"],
        )
        confidence = min(0.95, max(0.05, confidence + resumption_bonus))
        quality_score = min(
            100.0,
            max(
                0.0,
                orderliness["pullback_orderliness_score"] * 100.0
                + (8.0 if chosen_mode == "resumption_high" else 0.0),
            ),
        )
        readiness_penalty = max(0, tests_count - 1) * 15.0
        readiness_score = max(
            0.0,
            min(
                100.0,
                55.0 + (orderliness["pullback_orderliness_score"] * 35.0)
                + (12.0 if chosen_mode == "resumption_high" else -6.0)
                - readiness_penalty,
            ),
        )
        pullback_span_bars = latest_test_idx - first_test_idx + 1

        candidate = PatternCandidateModel(
            pattern=self.name,
            timeframe="daily",
            source_detector=self.name,
            pivot_price=pivot_price,
            pivot_type=chosen_mode,
            pivot_date=pivot_date,
            confidence=confidence,
            quality_score=quality_score,
            readiness_score=readiness_score,
            metrics={
                "ma_period_bars": _MA_PERIOD_BARS,
                "ma_touch_band_pct": _MA_TOUCH_BAND_PCT,
                "touch_separation_bars": _TOUCH_SEPARATION_BARS,
                "tests_count": tests_count,
                "is_first_test": is_first_test,
                "is_second_test": is_second_test,
                "latest_test_close": round(latest_close, 4),
                "latest_test_low": round(latest_low, 4),
                "latest_test_ma": round(latest_ma, 4),
                "ma_touch_distance_pct": round(ma_touch_distance_pct, 4),
                "pullback_span_bars": pullback_span_bars,
                "pullback_high_price": round(pullback_high_price, 4),
                "pullback_high_date": pullback_high_date,
                "resumption_high_price": (
                    round(float(resumption["high_price"]), 4)
                    if resumption["high_price"] is not None
                    else None
                ),
                "resumption_high_date": resumption["date"],
                "resumption_close": (
                    round(float(resumption["close"]), 4)
                    if resumption["close"] is not None
                    else None
                ),
                "resumption_ema10": (
                    round(float(resumption["ema10"]), 4)
                    if resumption["ema10"] is not None
                    else None
                ),
                "resumption_volume_ratio_20d": (
                    round(float(resumption["volume_ratio"]), 6)
                    if resumption["volume_ratio"] is not None
                    else None
                ),
                "resumption_trigger_offset_bars": (
                    int(resumption["index"]) - latest_test_idx
                    if resumption["index"] is not None
                    else None
                ),
                "pivot_mode_chosen": chosen_mode,
                "pivot_mode_alternate": alternate_mode,
                "pivot_choice_reason": chosen_reason,
                "pullback_orderliness_score": round(
                    orderliness["pullback_orderliness_score"], 6
                ),
                "pullback_avg_range_pct": round(
                    orderliness["pullback_avg_range_pct"], 6
                ),
                "pullback_close_volatility_pct": round(
                    orderliness["pullback_close_volatility_pct"], 6
                ),
                "pullback_lower_high_ratio": round(
                    orderliness["pullback_lower_high_ratio"], 6
                ),
            },
            checks={
                "ma_touch_detected": True,
                "ma_touch_within_band": ma_touch_distance_pct <= _MA_TOUCH_BAND_PCT,
                "is_first_test": is_first_test,
                "is_second_test": is_second_test,
                "pullback_orderly": orderliness["pullback_orderliness_score"]
                >= 0.5,
                "resumption_trigger_confirmed": chosen_mode == "resumption_high",
                "resumption_price_reclaimed_ema10": bool(
                    resumption["price_above_ema10"]
                ),
                "resumption_volume_supportive": bool(
                    resumption["volume_supportive"]
                ),
                "pivot_mode_pullback_high": chosen_mode == "pullback_high",
                "pivot_mode_resumption_high": chosen_mode
                == "resumption_high",
            },
            notes=(
                "ma_touch_test_counting_complete",
                "resumption_trigger_evaluated",
            ),
        )
        return PatternDetectorResult.detected(
            self.name,
            candidate,
            passed_checks=("ma_tests_identified",),
            warnings=normalized.warnings,
        )


def _resolve_ma_reference(frame: pd.DataFrame) -> pd.Series:
    for column in (
        "ma_21",
        "MA21",
        "sma_21",
        "ema_21",
        "ma21",
    ):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return frame["Close"].rolling(
        window=_MA_PERIOD_BARS,
        min_periods=_MA_PERIOD_BARS,
    ).mean()


def _ma_touch_mask(
    *,
    low: pd.Series,
    ma_reference: pd.Series,
    band_pct: float,
) -> pd.Series:
    ma_abs = ma_reference.abs().replace(0.0, float("nan"))
    distance_pct = (low - ma_reference).abs() / ma_abs * 100.0
    return distance_pct.le(band_pct) & ma_reference.notna()


def _distinct_touch_positions(
    *,
    touch_mask: pd.Series,
    separation_bars: int,
) -> list[int]:
    touch_indices = [idx for idx, touched in enumerate(touch_mask.tolist()) if touched]
    distinct: list[int] = []
    for idx in touch_indices:
        if not distinct or idx - distinct[-1] > separation_bars:
            distinct.append(idx)
    return distinct


def _pullback_orderliness(frame: pd.DataFrame, *, end_idx: int) -> dict[str, float]:
    start_idx = max(0, end_idx - _ORDERLINESS_LOOKBACK_BARS + 1)
    segment = frame.iloc[start_idx : end_idx + 1]
    if len(segment) < 3:
        return {
            "pullback_orderliness_score": 0.0,
            "pullback_avg_range_pct": 0.0,
            "pullback_close_volatility_pct": 0.0,
            "pullback_lower_high_ratio": 0.0,
        }

    high_diff = segment["High"].diff().dropna()
    lower_high_ratio = (
        float(high_diff.le(0).sum()) / float(len(high_diff))
        if len(high_diff) > 0
        else 0.0
    )
    range_pct = (
        (segment["High"] - segment["Low"])
        / segment["Close"].abs().replace(0.0, float("nan"))
        * 100.0
    ).dropna()
    avg_range_pct = float(range_pct.mean()) if not range_pct.empty else 0.0

    close_changes = segment["Close"].pct_change().dropna()
    close_volatility_pct = (
        float(close_changes.std(ddof=0) * 100.0)
        if not close_changes.empty
        else 0.0
    )

    range_component = max(0.0, 1.0 - (avg_range_pct / 6.0))
    volatility_component = max(0.0, 1.0 - (close_volatility_pct / 2.0))
    orderliness_score = min(
        1.0,
        max(
            0.0,
            0.55 * lower_high_ratio
            + 0.25 * range_component
            + 0.20 * volatility_component,
        ),
    )
    return {
        "pullback_orderliness_score": orderliness_score,
        "pullback_avg_range_pct": avg_range_pct,
        "pullback_close_volatility_pct": close_volatility_pct,
        "pullback_lower_high_ratio": lower_high_ratio,
    }


def _confidence_from_tests(*, tests_count: int, orderliness_score: float) -> float:
    if tests_count <= 1:
        test_component = 0.20
    elif tests_count == 2:
        test_component = 0.12
    else:
        test_component = max(0.0, 0.12 - (tests_count - 2) * 0.06)

    return min(
        0.95,
        max(
            0.05,
            0.30 + orderliness_score * 0.45 + test_component,
        ),
    )


def _argmax_index(series: pd.Series, *, start_idx: int, end_idx: int) -> int:
    segment = series.iloc[start_idx : end_idx + 1].to_numpy(dtype=float)
    return start_idx + int(segment.argmax())


def _find_resumption_candidate(
    frame: pd.DataFrame,
    *,
    start_idx: int,
    end_idx: int,
) -> dict[str, int | float | str | bool | None]:
    if start_idx > end_idx or start_idx >= len(frame):
        return {
            "index": None,
            "date": None,
            "high_price": None,
            "close": None,
            "ema10": None,
            "volume_ratio": None,
            "price_above_ema10": False,
            "volume_supportive": False,
        }

    close = frame["Close"]
    high = frame["High"]
    volume = frame["Volume"]
    ema10 = close.ewm(span=_RESUMPTION_EMA_SPAN, adjust=False).mean()
    volume20 = volume.rolling(window=20, min_periods=5).mean()

    for idx in range(start_idx, min(end_idx, len(frame) - 1) + 1):
        prev_idx = idx - 1
        if prev_idx < 0:
            continue

        close_now = float(close.iat[idx])
        close_prev = float(close.iat[prev_idx])
        ema_now = float(ema10.iat[idx])
        if pd.isna(ema_now):
            continue

        price_above_ema10 = close_now > ema_now
        price_up_day = close_now > close_prev
        if not (price_above_ema10 and price_up_day):
            continue

        rolling_vol = float(volume20.iat[idx])
        if rolling_vol <= 0.0 or pd.isna(rolling_vol):
            volume_ratio = 1.0
        else:
            volume_ratio = float(volume.iat[idx]) / rolling_vol
        volume_supportive = volume_ratio >= _RESUMPTION_MIN_VOLUME_RATIO
        if not volume_supportive:
            continue

        return {
            "index": idx,
            "date": frame.index[idx].date().isoformat(),
            "high_price": float(high.iat[idx]),
            "close": close_now,
            "ema10": ema_now,
            "volume_ratio": volume_ratio,
            "price_above_ema10": price_above_ema10,
            "volume_supportive": volume_supportive,
        }

    return {
        "index": None,
        "date": None,
        "high_price": None,
        "close": None,
        "ema10": None,
        "volume_ratio": None,
        "price_above_ema10": False,
        "volume_supportive": False,
    }
