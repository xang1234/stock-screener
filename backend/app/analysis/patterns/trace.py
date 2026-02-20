"""Score trace builder for field-level calculation auditability.

Pure module: no pandas, no I/O. Imports only types from ``models.py``
and weights from ``config.py``.
"""

from __future__ import annotations

from app.analysis.patterns.config import SETUP_SCORE_WEIGHTS
from app.analysis.patterns.models import FieldTrace, JsonScalar, ScoreTrace
from app.analysis.patterns.readiness import (
    BreakoutReadinessFeatures,
    BreakoutReadinessTraceInputs,
)


def _r(value: float | int | bool | None) -> JsonScalar:
    """Round numeric values to 6 decimal places; pass through None and bool."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    return round(value, 6)


def build_score_trace(
    features: BreakoutReadinessFeatures,
    trace_inputs: BreakoutReadinessTraceInputs,
    *,
    quality_score: float | None,
    readiness_score: float | None,
    setup_score: float | None,
) -> ScoreTrace:
    """Assemble all 11 trace entries from features and raw intermediates.

    Accepts ``BreakoutReadinessFeatures`` + ``BreakoutReadinessTraceInputs``
    structured objects plus 3 score synthesis kwargs. Uses ``_r()`` (round-to-6dp)
    helper. Never raises; ``None`` inputs produce ``None`` outputs.
    """
    wq, wr = SETUP_SCORE_WEIGHTS

    trace: ScoreTrace = {}

    # ── Composite setup_score ──────────────────────
    trace["setup_score"] = FieldTrace(
        formula=f"{wq} * quality_score + {wr} * readiness_score",
        inputs={
            "quality_score": _r(quality_score),
            "readiness_score": _r(readiness_score),
            "weight_quality": wq,
            "weight_readiness": wr,
        },
        output=_r(setup_score),
        unit="pct",
    )

    # ── distance_to_pivot_pct ──────────────────────
    trace["distance_to_pivot_pct"] = FieldTrace(
        formula="100 * (close - pivot_price) / pivot_price",
        inputs={
            "close": _r(trace_inputs.latest_close),
            "pivot_price": _r(trace_inputs.pivot_price),
        },
        output=_r(features.distance_to_pivot_pct),
        unit="pct",
    )

    # ── atr14_pct ──────────────────────────────────
    trace["atr14_pct"] = FieldTrace(
        formula="100 * ATR14 / close",
        inputs={
            "atr14": _r(trace_inputs.latest_atr14),
            "close": _r(trace_inputs.latest_close),
        },
        output=_r(features.atr14_pct),
        unit="pct",
    )

    # ── atr14_pct_trend ────────────────────────────
    trace["atr14_pct_trend"] = FieldTrace(
        formula="slope(atr14_pct, window=20)",
        inputs={
            "atr14_pct_current": _r(features.atr14_pct),
            "window": 20,
        },
        output=_r(features.atr14_pct_trend),
        unit="pct",
    )

    # ── bb_width_pct ───────────────────────────────
    trace["bb_width_pct"] = FieldTrace(
        formula="100 * (bb_upper - bb_lower) / bb_mid",
        inputs={
            "bb_upper": _r(trace_inputs.latest_bb_upper),
            "bb_lower": _r(trace_inputs.latest_bb_lower),
            "bb_mid": _r(trace_inputs.latest_bb_mid),
        },
        output=_r(features.bb_width_pct),
        unit="pct",
    )

    # ── bb_width_pctile_252 ────────────────────────
    trace["bb_width_pctile_252"] = FieldTrace(
        formula=f"pct_rank(bb_width_pct, window={trace_inputs.bb_pctile_window})",
        inputs={
            "bb_width_pct_current": _r(features.bb_width_pct),
            "window": trace_inputs.bb_pctile_window,
        },
        output=_r(features.bb_width_pctile_252),
        unit="pct",
    )

    # ── volume_vs_50d ──────────────────────────────
    trace["volume_vs_50d"] = FieldTrace(
        formula="volume / SMA(volume, 50)",
        inputs={
            "volume": _r(trace_inputs.latest_volume),
            "volume_sma_50": _r(trace_inputs.volume_sma_50),
        },
        output=_r(features.volume_vs_50d),
        unit="ratio",
    )

    # ── rs ─────────────────────────────────────────
    trace["rs"] = FieldTrace(
        formula="close / benchmark_close",
        inputs={
            "close": _r(trace_inputs.latest_close),
            "benchmark_close": _r(trace_inputs.latest_benchmark_close),
        },
        output=_r(features.rs),
        unit="ratio",
    )

    # ── rs_line_new_high ───────────────────────────
    trace["rs_line_new_high"] = FieldTrace(
        formula="rs_current >= max(rs[-252:])",
        inputs={
            "rs_current": _r(trace_inputs.rs_current),
            "rs_252_max": _r(trace_inputs.rs_252_max),
        },
        output=features.rs_line_new_high,
        unit="bool",
    )

    # ── rs_vs_spy_65d ──────────────────────────────
    trace["rs_vs_spy_65d"] = FieldTrace(
        formula="100 * (rs / rs[-65] - 1)",
        inputs={
            "rs_current": _r(trace_inputs.rs_current),
            "rs_65d_ago": _r(trace_inputs.rs_65d_ago),
        },
        output=_r(features.rs_vs_spy_65d),
        unit="pct",
    )

    # ── rs_vs_spy_trend_20d ────────────────────────
    trace["rs_vs_spy_trend_20d"] = FieldTrace(
        formula="slope(rs, window=20)",
        inputs={
            "rs_current": _r(trace_inputs.rs_current),
            "window": 20,
        },
        output=_r(features.rs_vs_spy_trend_20d),
        unit="ratio",
    )

    return trace


def _null_field_trace() -> FieldTrace:
    return FieldTrace(formula="", inputs={}, output=None, unit="")


def build_null_score_trace() -> ScoreTrace:
    """All-null trace for degraded paths (insufficient data, etc)."""
    return {
        "setup_score": _null_field_trace(),
        "distance_to_pivot_pct": _null_field_trace(),
        "atr14_pct": _null_field_trace(),
        "atr14_pct_trend": _null_field_trace(),
        "bb_width_pct": _null_field_trace(),
        "bb_width_pctile_252": _null_field_trace(),
        "volume_vs_50d": _null_field_trace(),
        "rs": _null_field_trace(),
        "rs_line_new_high": _null_field_trace(),
        "rs_vs_spy_65d": _null_field_trace(),
        "rs_vs_spy_trend_20d": _null_field_trace(),
    }
