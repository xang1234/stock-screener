"""Tests for SE-D6: Field-level calculation spec and trace metadata."""

import json

import numpy as np
import pandas as pd
import pytest

from app.analysis.patterns.config import SETUP_SCORE_WEIGHTS
from app.analysis.patterns.models import FieldTrace, ScoreTrace
from app.analysis.patterns.readiness import (
    BreakoutReadinessFeatures,
    BreakoutReadinessTraceInputs,
    compute_breakout_readiness_features,
    compute_breakout_readiness_features_with_trace,
)
from app.analysis.patterns.report import (
    validate_setup_engine_report_payload,
)
from app.analysis.patterns.trace import (
    build_null_score_trace,
    build_score_trace,
)
from app.scanners.setup_engine_scanner import build_setup_engine_payload


# ── Deterministic fixtures ─────────────────────────


def _make_features() -> BreakoutReadinessFeatures:
    return BreakoutReadinessFeatures(
        distance_to_pivot_pct=-1.131222,
        atr14_pct=2.434325,
        atr14_pct_trend=-0.01,
        bb_width_pct=5.99,
        bb_width_pctile_252=28.0,
        volume_vs_50d=1.208333,
        rs=2.427778,
        rs_line_new_high=True,
        rs_vs_spy_65d=15.6,
        rs_vs_spy_trend_20d=0.003,
    )


def _make_trace_inputs() -> BreakoutReadinessTraceInputs:
    return BreakoutReadinessTraceInputs(
        latest_close=218.5,
        latest_atr14=5.32,
        latest_bb_upper=225.1,
        latest_bb_lower=212.0,
        latest_bb_mid=218.5,
        latest_volume=1_450_000,
        volume_sma_50=1_200_000,
        pivot_price=221.0,
        bb_pctile_window=252,
        latest_benchmark_close=90.0,
        rs_current=2.427778,
        rs_252_max=2.427778,
        rs_65d_ago=2.1,
    )


EXPECTED_TRACE_FIELDS = {
    "setup_score",
    "distance_to_pivot_pct",
    "atr14_pct",
    "atr14_pct_trend",
    "bb_width_pct",
    "bb_width_pctile_252",
    "volume_vs_50d",
    "rs",
    "rs_line_new_high",
    "rs_vs_spy_65d",
    "rs_vs_spy_trend_20d",
}


# ── TestBuildScoreTrace ────────────────────────────


class TestBuildScoreTrace:
    """Unit tests for build_score_trace()."""

    def test_all_11_entries_present(self):
        trace = build_score_trace(
            _make_features(),
            _make_trace_inputs(),
            quality_score=75.0,
            readiness_score=82.0,
            setup_score=77.8,
        )
        assert set(trace.keys()) == EXPECTED_TRACE_FIELDS

    def test_each_entry_has_required_keys(self):
        trace = build_score_trace(
            _make_features(),
            _make_trace_inputs(),
            quality_score=75.0,
            readiness_score=82.0,
            setup_score=77.8,
        )
        for field_name, entry in trace.items():
            assert "formula" in entry, f"{field_name} missing formula"
            assert "inputs" in entry, f"{field_name} missing inputs"
            assert "output" in entry, f"{field_name} missing output"
            assert "unit" in entry, f"{field_name} missing unit"

    def test_setup_score_trace_entry(self):
        wq, wr = SETUP_SCORE_WEIGHTS
        trace = build_score_trace(
            _make_features(),
            _make_trace_inputs(),
            quality_score=75.0,
            readiness_score=82.0,
            setup_score=77.8,
        )
        entry = trace["setup_score"]
        assert entry["formula"] == f"{wq} * quality_score + {wr} * readiness_score"
        assert entry["inputs"]["quality_score"] == 75.0
        assert entry["inputs"]["readiness_score"] == 82.0
        assert entry["inputs"]["weight_quality"] == wq
        assert entry["inputs"]["weight_readiness"] == wr
        assert entry["output"] == 77.8
        assert entry["unit"] == "pct"

    def test_distance_to_pivot_trace_entry(self):
        trace = build_score_trace(
            _make_features(),
            _make_trace_inputs(),
            quality_score=75.0,
            readiness_score=82.0,
            setup_score=77.8,
        )
        entry = trace["distance_to_pivot_pct"]
        assert entry["formula"] == "100 * (close - pivot_price) / pivot_price"
        assert entry["inputs"]["close"] == 218.5
        assert entry["inputs"]["pivot_price"] == 221.0
        assert entry["output"] == pytest.approx(-1.131222)
        assert entry["unit"] == "pct"

    def test_atr14_pct_trace_entry(self):
        trace = build_score_trace(
            _make_features(),
            _make_trace_inputs(),
            quality_score=75.0,
            readiness_score=82.0,
            setup_score=77.8,
        )
        entry = trace["atr14_pct"]
        assert entry["formula"] == "100 * ATR14 / close"
        assert entry["inputs"]["atr14"] == 5.32
        assert entry["inputs"]["close"] == 218.5
        assert entry["output"] == pytest.approx(2.434325)
        assert entry["unit"] == "pct"

    def test_bb_width_pct_trace_entry(self):
        trace = build_score_trace(
            _make_features(),
            _make_trace_inputs(),
            quality_score=75.0,
            readiness_score=82.0,
            setup_score=77.8,
        )
        entry = trace["bb_width_pct"]
        assert entry["formula"] == "100 * (bb_upper - bb_lower) / bb_mid"
        assert entry["inputs"]["bb_upper"] == 225.1
        assert entry["inputs"]["bb_lower"] == 212.0
        assert entry["inputs"]["bb_mid"] == 218.5
        assert entry["output"] == pytest.approx(5.99)
        assert entry["unit"] == "pct"

    def test_volume_vs_50d_trace_entry(self):
        trace = build_score_trace(
            _make_features(),
            _make_trace_inputs(),
            quality_score=75.0,
            readiness_score=82.0,
            setup_score=77.8,
        )
        entry = trace["volume_vs_50d"]
        assert entry["formula"] == "volume / SMA(volume, 50)"
        assert entry["inputs"]["volume"] == 1_450_000
        assert entry["inputs"]["volume_sma_50"] == 1_200_000
        assert entry["output"] == pytest.approx(1.208333)
        assert entry["unit"] == "ratio"

    def test_rs_trace_entry(self):
        trace = build_score_trace(
            _make_features(),
            _make_trace_inputs(),
            quality_score=75.0,
            readiness_score=82.0,
            setup_score=77.8,
        )
        entry = trace["rs"]
        assert entry["formula"] == "close / benchmark_close"
        assert entry["inputs"]["close"] == 218.5
        assert entry["inputs"]["benchmark_close"] == 90.0
        assert entry["output"] == pytest.approx(2.427778)
        assert entry["unit"] == "ratio"

    def test_rs_line_new_high_trace_entry(self):
        trace = build_score_trace(
            _make_features(),
            _make_trace_inputs(),
            quality_score=75.0,
            readiness_score=82.0,
            setup_score=77.8,
        )
        entry = trace["rs_line_new_high"]
        assert entry["formula"] == "rs_current >= max(rs[-252:])"
        assert entry["inputs"]["rs_current"] == pytest.approx(2.427778)
        assert entry["inputs"]["rs_252_max"] == pytest.approx(2.427778)
        assert entry["output"] is True
        assert entry["unit"] == "bool"

    def test_rs_vs_spy_65d_trace_entry(self):
        trace = build_score_trace(
            _make_features(),
            _make_trace_inputs(),
            quality_score=75.0,
            readiness_score=82.0,
            setup_score=77.8,
        )
        entry = trace["rs_vs_spy_65d"]
        assert entry["formula"] == "100 * (rs / rs[-65] - 1)"
        assert entry["inputs"]["rs_current"] == pytest.approx(2.427778)
        assert entry["inputs"]["rs_65d_ago"] == 2.1
        assert entry["output"] == pytest.approx(15.6)
        assert entry["unit"] == "pct"

    def test_rs_vs_spy_trend_20d_trace_entry(self):
        trace = build_score_trace(
            _make_features(),
            _make_trace_inputs(),
            quality_score=75.0,
            readiness_score=82.0,
            setup_score=77.8,
        )
        entry = trace["rs_vs_spy_trend_20d"]
        assert entry["inputs"]["window"] == 20
        assert entry["output"] == 0.003
        assert entry["unit"] == "ratio"

    def test_null_propagation(self):
        """None inputs produce None outputs throughout the trace."""
        null_features = BreakoutReadinessFeatures(
            distance_to_pivot_pct=None,
            atr14_pct=None,
            atr14_pct_trend=None,
            bb_width_pct=None,
            bb_width_pctile_252=None,
            volume_vs_50d=None,
            rs=None,
            rs_line_new_high=False,
            rs_vs_spy_65d=None,
            rs_vs_spy_trend_20d=None,
        )
        null_inputs = BreakoutReadinessTraceInputs(
            latest_close=None,
            latest_atr14=None,
            latest_bb_upper=None,
            latest_bb_lower=None,
            latest_bb_mid=None,
            latest_volume=None,
            volume_sma_50=None,
            pivot_price=None,
            bb_pctile_window=252,
            latest_benchmark_close=None,
            rs_current=None,
            rs_252_max=None,
            rs_65d_ago=None,
        )
        trace = build_score_trace(
            null_features,
            null_inputs,
            quality_score=None,
            readiness_score=None,
            setup_score=None,
        )
        assert set(trace.keys()) == EXPECTED_TRACE_FIELDS
        for field_name, entry in trace.items():
            if field_name == "rs_line_new_high":
                assert entry["output"] is False
            else:
                assert entry["output"] is None, f"{field_name} should be None"

    def test_json_serialization(self):
        """Trace must be fully JSON-serializable."""
        trace = build_score_trace(
            _make_features(),
            _make_trace_inputs(),
            quality_score=75.0,
            readiness_score=82.0,
            setup_score=77.8,
        )
        serialized = json.dumps(trace)
        deserialized = json.loads(serialized)
        assert set(deserialized.keys()) == EXPECTED_TRACE_FIELDS

    def test_precision_6dp(self):
        """Numeric outputs are rounded to 6 decimal places max."""
        features = BreakoutReadinessFeatures(
            distance_to_pivot_pct=1.1234567890,
            atr14_pct=2.9876543210,
            atr14_pct_trend=0.00012345678,
            bb_width_pct=5.99999999,
            bb_width_pctile_252=28.123456789,
            volume_vs_50d=1.123456789,
            rs=2.123456789,
            rs_line_new_high=True,
            rs_vs_spy_65d=15.123456789,
            rs_vs_spy_trend_20d=0.001234567,
        )
        trace_inputs = _make_trace_inputs()
        trace = build_score_trace(
            features,
            trace_inputs,
            quality_score=75.1234567,
            readiness_score=82.9876543,
            setup_score=78.5678901,
        )
        for field_name, entry in trace.items():
            output = entry["output"]
            if isinstance(output, float):
                # round(value, 6) must be a fixpoint: rounding again changes nothing
                assert output == round(output, 6), (
                    f"{field_name} output {output} not rounded to 6dp"
                )

    def test_null_score_trace(self):
        """build_null_score_trace returns all 11 entries with None output."""
        trace = build_null_score_trace()
        assert set(trace.keys()) == EXPECTED_TRACE_FIELDS
        for field_name, entry in trace.items():
            assert entry["output"] is None
            assert entry["formula"] == ""
            assert entry["inputs"] == {}

    def test_null_score_trace_entries_are_distinct_objects(self):
        """Each null trace entry must be a separate dict (no shared mutable refs)."""
        trace = build_null_score_trace()
        entries = list(trace.values())
        for i, a in enumerate(entries):
            for b in entries[i + 1:]:
                assert a is not b, "Null trace entries must not alias the same dict"

    def test_unit_values_are_valid(self):
        """Each trace entry uses a recognized unit."""
        valid_units = {"pct", "ratio", "bool"}
        trace = build_score_trace(
            _make_features(),
            _make_trace_inputs(),
            quality_score=75.0,
            readiness_score=82.0,
            setup_score=77.8,
        )
        for field_name, entry in trace.items():
            assert entry["unit"] in valid_units, (
                f"{field_name} has unexpected unit: {entry['unit']}"
            )


# ── TestScoreTraceInPayload ────────────────────────


class TestScoreTraceInPayload:
    """Integration: trace absent by default, present when opt-in."""

    def test_trace_absent_by_default(self):
        payload = build_setup_engine_payload(
            quality_score=75.0,
            readiness_score=82.0,
            distance_to_pivot_pct=1.0,
        )
        assert "score_trace" not in payload["explain"]

    def test_trace_absent_without_trace_inputs(self):
        features = _make_features()
        payload = build_setup_engine_payload(
            quality_score=75.0,
            readiness_score=82.0,
            readiness_features=features,
            distance_to_pivot_pct=1.0,
            include_score_trace=True,
            readiness_trace_inputs=None,
        )
        assert "score_trace" not in payload["explain"]

    def test_trace_present_when_opted_in(self):
        features = _make_features()
        trace_inputs = _make_trace_inputs()
        payload = build_setup_engine_payload(
            quality_score=75.0,
            readiness_score=82.0,
            readiness_features=features,
            distance_to_pivot_pct=1.0,
            include_score_trace=True,
            readiness_trace_inputs=trace_inputs,
        )
        assert "score_trace" in payload["explain"]
        score_trace = payload["explain"]["score_trace"]
        assert set(score_trace.keys()) == EXPECTED_TRACE_FIELDS

    def test_trace_passes_report_validation(self):
        features = _make_features()
        trace_inputs = _make_trace_inputs()
        payload = build_setup_engine_payload(
            quality_score=75.0,
            readiness_score=82.0,
            readiness_features=features,
            distance_to_pivot_pct=1.0,
            include_score_trace=True,
            readiness_trace_inputs=trace_inputs,
        )
        errors = validate_setup_engine_report_payload(payload)
        assert errors == []


# ── TestComputeReadinessWithTrace ──────────────────


def _price_frame(periods: int = 320) -> pd.DataFrame:
    idx = pd.bdate_range("2025-01-02", periods=periods)
    close = np.linspace(100.0, 220.0, periods)
    return pd.DataFrame(
        {
            "Open": close * 0.995,
            "High": close * 1.02,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.linspace(1_000_000.0, 1_500_000.0, periods),
        },
        index=idx,
    )


class TestComputeReadinessWithTrace:
    """Acceptance: with_trace returns identical features; trace inputs reproduce values."""

    def test_with_trace_returns_same_features(self):
        """compute_breakout_readiness_features_with_trace returns same features."""
        frame = _price_frame()
        benchmark = pd.Series(
            np.linspace(50.0, 90.0, len(frame)),
            index=frame.index,
        )
        pivot_price = 210.0

        features_only = compute_breakout_readiness_features(
            frame,
            pivot_price=pivot_price,
            benchmark_close=benchmark,
        )
        features_with, trace_inputs = compute_breakout_readiness_features_with_trace(
            frame,
            pivot_price=pivot_price,
            benchmark_close=benchmark,
        )

        assert features_only == features_with

    def test_trace_inputs_reproduce_distance_to_pivot(self):
        """Trace inputs independently reproduce distance_to_pivot_pct."""
        frame = _price_frame()
        benchmark = pd.Series(
            np.linspace(50.0, 90.0, len(frame)),
            index=frame.index,
        )
        features, ti = compute_breakout_readiness_features_with_trace(
            frame,
            pivot_price=210.0,
            benchmark_close=benchmark,
        )
        if ti.latest_close is not None and ti.pivot_price is not None and ti.pivot_price != 0:
            recomputed = 100.0 * (ti.latest_close - ti.pivot_price) / ti.pivot_price
            assert features.distance_to_pivot_pct == pytest.approx(recomputed)

    def test_trace_inputs_reproduce_atr14_pct(self):
        """Trace inputs independently reproduce atr14_pct."""
        frame = _price_frame()
        features, ti = compute_breakout_readiness_features_with_trace(
            frame,
            pivot_price=210.0,
        )
        if ti.latest_atr14 is not None and ti.latest_close is not None and ti.latest_close != 0:
            recomputed = 100.0 * ti.latest_atr14 / ti.latest_close
            assert features.atr14_pct == pytest.approx(recomputed)

    def test_trace_inputs_reproduce_bb_width_pct(self):
        """Trace inputs independently reproduce bb_width_pct."""
        frame = _price_frame()
        features, ti = compute_breakout_readiness_features_with_trace(
            frame,
            pivot_price=210.0,
        )
        if (
            ti.latest_bb_upper is not None
            and ti.latest_bb_lower is not None
            and ti.latest_bb_mid is not None
            and ti.latest_bb_mid != 0
        ):
            recomputed = 100.0 * (ti.latest_bb_upper - ti.latest_bb_lower) / ti.latest_bb_mid
            assert features.bb_width_pct == pytest.approx(recomputed)

    def test_trace_inputs_reproduce_volume_vs_50d(self):
        """Trace inputs independently reproduce volume_vs_50d."""
        frame = _price_frame()
        features, ti = compute_breakout_readiness_features_with_trace(
            frame,
            pivot_price=210.0,
        )
        if ti.latest_volume is not None and ti.volume_sma_50 is not None and ti.volume_sma_50 != 0:
            recomputed = ti.latest_volume / ti.volume_sma_50
            assert features.volume_vs_50d == pytest.approx(recomputed)

    def test_trace_inputs_reproduce_rs(self):
        """Trace inputs independently reproduce rs."""
        frame = _price_frame()
        benchmark = pd.Series(
            np.linspace(50.0, 90.0, len(frame)),
            index=frame.index,
        )
        features, ti = compute_breakout_readiness_features_with_trace(
            frame,
            pivot_price=210.0,
            benchmark_close=benchmark,
        )
        if (
            ti.latest_close is not None
            and ti.latest_benchmark_close is not None
            and ti.latest_benchmark_close != 0
        ):
            recomputed = ti.latest_close / ti.latest_benchmark_close
            assert features.rs == pytest.approx(recomputed)

    def test_trace_inputs_reproduce_rs_vs_spy_65d(self):
        """Trace inputs independently reproduce rs_vs_spy_65d."""
        frame = _price_frame()
        benchmark = pd.Series(
            np.linspace(50.0, 90.0, len(frame)),
            index=frame.index,
        )
        features, ti = compute_breakout_readiness_features_with_trace(
            frame,
            pivot_price=210.0,
            benchmark_close=benchmark,
        )
        if ti.rs_current is not None and ti.rs_65d_ago is not None and ti.rs_65d_ago != 0:
            recomputed = 100.0 * (ti.rs_current / ti.rs_65d_ago - 1)
            assert features.rs_vs_spy_65d == pytest.approx(recomputed)

    def test_trace_inputs_reproduce_rs_line_new_high(self):
        """Trace inputs independently reproduce rs_line_new_high."""
        frame = _price_frame()
        benchmark = pd.Series(
            np.linspace(50.0, 90.0, len(frame)),
            index=frame.index,
        )
        features, ti = compute_breakout_readiness_features_with_trace(
            frame,
            pivot_price=210.0,
            benchmark_close=benchmark,
        )
        if ti.rs_current is not None and ti.rs_252_max is not None:
            recomputed = ti.rs_current >= ti.rs_252_max - 1e-12
            assert features.rs_line_new_high == recomputed

    def test_full_roundtrip_trace_in_payload(self):
        """Full integration: compute with trace, build payload, validate."""
        frame = _price_frame()
        benchmark = pd.Series(
            np.linspace(50.0, 90.0, len(frame)),
            index=frame.index,
        )
        features, trace_inputs = compute_breakout_readiness_features_with_trace(
            frame,
            pivot_price=210.0,
            benchmark_close=benchmark,
        )
        payload = build_setup_engine_payload(
            quality_score=75.0,
            readiness_score=82.0,
            readiness_features=features,
            include_score_trace=True,
            readiness_trace_inputs=trace_inputs,
        )
        errors = validate_setup_engine_report_payload(payload)
        assert errors == []
        assert "score_trace" in payload["explain"]
        assert set(payload["explain"]["score_trace"].keys()) == EXPECTED_TRACE_FIELDS
