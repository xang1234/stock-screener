"""Tests for SE-D5: Self-contained explain payload builder.

Comprehensive test suite with class-per-gate structure, exercising each
of the 8 readiness gates in isolation plus merge/trace/readiness semantics.
"""

import pytest

from app.analysis.patterns.config import (
    DEFAULT_SETUP_ENGINE_PARAMETERS,
    SetupEngineParameters,
)
from app.analysis.patterns.explain_builder import (
    ExplainBuilderInput,
    ExplainResult,
    build_explain_payload,
)
from app.analysis.patterns.readiness import (
    BreakoutReadinessFeatures,
    BreakoutReadinessTraceInputs,
)
from app.analysis.patterns.report import ExplainPayload, KeyLevels


# ── Factory helper ────────────────────────────────


def _default_input(**overrides) -> ExplainBuilderInput:
    """Build an ExplainBuilderInput with all-passing defaults.

    Override any field by keyword argument. Defaults produce a fully-passing
    8-gate scenario with no pre-existing failures.
    """
    defaults = dict(
        pre_existing_passed_checks=(),
        pre_existing_failed_checks=(),
        key_levels=KeyLevels(),
        pre_existing_invalidation_flags=(),
        setup_score=80.0,
        quality_score=75.0,
        readiness_score=82.0,
        distance_to_pivot_pct=1.0,
        atr14_pct=3.0,
        volume_vs_50d=1.2,
        rs_vs_spy_65d=5.0,
        rs_line_new_high=True,
        stage=2,
        ma_alignment_score=80.0,
        rs_rating=60.0,
        parameters=DEFAULT_SETUP_ENGINE_PARAMETERS,
        readiness_threshold_pct=70.0,
        include_score_trace=False,
        readiness_features=None,
        readiness_trace_inputs=None,
    )
    defaults.update(overrides)
    return ExplainBuilderInput(**defaults)


# ── Return type checks ───────────────────────────


class TestExplainBuilderReturnType:
    def test_returns_explain_result(self):
        result = build_explain_payload(_default_input())
        assert isinstance(result, ExplainResult)

    def test_explain_is_typed_payload(self):
        result = build_explain_payload(_default_input())
        assert isinstance(result.explain, ExplainPayload)

    def test_derived_ready_is_bool(self):
        result = build_explain_payload(_default_input())
        assert isinstance(result.derived_ready, bool)

    def test_to_payload_produces_dict(self):
        result = build_explain_payload(_default_input())
        payload = result.explain.to_payload()
        assert isinstance(payload, dict)
        assert "passed_checks" in payload
        assert "failed_checks" in payload
        assert "key_levels" in payload
        assert "invalidation_flags" in payload


# ── No-score fast path ───────────────────────────


class TestNoScoreFastPath:
    def test_no_score_skips_gates(self):
        result = build_explain_payload(_default_input(setup_score=None))
        assert result.derived_ready is False
        payload = result.explain.to_payload()
        # No gate checks should be appended
        assert "setup_score_ok" not in payload["passed_checks"]
        assert "setup_score_below_threshold" not in payload["failed_checks"]

    def test_no_score_preserves_pre_existing_checks(self):
        result = build_explain_payload(_default_input(
            setup_score=None,
            pre_existing_passed_checks=("detector_a_ok",),
            pre_existing_failed_checks=("detector_b_fail",),
        ))
        payload = result.explain.to_payload()
        assert "detector_a_ok" in payload["passed_checks"]
        assert "detector_b_fail" in payload["failed_checks"]

    def test_no_score_no_gate_checks_in_passed(self):
        gate_checks = {
            "setup_score_ok", "quality_floor_ok", "readiness_floor_ok",
            "in_early_zone", "atr14_within_limit", "volume_sufficient",
            "rs_leadership_ok", "stage_ok", "ma_alignment_ok", "rs_rating_ok",
        }
        result = build_explain_payload(_default_input(setup_score=None))
        payload = result.explain.to_payload()
        assert gate_checks.isdisjoint(set(payload["passed_checks"]))


# ── Gate 1: setup_score threshold ─────────────────


class TestGate1SetupScore:
    def test_score_above_threshold_passes(self):
        result = build_explain_payload(_default_input(setup_score=80.0))
        assert "setup_score_ok" in result.explain.to_payload()["passed_checks"]

    def test_score_at_threshold_passes(self):
        threshold = DEFAULT_SETUP_ENGINE_PARAMETERS.setup_score_min_pct
        result = build_explain_payload(_default_input(setup_score=threshold))
        assert "setup_score_ok" in result.explain.to_payload()["passed_checks"]

    def test_score_below_threshold_fails(self):
        result = build_explain_payload(_default_input(setup_score=10.0))
        assert "setup_score_below_threshold" in result.explain.to_payload()["failed_checks"]


# ── Gate 2: quality floor ────────────────────────


class TestGate2QualityFloor:
    def test_quality_above_threshold_passes(self):
        result = build_explain_payload(_default_input(quality_score=75.0))
        assert "quality_floor_ok" in result.explain.to_payload()["passed_checks"]

    def test_quality_at_threshold_passes(self):
        threshold = DEFAULT_SETUP_ENGINE_PARAMETERS.quality_score_min_pct
        result = build_explain_payload(_default_input(quality_score=threshold))
        assert "quality_floor_ok" in result.explain.to_payload()["passed_checks"]

    def test_quality_below_threshold_fails(self):
        result = build_explain_payload(_default_input(quality_score=20.0))
        assert "quality_below_threshold" in result.explain.to_payload()["failed_checks"]

    def test_quality_none_fails(self):
        result = build_explain_payload(_default_input(quality_score=None))
        assert "quality_below_threshold" in result.explain.to_payload()["failed_checks"]


# ── Gate 3: readiness floor ──────────────────────


class TestGate3ReadinessFloor:
    def test_readiness_above_threshold_passes(self):
        result = build_explain_payload(_default_input(readiness_score=85.0))
        assert "readiness_floor_ok" in result.explain.to_payload()["passed_checks"]

    def test_readiness_at_threshold_passes(self):
        result = build_explain_payload(_default_input(
            readiness_score=70.0,
            readiness_threshold_pct=70.0,
        ))
        assert "readiness_floor_ok" in result.explain.to_payload()["passed_checks"]

    def test_readiness_below_threshold_fails(self):
        result = build_explain_payload(_default_input(readiness_score=50.0))
        assert "readiness_below_threshold" in result.explain.to_payload()["failed_checks"]

    def test_readiness_none_fails(self):
        result = build_explain_payload(_default_input(readiness_score=None))
        assert "readiness_below_threshold" in result.explain.to_payload()["failed_checks"]


# ── Gate 4: early zone (non-permissive) ──────────


class TestGate4EarlyZone:
    def test_in_range_passes(self):
        result = build_explain_payload(_default_input(distance_to_pivot_pct=1.0))
        assert "in_early_zone" in result.explain.to_payload()["passed_checks"]

    def test_at_min_boundary_passes(self):
        min_val = DEFAULT_SETUP_ENGINE_PARAMETERS.early_zone_distance_to_pivot_pct_min
        result = build_explain_payload(_default_input(distance_to_pivot_pct=min_val))
        assert "in_early_zone" in result.explain.to_payload()["passed_checks"]

    def test_at_max_boundary_passes(self):
        max_val = DEFAULT_SETUP_ENGINE_PARAMETERS.early_zone_distance_to_pivot_pct_max
        result = build_explain_payload(_default_input(distance_to_pivot_pct=max_val))
        assert "in_early_zone" in result.explain.to_payload()["passed_checks"]

    def test_below_min_fails(self):
        result = build_explain_payload(_default_input(distance_to_pivot_pct=-10.0))
        assert "outside_early_zone" in result.explain.to_payload()["failed_checks"]

    def test_above_max_fails(self):
        result = build_explain_payload(_default_input(distance_to_pivot_pct=20.0))
        assert "outside_early_zone" in result.explain.to_payload()["failed_checks"]

    def test_none_fails_non_permissive(self):
        """Gate 4 is non-permissive: None distance -> outside_early_zone."""
        result = build_explain_payload(_default_input(distance_to_pivot_pct=None))
        assert "outside_early_zone" in result.explain.to_payload()["failed_checks"]


# ── Gate 5: ATR14 cap (permissive) ───────────────


class TestGate5Atr14:
    def test_within_limit_passes(self):
        result = build_explain_payload(_default_input(atr14_pct=3.0))
        assert "atr14_within_limit" in result.explain.to_payload()["passed_checks"]

    def test_at_limit_passes(self):
        limit = DEFAULT_SETUP_ENGINE_PARAMETERS.atr14_pct_max_for_ready
        result = build_explain_payload(_default_input(atr14_pct=limit))
        assert "atr14_within_limit" in result.explain.to_payload()["passed_checks"]

    def test_exceeds_limit_fails(self):
        result = build_explain_payload(_default_input(atr14_pct=20.0))
        assert "atr14_pct_exceeds_limit" in result.explain.to_payload()["failed_checks"]

    def test_none_passes_permissive(self):
        """Gate 5 is permissive: None ATR -> atr14_within_limit."""
        result = build_explain_payload(_default_input(atr14_pct=None))
        assert "atr14_within_limit" in result.explain.to_payload()["passed_checks"]


# ── Gate 6: Volume floor (permissive) ────────────


class TestGate6Volume:
    def test_sufficient_passes(self):
        result = build_explain_payload(_default_input(volume_vs_50d=1.5))
        assert "volume_sufficient" in result.explain.to_payload()["passed_checks"]

    def test_at_minimum_passes(self):
        minimum = DEFAULT_SETUP_ENGINE_PARAMETERS.volume_vs_50d_min_for_ready
        result = build_explain_payload(_default_input(volume_vs_50d=minimum))
        assert "volume_sufficient" in result.explain.to_payload()["passed_checks"]

    def test_below_minimum_fails(self):
        result = build_explain_payload(_default_input(volume_vs_50d=0.3))
        assert "volume_below_minimum" in result.explain.to_payload()["failed_checks"]

    def test_none_passes_permissive(self):
        """Gate 6 is permissive: None volume -> volume_sufficient."""
        result = build_explain_payload(_default_input(volume_vs_50d=None))
        assert "volume_sufficient" in result.explain.to_payload()["passed_checks"]


# ── Gate 7: RS leadership (permissive) ───────────


class TestGate7RsLeadership:
    def test_positive_rs_vs_spy_passes(self):
        result = build_explain_payload(_default_input(
            rs_vs_spy_65d=5.0, rs_line_new_high=False,
        ))
        assert "rs_leadership_ok" in result.explain.to_payload()["passed_checks"]

    def test_rs_line_new_high_alone_passes(self):
        result = build_explain_payload(_default_input(
            rs_vs_spy_65d=None, rs_line_new_high=True,
        ))
        assert "rs_leadership_ok" in result.explain.to_payload()["passed_checks"]

    def test_negative_rs_no_new_high_fails(self):
        result = build_explain_payload(_default_input(
            rs_vs_spy_65d=-2.0, rs_line_new_high=False,
        ))
        assert "rs_leadership_insufficient" in result.explain.to_payload()["failed_checks"]

    def test_zero_rs_no_new_high_fails(self):
        result = build_explain_payload(_default_input(
            rs_vs_spy_65d=0.0, rs_line_new_high=False,
        ))
        assert "rs_leadership_insufficient" in result.explain.to_payload()["failed_checks"]

    def test_both_none_passes_permissive(self):
        """Gate 7 is permissive: both None -> rs_leadership_ok."""
        result = build_explain_payload(_default_input(
            rs_vs_spy_65d=None, rs_line_new_high=False,
        ))
        assert "rs_leadership_ok" in result.explain.to_payload()["passed_checks"]

    def test_negative_rs_but_new_high_passes(self):
        result = build_explain_payload(_default_input(
            rs_vs_spy_65d=-2.0, rs_line_new_high=True,
        ))
        assert "rs_leadership_ok" in result.explain.to_payload()["passed_checks"]


# ── Gate 8: stage (semi-permissive) ──────────────


class TestGate8Stage:
    def test_stage_2_passes(self):
        result = build_explain_payload(_default_input(stage=2))
        assert "stage_ok" in result.explain.to_payload()["passed_checks"]

    def test_stage_1_passes(self):
        result = build_explain_payload(_default_input(stage=1))
        assert "stage_ok" in result.explain.to_payload()["passed_checks"]

    def test_stage_3_fails(self):
        result = build_explain_payload(_default_input(stage=3))
        assert "stage_not_ok" in result.explain.to_payload()["failed_checks"]

    def test_stage_4_fails(self):
        result = build_explain_payload(_default_input(stage=4))
        assert "stage_not_ok" in result.explain.to_payload()["failed_checks"]

    def test_stage_none_passes_semi_permissive(self):
        """Gate 8 is semi-permissive: None -> stage_ok."""
        result = build_explain_payload(_default_input(stage=None))
        assert "stage_ok" in result.explain.to_payload()["passed_checks"]


# ── Gate 9: MA alignment (permissive) ────────────


class TestGate9MaAlignment:
    def test_above_threshold_passes(self):
        result = build_explain_payload(_default_input(ma_alignment_score=80.0))
        assert "ma_alignment_ok" in result.explain.to_payload()["passed_checks"]

    def test_at_threshold_passes(self):
        threshold = DEFAULT_SETUP_ENGINE_PARAMETERS.context_ma_alignment_min_pct
        result = build_explain_payload(_default_input(ma_alignment_score=threshold))
        assert "ma_alignment_ok" in result.explain.to_payload()["passed_checks"]

    def test_below_threshold_fails(self):
        result = build_explain_payload(_default_input(ma_alignment_score=20.0))
        assert "ma_alignment_insufficient" in result.explain.to_payload()["failed_checks"]

    def test_none_passes_permissive(self):
        """Gate 9 is permissive: None -> ma_alignment_ok."""
        result = build_explain_payload(_default_input(ma_alignment_score=None))
        assert "ma_alignment_ok" in result.explain.to_payload()["passed_checks"]


# ── Gate 10: RS rating (permissive) ──────────────


class TestGate10RsRating:
    def test_above_threshold_passes(self):
        result = build_explain_payload(_default_input(rs_rating=70.0))
        assert "rs_rating_ok" in result.explain.to_payload()["passed_checks"]

    def test_at_threshold_passes(self):
        threshold = DEFAULT_SETUP_ENGINE_PARAMETERS.context_rs_rating_min
        result = build_explain_payload(_default_input(rs_rating=threshold))
        assert "rs_rating_ok" in result.explain.to_payload()["passed_checks"]

    def test_below_threshold_fails(self):
        result = build_explain_payload(_default_input(rs_rating=20.0))
        assert "rs_rating_insufficient" in result.explain.to_payload()["failed_checks"]

    def test_none_passes_permissive(self):
        """Gate 10 is permissive: None -> rs_rating_ok."""
        result = build_explain_payload(_default_input(rs_rating=None))
        assert "rs_rating_ok" in result.explain.to_payload()["passed_checks"]


# ── Pre-existing check merge ─────────────────────


class TestPreExistingCheckMerge:
    def test_detector_checks_prepended(self):
        result = build_explain_payload(_default_input(
            pre_existing_passed_checks=("detector_a_ok", "detector_b_ok"),
        ))
        passed = result.explain.to_payload()["passed_checks"]
        # Pre-existing checks appear before gate checks
        assert passed[0] == "detector_a_ok"
        assert passed[1] == "detector_b_ok"

    def test_key_levels_forwarded(self):
        levels = KeyLevels(levels={"pivot_price": 101.25, "stop_loss": 95.0})
        result = build_explain_payload(_default_input(key_levels=levels))
        payload = result.explain.to_payload()
        assert payload["key_levels"]["pivot_price"] == 101.25
        assert payload["key_levels"]["stop_loss"] == 95.0

    def test_invalidation_flags_preserved(self):
        result = build_explain_payload(_default_input(
            pre_existing_invalidation_flags=("volume_dry_up", "gap_risk"),
        ))
        payload = result.explain.to_payload()
        assert "volume_dry_up" in payload["invalidation_flags"]
        assert "gap_risk" in payload["invalidation_flags"]

    def test_empty_key_levels_produce_empty_dict(self):
        result = build_explain_payload(_default_input())
        assert result.explain.to_payload()["key_levels"] == {}


# ── Score trace opt-in ───────────────────────────


def _make_features() -> BreakoutReadinessFeatures:
    return BreakoutReadinessFeatures(
        distance_to_pivot_pct=-1.13,
        atr14_pct=2.43,
        atr14_pct_trend=-0.01,
        bb_width_pct=5.99,
        bb_width_pctile_252=28.0,
        volume_vs_50d=1.21,
        rs=2.43,
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
        latest_volume=1500000.0,
        volume_sma_50=1240000.0,
        pivot_price=221.0,
        bb_pctile_window=252,
        latest_benchmark_close=90.0,
        rs_current=2.43,
        rs_252_max=2.5,
        rs_65d_ago=2.1,
    )


class TestScoreTraceOptIn:
    def test_trace_included_when_opted_in(self):
        result = build_explain_payload(_default_input(
            include_score_trace=True,
            readiness_features=_make_features(),
            readiness_trace_inputs=_make_trace_inputs(),
        ))
        payload = result.explain.to_payload()
        assert "score_trace" in payload

    def test_trace_excluded_when_opted_out(self):
        result = build_explain_payload(_default_input(
            include_score_trace=False,
            readiness_features=_make_features(),
            readiness_trace_inputs=_make_trace_inputs(),
        ))
        payload = result.explain.to_payload()
        assert "score_trace" not in payload

    def test_trace_excluded_when_features_none(self):
        result = build_explain_payload(_default_input(
            include_score_trace=True,
            readiness_features=None,
            readiness_trace_inputs=_make_trace_inputs(),
        ))
        payload = result.explain.to_payload()
        assert "score_trace" not in payload

    def test_trace_excluded_when_trace_inputs_none(self):
        result = build_explain_payload(_default_input(
            include_score_trace=True,
            readiness_features=_make_features(),
            readiness_trace_inputs=None,
        ))
        payload = result.explain.to_payload()
        assert "score_trace" not in payload

    def test_trace_has_setup_score_field(self):
        result = build_explain_payload(_default_input(
            include_score_trace=True,
            readiness_features=_make_features(),
            readiness_trace_inputs=_make_trace_inputs(),
        ))
        trace = result.explain.to_payload()["score_trace"]
        assert "setup_score" in trace
        assert "formula" in trace["setup_score"]


# ── derived_ready semantics ──────────────────────


class TestDerivedReadySemantics:
    def test_all_gates_pass_is_ready(self):
        result = build_explain_payload(_default_input())
        assert result.derived_ready is True

    def test_single_gate_failure_blocks_ready(self):
        result = build_explain_payload(_default_input(setup_score=10.0))
        assert result.derived_ready is False

    def test_pre_existing_failure_blocks_ready(self):
        """Pre-existing failed checks count toward derived_ready."""
        result = build_explain_payload(_default_input(
            pre_existing_failed_checks=("detector_x_fail",),
        ))
        assert result.derived_ready is False

    def test_multiple_failures_all_recorded(self):
        result = build_explain_payload(_default_input(
            setup_score=10.0,
            quality_score=20.0,
            readiness_score=30.0,
            distance_to_pivot_pct=None,
        ))
        failed = result.explain.to_payload()["failed_checks"]
        assert "setup_score_below_threshold" in failed
        assert "quality_below_threshold" in failed
        assert "readiness_below_threshold" in failed
        assert "outside_early_zone" in failed
        assert result.derived_ready is False

    def test_custom_parameters_affect_gates(self):
        """Custom parameters should influence gate thresholds."""
        strict_params = SetupEngineParameters(setup_score_min_pct=95.0)
        result = build_explain_payload(_default_input(
            setup_score=80.0,
            parameters=strict_params,
        ))
        assert "setup_score_below_threshold" in result.explain.to_payload()["failed_checks"]
        assert result.derived_ready is False

    def test_custom_readiness_threshold_affects_gate3(self):
        result = build_explain_payload(_default_input(
            readiness_score=75.0,
            readiness_threshold_pct=80.0,
        ))
        assert "readiness_below_threshold" in result.explain.to_payload()["failed_checks"]
        assert result.derived_ready is False
