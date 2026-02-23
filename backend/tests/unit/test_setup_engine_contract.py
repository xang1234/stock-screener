"""Tests for canonical setup_engine contract and payload assembly."""

import pytest

from app.analysis.patterns.models import (
    PatternCandidateModel,
    SETUP_ENGINE_FIELD_SPECS,
    SETUP_ENGINE_NUMERIC_UNITS,
    SETUP_ENGINE_REQUIRED_KEYS,
    validate_setup_engine_payload,
)
from app.analysis.patterns.config import (
    SETUP_SCORE_WEIGHTS,
    SetupEngineParameters,
)
from app.analysis.patterns.policy import evaluate_setup_engine_data_policy
from app.analysis.patterns.report import ExplainPayload, SetupEngineReport
from app.scanners.setup_engine_scanner import (
    attach_setup_engine,
    build_setup_engine_payload_from_report,
    build_setup_engine_payload,
)


MANDATORY_V1_KEYS = {
    "setup_score",
    "quality_score",
    "readiness_score",
    "setup_ready",
    "pattern_primary",
    "pattern_confidence",
    "pivot_price",
    "pivot_type",
    "pivot_date",
    "distance_to_pivot_pct",
    "in_early_zone",
    "extended_from_pivot",
    "base_length_weeks",
    "base_depth_pct",
    "support_tests_count",
    "tight_closes_count",
    "atr14_pct",
    "atr14_pct_trend",
    "bb_width_pct",
    "bb_width_pctile_252",
    "bb_squeeze",
    "volume_vs_50d",
    "up_down_volume_ratio_10d",
    "quiet_days_10d",
    "rs",
    "rs_line_new_high",
    "rs_vs_spy_65d",
    "rs_vs_spy_trend_20d",
    "stage",
    "ma_alignment_score",
    "rs_rating",
    "candidates",
    "explain",
}


def test_required_keys_cover_mandatory_v1_fields():
    assert MANDATORY_V1_KEYS.issubset(set(SETUP_ENGINE_REQUIRED_KEYS))


def test_numeric_field_specs_use_explicit_units():
    numeric_specs = [
        spec for spec in SETUP_ENGINE_FIELD_SPECS
        if spec.type_name == "float"
    ]
    assert numeric_specs, "Expected at least one numeric setup_engine field"

    for spec in numeric_specs:
        assert spec.unit in SETUP_ENGINE_NUMERIC_UNITS, (
            f"{spec.name} must declare a supported unit"
        )


def test_build_payload_defaults_and_bool_semantics():
    payload = build_setup_engine_payload(
        quality_score=75.0,
        readiness_score=82.0,
        distance_to_pivot_pct=1.0,
        failed_checks=[],
    )

    assert payload["setup_ready"] is True
    assert payload["schema_version"] == "v1"
    assert payload["timeframe"] == "daily"
    assert payload["candidates"] == []
    # Gates produce passed_checks entries.
    assert "setup_score_ok" in payload["explain"]["passed_checks"]
    assert "quality_floor_ok" in payload["explain"]["passed_checks"]
    assert "readiness_floor_ok" in payload["explain"]["passed_checks"]
    assert "in_early_zone" in payload["explain"]["passed_checks"]
    assert payload["explain"]["failed_checks"] == []
    assert payload["atr14_pct_trend"] is None
    assert payload["bb_width_pct"] is None
    assert payload["rs"] is None
    assert payload["rs_vs_spy_65d"] is None
    assert payload["rs_vs_spy_trend_20d"] is None
    assert payload["stage"] is None
    assert payload["ma_alignment_score"] is None
    assert payload["rs_rating"] is None


def test_build_payload_failed_checks_forces_not_ready():
    payload = build_setup_engine_payload(
        readiness_score=95.0,
        failed_checks=["trend_break"],
    )

    assert payload["setup_ready"] is False


def test_build_payload_rejects_non_snake_case_candidate_keys():
    with pytest.raises(ValueError, match="snake_case"):
        build_setup_engine_payload(
            candidates=[
                {
                    "pattern": "cup_handle",
                    "confidencePct": 88.0,
                }
            ]
        )


def test_attach_setup_engine_uses_top_level_key():
    payload = build_setup_engine_payload(setup_score=71.5)
    merged = attach_setup_engine({"legacy_metric": 1}, payload)

    assert merged["legacy_metric"] == 1
    assert "setup_engine" in merged
    assert merged["setup_engine"]["setup_score"] == 71.5


def test_validator_flags_missing_required_key():
    payload = build_setup_engine_payload(setup_score=75.0)
    payload.pop("setup_score")

    errors = validate_setup_engine_payload(payload)
    assert any("setup_score" in err for err in errors)


def test_readiness_threshold_can_come_from_parameters():
    params = SetupEngineParameters(readiness_score_ready_min_pct=80.0)
    payload = build_setup_engine_payload(
        readiness_score=75.0,
        failed_checks=[],
        parameters=params,
    )
    assert payload["setup_ready"] is False


def test_policy_insufficient_nulls_primary_fields():
    policy = evaluate_setup_engine_data_policy(
        daily_bars=100,
        weekly_bars=30,
        benchmark_bars=20,
        current_week_sessions=1,
    )
    payload = build_setup_engine_payload(
        setup_score=88.0,
        pattern_primary="vcp",
        candidates=[{"pattern": "vcp", "confidence_pct": 80.0}],
        data_policy_result=policy,
    )

    assert payload["setup_score"] is None
    assert payload["pattern_primary"] is None
    assert payload["candidates"] == []
    assert payload["setup_ready"] is False
    assert payload["atr14_pct_trend"] is None
    assert payload["bb_width_pct"] is None
    assert payload["rs"] is None
    assert payload["rs_vs_spy_65d"] is None
    assert payload["rs_vs_spy_trend_20d"] is None
    assert payload["stage"] is None
    assert payload["ma_alignment_score"] is None
    assert payload["rs_rating"] is None
    assert "insufficient_data" in payload["explain"]["failed_checks"]


def test_candidate_confidence_ratio_is_normalized_to_pct():
    payload = build_setup_engine_payload(
        candidates=[
            {
                "pattern": "vcp",
                "timeframe": "daily",
                "confidence": 0.81,
            }
        ]
    )
    candidate = payload["candidates"][0]
    assert candidate["confidence"] == pytest.approx(0.81)
    assert candidate["confidence_pct"] == pytest.approx(81.0)


def test_candidate_model_input_is_supported():
    candidate_model = PatternCandidateModel(
        pattern="three_weeks_tight",
        timeframe="weekly",
        confidence=0.63,
        metrics={"weeks_tight": 3},
        checks={"tight_band_ok": True},
        notes=("strict_mode",),
    )
    payload = build_setup_engine_payload(candidates=[candidate_model])
    candidate = payload["candidates"][0]
    assert candidate["pattern"] == "three_weeks_tight"
    assert candidate["checks"]["tight_band_ok"] is True
    assert candidate["notes"] == ["strict_mode"]


def test_pattern_confidence_falls_back_from_primary_candidate():
    payload = build_setup_engine_payload(
        pattern_primary="vcp",
        candidates=[
            {
                "pattern": "vcp",
                "timeframe": "daily",
                "confidence": 0.73,
            }
        ],
    )
    assert payload["pattern_confidence"] == pytest.approx(73.0)


def test_build_payload_from_typed_report():
    report = SetupEngineReport(
        timeframe="daily",
        setup_ready=False,
        setup_score=65.0,
        quality_score=60.0,
        readiness_score=55.0,
        pattern_primary="vcp",
        pattern_confidence_pct=58.0,
        explain=ExplainPayload(),
    )
    payload = build_setup_engine_payload_from_report(report)
    assert payload["schema_version"] == "v1"
    assert payload["pattern_primary"] == "vcp"


def test_build_payload_accepts_central_readiness_features_mapping():
    payload = build_setup_engine_payload(
        readiness_features={
            "distance_to_pivot_pct": -1.2,
            "atr14_pct": 3.4,
            "atr14_pct_trend": -0.02,
            "bb_width_pct": 7.8,
            "bb_width_pctile_252": 26.0,
            "volume_vs_50d": 1.15,
            "bb_squeeze": False,
            "quiet_days_10d": 2,
            "up_down_volume_ratio_10d": 1.4,
            "rs": 1.32,
            "rs_line_new_high": True,
            "rs_vs_spy_65d": 8.5,
            "rs_vs_spy_trend_20d": 0.004,
        }
    )

    assert payload["distance_to_pivot_pct"] == pytest.approx(-1.2)
    assert payload["atr14_pct"] == pytest.approx(3.4)
    assert payload["atr14_pct_trend"] == pytest.approx(-0.02)
    assert payload["bb_width_pct"] == pytest.approx(7.8)
    assert payload["bb_width_pctile_252"] == pytest.approx(26.0)
    assert payload["volume_vs_50d"] == pytest.approx(1.15)
    assert payload["bb_squeeze"] is False
    assert payload["quiet_days_10d"] == 2
    assert payload["up_down_volume_ratio_10d"] == pytest.approx(1.4)
    assert payload["rs"] == pytest.approx(1.32)
    assert payload["rs_line_new_high"] is True
    assert payload["rs_vs_spy_65d"] == pytest.approx(8.5)
    assert payload["rs_vs_spy_trend_20d"] == pytest.approx(0.004)


# ── SE-D4: Setup Score Synthesis and Gating Tests ─────────────────


def _all_gates_pass_kwargs():
    """Return kwargs that satisfy all 10 gates for setup_ready=True."""
    return dict(
        quality_score=75.0,
        readiness_score=82.0,
        distance_to_pivot_pct=1.0,
        atr14_pct=3.5,
        volume_vs_50d=0.6,
        rs_vs_spy_65d=5.0,
        rs_line_new_high=False,
        stage=2,
        ma_alignment_score=80.0,
        rs_rating=60.0,
        candidates=[{
            "pattern": "vcp",
            "timeframe": "daily",
            "confidence": 0.80,
        }],
        pattern_primary="vcp",
        failed_checks=[],
    )


class TestSetupScoreSynthesis:
    """Tests for auto-computing setup_score from quality and readiness."""

    def test_setup_score_auto_computed_from_quality_and_readiness(self):
        payload = build_setup_engine_payload(
            quality_score=80.0,
            readiness_score=70.0,
            distance_to_pivot_pct=1.0,
        )
        wq, wr = SETUP_SCORE_WEIGHTS
        expected = wq * 80.0 + wr * 70.0
        assert payload["setup_score"] == pytest.approx(expected)

    def test_setup_score_explicit_overrides_auto_computation(self):
        payload = build_setup_engine_payload(
            setup_score=42.0,
            quality_score=80.0,
            readiness_score=70.0,
        )
        assert payload["setup_score"] == pytest.approx(42.0)

    def test_setup_score_none_when_quality_missing(self):
        payload = build_setup_engine_payload(readiness_score=80.0)
        assert payload["setup_score"] is None

    def test_setup_score_none_when_readiness_missing(self):
        payload = build_setup_engine_payload(quality_score=80.0)
        assert payload["setup_score"] is None

    def test_setup_score_extracted_from_primary_candidate(self):
        """When quality/readiness not passed but candidates+pattern_primary are, extract from candidate."""
        payload = build_setup_engine_payload(
            pattern_primary="vcp",
            distance_to_pivot_pct=1.0,
            candidates=[{
                "pattern": "vcp",
                "timeframe": "daily",
                "quality_score": 72.0,
                "readiness_score": 78.0,
                "confidence": 0.75,
                "checks": {"stage_ok": True},
            }],
        )
        wq, wr = SETUP_SCORE_WEIGHTS
        expected = wq * 72.0 + wr * 78.0
        assert payload["setup_score"] == pytest.approx(expected)
        assert payload["quality_score"] == pytest.approx(72.0)
        assert payload["readiness_score"] == pytest.approx(78.0)


class TestSetupReadyGates:
    """Tests for the 9-gate setup_ready logic."""

    def test_setup_ready_requires_setup_score_above_threshold(self):
        params = SetupEngineParameters(setup_score_min_pct=80.0)
        payload = build_setup_engine_payload(
            quality_score=70.0,
            readiness_score=70.0,
            distance_to_pivot_pct=1.0,
            parameters=params,
        )
        # setup_score = 0.60*70 + 0.40*70 = 70.0 < 80.0
        assert payload["setup_ready"] is False
        assert "setup_score_below_threshold" in payload["explain"]["failed_checks"]

    def test_setup_ready_requires_quality_floor(self):
        params = SetupEngineParameters(
            quality_score_min_pct=80.0,
            readiness_score_ready_min_pct=85.0,
        )
        payload = build_setup_engine_payload(
            quality_score=75.0,
            readiness_score=85.0,
            distance_to_pivot_pct=1.0,
            parameters=params,
        )
        assert payload["setup_ready"] is False
        assert "quality_below_threshold" in payload["explain"]["failed_checks"]

    def test_setup_ready_requires_readiness_floor(self):
        params = SetupEngineParameters(readiness_score_ready_min_pct=85.0)
        payload = build_setup_engine_payload(
            quality_score=80.0,
            readiness_score=75.0,
            distance_to_pivot_pct=1.0,
            parameters=params,
        )
        assert payload["setup_ready"] is False
        assert "readiness_below_threshold" in payload["explain"]["failed_checks"]

    def test_setup_ready_requires_early_zone(self):
        payload = build_setup_engine_payload(
            quality_score=80.0,
            readiness_score=85.0,
            distance_to_pivot_pct=10.0,  # way above early_zone_max=3.0
        )
        assert payload["setup_ready"] is False
        assert "outside_early_zone" in payload["explain"]["failed_checks"]

    def test_setup_ready_rs_ok_either_condition(self):
        # rs_vs_spy_65d > 0 alone should pass
        payload1 = build_setup_engine_payload(
            **{**_all_gates_pass_kwargs(), "rs_vs_spy_65d": 2.0, "rs_line_new_high": False}
        )
        assert "rs_leadership_ok" in payload1["explain"]["passed_checks"]

        # rs_line_new_high alone should pass
        payload2 = build_setup_engine_payload(
            **{**_all_gates_pass_kwargs(), "rs_vs_spy_65d": -1.0, "rs_line_new_high": True}
        )
        assert "rs_leadership_ok" in payload2["explain"]["passed_checks"]

    def test_setup_ready_rs_ok_permissive_when_unavailable(self):
        kwargs = _all_gates_pass_kwargs()
        kwargs["rs_vs_spy_65d"] = None
        kwargs["rs_line_new_high"] = None
        payload = build_setup_engine_payload(**kwargs)
        assert "rs_leadership_ok" in payload["explain"]["passed_checks"]
        assert "rs_leadership_insufficient" not in payload["explain"]["failed_checks"]

    def test_setup_ready_rs_fails_when_negative_and_no_new_high(self):
        kwargs = _all_gates_pass_kwargs()
        kwargs["rs_vs_spy_65d"] = -3.0
        kwargs["rs_line_new_high"] = False
        payload = build_setup_engine_payload(**kwargs)
        assert payload["setup_ready"] is False
        assert "rs_leadership_insufficient" in payload["explain"]["failed_checks"]

    def test_setup_ready_atr14_gate(self):
        kwargs = _all_gates_pass_kwargs()
        kwargs["atr14_pct"] = 15.0  # exceeds default max of 8.0
        payload = build_setup_engine_payload(**kwargs)
        assert payload["setup_ready"] is False
        assert "atr14_pct_exceeds_limit" in payload["explain"]["failed_checks"]

    def test_setup_ready_volume_gate(self):
        kwargs = _all_gates_pass_kwargs()
        kwargs["volume_vs_50d"] = 1.5  # above default dry-up cap of 0.8
        payload = build_setup_engine_payload(**kwargs)
        assert payload["setup_ready"] is False
        assert "volume_below_minimum" in payload["explain"]["failed_checks"]

    def test_setup_ready_stage_fails_when_topping(self):
        kwargs = _all_gates_pass_kwargs()
        kwargs["stage"] = 3  # Stage 3 = Topping
        payload = build_setup_engine_payload(**kwargs)
        assert payload["setup_ready"] is False
        assert "stage_not_ok" in payload["explain"]["failed_checks"]

    def test_setup_ready_all_gates_pass(self):
        payload = build_setup_engine_payload(**_all_gates_pass_kwargs())
        assert payload["setup_ready"] is True
        passed = payload["explain"]["passed_checks"]
        assert "setup_score_ok" in passed
        assert "quality_floor_ok" in passed
        assert "readiness_floor_ok" in passed
        assert "in_early_zone" in passed
        assert "atr14_within_limit" in passed
        assert "volume_sufficient" in passed
        assert "rs_leadership_ok" in passed
        assert "stage_ok" in passed
        assert "ma_alignment_ok" in passed
        assert "rs_rating_ok" in passed
        assert payload["explain"]["failed_checks"] == []


class TestGateExplainability:
    """Tests for gate failure/pass check entries."""

    def test_context_gate_failures_appear_in_failed_checks(self):
        payload = build_setup_engine_payload(
            quality_score=50.0,  # below default 60.0 floor
            readiness_score=60.0,  # below default 70.0 floor
            distance_to_pivot_pct=15.0,  # outside early zone
        )
        failed = payload["explain"]["failed_checks"]
        assert "quality_below_threshold" in failed
        assert "readiness_below_threshold" in failed
        assert "outside_early_zone" in failed

    def test_context_gate_passes_appear_in_passed_checks(self):
        payload = build_setup_engine_payload(**_all_gates_pass_kwargs())
        passed = payload["explain"]["passed_checks"]
        assert len(passed) >= 10  # All 10 gate checks should be in passed

    def test_gates_skipped_when_no_setup_score(self):
        """Insufficient data produces no gate entries."""
        payload = build_setup_engine_payload()  # No quality/readiness → no setup_score
        assert payload["setup_score"] is None
        assert payload["setup_ready"] is False
        # No gate check names should appear
        gate_names = {
            "setup_score_ok", "setup_score_below_threshold",
            "quality_floor_ok", "quality_below_threshold",
            "readiness_floor_ok", "readiness_below_threshold",
            "in_early_zone", "outside_early_zone",
            "atr14_within_limit", "atr14_pct_exceeds_limit",
            "volume_sufficient", "volume_below_minimum",
            "rs_leadership_ok", "rs_leadership_insufficient",
            "stage_ok", "stage_not_ok",
            "ma_alignment_ok", "ma_alignment_insufficient",
            "rs_rating_ok", "rs_rating_insufficient",
        }
        all_checks = set(payload["explain"]["passed_checks"]) | set(payload["explain"]["failed_checks"])
        assert all_checks.isdisjoint(gate_names)
