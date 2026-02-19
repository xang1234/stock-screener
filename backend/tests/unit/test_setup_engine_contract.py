"""Tests for canonical setup_engine contract and payload assembly."""

import pytest

from app.analysis.patterns.models import (
    SETUP_ENGINE_FIELD_SPECS,
    SETUP_ENGINE_NUMERIC_UNITS,
    SETUP_ENGINE_REQUIRED_KEYS,
    validate_setup_engine_payload,
)
from app.analysis.patterns.config import SetupEngineParameters
from app.analysis.patterns.policy import evaluate_setup_engine_data_policy
from app.scanners.setup_engine_scanner import (
    attach_setup_engine,
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
    "atr14_pct",
    "bb_width_pctile_252",
    "volume_vs_50d",
    "rs_line_new_high",
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
        readiness_score=82.0,
        failed_checks=[],
    )

    assert payload["setup_ready"] is True
    assert payload["schema_version"] == "v1"
    assert payload["timeframe"] == "daily"
    assert payload["candidates"] == []
    assert payload["explain"]["passed_checks"] == []
    assert payload["explain"]["failed_checks"] == []


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
    assert "insufficient_data" in payload["explain"]["failed_checks"]
