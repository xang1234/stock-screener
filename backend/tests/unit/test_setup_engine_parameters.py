"""Tests for Setup Engine parameter governance."""

import pytest

from app.analysis.patterns.config import (
    DEFAULT_SETUP_ENGINE_PARAMETERS,
    SETUP_ENGINE_PARAMETER_SPECS,
    SetupEngineParameters,
    build_setup_engine_parameters,
    validate_setup_engine_parameters,
)


def test_specs_match_dataclass_fields():
    spec_names = {spec.name for spec in SETUP_ENGINE_PARAMETER_SPECS}
    param_names = set(DEFAULT_SETUP_ENGINE_PARAMETERS.__dataclass_fields__.keys())
    assert spec_names == param_names


def test_default_parameters_are_valid():
    errors = validate_setup_engine_parameters(DEFAULT_SETUP_ENGINE_PARAMETERS)
    assert errors == []


def test_build_parameters_supports_numeric_override():
    params = build_setup_engine_parameters({"readiness_score_ready_min_pct": 72})
    assert params.readiness_score_ready_min_pct == 72.0


def test_build_parameters_rejects_unknown_key():
    with pytest.raises(KeyError):
        build_setup_engine_parameters({"unknown_threshold": 1.0})


def test_validation_catches_contradictory_strict_relaxed():
    params = SetupEngineParameters(
        three_weeks_tight_max_contraction_pct_strict=2.0,
        three_weeks_tight_max_contraction_pct_relaxed=1.0,
    )
    errors = validate_setup_engine_parameters(params)
    assert any("strict must be <= relaxed" in err for err in errors)
