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


def test_setup_score_min_pct_bounds():
    spec = next(
        s for s in SETUP_ENGINE_PARAMETER_SPECS if s.name == "setup_score_min_pct"
    )
    assert spec.default_value == 65.0
    assert spec.min_value == 30.0
    assert spec.max_value == 95.0
    assert spec.unit == "pct"
    assert spec.profile == "baseline"

    # Out of bounds triggers validation error.
    params = SetupEngineParameters(setup_score_min_pct=10.0)
    errors = validate_setup_engine_parameters(params)
    assert any("setup_score_min_pct" in err for err in errors)


def test_context_ma_alignment_min_pct_bounds():
    spec = next(
        s for s in SETUP_ENGINE_PARAMETER_SPECS if s.name == "context_ma_alignment_min_pct"
    )
    assert spec.default_value == 60.0
    assert spec.min_value == 0.0
    assert spec.max_value == 100.0
    assert spec.unit == "pct"
    assert spec.profile == "baseline"


def test_context_rs_rating_min_bounds():
    spec = next(
        s for s in SETUP_ENGINE_PARAMETER_SPECS if s.name == "context_rs_rating_min"
    )
    assert spec.default_value == 50.0
    assert spec.min_value == 0.0
    assert spec.max_value == 100.0
    assert spec.unit == "pct"
    assert spec.profile == "baseline"


# ── Operational invalidation threshold specs (SE-E7) ──


def test_too_extended_pivot_distance_pct_spec():
    spec = next(
        s for s in SETUP_ENGINE_PARAMETER_SPECS if s.name == "too_extended_pivot_distance_pct"
    )
    assert spec.default_value == 10.0
    assert spec.min_value == 2.0
    assert spec.max_value == 50.0
    assert spec.unit == "pct"


def test_breaks_50d_support_cushion_pct_spec():
    spec = next(
        s for s in SETUP_ENGINE_PARAMETER_SPECS if s.name == "breaks_50d_support_cushion_pct"
    )
    assert spec.default_value == 0.0
    assert spec.min_value == 0.0
    assert spec.max_value == 10.0
    assert spec.unit == "pct"


def test_low_liquidity_adtv_min_usd_spec():
    spec = next(
        s for s in SETUP_ENGINE_PARAMETER_SPECS if s.name == "low_liquidity_adtv_min_usd"
    )
    assert spec.default_value == 1_000_000.0
    assert spec.min_value == 0.0
    assert spec.max_value == 100_000_000.0
    assert spec.unit == "usd"


def test_earnings_soon_window_days_spec():
    spec = next(
        s for s in SETUP_ENGINE_PARAMETER_SPECS if s.name == "earnings_soon_window_days"
    )
    assert spec.default_value == 21.0
    assert spec.min_value == 1.0
    assert spec.max_value == 90.0
    assert spec.unit == "days"
