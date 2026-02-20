"""Tests for analysis.patterns package skeleton boundaries."""

import inspect

import app.analysis.patterns.aggregator as aggregator_module
import app.analysis.patterns.config as config_module
import app.analysis.patterns.cup_handle as cup_entry_module
import app.analysis.patterns.detectors.base as detector_base_module
import app.analysis.patterns.detectors.cup_with_handle as cup_module
import app.analysis.patterns.detectors.double_bottom as db_module
import app.analysis.patterns.detectors.vcp as vcp_module
import app.analysis.patterns.first_pullback as first_pullback_module
import app.analysis.patterns.high_tight_flag as htf_module
import app.analysis.patterns.nr7_inside_day as nr7_module
import app.analysis.patterns.policy as policy_module
import app.analysis.patterns.normalization as normalization_module
import app.analysis.patterns.report as report_module
import app.analysis.patterns.technicals as technicals_module
import app.analysis.patterns.three_weeks_tight as three_weeks_tight_module
import app.analysis.patterns.vcp_wrapper as vcp_wrapper_module
import app.analysis.patterns as patterns_public_api
from app.analysis.patterns.aggregator import SetupEngineAggregator
from app.analysis.patterns.config import DEFAULT_SETUP_ENGINE_PARAMETERS
from app.analysis.patterns.detectors.base import PatternDetectorInput
from app.analysis.patterns.policy import evaluate_setup_engine_data_policy


def test_aggregator_runs_with_stub_detectors_and_no_throw():
    aggregator = SetupEngineAggregator()
    detector_input = PatternDetectorInput(
        symbol="AAPL",
        timeframe="daily",
        daily_bars=260,
        weekly_bars=60,
        features={},
    )

    result = aggregator.aggregate(
        detector_input,
        parameters=DEFAULT_SETUP_ENGINE_PARAMETERS,
    )

    assert result.pattern_primary is None
    assert "no_primary_pattern" in result.failed_checks


def test_aggregator_applies_policy_insufficient_gate():
    aggregator = SetupEngineAggregator()
    detector_input = PatternDetectorInput(
        symbol="AAPL",
        timeframe="daily",
        daily_bars=120,
        weekly_bars=30,
        features={},
    )
    policy = evaluate_setup_engine_data_policy(
        daily_bars=120,
        weekly_bars=30,
        benchmark_bars=100,
        current_week_sessions=2,
    )

    result = aggregator.aggregate(
        detector_input,
        parameters=DEFAULT_SETUP_ENGINE_PARAMETERS,
        policy_result=policy,
    )

    assert result.pattern_primary is None
    assert "insufficient_data" in result.failed_checks
    assert "data_policy:insufficient" in result.invalidation_flags
    assert result.candidates == ()


def test_analysis_layer_modules_do_not_import_scanner_layer():
    for module in (
        aggregator_module,
        config_module,
        policy_module,
        normalization_module,
        report_module,
        technicals_module,
        cup_entry_module,
        three_weeks_tight_module,
        htf_module,
        nr7_module,
        first_pullback_module,
        vcp_wrapper_module,
        detector_base_module,
        cup_module,
        db_module,
        vcp_module,
    ):
        source = inspect.getsource(module)
        assert "app.scanners" not in source


def test_stub_modules_reference_followup_bead_todos():
    expected_todos = {
        "SE-C1": vcp_wrapper_module,
        "SE-C2": three_weeks_tight_module,
        "SE-C5": nr7_module,
    }
    for todo, module in expected_todos.items():
        source = inspect.getsource(module)
        assert f"TODO({todo})" in source


def test_public_api_exports_stable_symbols_only():
    exported = set(patterns_public_api.__all__)
    # Stable surfaces: candidate schema, utilities, and detector entrypoints.
    assert "PatternCandidate" in exported
    assert "coerce_pattern_candidate" in exported
    assert "resample_ohlcv" in exported
    assert "normalize_ohlcv_frame" in exported
    assert "VCPWrapperDetector" in exported
    assert "ThreeWeeksTightDetector" in exported
    assert "SetupEngineReport" in exported
    # Internal governance/policy modules are intentionally not wildcard-exported.
    assert "SetupEngineParameters" not in exported
    assert "evaluate_setup_engine_data_policy" not in exported
