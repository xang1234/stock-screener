"""Tests for analysis.patterns package skeleton boundaries."""

import inspect

import app.analysis.patterns.aggregator as aggregator_module
import app.analysis.patterns.config as config_module
import app.analysis.patterns.detectors.base as detector_base_module
import app.analysis.patterns.detectors.cup_with_handle as cup_module
import app.analysis.patterns.detectors.double_bottom as db_module
import app.analysis.patterns.detectors.vcp as vcp_module
import app.analysis.patterns.policy as policy_module
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
        detector_base_module,
        cup_module,
        db_module,
        vcp_module,
    ):
        source = inspect.getsource(module)
        assert "app.scanners" not in source
