"""Tests for deterministic detector orchestration in the aggregator."""

import threading
from unittest.mock import patch

from app.analysis.patterns.aggregator import SetupEngineAggregator, _pick_primary_candidate
from app.analysis.patterns.config import DEFAULT_SETUP_ENGINE_PARAMETERS
from app.analysis.patterns.detectors.base import (
    PatternDetector,
    PatternDetectorInput,
    PatternDetectorResult,
)
from app.analysis.patterns.models import PatternCandidateModel


def _detector_input() -> PatternDetectorInput:
    return PatternDetectorInput(
        symbol="AAPL",
        timeframe="daily",
        daily_bars=260,
        weekly_bars=60,
        features={},
    )


def test_aggregator_execution_trace_preserves_detector_order():
    class _DetectorAlpha(PatternDetector):
        name = "detector_alpha"

        def detect(self, detector_input, parameters):
            del detector_input, parameters
            return PatternDetectorResult.detected(
                self.name,
                PatternCandidateModel(
                    pattern="vcp",
                    timeframe="daily",
                    source_detector=self.name,
                    quality_score=80.0,
                    readiness_score=78.0,
                    confidence=0.74,
                ),
                passed_checks=("alpha_pass",),
                warnings=("alpha_warning",),
            )

    class _DetectorBeta(PatternDetector):
        name = "detector_beta"

        def detect(self, detector_input, parameters):
            del detector_input, parameters
            return PatternDetectorResult.no_detection(
                self.name,
                failed_checks=("beta_miss",),
                warnings=("beta_warning",),
            )

    agg = SetupEngineAggregator(detectors=[_DetectorAlpha(), _DetectorBeta()])
    first = agg.aggregate(_detector_input(), parameters=DEFAULT_SETUP_ENGINE_PARAMETERS)
    second = agg.aggregate(_detector_input(), parameters=DEFAULT_SETUP_ENGINE_PARAMETERS)

    assert [c["source_detector"] for c in first.candidates] == [
        c["source_detector"] for c in second.candidates
    ]
    assert [trace.detector_name for trace in first.detector_traces] == [
        "detector_alpha",
        "detector_beta",
    ]
    assert [trace.execution_index for trace in first.detector_traces] == [0, 1]
    assert first.detector_traces[0].outcome == "detected"
    assert first.detector_traces[1].outcome == "not_detected"
    assert first.pattern_primary == "vcp"
    assert "detector_pipeline_executed" in first.passed_checks


def test_aggregator_runs_detectors_serially_by_default():
    execution_order: list[str] = []

    class _FirstDetector(PatternDetector):
        name = "first"

        def detect(self, detector_input, parameters):
            del detector_input, parameters
            execution_order.append(self.name)
            return PatternDetectorResult.no_detection(self.name)

    class _SecondDetector(PatternDetector):
        name = "second"

        def detect(self, detector_input, parameters):
            del detector_input, parameters
            execution_order.append(self.name)
            return PatternDetectorResult.no_detection(self.name)

    agg = SetupEngineAggregator(detectors=[_FirstDetector(), _SecondDetector()])

    with patch(
        "app.analysis.patterns.aggregator.ThreadPoolExecutor",
        side_effect=AssertionError("default detector execution must not spawn a pool"),
    ):
        result = agg.aggregate(_detector_input(), parameters=DEFAULT_SETUP_ENGINE_PARAMETERS)

    assert execution_order == ["first", "second"]
    assert [trace.detector_name for trace in result.detector_traces] == [
        "first",
        "second",
    ]


def test_aggregator_runs_detectors_concurrently_while_preserving_trace_order():
    barrier = threading.Barrier(2)
    overlapped: list[str] = []

    class _FirstDetector(PatternDetector):
        name = "first"

        def detect(self, detector_input, parameters):
            del detector_input, parameters
            barrier.wait(timeout=1.0)
            overlapped.append(self.name)
            return PatternDetectorResult.no_detection(self.name)

    class _SecondDetector(PatternDetector):
        name = "second"

        def detect(self, detector_input, parameters):
            del detector_input, parameters
            barrier.wait(timeout=1.0)
            overlapped.append(self.name)
            return PatternDetectorResult.no_detection(self.name)

    agg = SetupEngineAggregator(
        detectors=[_FirstDetector(), _SecondDetector()],
        detector_workers=2,
    )

    result = agg.aggregate(_detector_input(), parameters=DEFAULT_SETUP_ENGINE_PARAMETERS)

    assert sorted(overlapped) == ["first", "second"]
    assert [trace.detector_name for trace in result.detector_traces] == [
        "first",
        "second",
    ]
    assert [trace.execution_index for trace in result.detector_traces] == [0, 1]


def test_aggregator_trace_includes_detector_errors():
    class _BrokenDetector(PatternDetector):
        name = "detector_broken"

        def detect(self, detector_input, parameters):
            del detector_input, parameters
            raise RuntimeError("boom")

    agg = SetupEngineAggregator(detectors=[_BrokenDetector()])
    result = agg.aggregate(_detector_input(), parameters=DEFAULT_SETUP_ENGINE_PARAMETERS)

    assert "detector_broken:error" in result.failed_checks
    assert len(result.detector_traces) == 1
    trace = result.detector_traces[0]
    assert trace.outcome == "error"
    assert trace.error_detail is not None
    assert "RuntimeError: boom" in trace.error_detail


def test_primary_tie_break_prefers_structural_pattern_when_scores_are_close():
    trigger_candidate = {
        "pattern": "nr7_inside_day",
        "timeframe": "daily",
        "source_detector": "nr7_inside_day",
        "quality_score": 80.0,
        "readiness_score": 80.0,
        "confidence": 0.74,
    }
    structural_candidate = {
        "pattern": "vcp",
        "timeframe": "daily",
        "source_detector": "vcp",
        "quality_score": 80.0,
        "readiness_score": 80.0,
        "confidence": 0.73,
    }

    primary, tie_break_applied = _pick_primary_candidate(
        [trigger_candidate, structural_candidate]
    )

    assert primary is not None
    assert primary["pattern"] == "vcp"
    assert tie_break_applied is True


def test_aggregator_falls_back_when_no_candidate_meets_confidence_floor():
    class _LowConfidenceDetector(PatternDetector):
        name = "detector_low_confidence"

        def detect(self, detector_input, parameters):
            del detector_input, parameters
            return PatternDetectorResult.detected(
                self.name,
                PatternCandidateModel(
                    pattern="detector_low_confidence",
                    timeframe="daily",
                    source_detector=self.name,
                    quality_score=None,
                    readiness_score=None,
                    confidence=0.10,
                ),
            )

    agg = SetupEngineAggregator(detectors=[_LowConfidenceDetector()])
    result = agg.aggregate(_detector_input(), parameters=DEFAULT_SETUP_ENGINE_PARAMETERS)

    assert result.pattern_primary == "detector_low_confidence"
    assert "primary_pattern_fallback_selected" in result.passed_checks
    assert "primary_pattern_below_confidence_floor" in result.failed_checks
