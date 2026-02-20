"""Comprehensive tests for the detector interface contract (SE-B5).

Covers:
- DetectorOutcome enum values
- PatternDetectorResult.outcome derivation
- PatternDetectorResult.candidate backward-compat property
- PatternDetectorResult.candidates with 0, 1, and multiple entries
- Factory methods: detected, no_detection, insufficient_data, not_implemented, error
- insufficient_data() with normalized= merges checks/warnings
- detect_safe() exception guarding, passthrough, name mismatch, invalid candidates
- __init_subclass__ rejects non-snake_case names and missing name attribute
- All 7 default detectors produce valid PatternDetectorResult via detect_safe()
- Aggregator end-to-end with new detect_safe() path
"""

import pytest

from app.analysis.patterns.config import DEFAULT_SETUP_ENGINE_PARAMETERS
from app.analysis.patterns.detectors import (
    DetectorOutcome,
    PatternDetector,
    PatternDetectorInput,
    PatternDetectorResult,
    default_pattern_detectors,
)
from app.analysis.patterns.models import PatternCandidateModel, is_snake_case
from app.analysis.patterns.normalization import NormalizedOHLCV


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(**overrides):
    defaults = dict(symbol="TEST", timeframe="daily", daily_bars=260, weekly_bars=60, features={})
    defaults.update(overrides)
    return PatternDetectorInput(**defaults)


def _make_candidate(**overrides):
    defaults = dict(pattern="test_pattern", timeframe="daily", confidence=0.8)
    defaults.update(overrides)
    return PatternCandidateModel(**defaults)


def _make_normalized(
    prerequisites_ok=True,
    failed_checks=(),
    warnings=(),
):
    """Create a NormalizedOHLCV for testing factory merge behavior."""
    return NormalizedOHLCV(
        timeframe="daily",
        frame=None,
        checks={"frame_present": prerequisites_ok},
        failed_checks=failed_checks,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# DetectorOutcome enum
# ---------------------------------------------------------------------------

class TestDetectorOutcome:

    def test_all_five_values_exist(self):
        expected = {"detected", "not_detected", "insufficient_data", "not_implemented", "error"}
        actual = {e.value for e in DetectorOutcome}
        assert actual == expected

    def test_values_are_snake_case(self):
        for member in DetectorOutcome:
            assert is_snake_case(member.value), f"{member.name}.value is not snake_case"


# ---------------------------------------------------------------------------
# PatternDetectorResult.outcome derivation
# ---------------------------------------------------------------------------

class TestOutcomeDerivation:

    def test_detected_when_candidates_present(self):
        result = PatternDetectorResult(
            detector_name="test",
            candidates=(_make_candidate(),),
        )
        assert result.outcome == DetectorOutcome.DETECTED

    def test_not_detected_when_empty(self):
        result = PatternDetectorResult(detector_name="test")
        assert result.outcome == DetectorOutcome.NOT_DETECTED

    def test_not_implemented(self):
        result = PatternDetectorResult(
            detector_name="test",
            failed_checks=("not_implemented",),
        )
        assert result.outcome == DetectorOutcome.NOT_IMPLEMENTED

    def test_error(self):
        result = PatternDetectorResult(
            detector_name="test",
            failed_checks=("error",),
        )
        assert result.outcome == DetectorOutcome.ERROR

    def test_insufficient_data(self):
        result = PatternDetectorResult(
            detector_name="test",
            failed_checks=("insufficient_data", "daily_bars_lt_120"),
        )
        assert result.outcome == DetectorOutcome.INSUFFICIENT_DATA

    def test_candidates_override_failed_checks(self):
        """If candidates exist, outcome is DETECTED even with error in failed_checks."""
        result = PatternDetectorResult(
            detector_name="test",
            candidates=(_make_candidate(),),
            failed_checks=("error",),
        )
        assert result.outcome == DetectorOutcome.DETECTED


# ---------------------------------------------------------------------------
# Backward-compat candidate property
# ---------------------------------------------------------------------------

class TestCandidateBackwardCompat:

    def test_candidate_returns_none_when_empty(self):
        result = PatternDetectorResult(detector_name="test")
        assert result.candidate is None

    def test_candidate_returns_first(self):
        c1 = _make_candidate(pattern="first")
        c2 = _make_candidate(pattern="second")
        result = PatternDetectorResult(
            detector_name="test", candidates=(c1, c2)
        )
        assert result.candidate is c1

    def test_candidates_tuple_with_zero_entries(self):
        result = PatternDetectorResult(detector_name="test")
        assert result.candidates == ()

    def test_candidates_tuple_with_one_entry(self):
        c = _make_candidate()
        result = PatternDetectorResult(detector_name="test", candidates=(c,))
        assert result.candidates == (c,)
        assert len(result.candidates) == 1

    def test_candidates_tuple_with_multiple_entries(self):
        c1 = _make_candidate(pattern="a")
        c2 = _make_candidate(pattern="b")
        c3 = _make_candidate(pattern="c")
        result = PatternDetectorResult(
            detector_name="test", candidates=(c1, c2, c3)
        )
        assert len(result.candidates) == 3


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------

class TestFactoryMethods:

    def test_detected_single_candidate(self):
        c = _make_candidate()
        result = PatternDetectorResult.detected("my_det", c)
        assert result.outcome == DetectorOutcome.DETECTED
        assert result.candidates == (c,)
        assert "detected" in result.passed_checks

    def test_detected_multiple_candidates(self):
        c1 = _make_candidate(pattern="a")
        c2 = _make_candidate(pattern="b")
        result = PatternDetectorResult.detected("my_det", [c1, c2])
        assert len(result.candidates) == 2
        assert result.outcome == DetectorOutcome.DETECTED

    def test_detected_with_extra_passed_checks(self):
        c = _make_candidate()
        result = PatternDetectorResult.detected(
            "my_det", c, passed_checks=("volume_ok",)
        )
        assert "detected" in result.passed_checks
        assert "volume_ok" in result.passed_checks

    def test_no_detection(self):
        result = PatternDetectorResult.no_detection("my_det")
        assert result.outcome == DetectorOutcome.NOT_DETECTED
        assert result.candidates == ()
        assert "not_detected" in result.failed_checks

    def test_no_detection_with_extra_failed_checks(self):
        result = PatternDetectorResult.no_detection(
            "my_det", failed_checks=("contraction_too_shallow",)
        )
        assert "not_detected" in result.failed_checks
        assert "contraction_too_shallow" in result.failed_checks

    def test_insufficient_data_basic(self):
        result = PatternDetectorResult.insufficient_data("my_det")
        assert result.outcome == DetectorOutcome.INSUFFICIENT_DATA
        assert "insufficient_data" in result.failed_checks

    def test_insufficient_data_with_normalized(self):
        norm = _make_normalized(
            prerequisites_ok=False,
            failed_checks=("daily_bars_lt_120",),
            warnings=("ohlcv_sorted_chronologically",),
        )
        result = PatternDetectorResult.insufficient_data(
            "my_det", normalized=norm
        )
        assert "insufficient_data" in result.failed_checks
        assert "daily_bars_lt_120" in result.failed_checks
        assert "ohlcv_sorted_chronologically" in result.warnings

    def test_insufficient_data_with_extra_checks_and_warnings(self):
        norm = _make_normalized(
            failed_checks=("weekly_bars_lt_8",),
            warnings=("norm_warn",),
        )
        result = PatternDetectorResult.insufficient_data(
            "my_det",
            normalized=norm,
            failed_checks=("extra_fail",),
            warnings=("extra_warn",),
        )
        assert "weekly_bars_lt_8" in result.failed_checks
        assert "extra_fail" in result.failed_checks
        assert "norm_warn" in result.warnings
        assert "extra_warn" in result.warnings

    def test_not_implemented(self):
        result = PatternDetectorResult.not_implemented("my_det")
        assert result.outcome == DetectorOutcome.NOT_IMPLEMENTED
        assert "not_implemented" in result.failed_checks

    def test_not_implemented_with_warnings(self):
        result = PatternDetectorResult.not_implemented(
            "my_det", warnings=("stub_warning",)
        )
        assert "stub_warning" in result.warnings

    def test_error_factory(self):
        exc = RuntimeError("boom")
        result = PatternDetectorResult.error("my_det", exc)
        assert result.outcome == DetectorOutcome.ERROR
        assert "error" in result.failed_checks
        assert result.error_detail == "RuntimeError: boom"

    def test_error_factory_with_warnings(self):
        exc = ValueError("bad input")
        result = PatternDetectorResult.error(
            "my_det", exc, warnings=("partial_data",)
        )
        assert "partial_data" in result.warnings
        assert "ValueError: bad input" in result.error_detail


# ---------------------------------------------------------------------------
# detect_safe() template method
# ---------------------------------------------------------------------------

class TestDetectSafe:

    def test_catches_exception_returns_error(self):
        class _BrokenDetector(PatternDetector):
            name = "broken_test"

            def detect(self, detector_input, parameters):
                raise RuntimeError("kaboom")

        det = _BrokenDetector()
        result = det.detect_safe(_make_input(), DEFAULT_SETUP_ENGINE_PARAMETERS)
        assert result.outcome == DetectorOutcome.ERROR
        assert "RuntimeError: kaboom" in result.error_detail

    def test_passes_through_valid_result(self):
        class _GoodDetector(PatternDetector):
            name = "good_test"

            def detect(self, detector_input, parameters):
                return PatternDetectorResult.not_implemented(self.name)

        det = _GoodDetector()
        result = det.detect_safe(_make_input(), DEFAULT_SETUP_ENGINE_PARAMETERS)
        assert result.outcome == DetectorOutcome.NOT_IMPLEMENTED

    def test_coerces_candidates_to_canonical_shape(self):
        """detect_safe() returns PatternCandidate dicts, not PatternCandidateModel."""

        class _DetectorWithCandidate(PatternDetector):
            name = "coerce_test"

            def detect(self, detector_input, parameters):
                model = PatternCandidateModel(
                    pattern="vcp", timeframe="daily", confidence=0.75,
                )
                return PatternDetectorResult.detected(self.name, model)

        det = _DetectorWithCandidate()
        result = det.detect_safe(_make_input(), DEFAULT_SETUP_ENGINE_PARAMETERS)
        assert result.outcome == DetectorOutcome.DETECTED
        # Candidates should be coerced dicts, not PatternCandidateModel instances
        cand = result.candidates[0]
        assert isinstance(cand, dict)
        assert cand["pattern"] == "vcp"
        assert cand["confidence_pct"] == pytest.approx(75.0)

    def test_catches_exception_in_detect_body(self):
        """Exceptions during detect() (e.g. invalid model construction) are caught."""

        class _ConstructionErrorDetector(PatternDetector):
            name = "construct_err_test"

            def detect(self, detector_input, parameters):
                # Raises ValueError during __post_init__ — before result is built
                PatternCandidateModel(
                    pattern="test", timeframe="daily", setup_score=999.0
                )

        det = _ConstructionErrorDetector()
        result = det.detect_safe(_make_input(), DEFAULT_SETUP_ENGINE_PARAMETERS)
        assert result.outcome == DetectorOutcome.ERROR
        assert "setup_score" in result.error_detail

    def test_catches_name_mismatch(self):
        class _MismatchDetector(PatternDetector):
            name = "mismatch_test"

            def detect(self, detector_input, parameters):
                return PatternDetectorResult.not_implemented("wrong_name")

        det = _MismatchDetector()
        result = det.detect_safe(_make_input(), DEFAULT_SETUP_ENGINE_PARAMETERS)
        assert result.outcome == DetectorOutcome.ERROR
        assert "mismatch" in result.error_detail.lower()

    def test_catches_invalid_candidate(self):
        """Candidate validation loop catches bad data in raw dicts.

        Uses a PatternCandidate dict (not PatternCandidateModel) because
        TypedDict has no runtime validation — the invalid setup_score
        survives construction and is only caught during coercion inside
        detect_safe().
        """

        class _BadCandidateDetector(PatternDetector):
            name = "bad_cand_test"

            def detect(self, detector_input, parameters):
                # Raw dict bypasses PatternCandidateModel.__post_init__
                bad = {"pattern": "test", "timeframe": "daily", "setup_score": 999.0}
                return PatternDetectorResult(
                    detector_name=self.name, candidates=(bad,)
                )

        det = _BadCandidateDetector()
        result = det.detect_safe(_make_input(), DEFAULT_SETUP_ENGINE_PARAMETERS)
        assert result.outcome == DetectorOutcome.ERROR
        assert result.error_detail is not None
        assert "setup_score" in result.error_detail


# ---------------------------------------------------------------------------
# __init_subclass__ validation
# ---------------------------------------------------------------------------

class TestInitSubclass:

    def test_rejects_non_snake_case_name(self):
        with pytest.raises(TypeError, match="snake_case"):
            class _BadName(PatternDetector):
                name = "CamelCase"

                def detect(self, detector_input, parameters):
                    pass

    def test_rejects_missing_name(self):
        with pytest.raises(TypeError, match="name"):
            class _NoName(PatternDetector):
                def detect(self, detector_input, parameters):
                    pass

    def test_accepts_valid_snake_case(self):
        # Should not raise
        class _ValidDetector(PatternDetector):
            name = "valid_test_detector"

            def detect(self, detector_input, parameters):
                return PatternDetectorResult.not_implemented(self.name)

        assert _ValidDetector.name == "valid_test_detector"

    def test_skips_abstract_intermediary(self):
        """Abstract subclass without name should not raise."""
        import abc

        class _AbstractMiddle(PatternDetector):
            @abc.abstractmethod
            def extra_method(self):
                pass

        # Should not raise — it has abstract methods so validation is skipped


# ---------------------------------------------------------------------------
# All 7 default detectors produce valid results via detect_safe()
# ---------------------------------------------------------------------------

class TestAllDefaultDetectors:

    @pytest.fixture()
    def detector_input(self):
        return _make_input(daily_bars=260, weekly_bars=60)

    @pytest.fixture()
    def detector_input_insufficient(self):
        return _make_input(daily_bars=5, weekly_bars=2)

    def test_all_detectors_have_snake_case_names(self):
        for det in default_pattern_detectors():
            assert is_snake_case(det.name), f"{det.__class__.__name__}.name={det.name!r}"

    def test_all_detectors_return_valid_result_via_detect_safe(self, detector_input):
        for det in default_pattern_detectors():
            result = det.detect_safe(detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS)
            assert isinstance(result, PatternDetectorResult)
            assert result.detector_name == det.name
            assert result.outcome in set(DetectorOutcome)

    def test_default_detectors_return_contract_outcomes_with_sufficient_bars(
        self, detector_input
    ):
        for det in default_pattern_detectors():
            result = det.detect_safe(detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS)
            assert result.outcome in {
                DetectorOutcome.NOT_IMPLEMENTED,
                DetectorOutcome.INSUFFICIENT_DATA,
            }, (
                f"{det.name} expected NOT_IMPLEMENTED/INSUFFICIENT_DATA, got {result.outcome}"
            )

    def test_detectors_with_insufficient_bars_return_insufficient_data(
        self, detector_input_insufficient
    ):
        """At least some detectors should report insufficient data with tiny bar counts."""
        insufficient_count = 0
        for det in default_pattern_detectors():
            result = det.detect_safe(
                detector_input_insufficient, DEFAULT_SETUP_ENGINE_PARAMETERS
            )
            if result.outcome == DetectorOutcome.INSUFFICIENT_DATA:
                insufficient_count += 1
                assert DetectorOutcome.INSUFFICIENT_DATA.value in result.failed_checks

        assert insufficient_count > 0, "Expected at least one detector to report insufficient data"


# ---------------------------------------------------------------------------
# Aggregator end-to-end with detect_safe() path
# ---------------------------------------------------------------------------

class TestAggregatorWithDetectSafe:

    def test_aggregator_runs_without_exceptions(self):
        from app.analysis.patterns.aggregator import SetupEngineAggregator

        agg = SetupEngineAggregator()
        result = agg.aggregate(
            _make_input(), parameters=DEFAULT_SETUP_ENGINE_PARAMETERS
        )
        assert result.pattern_primary is None
        assert "no_primary_pattern" in result.failed_checks

    def test_aggregator_handles_broken_detector(self):
        from app.analysis.patterns.aggregator import SetupEngineAggregator

        class _BrokenDetector(PatternDetector):
            name = "broken_agg_test"

            def detect(self, detector_input, parameters):
                raise RuntimeError("intentional failure")

        agg = SetupEngineAggregator(detectors=[_BrokenDetector()])
        result = agg.aggregate(
            _make_input(), parameters=DEFAULT_SETUP_ENGINE_PARAMETERS
        )
        assert "broken_agg_test:error" in result.failed_checks
        assert any("RuntimeError" in d for d in result.diagnostics)

    def test_aggregator_collects_checks_from_healthy_detectors(self):
        from app.analysis.patterns.aggregator import SetupEngineAggregator

        agg = SetupEngineAggregator()
        result = agg.aggregate(
            _make_input(), parameters=DEFAULT_SETUP_ENGINE_PARAMETERS
        )
        # Some detectors are still stubs, so "not_implemented" should be present.
        assert "not_implemented" in result.failed_checks
