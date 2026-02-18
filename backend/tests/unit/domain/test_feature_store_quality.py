"""Tests for feature store data-quality checks and publish readiness.

Verifies:
- DQResult construction and frozen behaviour
- check_row_count: passing, failing, boundary, zero expected, custom severity
- check_null_rate: passing, failing, zero total, boundary
- check_score_distribution: in-range, out-of-range, empty, single, message
- check_rating_distribution: multiple distinct, single value, empty, custom
- check_symbol_coverage: full, partial, low, empty universe, extra symbols
- DQThresholds: validation of threshold ranges
- run_all_dq_checks: 5 results, correct check names, default/custom thresholds
- is_publishable: critical pass/fail, warnings-only, empty
- evaluate_publish_readiness: status gating, DQ partitioning, reason property
- evaluate_force_publish: quarantined allowed, non-quarantined blocked
"""

from __future__ import annotations

import pytest

from app.domain.feature_store.models import DQSeverity, RunStatus, validate_transition
from app.domain.feature_store.quality import (
    DQ_NULL_RATE,
    DQ_RATING_DISTRIBUTION,
    DQ_ROW_COUNT,
    DQ_SCORE_DISTRIBUTION,
    DQ_SYMBOL_COVERAGE,
    DQInputs,
    DQResult,
    DQThresholds,
    check_null_rate,
    check_rating_distribution,
    check_row_count,
    check_score_distribution,
    check_symbol_coverage,
    is_publishable,
    run_all_dq_checks,
)
from app.domain.feature_store.publish_policy import (
    evaluate_force_publish,
    evaluate_publish_readiness,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_dq(
    passed: bool = True,
    severity: DQSeverity = DQSeverity.CRITICAL,
    name: str = "test_check",
) -> DQResult:
    """Factory for a minimal DQResult."""
    return DQResult(
        check_name=name,
        passed=passed,
        severity=severity,
        actual_value=0.95 if passed else 0.5,
        threshold=0.9,
        message="test",
    )


# ── DQResult Tests ───────────────────────────────────────────────────


class TestDQResult:
    """DQResult construction and immutability."""

    def test_construction(self):
        r = _make_dq()
        assert r.check_name == "test_check"
        assert r.passed is True
        assert r.severity == DQSeverity.CRITICAL

    def test_frozen(self):
        r = _make_dq()
        with pytest.raises(AttributeError):
            r.passed = False  # type: ignore[misc]

    def test_severity_field(self):
        r = _make_dq(severity=DQSeverity.WARNING)
        assert r.severity == DQSeverity.WARNING


# ── check_row_count Tests ────────────────────────────────────────────


class TestCheckRowCount:
    """Row count DQ check."""

    def test_passing(self):
        result = check_row_count(expected=100, actual=95)
        assert result.passed is True
        assert result.check_name == "row_count"

    def test_failing(self):
        result = check_row_count(expected=100, actual=80)
        assert result.passed is False

    def test_exact_threshold_passes(self):
        result = check_row_count(expected=100, actual=90, threshold=0.9)
        assert result.passed is True
        assert result.actual_value == pytest.approx(0.9)

    def test_just_below_threshold_fails(self):
        result = check_row_count(expected=100, actual=89, threshold=0.9)
        assert result.passed is False
        assert result.actual_value == pytest.approx(0.89)

    def test_zero_expected_passes(self):
        result = check_row_count(expected=0, actual=0)
        assert result.passed is True

    def test_custom_severity(self):
        result = check_row_count(
            expected=100, actual=50, severity=DQSeverity.WARNING
        )
        assert result.severity == DQSeverity.WARNING

    def test_message_contains_counts(self):
        result = check_row_count(expected=100, actual=95)
        assert "actual=95" in result.message
        assert "expected=100" in result.message


# ── check_null_rate Tests ────────────────────────────────────────────


class TestCheckNullRate:
    """Null rate DQ check."""

    def test_passing(self):
        result = check_null_rate(column_nulls=2, total=100)
        assert result.passed is True
        assert result.check_name == "null_rate"

    def test_failing(self):
        result = check_null_rate(column_nulls=10, total=100)
        assert result.passed is False

    def test_zero_total_passes(self):
        result = check_null_rate(column_nulls=0, total=0)
        assert result.passed is True

    def test_exact_threshold_passes(self):
        result = check_null_rate(column_nulls=5, total=100, max_rate=0.05)
        assert result.passed is True
        assert result.actual_value == pytest.approx(0.05)

    def test_just_above_threshold_fails(self):
        result = check_null_rate(column_nulls=6, total=100, max_rate=0.05)
        assert result.passed is False

    def test_message_contains_counts(self):
        result = check_null_rate(column_nulls=3, total=100)
        assert "nulls=3" in result.message
        assert "total=100" in result.message


# ── check_score_distribution Tests ───────────────────────────────────


class TestCheckScoreDistribution:
    """Score distribution DQ check."""

    def test_in_range_mean(self):
        scores = [40.0, 50.0, 60.0]  # mean = 50
        result = check_score_distribution(scores, (30.0, 70.0))
        assert result.passed is True

    def test_out_of_range_mean(self):
        scores = [90.0, 95.0, 100.0]  # mean = 95
        result = check_score_distribution(scores, (30.0, 70.0))
        assert result.passed is False

    def test_empty_scores_fail(self):
        result = check_score_distribution([], (30.0, 70.0))
        assert result.passed is False
        assert "no scores" in result.message

    def test_single_score(self):
        result = check_score_distribution([50.0], (40.0, 60.0))
        assert result.passed is True
        assert result.actual_value == pytest.approx(50.0)

    def test_custom_severity(self):
        result = check_score_distribution(
            [50.0], (40.0, 60.0), severity=DQSeverity.CRITICAL
        )
        assert result.severity == DQSeverity.CRITICAL

    def test_message_includes_range(self):
        result = check_score_distribution([50.0], (30.0, 70.0))
        assert "[30.0, 70.0]" in result.message

    def test_threshold_is_midpoint(self):
        result = check_score_distribution([50.0], (30.0, 70.0))
        assert result.threshold == pytest.approx(50.0)

    def test_boundary_low(self):
        """Mean exactly at low bound passes."""
        result = check_score_distribution([30.0], (30.0, 70.0))
        assert result.passed is True

    def test_boundary_high(self):
        """Mean exactly at high bound passes."""
        result = check_score_distribution([70.0], (30.0, 70.0))
        assert result.passed is True


# ── is_publishable Tests ─────────────────────────────────────────────


class TestIsPublishable:
    """Aggregate publishability from DQ results."""

    def test_all_critical_pass(self):
        results = [_make_dq(passed=True), _make_dq(passed=True)]
        assert is_publishable(results) is True

    def test_one_critical_fails(self):
        results = [_make_dq(passed=True), _make_dq(passed=False)]
        assert is_publishable(results) is False

    def test_warnings_only_failures_still_publishable(self):
        results = [
            _make_dq(passed=True, severity=DQSeverity.CRITICAL),
            _make_dq(passed=False, severity=DQSeverity.WARNING),
        ]
        assert is_publishable(results) is True

    def test_empty_list_publishable(self):
        assert is_publishable([]) is True


# ── evaluate_publish_readiness Tests ─────────────────────────────────


class TestEvaluatePublishReadiness:
    """Publish readiness policy evaluation."""

    def test_completed_all_pass(self):
        dq = [_make_dq(passed=True)]
        decision = evaluate_publish_readiness(RunStatus.COMPLETED, dq)
        assert decision.allowed is True
        assert decision.blocking_checks == ()
        assert decision.reason == "All critical checks passed"

    def test_completed_critical_fail(self):
        dq = [_make_dq(passed=False, severity=DQSeverity.CRITICAL, name="row_count")]
        decision = evaluate_publish_readiness(RunStatus.COMPLETED, dq)
        assert decision.allowed is False
        assert len(decision.blocking_checks) == 1
        assert "row_count" in decision.reason

    def test_running_status_blocks(self):
        decision = evaluate_publish_readiness(RunStatus.RUNNING, [])
        assert decision.allowed is False
        assert decision.reason == "Run is not in COMPLETED status"

    def test_failed_status_blocks(self):
        decision = evaluate_publish_readiness(RunStatus.FAILED, [])
        assert decision.allowed is False
        assert decision.reason == "Run is not in COMPLETED status"

    def test_published_status_blocks(self):
        decision = evaluate_publish_readiness(RunStatus.PUBLISHED, [])
        assert decision.allowed is False
        assert decision.reason == "Run is not in COMPLETED status"

    def test_completed_only_warnings_fail(self):
        dq = [
            _make_dq(passed=True, severity=DQSeverity.CRITICAL),
            _make_dq(passed=False, severity=DQSeverity.WARNING, name="null_rate"),
        ]
        decision = evaluate_publish_readiness(RunStatus.COMPLETED, dq)
        assert decision.allowed is True
        assert len(decision.warnings) == 1
        assert decision.warnings[0].check_name == "null_rate"

    def test_blocking_and_warnings_partitioned(self):
        dq = [
            _make_dq(passed=False, severity=DQSeverity.CRITICAL, name="row_count"),
            _make_dq(passed=False, severity=DQSeverity.WARNING, name="null_rate"),
            _make_dq(passed=True, severity=DQSeverity.CRITICAL, name="other"),
        ]
        decision = evaluate_publish_readiness(RunStatus.COMPLETED, dq)
        assert decision.allowed is False
        assert len(decision.blocking_checks) == 1
        assert len(decision.warnings) == 1
        assert decision.blocking_checks[0].check_name == "row_count"
        assert decision.warnings[0].check_name == "null_rate"

    def test_publish_decision_frozen(self):
        decision = evaluate_publish_readiness(RunStatus.COMPLETED, [])
        with pytest.raises(AttributeError):
            decision.allowed = False  # type: ignore[misc]

    def test_non_completed_has_empty_checks(self):
        """Non-COMPLETED status returns empty blocking_checks (not status as DQ)."""
        decision = evaluate_publish_readiness(RunStatus.RUNNING, [_make_dq()])
        assert decision.blocking_checks == ()
        assert decision.warnings == ()


# ── check_rating_distribution Tests ─────────────────────────────────


class TestCheckRatingDistribution:
    """Rating distribution DQ check."""

    def test_multiple_distinct_ratings_pass(self):
        result = check_rating_distribution([1, 2, 3, 4, 5])
        assert result.passed is True
        assert result.check_name == DQ_RATING_DISTRIBUTION

    def test_single_value_fails(self):
        result = check_rating_distribution([3, 3, 3, 3])
        assert result.passed is False

    def test_empty_ratings_fail(self):
        result = check_rating_distribution([])
        assert result.passed is False
        assert result.actual_value == 0.0

    def test_exact_threshold_passes(self):
        result = check_rating_distribution([1, 2], min_distinct=2)
        assert result.passed is True
        assert result.actual_value == 2.0

    def test_below_threshold_fails(self):
        result = check_rating_distribution([1], min_distinct=2)
        assert result.passed is False

    def test_custom_severity(self):
        result = check_rating_distribution(
            [1, 2], severity=DQSeverity.CRITICAL
        )
        assert result.severity == DQSeverity.CRITICAL

    def test_custom_min_distinct(self):
        result = check_rating_distribution([1, 2, 3], min_distinct=3)
        assert result.passed is True

    def test_message_contains_count(self):
        result = check_rating_distribution([1, 2, 3])
        assert "3 distinct" in result.message


# ── check_symbol_coverage Tests ─────────────────────────────────────


class TestCheckSymbolCoverage:
    """Symbol coverage DQ check."""

    def test_full_coverage(self):
        result = check_symbol_coverage(
            ["AAPL", "MSFT", "GOOGL"], ["AAPL", "MSFT", "GOOGL"]
        )
        assert result.passed is True
        assert result.check_name == DQ_SYMBOL_COVERAGE

    def test_partial_above_threshold(self):
        universe = [f"SYM{i}" for i in range(10)]
        results = [f"SYM{i}" for i in range(9)]  # 9/10 = 90%
        result = check_symbol_coverage(universe, results, threshold=0.9)
        assert result.passed is True

    def test_low_coverage_fails(self):
        universe = [f"SYM{i}" for i in range(10)]
        results = [f"SYM{i}" for i in range(5)]  # 5/10 = 50%
        result = check_symbol_coverage(universe, results, threshold=0.9)
        assert result.passed is False

    def test_empty_universe_passes(self):
        result = check_symbol_coverage([], ["AAPL"])
        assert result.passed is True
        assert result.actual_value == 1.0

    def test_empty_results_fails(self):
        result = check_symbol_coverage(["AAPL", "MSFT"], [])
        assert result.passed is False
        assert result.actual_value == 0.0

    def test_extra_symbols_ignored(self):
        """Result symbols not in universe don't affect coverage."""
        result = check_symbol_coverage(
            ["AAPL", "MSFT"], ["AAPL", "MSFT", "EXTRA"]
        )
        assert result.passed is True
        assert result.actual_value == 1.0

    def test_custom_severity(self):
        result = check_symbol_coverage(
            ["AAPL"], ["AAPL"], severity=DQSeverity.CRITICAL
        )
        assert result.severity == DQSeverity.CRITICAL

    def test_message_contains_counts(self):
        result = check_symbol_coverage(
            ["AAPL", "MSFT", "GOOGL"], ["AAPL", "MSFT"]
        )
        assert "results=2" in result.message
        assert "universe=3" in result.message


# ── DQThresholds Tests ──────────────────────────────────────────────


class TestDQThresholds:
    """DQThresholds validation."""

    def test_valid_defaults(self):
        t = DQThresholds()
        assert t.row_count_threshold == 0.9
        assert t.null_max_rate == 0.05
        assert t.score_mean_range == (20.0, 80.0)
        assert t.min_distinct_ratings == 2
        assert t.symbol_coverage_threshold == 0.9

    def test_row_count_threshold_too_high(self):
        with pytest.raises(ValueError, match="row_count_threshold"):
            DQThresholds(row_count_threshold=1.5)

    def test_row_count_threshold_negative(self):
        with pytest.raises(ValueError, match="row_count_threshold"):
            DQThresholds(row_count_threshold=-0.1)

    def test_null_max_rate_too_high(self):
        with pytest.raises(ValueError, match="null_max_rate"):
            DQThresholds(null_max_rate=1.5)

    def test_null_max_rate_negative(self):
        with pytest.raises(ValueError, match="null_max_rate"):
            DQThresholds(null_max_rate=-0.1)

    def test_score_mean_range_inverted(self):
        with pytest.raises(ValueError, match="score_mean_range"):
            DQThresholds(score_mean_range=(80.0, 20.0))

    def test_score_mean_range_equal(self):
        with pytest.raises(ValueError, match="score_mean_range"):
            DQThresholds(score_mean_range=(50.0, 50.0))

    def test_min_distinct_ratings_zero(self):
        with pytest.raises(ValueError, match="min_distinct_ratings"):
            DQThresholds(min_distinct_ratings=0)

    def test_min_distinct_ratings_negative(self):
        with pytest.raises(ValueError, match="min_distinct_ratings"):
            DQThresholds(min_distinct_ratings=-1)

    def test_symbol_coverage_threshold_too_high(self):
        with pytest.raises(ValueError, match="symbol_coverage_threshold"):
            DQThresholds(symbol_coverage_threshold=1.5)

    def test_symbol_coverage_threshold_negative(self):
        with pytest.raises(ValueError, match="symbol_coverage_threshold"):
            DQThresholds(symbol_coverage_threshold=-0.1)

    def test_frozen(self):
        t = DQThresholds()
        with pytest.raises(AttributeError):
            t.row_count_threshold = 0.5  # type: ignore[misc]


# ── run_all_dq_checks Tests ─────────────────────────────────────────


class TestRunAllDQChecks:
    """Orchestrator returning all five DQ checks."""

    @pytest.fixture
    def good_inputs(self) -> DQInputs:
        return DQInputs(
            expected_row_count=10,
            actual_row_count=10,
            null_score_count=0,
            total_row_count=10,
            scores=tuple(50.0 + i for i in range(10)),
            ratings=(1, 2, 3, 4, 5, 1, 2, 3, 4, 5),
            universe_symbols=tuple(f"SYM{i}" for i in range(10)),
            result_symbols=tuple(f"SYM{i}" for i in range(10)),
        )

    def test_returns_five_results(self, good_inputs):
        results = run_all_dq_checks(good_inputs, DQThresholds())
        assert len(results) == 5

    def test_check_names_match_constants(self, good_inputs):
        results = run_all_dq_checks(good_inputs, DQThresholds())
        names = [r.check_name for r in results]
        assert names == [
            DQ_ROW_COUNT,
            DQ_NULL_RATE,
            DQ_SCORE_DISTRIBUTION,
            DQ_RATING_DISTRIBUTION,
            DQ_SYMBOL_COVERAGE,
        ]

    def test_all_pass_with_good_data(self, good_inputs):
        results = run_all_dq_checks(good_inputs, DQThresholds())
        assert all(r.passed for r in results)

    def test_custom_thresholds_applied(self, good_inputs):
        strict = DQThresholds(row_count_threshold=1.0, min_distinct_ratings=10)
        results = run_all_dq_checks(good_inputs, strict)
        row_count = results[0]
        assert row_count.passed is True  # 10/10 = 100%
        rating_dist = results[3]
        assert rating_dist.passed is False  # only 5 distinct < 10


# ── State Transition: QUARANTINED → PUBLISHED ───────────────────────


class TestStateTransitionQuarantinedToPublished:
    """Verify the new QUARANTINED → PUBLISHED transition."""

    def test_quarantined_to_published_valid(self):
        validate_transition(RunStatus.QUARANTINED, RunStatus.PUBLISHED)


# ── evaluate_force_publish Tests ────────────────────────────────────


class TestEvaluateForcePublish:
    """Force-publish policy evaluation."""

    def test_quarantined_allowed(self):
        decision = evaluate_force_publish(RunStatus.QUARANTINED)
        assert decision.allowed is True

    def test_completed_blocked(self):
        decision = evaluate_force_publish(RunStatus.COMPLETED)
        assert decision.allowed is False

    def test_running_blocked(self):
        decision = evaluate_force_publish(RunStatus.RUNNING)
        assert decision.allowed is False

    def test_published_blocked(self):
        decision = evaluate_force_publish(RunStatus.PUBLISHED)
        assert decision.allowed is False

    def test_failed_blocked(self):
        decision = evaluate_force_publish(RunStatus.FAILED)
        assert decision.allowed is False
