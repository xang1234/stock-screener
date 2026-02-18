"""Tests for feature store domain models.

Verifies:
- Enum values and str behaviour
- RunStats validation (frozen, invariants)
- FeatureRunDomain construction and defaults
- SnapshotRef construction
- State transition validation (valid, invalid, terminal)
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from app.domain.common.errors import InvalidTransitionError
from app.domain.feature_store.models import (
    DQSeverity,
    FeatureRunDomain,
    RunStats,
    RunStatus,
    RunType,
    SnapshotRef,
    validate_transition,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_stats(**overrides: object) -> RunStats:
    """Factory for a minimal RunStats."""
    defaults = {
        "total_symbols": 100,
        "processed_symbols": 95,
        "failed_symbols": 5,
        "duration_seconds": 12.5,
    }
    defaults.update(overrides)
    return RunStats(**defaults)  # type: ignore[arg-type]


def _make_run(**overrides: object) -> FeatureRunDomain:
    """Factory for a minimal FeatureRunDomain."""
    defaults: dict[str, object] = {
        "id": 1,
        "as_of_date": date(2026, 1, 15),
        "run_type": RunType.DAILY_SNAPSHOT,
        "status": RunStatus.COMPLETED,
        "created_at": datetime(2026, 1, 15, 8, 0, 0),
        "completed_at": datetime(2026, 1, 15, 8, 5, 0),
        "correlation_id": "abc-123",
        "code_version": "v1.0.0",
        "universe_hash": "sha256:deadbeef",
        "input_hash": "sha256:cafebabe",
    }
    defaults.update(overrides)
    return FeatureRunDomain(**defaults)  # type: ignore[arg-type]


# ── Enum Tests ───────────────────────────────────────────────────────


class TestRunEnums:
    """RunStatus, RunType, and DQSeverity enum values and str behaviour."""

    def test_run_status_values(self):
        assert RunStatus.RUNNING == "running"
        assert RunStatus.COMPLETED == "completed"
        assert RunStatus.FAILED == "failed"
        assert RunStatus.QUARANTINED == "quarantined"
        assert RunStatus.PUBLISHED == "published"

    def test_run_type_values(self):
        assert RunType.DAILY_SNAPSHOT == "daily_snapshot"
        assert RunType.BACKFILL == "backfill"
        assert RunType.MANUAL == "manual"

    def test_dq_severity_values(self):
        assert DQSeverity.CRITICAL == "critical"
        assert DQSeverity.WARNING == "warning"

    def test_run_status_is_str(self):
        """(str, Enum) pattern allows direct string comparison."""
        assert isinstance(RunStatus.RUNNING, str)

    def test_run_type_is_str(self):
        assert isinstance(RunType.DAILY_SNAPSHOT, str)

    def test_dq_severity_is_str(self):
        assert isinstance(DQSeverity.CRITICAL, str)

    def test_all_run_statuses_present(self):
        assert len(RunStatus) == 5

    def test_all_run_types_present(self):
        assert len(RunType) == 3


# ── RunStats Tests ───────────────────────────────────────────────────


class TestRunStats:
    """RunStats construction, immutability, and validation."""

    def test_valid_construction(self):
        stats = _make_stats()
        assert stats.total_symbols == 100
        assert stats.processed_symbols == 95
        assert stats.failed_symbols == 5
        assert stats.duration_seconds == 12.5

    def test_frozen(self):
        stats = _make_stats()
        with pytest.raises(AttributeError):
            stats.total_symbols = 200  # type: ignore[misc]

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError, match="duration_seconds must be >= 0"):
            _make_stats(duration_seconds=-1.0)

    def test_processed_plus_failed_exceeds_total_raises(self):
        with pytest.raises(ValueError, match="exceeds total"):
            _make_stats(total_symbols=10, processed_symbols=8, failed_symbols=5)

    def test_zero_duration_valid(self):
        stats = _make_stats(duration_seconds=0.0)
        assert stats.duration_seconds == 0.0

    def test_all_zeros_valid(self):
        stats = RunStats(
            total_symbols=0,
            processed_symbols=0,
            failed_symbols=0,
            duration_seconds=0.0,
        )
        assert stats.total_symbols == 0

    def test_processed_plus_failed_equals_total_valid(self):
        stats = _make_stats(total_symbols=10, processed_symbols=7, failed_symbols=3)
        assert stats.processed_symbols + stats.failed_symbols == stats.total_symbols


# ── FeatureRunDomain Tests ───────────────────────────────────────────


class TestFeatureRunDomain:
    """FeatureRunDomain construction, defaults, and immutability."""

    def test_valid_construction(self):
        run = _make_run()
        assert run.id == 1
        assert run.as_of_date == date(2026, 1, 15)
        assert run.run_type == RunType.DAILY_SNAPSHOT
        assert run.status == RunStatus.COMPLETED

    def test_frozen(self):
        run = _make_run()
        with pytest.raises(AttributeError):
            run.status = RunStatus.PUBLISHED  # type: ignore[misc]

    def test_default_stats_none(self):
        run = _make_run()
        assert run.stats is None

    def test_default_warnings_empty_tuple(self):
        run = _make_run()
        assert run.warnings == ()

    def test_with_stats(self):
        stats = _make_stats()
        run = _make_run(stats=stats)
        assert run.stats is stats

    def test_with_warnings(self):
        run = _make_run(warnings=("low coverage", "stale data"))
        assert run.warnings == ("low coverage", "stale data")

    def test_none_id_for_unsaved(self):
        run = _make_run(id=None)
        assert run.id is None

    def test_none_completed_at(self):
        run = _make_run(completed_at=None)
        assert run.completed_at is None


# ── SnapshotRef Tests ────────────────────────────────────────────────


class TestSnapshotRef:
    """SnapshotRef construction and immutability."""

    def test_valid_construction(self):
        ref = SnapshotRef(
            run_id=42,
            as_of_date=date(2026, 1, 15),
            status=RunStatus.PUBLISHED,
        )
        assert ref.run_id == 42
        assert ref.as_of_date == date(2026, 1, 15)
        assert ref.status == RunStatus.PUBLISHED

    def test_frozen(self):
        ref = SnapshotRef(
            run_id=42,
            as_of_date=date(2026, 1, 15),
            status=RunStatus.PUBLISHED,
        )
        with pytest.raises(AttributeError):
            ref.run_id = 99  # type: ignore[misc]


# ── Transition Validation Tests ──────────────────────────────────────


class TestValidateTransition:
    """State machine transition checks."""

    @pytest.mark.parametrize(
        "current, target",
        [
            (RunStatus.RUNNING, RunStatus.COMPLETED),
            (RunStatus.RUNNING, RunStatus.FAILED),
            (RunStatus.COMPLETED, RunStatus.PUBLISHED),
            (RunStatus.COMPLETED, RunStatus.QUARANTINED),
            (RunStatus.QUARANTINED, RunStatus.PUBLISHED),
        ],
    )
    def test_valid_transitions(self, current: RunStatus, target: RunStatus):
        validate_transition(current, target)  # should not raise

    @pytest.mark.parametrize(
        "current, target",
        [
            (RunStatus.RUNNING, RunStatus.PUBLISHED),
            (RunStatus.RUNNING, RunStatus.QUARANTINED),
            (RunStatus.COMPLETED, RunStatus.RUNNING),
            (RunStatus.COMPLETED, RunStatus.FAILED),
            (RunStatus.FAILED, RunStatus.RUNNING),
            (RunStatus.FAILED, RunStatus.COMPLETED),
            (RunStatus.PUBLISHED, RunStatus.RUNNING),
            (RunStatus.PUBLISHED, RunStatus.COMPLETED),
            (RunStatus.QUARANTINED, RunStatus.RUNNING),
            (RunStatus.QUARANTINED, RunStatus.COMPLETED),
        ],
    )
    def test_invalid_transitions_raise(
        self, current: RunStatus, target: RunStatus
    ):
        with pytest.raises(InvalidTransitionError) as exc_info:
            validate_transition(current, target)
        assert exc_info.value.current is current
        assert exc_info.value.target is target

    @pytest.mark.parametrize(
        "terminal",
        [RunStatus.FAILED, RunStatus.PUBLISHED],
    )
    def test_terminal_states_reject_all(self, terminal: RunStatus):
        """Terminal states have no valid outgoing transitions."""
        for target in RunStatus:
            if target == terminal:
                continue
            with pytest.raises(InvalidTransitionError):
                validate_transition(terminal, target)

    def test_self_transition_invalid(self):
        """A status cannot transition to itself."""
        for status in RunStatus:
            with pytest.raises(InvalidTransitionError):
                validate_transition(status, status)
