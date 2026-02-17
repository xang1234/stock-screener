"""Domain models for the feature store bounded context.

Pure value objects and enums that represent feature-run lifecycle,
data quality, and publish policy — independently of any infrastructure
(ORM, HTTP, caching).  All dataclasses use frozen=True for immutability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

from ..common.errors import InvalidTransitionError


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RunStatus(str, Enum):
    """Lifecycle states of a feature run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    PUBLISHED = "published"


class RunType(str, Enum):
    """How the feature run was initiated."""

    DAILY_SNAPSHOT = "daily_snapshot"
    BACKFILL = "backfill"
    MANUAL = "manual"


class DQSeverity(str, Enum):
    """Severity level of a data-quality check."""

    CRITICAL = "critical"  # blocks publishing
    WARNING = "warning"  # logged, does not block


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------


_VALID_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.RUNNING: frozenset({RunStatus.COMPLETED, RunStatus.FAILED}),
    RunStatus.COMPLETED: frozenset({RunStatus.PUBLISHED, RunStatus.QUARANTINED}),
    RunStatus.FAILED: frozenset(),
    RunStatus.PUBLISHED: frozenset(),
    RunStatus.QUARANTINED: frozenset(),
}


def validate_transition(current: RunStatus, target: RunStatus) -> None:
    """Raise InvalidTransitionError if *current* → *target* is illegal."""
    allowed = _VALID_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise InvalidTransitionError(current, target)


# ---------------------------------------------------------------------------
# Value Objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunStats:
    """Aggregate statistics for a completed (or failed) feature run."""

    total_symbols: int
    processed_symbols: int
    failed_symbols: int
    duration_seconds: float

    def __post_init__(self) -> None:
        if self.duration_seconds < 0:
            raise ValueError(
                f"duration_seconds must be >= 0, got {self.duration_seconds}"
            )
        if self.processed_symbols + self.failed_symbols > self.total_symbols:
            raise ValueError(
                f"processed ({self.processed_symbols}) + failed "
                f"({self.failed_symbols}) exceeds total ({self.total_symbols})"
            )


@dataclass(frozen=True)
class FeatureRunDomain:
    """Pure domain representation of a feature run.

    Uses tuple (not list) for warnings to ensure hashability
    and immutability of the frozen dataclass.
    """

    id: int | None
    as_of_date: date
    run_type: RunType
    status: RunStatus
    created_at: datetime
    completed_at: datetime | None  # None if still running or failed early
    correlation_id: str | None
    code_version: str | None
    universe_hash: str | None
    input_hash: str | None
    stats: RunStats | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SnapshotRef:
    """Lightweight pointer to a published snapshot for a given date."""

    run_id: int
    as_of_date: date
    status: RunStatus


# ---------------------------------------------------------------------------
# Feature Row Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureRowWrite:
    """Input for upserting a single snapshot row.

    Used by the use case to pass scored data into the repository
    without coupling to ORM models.
    """

    symbol: str
    as_of_date: date
    composite_score: float | None
    overall_rating: int | None
    passes_count: int | None
    details: dict[str, Any] | None


@dataclass(frozen=True)
class FeatureRow:
    """Output from feature store queries — a single scored row.

    Includes the run_id so callers know which run produced this data.
    """

    run_id: int
    symbol: str
    as_of_date: date
    composite_score: float | None
    overall_rating: int | None
    passes_count: int | None
    details: dict[str, Any] | None


@dataclass(frozen=True)
class FeaturePage:
    """Paginated query results from the feature store."""

    items: tuple[FeatureRow, ...]
    total: int
    page: int
    per_page: int

    @property
    def total_pages(self) -> int:
        if self.total == 0:
            return 0
        return (self.total + self.per_page - 1) // self.per_page


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "RunStatus",
    "RunType",
    "DQSeverity",
    "validate_transition",
    "RunStats",
    "FeatureRunDomain",
    "SnapshotRef",
    "FeatureRowWrite",
    "FeatureRow",
    "FeaturePage",
]
