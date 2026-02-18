"""Pydantic schemas for feature store API endpoints.

Contains response models for feature run listing and run comparison,
with ``from_domain()`` classmethods for domain→HTTP mapping.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Self

from pydantic import BaseModel

from ..domain.feature_store.models import RunStats
from ..use_cases.feature_store.compare_runs import (
    CompareRunsResult,
    CompareSummary,
    SymbolDelta,
    SymbolEntry,
)
from ..use_cases.feature_store.list_runs import FeatureRunSummary


# ---------------------------------------------------------------------------
# List runs response models
# ---------------------------------------------------------------------------


class RunStatsResponse(BaseModel):
    """Aggregate statistics for a feature run."""

    total_symbols: int
    processed_symbols: int
    failed_symbols: int
    duration_seconds: float

    @classmethod
    def from_domain(cls, stats: RunStats) -> Self:
        """Map domain RunStats to HTTP response."""
        return cls(
            total_symbols=stats.total_symbols,
            processed_symbols=stats.processed_symbols,
            failed_symbols=stats.failed_symbols,
            duration_seconds=stats.duration_seconds,
        )


class FeatureRunResponse(BaseModel):
    """Summary of a single feature run."""

    id: int
    as_of_date: date
    run_type: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    published_at: Optional[str] = None
    row_count: int
    is_latest_published: bool
    stats: Optional[RunStatsResponse] = None
    warnings: list[str]

    @classmethod
    def from_domain(cls, run: FeatureRunSummary) -> Self:
        """Map domain FeatureRunSummary to HTTP response."""
        stats = RunStatsResponse.from_domain(run.stats) if run.stats is not None else None
        return cls(
            id=run.id,
            as_of_date=run.as_of_date,
            run_type=run.run_type,
            status=run.status,
            created_at=run.created_at.isoformat(),
            completed_at=run.completed_at.isoformat() if run.completed_at else None,
            published_at=run.published_at.isoformat() if run.published_at else None,
            row_count=run.row_count,
            is_latest_published=run.is_latest_published,
            stats=stats,
            warnings=list(run.warnings),
        )


class ListRunsResponse(BaseModel):
    """Response model for listing feature runs."""

    runs: list[FeatureRunResponse]


# ---------------------------------------------------------------------------
# Compare runs response models
# ---------------------------------------------------------------------------


class SymbolEntryResponse(BaseModel):
    """A symbol with its score and rating (for added/removed lists)."""

    symbol: str
    score: Optional[float] = None
    rating: Optional[str] = None

    @classmethod
    def from_domain(cls, entry: SymbolEntry) -> Self:
        """Map domain SymbolEntry to HTTP response."""
        return cls(symbol=entry.symbol, score=entry.score, rating=entry.rating)


class SymbolDeltaResponse(BaseModel):
    """A symbol's score change between two runs."""

    symbol: str
    score_a: Optional[float] = None
    score_b: Optional[float] = None
    score_delta: float
    rating_a: Optional[str] = None
    rating_b: Optional[str] = None

    @classmethod
    def from_domain(cls, delta: SymbolDelta) -> Self:
        """Map domain SymbolDelta to HTTP response."""
        return cls(
            symbol=delta.symbol,
            score_a=delta.score_a,
            score_b=delta.score_b,
            score_delta=delta.score_delta,
            rating_a=delta.rating_a,
            rating_b=delta.rating_b,
        )


class CompareSummaryResponse(BaseModel):
    """Aggregate statistics for a run comparison."""

    total_common: int
    upgraded_count: int
    downgraded_count: int
    avg_score_change: float

    @classmethod
    def from_domain(cls, summary: CompareSummary) -> Self:
        """Map domain CompareSummary to HTTP response."""
        return cls(
            total_common=summary.total_common,
            upgraded_count=summary.upgraded_count,
            downgraded_count=summary.downgraded_count,
            avg_score_change=summary.avg_score_change,
        )


class CompareRunsResponse(BaseModel):
    """Response model for comparing two feature runs."""

    run_a_id: int
    run_b_id: int
    run_a_date: date
    run_b_date: date
    summary: CompareSummaryResponse
    added: list[SymbolEntryResponse]
    removed: list[SymbolEntryResponse]
    movers: list[SymbolDeltaResponse]

    @classmethod
    def from_domain(cls, result: CompareRunsResult) -> Self:
        """Map domain CompareRunsResult to HTTP response."""
        return cls(
            run_a_id=result.run_a_id,
            run_b_id=result.run_b_id,
            run_a_date=result.run_a_date,
            run_b_date=result.run_b_date,
            summary=CompareSummaryResponse.from_domain(result.summary),
            added=[SymbolEntryResponse.from_domain(e) for e in result.added],
            removed=[SymbolEntryResponse.from_domain(e) for e in result.removed],
            movers=[SymbolDeltaResponse.from_domain(d) for d in result.movers],
        )
