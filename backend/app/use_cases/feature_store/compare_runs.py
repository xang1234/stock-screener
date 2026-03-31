"""CompareFeatureRunsUseCase — compare two feature runs side-by-side.

Identifies added/removed symbols, score movers, and rating changes.
All computation is pure Python over two dicts — no extra SQL queries
beyond the two bulk-fetch calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.domain.common.errors import ValidationError
from app.domain.common.uow import UnitOfWork
from app.domain.feature_store.models import INT_TO_RATING


# ── Query (input) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CompareRunsQuery:
    """Immutable value object describing the compare-runs request."""

    run_a: int
    run_b: int
    limit: int = 50

    def __post_init__(self) -> None:
        if self.run_a == self.run_b:
            raise ValidationError("run_a and run_b must be different")
        if not 1 <= self.limit <= 500:
            raise ValidationError("limit must be between 1 and 500")


# ── Result (output) ────────────────────────────────────────────────────


@dataclass(frozen=True)
class SymbolEntry:
    """A symbol with its score and rating (for added/removed lists)."""

    symbol: str
    score: float | None
    rating: str | None


@dataclass(frozen=True)
class SymbolDelta:
    """A symbol's score change between two runs."""

    symbol: str
    score_a: float | None
    score_b: float | None
    score_delta: float
    rating_a: str | None
    rating_b: str | None


@dataclass(frozen=True)
class CompareSummary:
    """Aggregate statistics for the comparison."""

    total_common: int
    upgraded_count: int  # rating improved (higher int value)
    downgraded_count: int  # rating worsened
    avg_score_change: float  # mean of all score deltas


@dataclass(frozen=True)
class CompareRunsResult:
    """What the use case returns to the caller."""

    run_a_id: int
    run_b_id: int
    run_a_date: date
    run_b_date: date
    summary: CompareSummary
    added: tuple[SymbolEntry, ...]  # in B not A (with B scores)
    removed: tuple[SymbolEntry, ...]  # in A not B (with A scores)
    movers: tuple[SymbolDelta, ...]  # sorted by abs(score_delta) DESC


# ── Use Case ───────────────────────────────────────────────────────────


class CompareFeatureRunsUseCase:
    """Compare two feature runs and return diffs."""

    def execute(
        self, uow: UnitOfWork, query: CompareRunsQuery
    ) -> CompareRunsResult:
        with uow:
            run_a = uow.feature_runs.get_run(query.run_a)
            run_b = uow.feature_runs.get_run(query.run_b)
            scores_a = uow.feature_store.get_scores_for_run(query.run_a)
            scores_b = uow.feature_store.get_scores_for_run(query.run_b)

        syms_a, syms_b = set(scores_a), set(scores_b)

        # Added/removed with their scores
        added = tuple(
            sorted(
                (
                    SymbolEntry(s, scores_b[s][0], INT_TO_RATING.get(scores_b[s][1]))
                    for s in syms_b - syms_a
                ),
                key=lambda e: (-(e.score or 0), e.symbol),
            )
        )
        removed = tuple(
            sorted(
                (
                    SymbolEntry(s, scores_a[s][0], INT_TO_RATING.get(scores_a[s][1]))
                    for s in syms_a - syms_b
                ),
                key=lambda e: (-(e.score or 0), e.symbol),
            )
        )

        # Movers + summary stats
        common = syms_a & syms_b
        all_deltas: list[SymbolDelta] = []
        upgraded = downgraded = 0
        for sym in common:
            sa, ra = scores_a[sym]
            sb, rb = scores_b[sym]
            delta = (sb or 0) - (sa or 0)
            if ra is not None and rb is not None:
                if rb > ra:
                    upgraded += 1
                elif rb < ra:
                    downgraded += 1
            if abs(delta) > 0.01:
                all_deltas.append(SymbolDelta(
                    symbol=sym,
                    score_a=sa,
                    score_b=sb,
                    score_delta=round(delta, 2),
                    rating_a=INT_TO_RATING.get(ra),
                    rating_b=INT_TO_RATING.get(rb),
                ))

        all_deltas.sort(key=lambda d: (-abs(d.score_delta), d.symbol))
        avg_change = (
            sum(d.score_delta for d in all_deltas) / len(all_deltas)
            if all_deltas
            else 0.0
        )

        return CompareRunsResult(
            run_a_id=run_a.id,
            run_b_id=run_b.id,
            run_a_date=run_a.as_of_date,
            run_b_date=run_b.as_of_date,
            summary=CompareSummary(
                total_common=len(common),
                upgraded_count=upgraded,
                downgraded_count=downgraded,
                avg_score_change=round(avg_change, 2),
            ),
            added=added,
            removed=removed,
            movers=tuple(all_deltas[: query.limit]),
        )
