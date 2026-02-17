"""BuildDailyFeatureSnapshotUseCase — builds a complete feature snapshot for a trading day.

This use case contains the business rules for building a daily feature
snapshot:
  1. Validate as_of_date is a US trading day
  2. Start a feature run (status=RUNNING)
  3. Resolve universe of active symbols and save snapshot
  4. Scan each symbol via StockScanner in chunks (with progress + cancellation)
  5. Upsert feature rows after each chunk (checkpoint)
  6. Mark run COMPLETED with stats
  7. Run DQ checks (row count, null rate, score distribution)
  8. Publish or quarantine based on DQ results
  9. Emit completion ProgressEvent

The use case depends ONLY on domain ports — never on SQLAlchemy, Celery,
Redis, or any other infrastructure.

Zero Celery imports.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Iterator, Sequence

import pandas_market_calendars as mcal

from app.domain.common.errors import ValidationError
from app.domain.common.uow import UnitOfWork
from app.domain.feature_store.models import (
    FeatureRowWrite,
    RunStats,
    RunStatus,
    RunType,
)
from app.domain.feature_store.publish_policy import evaluate_publish_readiness
from app.domain.feature_store.quality import (
    DQResult,
    check_null_rate,
    check_row_count,
    check_score_distribution,
)
from app.domain.scanning.models import ProgressEvent
from app.domain.scanning.ports import (
    CancellationToken,
    ProgressSink,
    StockScanner,
)

logger = logging.getLogger(__name__)

# Singleton calendar instance (thread-safe, immutable schedule data).
_nyse_calendar = mcal.get_calendar("NYSE")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_us_trading_day(dt: date) -> bool:
    """Check if *dt* is a valid US trading day using the NYSE calendar."""
    schedule = _nyse_calendar.schedule(start_date=dt, end_date=dt)
    return len(schedule) > 0


def _chunked(seq: Sequence[str], size: int) -> Iterator[list[str]]:
    """Yield successive chunks of *size* from *seq*."""
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


_RATING_TO_INT: dict[str, int] = {
    "Strong Buy": 5,
    "Buy": 4,
    "Watch": 3,
    "Pass": 2,
    "Error": 1,
}


def _map_orchestrator_to_feature_row(
    symbol: str,
    as_of_date: date,
    result_dict: dict,
) -> FeatureRowWrite:
    """Map ScanOrchestrator dict output to a FeatureRowWrite domain object.

    Pure function — no I/O, fully deterministic.
    """
    return FeatureRowWrite(
        symbol=symbol,
        as_of_date=as_of_date,
        composite_score=result_dict.get("composite_score"),
        overall_rating=_RATING_TO_INT.get(result_dict.get("rating", ""), 3),
        passes_count=result_dict.get("screeners_passed"),
        details=result_dict.get("details"),
    )


# ── Command (input) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class BuildDailySnapshotCommand:
    """Immutable value object describing the snapshot to build."""

    as_of_date: date
    screener_names: list[str]
    universe_def: object  # UniverseSpec or similar
    criteria: dict = field(default_factory=dict)
    composite_method: str = "weighted_average"
    chunk_size: int = 50
    correlation_id: str | None = None
    code_version: str | None = None

    # DQ thresholds (overridable per-invocation)
    row_count_threshold: float = 0.9
    null_max_rate: float = 0.05
    score_mean_range: tuple[float, float] = (20.0, 80.0)

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if not 0.0 <= self.row_count_threshold <= 1.0:
            raise ValueError("row_count_threshold must be in [0.0, 1.0]")
        if not 0.0 <= self.null_max_rate <= 1.0:
            raise ValueError("null_max_rate must be in [0.0, 1.0]")
        low, high = self.score_mean_range
        if low >= high:
            raise ValueError(
                f"score_mean_range must be (low, high) with low < high, "
                f"got ({low}, {high})"
            )


# ── Result (output) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class BuildDailySnapshotResult:
    """What the use case returns to the caller."""

    run_id: int
    status: str  # published | quarantined | failed
    total_symbols: int
    processed_symbols: int
    failed_symbols: int
    dq_passed: bool
    warnings: tuple[str, ...]


# ── Use Case ─────────────────────────────────────────────────────────────


class BuildDailyFeatureSnapshotUseCase:
    """Build a complete feature snapshot for a trading day.

    The constructor accepts infrastructure collaborators through ports.
    ``execute()`` receives a fresh UoW per invocation.
    """

    def __init__(self, scanner: StockScanner) -> None:
        self._scanner = scanner

    def execute(
        self,
        uow: UnitOfWork,
        cmd: BuildDailySnapshotCommand,
        progress: ProgressSink,
        cancel: CancellationToken,
    ) -> BuildDailySnapshotResult:
        """Run the full snapshot build lifecycle inside a single UoW."""
        # ── Validate trading day ────────────────────────────────
        if not _is_us_trading_day(cmd.as_of_date):
            raise ValidationError(
                f"as_of_date {cmd.as_of_date} is not a US trading day"
            )

        with uow:
            # Start run BEFORE the try-except so run_id is available
            # for best-effort error handling.
            run = uow.feature_runs.start_run(
                as_of_date=cmd.as_of_date,
                run_type=RunType.DAILY_SNAPSHOT,
                code_version=cmd.code_version,
                correlation_id=cmd.correlation_id,
            )
            run_id = run.id
            uow.commit()
            logger.info(
                "Started feature run %d for %s", run_id, cmd.as_of_date
            )

            try:
                return self._run(uow, run_id, cmd, progress, cancel)
            except Exception:
                # Best-effort: don't attempt status transition here.
                # The run may be in RUNNING or COMPLETED depending on
                # where the error occurred.  A stale-run cleanup task
                # can reap stuck runs that haven't progressed.
                logger.exception("Feature run %d failed", run_id)
                raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(
        self,
        uow: UnitOfWork,
        run_id: int,
        cmd: BuildDailySnapshotCommand,
        progress: ProgressSink,
        cancel: CancellationToken,
    ) -> BuildDailySnapshotResult:
        start_time = time.monotonic()

        # ── 1. Resolve universe ─────────────────────────────────
        symbols = uow.universe.resolve_symbols(cmd.universe_def)
        if not symbols:
            raise ValidationError("Universe resolved to zero symbols")
        total = len(symbols)
        logger.info("Resolved %d symbols for run %d", total, run_id)

        # ── 2. Save universe snapshot ───────────────────────────
        uow.feature_store.save_run_universe_symbols(run_id, symbols)
        uow.commit()

        # ── 3. Scan symbols in chunks ───────────────────────────
        processed = 0
        failed = 0
        all_rows: list[FeatureRowWrite] = []

        for chunk in _chunked(symbols, cmd.chunk_size):
            # 3a — Cancellation gate
            if cancel.is_cancelled():
                duration = time.monotonic() - start_time
                stats = RunStats(
                    total_symbols=total,
                    processed_symbols=processed - failed,
                    failed_symbols=failed,
                    duration_seconds=round(duration, 2),
                )
                uow.feature_runs.mark_completed(
                    run_id, stats, warnings=("Cancelled by user",)
                )
                uow.commit()
                logger.info(
                    "Run %d cancelled at %d/%d", run_id, processed, total
                )
                return BuildDailySnapshotResult(
                    run_id=run_id,
                    status=RunStatus.COMPLETED.value,
                    total_symbols=total,
                    processed_symbols=processed,
                    failed_symbols=failed,
                    dq_passed=False,
                    warnings=("Cancelled by user",),
                )

            # 3b — Scan each symbol in the chunk
            chunk_rows: list[FeatureRowWrite] = []
            for symbol in chunk:
                sym = symbol.upper()
                try:
                    result = self._scanner.scan_stock_multi(
                        symbol=sym,
                        screener_names=cmd.screener_names,
                        criteria=cmd.criteria,
                        composite_method=cmd.composite_method,
                    )
                    if result and "error" not in result:
                        row = _map_orchestrator_to_feature_row(
                            sym, cmd.as_of_date, result
                        )
                        chunk_rows.append(row)
                    else:
                        failed += 1
                except Exception:
                    logger.debug(
                        "Error scanning %s in run %d",
                        sym,
                        run_id,
                        exc_info=True,
                    )
                    failed += 1
                processed += 1

            # 3c — Persist chunk (checkpoint)
            if chunk_rows:
                uow.feature_store.upsert_snapshot_rows(run_id, chunk_rows)
                all_rows.extend(chunk_rows)
            uow.commit()

            # 3d — Progress reporting
            elapsed = time.monotonic() - start_time
            throughput = processed / elapsed if elapsed > 0 else 0.0
            remaining = total - processed
            eta = remaining / throughput if throughput > 0 else None

            progress.emit(
                ProgressEvent(
                    current=processed,
                    total=total,
                    passed=len(all_rows),
                    failed=failed,
                    throughput=(
                        round(throughput, 2) if throughput > 0 else None
                    ),
                    eta_seconds=round(eta) if eta is not None else None,
                )
            )

        # ── 4. Mark completed ───────────────────────────────────
        duration = time.monotonic() - start_time
        stats = RunStats(
            total_symbols=total,
            processed_symbols=processed - failed,
            failed_symbols=failed,
            duration_seconds=round(duration, 2),
        )
        uow.feature_runs.mark_completed(run_id, stats)
        uow.commit()
        logger.info(
            "Run %d completed: %d processed, %d failed, %.1fs",
            run_id,
            processed,
            failed,
            duration,
        )

        # ── 5. DQ checks ───────────────────────────────────────
        dq_results = self._run_dq_checks(uow, run_id, cmd, total, all_rows)

        # ── 6. Publish or quarantine ────────────────────────────
        run = uow.feature_runs.get_run(run_id)
        decision = evaluate_publish_readiness(run.status, dq_results)
        dq_warnings = tuple(r.message for r in dq_results if not r.passed)

        if decision.allowed:
            uow.feature_runs.publish_atomically(run_id)
            uow.commit()
            logger.info("Run %d published", run_id)
            status = RunStatus.PUBLISHED.value
        else:
            uow.feature_runs.mark_quarantined(run_id, dq_results)
            uow.commit()
            logger.warning("Run %d quarantined: %s", run_id, decision.reason)
            status = RunStatus.QUARANTINED.value

        # ── 7. Completion progress ─────────────────────────────
        progress.emit(
            ProgressEvent(
                current=total,
                total=total,
                passed=len(all_rows),
                failed=failed,
            )
        )

        return BuildDailySnapshotResult(
            run_id=run_id,
            status=status,
            total_symbols=total,
            processed_symbols=processed,
            failed_symbols=failed,
            dq_passed=decision.allowed,
            warnings=dq_warnings,
        )

    def _run_dq_checks(
        self,
        uow: UnitOfWork,
        run_id: int,
        cmd: BuildDailySnapshotCommand,
        expected_count: int,
        rows: list[FeatureRowWrite],
    ) -> list[DQResult]:
        """Execute data quality checks against the completed run."""
        results: list[DQResult] = []

        # Row count: compare DB count to expected universe size
        actual_count = uow.feature_store.count_by_run_id(run_id)
        results.append(
            check_row_count(
                expected=expected_count,
                actual=actual_count,
                threshold=cmd.row_count_threshold,
            )
        )

        # Null rate on composite_score (using in-memory rows)
        nulls = sum(1 for r in rows if r.composite_score is None)
        results.append(
            check_null_rate(
                column_nulls=nulls,
                total=len(rows),
                max_rate=cmd.null_max_rate,
            )
        )

        # Score distribution (using in-memory rows)
        scores = [
            r.composite_score for r in rows if r.composite_score is not None
        ]
        results.append(
            check_score_distribution(
                scores=scores,
                expected_mean_range=cmd.score_mean_range,
            )
        )

        return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "BuildDailyFeatureSnapshotUseCase",
    "BuildDailySnapshotCommand",
    "BuildDailySnapshotResult",
]
