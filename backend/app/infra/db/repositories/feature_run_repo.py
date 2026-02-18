"""SQLAlchemy implementation of FeatureRunRepository."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timezone

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.domain.common.errors import EntityNotFoundError
from app.domain.feature_store.models import (
    FeatureRunDomain,
    RunStats,
    RunStatus,
    RunType,
    validate_transition,
)
from app.domain.feature_store.ports import FeatureRunRepository
from app.infra.db.models.feature_store import StockFeatureDaily
from app.domain.feature_store.quality import DQResult
from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer


class SqlFeatureRunRepository(FeatureRunRepository):
    """Persist and retrieve feature run lifecycle records via SQLAlchemy."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def start_run(
        self,
        as_of_date,
        run_type,
        code_version=None,
        universe_hash=None,
        input_hash=None,
        correlation_id=None,
    ) -> FeatureRunDomain:
        row = FeatureRun(
            as_of_date=as_of_date,
            run_type=run_type.value if isinstance(run_type, RunType) else run_type,
            status=RunStatus.RUNNING.value,
            code_version=code_version,
            universe_hash=universe_hash,
            input_hash=input_hash,
            correlation_id=correlation_id,
        )
        self._session.add(row)
        self._session.flush()
        return self._to_domain(row)

    def mark_completed(
        self,
        run_id,
        stats,
        warnings=(),
    ) -> FeatureRunDomain:
        row = self._get_or_raise(run_id)
        validate_transition(RunStatus(row.status), RunStatus.COMPLETED)

        row.status = RunStatus.COMPLETED.value
        row.completed_at = datetime.now(timezone.utc)
        row.stats_json = {
            "total_symbols": stats.total_symbols,
            "processed_symbols": stats.processed_symbols,
            "failed_symbols": stats.failed_symbols,
            "duration_seconds": stats.duration_seconds,
        }
        row.warnings_json = list(warnings)
        self._session.flush()
        return self._to_domain(row)

    def mark_quarantined(
        self,
        run_id,
        dq_results,
    ) -> FeatureRunDomain:
        row = self._get_or_raise(run_id)
        validate_transition(RunStatus(row.status), RunStatus.QUARANTINED)

        row.status = RunStatus.QUARANTINED.value
        row.warnings_json = [
            {
                "check_name": r.check_name,
                "passed": r.passed,
                "severity": r.severity.value,
                "actual_value": r.actual_value,
                "threshold": r.threshold,
                "message": r.message,
            }
            for r in dq_results
        ]
        self._session.flush()
        return self._to_domain(row)

    def publish_atomically(self, run_id) -> FeatureRunDomain:
        row = self._get_or_raise(run_id)
        validate_transition(RunStatus(row.status), RunStatus.PUBLISHED)

        # Update run status
        row.status = RunStatus.PUBLISHED.value
        row.published_at = datetime.now(timezone.utc)

        # Upsert the pointer in the same flush
        pointer = (
            self._session.query(FeatureRunPointer)
            .filter(FeatureRunPointer.key == "latest_published")
            .first()
        )
        if pointer is None:
            pointer = FeatureRunPointer(key="latest_published", run_id=run_id)
            self._session.add(pointer)
        else:
            pointer.run_id = run_id

        self._session.flush()
        return self._to_domain(row)

    def get_latest_published(self) -> FeatureRunDomain | None:
        pointer = (
            self._session.query(FeatureRunPointer)
            .filter(FeatureRunPointer.key == "latest_published")
            .first()
        )
        if pointer is None:
            return None

        row = self._session.get(FeatureRun, pointer.run_id)
        if row is None:
            return None
        return self._to_domain(row)

    def get_run(self, run_id) -> FeatureRunDomain:
        row = self._get_or_raise(run_id)
        return self._to_domain(row)

    def list_runs_with_counts(
        self,
        *,
        status: RunStatus | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        limit: int = 50,
    ) -> Sequence[tuple[FeatureRunDomain, int, bool]]:
        # Subquery: latest_published pointer
        pointer_sq = (
            self._session.query(FeatureRunPointer.run_id)
            .filter(FeatureRunPointer.key == "latest_published")
            .scalar_subquery()
        )

        q = (
            self._session.query(
                FeatureRun,
                func.coalesce(
                    self._session.query(func.count(StockFeatureDaily.symbol))
                    .filter(StockFeatureDaily.run_id == FeatureRun.id)
                    .correlate(FeatureRun)
                    .scalar_subquery(),
                    0,
                ).label("row_count"),
                case(
                    (FeatureRun.id == pointer_sq, True),
                    else_=False,
                ).label("is_latest"),
            )
        )

        if status is not None:
            q = q.filter(FeatureRun.status == status.value)
        if date_from is not None:
            q = q.filter(FeatureRun.as_of_date >= date_from)
        if date_to is not None:
            q = q.filter(FeatureRun.as_of_date <= date_to)

        q = q.order_by(FeatureRun.created_at.desc()).limit(limit)

        results: list[tuple[FeatureRunDomain, int, bool]] = []
        for row, count, is_latest in q.all():
            results.append((self._to_domain(row), count, bool(is_latest)))
        return results

    # -- Private helpers ---------------------------------------------------

    def _get_or_raise(self, run_id: int) -> FeatureRun:
        """Fetch by PK or raise EntityNotFoundError."""
        row = self._session.get(FeatureRun, run_id)
        if row is None:
            raise EntityNotFoundError("FeatureRun", run_id)
        return row

    @staticmethod
    def _to_domain(row: FeatureRun) -> FeatureRunDomain:
        """Map ORM model to domain value object."""
        stats = None
        if row.stats_json:
            stats = RunStats(
                total_symbols=row.stats_json["total_symbols"],
                processed_symbols=row.stats_json["processed_symbols"],
                failed_symbols=row.stats_json["failed_symbols"],
                duration_seconds=row.stats_json["duration_seconds"],
            )

        warnings: tuple[str, ...] = ()
        if row.warnings_json and isinstance(row.warnings_json, list):
            # warnings_json may contain plain strings (from mark_completed)
            # or dicts (from mark_quarantined); extract strings only
            warnings = tuple(
                w if isinstance(w, str) else w.get("message", "")
                for w in row.warnings_json
            )

        return FeatureRunDomain(
            id=row.id,
            as_of_date=row.as_of_date,
            run_type=RunType(row.run_type),
            status=RunStatus(row.status),
            created_at=row.created_at,
            completed_at=row.completed_at,
            published_at=row.published_at,
            correlation_id=row.correlation_id,
            code_version=row.code_version,
            universe_hash=row.universe_hash,
            input_hash=row.input_hash,
            stats=stats,
            warnings=warnings,
        )
