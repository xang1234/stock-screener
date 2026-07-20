from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from sqlalchemy.orm import Session

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
    GroupSnapshotIdentity,
)


class GroupSnapshotStatus(StrEnum):
    EXISTING = "existing"
    PROCESSED = "processed"
    EMPTY = "empty"
    ERRORED = "errored"


@dataclass(frozen=True)
class GroupSnapshotResult:
    identity: GroupSnapshotIdentity
    status: GroupSnapshotStatus
    row_count: int
    market_rs_run_id: int | None
    reason_code: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class GroupBackfillReport:
    results: tuple[GroupSnapshotResult, ...]

    @property
    def processed(self) -> int:
        return sum(
            item.status is GroupSnapshotStatus.PROCESSED for item in self.results
        )

    @property
    def existing(self) -> int:
        return sum(item.status is GroupSnapshotStatus.EXISTING for item in self.results)

    @property
    def errors(self) -> int:
        return sum(item.status is GroupSnapshotStatus.ERRORED for item in self.results)


class GroupRankSnapshotCoordinator:
    def __init__(
        self,
        *,
        reader,
        market_rs_snapshot_service,
        canonical_group_service,
        legacy_group_service,
    ) -> None:
        self.reader = reader
        self.market_rs_snapshot_service = market_rs_snapshot_service
        self.canonical_group_service = canonical_group_service
        self.legacy_group_service = legacy_group_service

    def ensure_snapshot(
        self,
        db: Session,
        *,
        identity: GroupSnapshotIdentity,
    ) -> GroupSnapshotResult:
        existing = self.reader.load_exact(
            db,
            identity=identity,
            include_top_symbol_names=False,
        )
        if existing:
            return self._result(identity, GroupSnapshotStatus.EXISTING, existing)

        if identity.formula_version == BALANCED_RS_FORMULA_VERSION:
            run = self.market_rs_snapshot_service.calculate(
                db,
                market=identity.market,
                as_of_date=identity.as_of_date,
                formula_version=identity.formula_version,
            )
            self.canonical_group_service.calculate_and_store(
                db,
                market=identity.market,
                as_of_date=identity.as_of_date,
                formula_version=identity.formula_version,
            )
            rows = self.reader.load_exact(
                db,
                identity=identity,
                include_top_symbol_names=False,
            )
            if rows and {row.get("market_rs_run_id") for row in rows} != {run.id}:
                raise RuntimeError(
                    "Group snapshot does not reference the exact Market RS run"
                )
        elif identity.formula_version == LEGACY_RS_FORMULA_VERSION:
            self.legacy_group_service.calculate_group_rankings(
                db,
                identity.as_of_date,
                market=identity.market,
                formula_version=LEGACY_RS_FORMULA_VERSION,
            )
            rows = self.reader.load_exact(
                db,
                identity=identity,
                include_top_symbol_names=False,
            )
        else:
            raise ValueError(
                f"Unsupported Group RS formula: {identity.formula_version}"
            )

        status = GroupSnapshotStatus.PROCESSED if rows else GroupSnapshotStatus.EMPTY
        return self._result(identity, status, rows)

    def backfill(
        self,
        db: Session,
        *,
        identities: Iterable[GroupSnapshotIdentity],
        continue_on_error: bool,
    ) -> GroupBackfillReport:
        results: list[GroupSnapshotResult] = []
        for identity in sorted(identities, key=lambda item: item.as_of_date):
            try:
                results.append(self.ensure_snapshot(db, identity=identity))
            except Exception as exc:
                db.rollback()
                if not continue_on_error:
                    raise
                results.append(
                    GroupSnapshotResult(
                        identity=identity,
                        status=GroupSnapshotStatus.ERRORED,
                        row_count=0,
                        market_rs_run_id=None,
                        reason_code=type(exc).__name__,
                        error=str(exc),
                    )
                )
        return GroupBackfillReport(results=tuple(results))

    @staticmethod
    def _result(identity, status, rows) -> GroupSnapshotResult:
        run_ids = {row.get("market_rs_run_id") for row in rows}
        run_id = next(iter(run_ids)) if len(run_ids) == 1 else None
        return GroupSnapshotResult(
            identity=identity,
            status=status,
            row_count=len(rows),
            market_rs_run_id=run_id,
        )
