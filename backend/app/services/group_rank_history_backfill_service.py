"""Canonical recent group-rank history backfill used by static exports."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import Any, Callable, Protocol

from sqlalchemy.orm import Session

from app.models.industry import IBDGroupRank
from app.domain.relative_strength import GroupSnapshotIdentity
from app.services.group_rank_snapshot_coordinator import GroupSnapshotStatus


DEFAULT_GROUP_RANK_HISTORY_LOOKBACK_DAYS = 100


class TradingDayRange(Protocol):
    def trading_days(self, market: str, start: date, end: date) -> list[date]: ...


SessionFactory = Callable[[], AbstractContextManager[Session]]


class GroupRankHistoryBackfillStatus(StrEnum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    ERRORED = "errored"


@dataclass(frozen=True)
class GroupRankHistoryBackfillResult:
    status: GroupRankHistoryBackfillStatus
    market: str
    as_of_date: date
    lookback_start_date: date
    missing_dates: int = 0
    processed: int = 0
    skipped: int = 0
    errors: int = 0
    total_dates: int = 0
    duration_seconds: float | None = None
    reason: str | None = None
    error: str | None = None

    @property
    def ready_for_enrichment(self) -> bool:
        return self.status in {
            GroupRankHistoryBackfillStatus.COMPLETED,
            GroupRankHistoryBackfillStatus.SKIPPED,
        }

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status.value,
            "market": self.market,
            "as_of_date": self.as_of_date.isoformat(),
            "lookback_start_date": self.lookback_start_date.isoformat(),
            "missing_dates": self.missing_dates,
            "processed": self.processed,
            "errors": self.errors,
        }
        if self.total_dates:
            result["total_dates"] = self.total_dates
        if self.skipped:
            result["skipped"] = self.skipped
        if self.duration_seconds is not None:
            result["duration_seconds"] = self.duration_seconds
        if self.reason is not None:
            result["reason"] = self.reason
        if self.error is not None:
            result["error"] = self.error
        return result


@dataclass(frozen=True)
class GroupRankHistoryBackfillService:
    """Fill missing recent group-rank dates for one market."""

    session_factory: SessionFactory
    calendar_service: TradingDayRange
    group_snapshot_coordinator: Any
    lookback_days: int = DEFAULT_GROUP_RANK_HISTORY_LOOKBACK_DAYS

    def backfill(
        self,
        *,
        as_of_date: date,
        formula_version: str,
        market: str = "US",
    ) -> GroupRankHistoryBackfillResult:
        normalized_market = str(market or "US").strip().upper()
        start_date = as_of_date - timedelta(days=self.lookback_days)
        desired_dates = self.calendar_service.trading_days(
            normalized_market,
            start_date,
            as_of_date,
        )

        with self.session_factory() as db:
            existing_dates = {
                record_date
                for record_date, in db.query(IBDGroupRank.date)
                .filter(
                    IBDGroupRank.date >= start_date,
                    IBDGroupRank.date <= as_of_date,
                    IBDGroupRank.market == normalized_market,
                    IBDGroupRank.rs_formula_version == formula_version,
                )
                .distinct()
                .all()
            }
            missing_dates = [
                calculation_date
                for calculation_date in desired_dates
                if calculation_date not in existing_dates
            ]
            if not missing_dates:
                return GroupRankHistoryBackfillResult(
                    status=GroupRankHistoryBackfillStatus.SKIPPED,
                    market=normalized_market,
                    as_of_date=as_of_date,
                    lookback_start_date=start_date,
                )

            try:
                report = self.group_snapshot_coordinator.backfill(
                    db,
                    identities=tuple(
                        GroupSnapshotIdentity(
                            normalized_market,
                            calculation_date,
                            formula_version,
                        )
                        for calculation_date in missing_dates
                    ),
                    continue_on_error=True,
                )
            except Exception as exc:
                return GroupRankHistoryBackfillResult(
                    status=GroupRankHistoryBackfillStatus.ERRORED,
                    market=normalized_market,
                    as_of_date=as_of_date,
                    lookback_start_date=start_date,
                    missing_dates=len(missing_dates),
                    errors=len(missing_dates),
                    error=str(exc),
                )
            errors = report.errors
            error_messages = [item.error for item in report.results if item.error]
            error = "; ".join(error_messages) if error_messages else None
            skipped = sum(
                item.status
                in {GroupSnapshotStatus.EXISTING, GroupSnapshotStatus.EMPTY}
                for item in report.results
            )
            return GroupRankHistoryBackfillResult(
                status=(
                    GroupRankHistoryBackfillStatus.ERRORED
                    if errors or error
                    else GroupRankHistoryBackfillStatus.COMPLETED
                ),
                market=normalized_market,
                as_of_date=as_of_date,
                lookback_start_date=start_date,
                missing_dates=len(missing_dates),
                processed=report.processed + report.existing,
                skipped=skipped,
                errors=errors,
                total_dates=len(report.results),
                error=error,
            )


__all__ = [
    "DEFAULT_GROUP_RANK_HISTORY_LOOKBACK_DAYS",
    "GroupRankHistoryBackfillResult",
    "GroupRankHistoryBackfillService",
    "GroupRankHistoryBackfillStatus",
]
