"""Canonical recent group-rank history backfill used by static exports."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import Any, Callable, Protocol, TypedDict

from sqlalchemy.orm import Session

from app.models.industry import IBDGroupRank


DEFAULT_GROUP_RANK_HISTORY_LOOKBACK_DAYS = 100


class TradingDayRange(Protocol):
    def trading_days(self, market: str, start: date, end: date) -> list[date]: ...


class GroupRankGapFillStats(TypedDict, total=False):
    total_dates: int
    processed: int
    skipped: int
    errors: int
    duration_seconds: float
    error: str


class GroupRankGapFiller(Protocol):
    def fill_gaps_optimized(
        self,
        db: Session,
        dates: list[date],
        *,
        market: str,
    ) -> GroupRankGapFillStats: ...


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
    group_rank_service: GroupRankGapFiller
    lookback_days: int = DEFAULT_GROUP_RANK_HISTORY_LOOKBACK_DAYS

    def backfill(
        self,
        *,
        as_of_date: date,
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
                stats = self.group_rank_service.fill_gaps_optimized(
                    db,
                    missing_dates,
                    market=normalized_market,
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
            errors = int(stats.get("errors") or 0)
            error = str(stats["error"]) if stats.get("error") else None
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
                processed=int(stats.get("processed") or 0),
                skipped=int(stats.get("skipped") or 0),
                errors=errors,
                total_dates=int(stats.get("total_dates") or 0),
                duration_seconds=(
                    float(stats["duration_seconds"])
                    if stats.get("duration_seconds") is not None
                    else None
                ),
                error=error,
            )


__all__ = [
    "DEFAULT_GROUP_RANK_HISTORY_LOOKBACK_DAYS",
    "GroupRankHistoryBackfillResult",
    "GroupRankHistoryBackfillService",
    "GroupRankHistoryBackfillStatus",
]
