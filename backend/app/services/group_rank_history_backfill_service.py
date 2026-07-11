"""Canonical recent group-rank history backfill used by static exports."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Protocol

from sqlalchemy.orm import Session

from app.models.industry import IBDGroupRank


DEFAULT_GROUP_RANK_HISTORY_LOOKBACK_DAYS = 100


class TradingDayRange(Protocol):
    def trading_days(self, market: str, start: date, end: date) -> list[date]: ...


class GroupRankGapFiller(Protocol):
    def fill_gaps_optimized(
        self,
        db: Session,
        dates: list[date],
        *,
        market: str,
    ) -> dict[str, Any]: ...


SessionFactory = Callable[[], AbstractContextManager[Session]]


@dataclass(frozen=True)
class GroupRankHistoryBackfillService:
    """Fill missing recent group-rank dates for one market."""

    session_factory: SessionFactory
    calendar_service: TradingDayRange
    group_rank_service: GroupRankGapFiller
    lookback_days: int = DEFAULT_GROUP_RANK_HISTORY_LOOKBACK_DAYS

    def backfill(self, *, as_of_date: date, market: str = "US") -> dict[str, Any]:
        normalized_market = str(market or "US").strip().upper()
        start_date = as_of_date - timedelta(days=self.lookback_days)
        desired_dates = self.calendar_service.trading_days(
            normalized_market,
            start_date,
            as_of_date,
        )

        with self.session_factory() as db:
            if not hasattr(db, "query"):
                return _result(
                    status="skipped",
                    market=normalized_market,
                    as_of_date=as_of_date,
                    start_date=start_date,
                    reason="session_factory_stub",
                )
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
                return _result(
                    status="skipped",
                    market=normalized_market,
                    as_of_date=as_of_date,
                    start_date=start_date,
                )

            try:
                stats = self.group_rank_service.fill_gaps_optimized(
                    db,
                    missing_dates,
                    market=normalized_market,
                )
            except Exception as exc:
                return _result(
                    status="errored",
                    market=normalized_market,
                    as_of_date=as_of_date,
                    start_date=start_date,
                    missing_dates=len(missing_dates),
                    errors=len(missing_dates),
                    error=str(exc),
                )
            return {
                **stats,
                **_result(
                    status="completed",
                    market=normalized_market,
                    as_of_date=as_of_date,
                    start_date=start_date,
                    missing_dates=len(missing_dates),
                    processed=int(stats.get("processed") or 0),
                    errors=int(stats.get("errors") or 0),
                ),
            }


def _result(
    *,
    status: str,
    market: str,
    as_of_date: date,
    start_date: date,
    missing_dates: int = 0,
    processed: int = 0,
    errors: int = 0,
    reason: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "market": market,
        "as_of_date": as_of_date.isoformat(),
        "lookback_start_date": start_date.isoformat(),
        "missing_dates": missing_dates,
        "processed": processed,
        "errors": errors,
    }
    if reason is not None:
        result["reason"] = reason
    if error is not None:
        result["error"] = error
    return result


__all__ = [
    "DEFAULT_GROUP_RANK_HISTORY_LOOKBACK_DAYS",
    "GroupRankHistoryBackfillService",
]
