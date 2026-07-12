"""History providers for Relative Rotation Graph inputs."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Protocol, Sequence, Tuple


RRGHistoryResult = Tuple[
    str | None,
    dict[str, dict[str, Any]],
    dict[str, list[tuple[date, float, int]]],
]


class RRGHistoryProvider(Protocol):
    """Source RRG-ready group-ranking history for one market."""

    def get_all_groups_history(
        self,
        db: Any,
        *,
        market: str,
        days: int,
        as_of_date: date | None = None,
    ) -> RRGHistoryResult:
        """Return latest date, current ranking metadata, and daily RS series."""


class USGroupRankHistoryProvider:
    """Read US RRG history from persisted IBD group-rank rows."""

    def __init__(self, group_rank_service: Any) -> None:
        self._group_rank_service = group_rank_service

    def get_all_groups_history(
        self,
        db: Any,
        *,
        market: str,
        days: int,
        as_of_date: date | None = None,
    ) -> RRGHistoryResult:
        from datetime import date as _date

        from app.models.industry import IBDGroupRank

        current = self._group_rank_service.get_current_rankings(
            db,
            limit=197,
            market=market,
            calculation_date=as_of_date,
        )
        if not current:
            return None, {}, {}

        latest_date = current[0]["date"]
        meta = {row["industry_group"]: row for row in current}
        latest_day = _date.fromisoformat(latest_date)
        cutoff = latest_day - timedelta(days=days)
        rows = (
            db.query(
                IBDGroupRank.industry_group,
                IBDGroupRank.date,
                IBDGroupRank.avg_rs_rating,
                IBDGroupRank.num_stocks,
            )
            .filter(
                IBDGroupRank.market == market,
                IBDGroupRank.date >= cutoff,
                IBDGroupRank.date <= latest_day,
            )
            .order_by(IBDGroupRank.industry_group, IBDGroupRank.date)
            .all()
        )
        return latest_date, meta, _collect_group_series(rows)


class MarketDispatchRRGHistoryProvider:
    """Dispatch to the market-appropriate RRG history provider."""

    def __init__(
        self,
        *,
        us_provider: RRGHistoryProvider,
        non_us_provider: RRGHistoryProvider,
        us_market: str = "US",
    ) -> None:
        self._us_provider = us_provider
        self._non_us_provider = non_us_provider
        self._us_market = str(us_market or "").upper()

    def get_all_groups_history(
        self,
        db: Any,
        *,
        market: str,
        days: int,
        as_of_date: date | None = None,
    ) -> RRGHistoryResult:
        provider = (
            self._us_provider
            if str(market or "").upper() == self._us_market
            else self._non_us_provider
        )
        return provider.get_all_groups_history(
            db,
            market=market,
            days=days,
            as_of_date=as_of_date,
        )


def build_rrg_history_provider(
    *,
    group_rank_service: Any,
    market_group_ranking_service: Any,
) -> RRGHistoryProvider:
    return MarketDispatchRRGHistoryProvider(
        us_provider=USGroupRankHistoryProvider(group_rank_service),
        non_us_provider=market_group_ranking_service,
    )


def _collect_group_series(
    rows: Sequence[Tuple[str, date, float, int | None]],
) -> dict[str, list[tuple[date, float, int]]]:
    series: dict[str, list[tuple[date, float, int]]] = defaultdict(list)
    for group, d, rs, ns in rows:
        series[group].append((d, float(rs), int(ns or 0)))
    return dict(series)


__all__ = [
    "MarketDispatchRRGHistoryProvider",
    "RRGHistoryProvider",
    "RRGHistoryResult",
    "USGroupRankHistoryProvider",
    "build_rrg_history_provider",
]
