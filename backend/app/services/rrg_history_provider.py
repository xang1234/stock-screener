"""History providers for Relative Rotation Graph inputs."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Protocol, Sequence, Tuple

from app.domain.relative_strength import LEGACY_RS_FORMULA_VERSION

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


class StoredGroupRankHistoryProvider:
    """Read one active-formula Group history for every supported Market."""

    def __init__(self, group_rank_service: Any, market_rs_repository: Any) -> None:
        self._group_rank_service = group_rank_service
        self._market_rs_repository = market_rs_repository

    def get_all_groups_history(
        self,
        db: Any,
        *,
        market: str,
        days: int,
        as_of_date: date | None = None,
    ) -> RRGHistoryResult:
        from app.models.industry import IBDGroupRank

        normalized_market = str(market or "").strip().upper()
        formula_version = self._market_rs_repository.active_formula(
            db,
            market=normalized_market,
        )
        current = self._group_rank_service.get_current_rankings(
            db,
            limit=197,
            market=normalized_market,
            calculation_date=as_of_date,
            formula_version=formula_version,
        )
        if not current:
            return None, {}, {}

        latest_date = current[0]["date"]
        meta = {row["industry_group"]: row for row in current}
        cutoff = date.fromisoformat(latest_date) - timedelta(days=days)
        latest_day = date.fromisoformat(latest_date)
        rows = (
            db.query(
                IBDGroupRank.industry_group,
                IBDGroupRank.date,
                IBDGroupRank.avg_rs_rating,
                IBDGroupRank.num_stocks,
            )
            .filter(
                IBDGroupRank.market == normalized_market,
                IBDGroupRank.rs_formula_version == formula_version,
                IBDGroupRank.date >= cutoff,
                IBDGroupRank.date <= latest_day,
            )
            .order_by(IBDGroupRank.industry_group, IBDGroupRank.date)
            .all()
        )
        return latest_date, meta, _collect_group_series(rows)


class _LegacyFormulaRepository:
    @staticmethod
    def active_formula(_db: Any, *, market: str) -> str:  # noqa: ARG004
        return LEGACY_RS_FORMULA_VERSION


class USGroupRankHistoryProvider(StoredGroupRankHistoryProvider):
    """Backward-compatible legacy provider retained for focused unit tests."""

    def __init__(self, group_rank_service: Any) -> None:
        super().__init__(
            group_rank_service,
            getattr(
                group_rank_service,
                "market_rs_repository",
                _LegacyFormulaRepository(),
            ),
        )


def build_rrg_history_provider(
    *,
    group_rank_service: Any,
    market_rs_repository: Any | None = None,
) -> RRGHistoryProvider:
    repository = market_rs_repository or getattr(
        group_rank_service,
        "market_rs_repository",
        None,
    )
    if repository is None:
        raise ValueError("Market RS repository is required for RRG history")
    return StoredGroupRankHistoryProvider(group_rank_service, repository)


def _collect_group_series(
    rows: Sequence[Tuple[str, date, float, int | None]],
) -> dict[str, list[tuple[date, float, int]]]:
    series: dict[str, list[tuple[date, float, int]]] = defaultdict(list)
    for group, d, rs, ns in rows:
        series[group].append((d, float(rs), int(ns or 0)))
    return dict(series)


__all__ = [
    "RRGHistoryProvider",
    "RRGHistoryResult",
    "StoredGroupRankHistoryProvider",
    "USGroupRankHistoryProvider",
    "build_rrg_history_provider",
]
