"""History providers for Relative Rotation Graph inputs."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Protocol, Sequence, Tuple

from app.domain.relative_strength import (
    LEGACY_RS_FORMULA_VERSION,
    GroupSnapshotIdentity,
)
from app.services.group_rank_snapshot_reader import GroupRankSnapshotReader

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

    def __init__(
        self,
        group_rank_service: Any,
        market_rs_repository: Any,
        snapshot_reader: GroupRankSnapshotReader | None = None,
    ) -> None:
        self._market_rs_repository = market_rs_repository
        self._snapshot_reader = snapshot_reader or GroupRankSnapshotReader()

    def get_all_groups_history(
        self,
        db: Any,
        *,
        market: str,
        days: int,
        as_of_date: date | None = None,
    ) -> RRGHistoryResult:
        normalized_market = str(market or "").strip().upper()
        formula_version = self._market_rs_repository.active_formula(
            db,
            market=normalized_market,
        )
        through_date = as_of_date or date.max
        dates = self._snapshot_reader.available_dates(
            db,
            market=normalized_market,
            formula_version=formula_version,
            through_date=through_date,
        )
        if not dates:
            return None, {}, {}
        latest_day = dates[-1]
        cutoff = latest_day - timedelta(days=days)
        selected_dates = tuple(item for item in dates if cutoff <= item <= latest_day)
        snapshots = [
            (
                snapshot_date,
                self._snapshot_reader.load_exact(
                    db,
                    identity=GroupSnapshotIdentity(
                        normalized_market,
                        snapshot_date,
                        formula_version,
                    ),
                    include_top_symbol_names=False,
                ),
            )
            for snapshot_date in selected_dates
        ]
        current = snapshots[-1][1]
        latest_date = latest_day.isoformat()
        meta = {row["industry_group"]: row for row in current}
        rows = [
            (
                str(row["industry_group"]),
                snapshot_date,
                float(row["avg_rs_rating"]),
                int(row.get("num_stocks") or 0),
            )
            for snapshot_date, snapshot in snapshots
            for row in snapshot
        ]
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
    snapshot_reader: GroupRankSnapshotReader | None = None,
) -> RRGHistoryProvider:
    repository = market_rs_repository or getattr(
        group_rank_service,
        "market_rs_repository",
        None,
    )
    if repository is None:
        raise ValueError("Market RS repository is required for RRG history")
    return StoredGroupRankHistoryProvider(
        group_rank_service,
        repository,
        snapshot_reader=snapshot_reader,
    )


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
