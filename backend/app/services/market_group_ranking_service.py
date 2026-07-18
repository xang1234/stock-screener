"""Compatibility adapter for market-aware stored Group rankings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy.orm import Session


@dataclass(frozen=True)
class GroupRankSnapshot:
    date: str | None
    ranks_by_group: dict[str, int]


class MarketGroupRankingService:
    """Delegate legacy market-aware calls to the canonical stored service."""

    def __init__(
        self,
        *,
        group_rank_service: Any | None = None,
    ) -> None:
        self._group_rank_service = group_rank_service

    def _stored_group_rank_service(self) -> Any:
        if self._group_rank_service is None:
            # Imported lazily to avoid a module cycle through runtime wiring.
            from app.wiring.bootstrap import get_group_rank_service

            self._group_rank_service = get_group_rank_service()
        return self._group_rank_service

    def get_current_rankings(
        self,
        db: Session,
        *,
        market: str,
        limit: int = 197,
        calculation_date: date | None = None,
        include_rank_changes: bool = True,
    ) -> list[dict[str, Any]]:
        del include_rank_changes  # Stored service owns versioned rank-change lookup.
        return self._stored_group_rank_service().get_current_rankings(
            db,
            limit=limit,
            calculation_date=calculation_date,
            market=str(market or "").strip().upper(),
        )

    def get_current_rank_map(
        self,
        db: Session,
        *,
        market: str,
        calculation_date: date | None = None,
    ) -> dict[str, int]:
        return self.get_current_rank_snapshot(
            db,
            market=market,
            calculation_date=calculation_date,
        ).ranks_by_group

    def get_current_rank_snapshot(
        self,
        db: Session,
        *,
        market: str,
        calculation_date: date | None = None,
    ) -> GroupRankSnapshot:
        rankings = self.get_current_rankings(
            db,
            market=market,
            limit=10_000,
            calculation_date=calculation_date,
            include_rank_changes=False,
        )
        ranking_date = next(
            (str(row["date"]) for row in rankings if row.get("date")),
            None,
        )
        return GroupRankSnapshot(
            date=ranking_date,
            ranks_by_group={
                str(row["industry_group"]): int(row["rank"])
                for row in rankings
                if row.get("industry_group") and row.get("rank") is not None
            },
        )

    def get_rank_movers(
        self,
        db: Session,
        *,
        market: str,
        period: str = "1w",
        limit: int = 20,
        calculation_date: date | None = None,
    ) -> dict[str, Any]:
        return self._stored_group_rank_service().get_rank_movers(
            db,
            period=period,
            limit=limit,
            calculation_date=calculation_date,
            market=str(market or "").strip().upper(),
        )

    def get_group_history(
        self,
        db: Session,
        *,
        market: str,
        industry_group: str,
        days: int = 180,
    ) -> dict[str, Any]:
        return self._stored_group_rank_service().get_group_history(
            db,
            industry_group,
            days=days,
            market=str(market or "").strip().upper(),
        )

_market_group_ranking_service: MarketGroupRankingService | None = None


def get_market_group_ranking_service() -> MarketGroupRankingService:
    global _market_group_ranking_service
    if _market_group_ranking_service is None:
        _market_group_ranking_service = MarketGroupRankingService()
    return _market_group_ranking_service


__all__ = [
    "MarketGroupRankingService",
    "get_market_group_ranking_service",
]
