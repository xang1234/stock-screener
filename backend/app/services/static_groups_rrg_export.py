"""Static-site RRG payload builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sqlalchemy import inspect

from app.domain.markets.catalog import get_market_catalog
from app.models.industry import IBDGroupRank, IBDIndustryGroup
from app.models.stock_universe import StockUniverse
from app.services.market_taxonomy_service import get_market_taxonomy_service
from app.services.rrg_history_provider import PersistedGroupRankHistoryProvider
from app.services.rrg_service import RRGService
from app.wiring.bootstrap import get_group_rank_service


class StaticGroupsRRGUnavailableError(RuntimeError):
    """Raised when static RRG cannot be exported from the current DB."""

    def __init__(self, *, section: str, reason: str) -> None:
        self.section = section
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class StaticGroupsRRGPayloadBuilder:
    """Build the offline RRG bundle consumed by static group-ranking pages."""

    schema_version: str
    rrg_service: RRGService
    market_catalog: Any = field(default_factory=get_market_catalog)

    @classmethod
    def from_runtime_services(
        cls,
        *,
        schema_version: str,
    ) -> "StaticGroupsRRGPayloadBuilder":
        return cls(
            schema_version=schema_version,
            rrg_service=RRGService(
                history_provider=PersistedGroupRankHistoryProvider(
                    get_group_rank_service()
                ),
                taxonomy_service=get_market_taxonomy_service(),
            ),
        )

    def build(
        self,
        *,
        db: Any,
        generated_at: str,
        expected_as_of_date: date,
        market: str,
    ) -> dict[str, Any]:
        normalized_market = str(market or "US").strip().upper()
        self._preflight_tables(db, normalized_market)
        requested_scopes = self.market_catalog.rrg_scopes_for_market(normalized_market)
        if not requested_scopes:
            raise StaticGroupsRRGUnavailableError(
                section=f"{normalized_market} rrg",
                reason=f"RRG is not enabled for market {normalized_market}.",
            )

        scopes = self.rrg_service.get_rrg_scopes(
            db,
            market=normalized_market,
            scopes=requested_scopes,
            as_of_date=expected_as_of_date,
        )

        groups_rrg = scopes["groups"]
        available_scopes = [
            scope for scope in requested_scopes
            if scopes.get(scope, {}).get("groups")
        ]

        if not groups_rrg.get("groups"):
            raise StaticGroupsRRGUnavailableError(
                section=f"{normalized_market} rrg",
                reason=(
                    "No RRG data could be computed (group-rank history is too "
                    "short or absent for this market)."
                ),
            )

        expected_date = expected_as_of_date.isoformat()
        rrg_date = groups_rrg.get("date")
        if rrg_date != expected_date:
            raise StaticGroupsRRGUnavailableError(
                section=f"{normalized_market} rrg",
                reason=(
                    f"RRG data date {rrg_date or 'none'} does not match static "
                    f"export date {expected_date}."
                ),
            )

        return {
            "schema_version": self.schema_version,
            "generated_at": generated_at,
            "available": True,
            "market": normalized_market,
            "as_of_date": expected_date,
            "available_scopes": available_scopes,
            "payload": {scope: scopes[scope] for scope in requested_scopes},
        }

    def _preflight_tables(self, db: Any, market: str) -> None:
        missing_tables = _missing_required_tables(
            db,
            self._required_table_names(market),
        )
        if missing_tables:
            raise StaticGroupsRRGUnavailableError(
                section=f"{market} rrg",
                reason=(
                    "RRG source tables are unavailable for this export database: "
                    f"{', '.join(missing_tables)}."
                ),
            )

    def _required_table_names(self, market: str) -> tuple[str, ...]:
        table_names = {IBDGroupRank.__tablename__}
        if market == "US" and "sectors" in self.market_catalog.rrg_scopes_for_market(market):
            table_names.update(
                {
                    IBDIndustryGroup.__tablename__,
                    StockUniverse.__tablename__,
                }
            )
        return tuple(sorted(table_names))


def _missing_required_tables(db: Any, table_names: tuple[str, ...]) -> list[str]:
    bind = db.get_bind() if callable(getattr(db, "get_bind", None)) else db
    inspector = inspect(bind)
    return [
        table_name for table_name in table_names
        if not inspector.has_table(table_name)
    ]


__all__ = [
    "StaticGroupsRRGUnavailableError",
    "StaticGroupsRRGPayloadBuilder",
]
