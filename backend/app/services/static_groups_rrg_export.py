"""Static-site RRG payload builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import inspect

from app.domain.markets.catalog import MarketCatalog, get_market_catalog
from app.models.industry import IBDIndustryGroup
from app.models.stock_universe import StockUniverse
from app.services.rrg_service import RRGService
from app.services.static_rrg_history_bundle import (
    StaticRRGHistoryBundleError,
    StaticRRGHistoryBundleService,
    StaticRRGHistoryPreparation,
    StaticRRGHistoryState,
    StaticRRGHistoryUnavailableError,
    build_static_rrg_service,
)
from app.services.static_rrg_history_contract import normalize_static_rrg_market


class StaticGroupsRRGUnavailableError(RuntimeError):
    """Raised when static RRG cannot be exported for the requested snapshot."""

    def __init__(self, *, section: str, reason: str) -> None:
        self.section = section
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class StaticGroupsRRGPayloadBuilder:
    """Build the offline RRG bundle consumed by static group-ranking pages."""

    schema_version: str
    rrg_service: RRGService
    market_catalog: MarketCatalog = field(default_factory=get_market_catalog)

    def build(
        self,
        *,
        db: Any,
        generated_at: str,
        expected_as_of_date: date,
        market: str,
    ) -> dict[str, Any]:
        normalized_market = normalize_static_rrg_market(market)
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
        table_names: set[str] = set()
        if market == "US" and "sectors" in self.market_catalog.rrg_scopes_for_market(market):
            table_names.update(
                {
                    IBDIndustryGroup.__tablename__,
                    StockUniverse.__tablename__,
                }
            )
        return tuple(sorted(table_names))


class StaticGroupsRRGPayloadSource(Protocol):
    """Build the optional static RRG payload without exposing input storage."""

    def build(
        self,
        *,
        db: Any,
        generated_at: str,
        expected_as_of_date: date,
        market: str,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class StaticGroupsRRGDatabasePayloadSource:
    """Build static RRG directly from the current export database."""

    schema_version: str
    history_service: StaticRRGHistoryBundleService = field(
        default_factory=StaticRRGHistoryBundleService
    )

    def build(
        self,
        *,
        db: Any,
        generated_at: str,
        expected_as_of_date: date,
        market: str,
    ) -> dict[str, Any]:
        normalized_market = normalize_static_rrg_market(market)
        try:
            state = self.history_service.build(
                db,
                market=normalized_market,
                through_date=expected_as_of_date,
            )
        except (StaticRRGHistoryBundleError, StaticRRGHistoryUnavailableError) as exc:
            raise StaticGroupsRRGUnavailableError(
                section=f"{normalized_market} rrg",
                reason=str(exc),
            ) from exc

        return _build_payload_from_state(
            schema_version=self.schema_version,
            state=state,
            db=db,
            generated_at=generated_at,
            expected_as_of_date=expected_as_of_date,
            market=normalized_market,
        )


@dataclass
class StaticGroupsRRGRollingHistoryPayloadSource:
    """Prepare, serve, and persist one market's rolling RRG artifact."""

    schema_version: str
    market: str
    directory: Path
    history_service: StaticRRGHistoryBundleService = field(
        default_factory=StaticRRGHistoryBundleService
    )
    _preparation: StaticRRGHistoryPreparation | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._preparation.warnings if self._preparation is not None else ()

    def build(
        self,
        *,
        db: Any,
        generated_at: str,
        expected_as_of_date: date,
        market: str,
    ) -> dict[str, Any]:
        expected_market = normalize_static_rrg_market(self.market)
        requested_market = normalize_static_rrg_market(market)
        if requested_market != expected_market:
            raise ValueError(
                f"Rolling RRG source for {expected_market} cannot build {requested_market}."
            )

        preparation = self.history_service.prepare(
            db,
            market=expected_market,
            through_date=expected_as_of_date,
            directory=self.directory,
        )
        self._preparation = preparation
        if preparation.state is None:
            reason = (
                preparation.warnings[-1]
                if preparation.warnings
                else f"RRG is not enabled for market {expected_market}."
            )
            raise StaticGroupsRRGUnavailableError(
                section=f"{expected_market} rrg",
                reason=reason,
            )

        return _build_payload_from_state(
            schema_version=self.schema_version,
            state=preparation.state,
            db=db,
            generated_at=generated_at,
            expected_as_of_date=expected_as_of_date,
            market=expected_market,
        )

    def persist(self, *, exported_as_of_date: date) -> dict[str, Any] | None:
        if self._preparation is None:
            return None
        return self.history_service.persist(
            self._preparation,
            exported_as_of_date=exported_as_of_date,
        )


def _build_payload_from_state(
    *,
    schema_version: str,
    state: StaticRRGHistoryState,
    db: Any,
    generated_at: str,
    expected_as_of_date: date,
    market: str,
) -> dict[str, Any]:
    return StaticGroupsRRGPayloadBuilder(
        schema_version=schema_version,
        rrg_service=build_static_rrg_service(state),
    ).build(
        db=db,
        generated_at=generated_at,
        expected_as_of_date=expected_as_of_date,
        market=market,
    )


def _missing_required_tables(db: Any, table_names: tuple[str, ...]) -> list[str]:
    bind = db.get_bind() if callable(getattr(db, "get_bind", None)) else db
    inspector = inspect(bind)
    return [
        table_name for table_name in table_names
        if not inspector.has_table(table_name)
    ]


__all__ = [
    "StaticGroupsRRGDatabasePayloadSource",
    "StaticGroupsRRGUnavailableError",
    "StaticGroupsRRGPayloadBuilder",
    "StaticGroupsRRGPayloadSource",
    "StaticGroupsRRGRollingHistoryPayloadSource",
]
