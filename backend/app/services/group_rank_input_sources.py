"""Source ports and default adapters for group-ranking inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from sqlalchemy.orm import Session

from ..models.stock_universe import StockUniverse
from .ibd_industry_service import IBDIndustryService
from .stock_universe_service import StockUniverseService


class GroupRankUniverseSource(Protocol):
    def active_symbols(
        self,
        db: Session,
        market: str,
    ) -> frozenset[str]: ...

    def symbol_names(
        self,
        db: Session,
        symbols: Sequence[str],
    ) -> dict[str, str | None]: ...


class GroupRankTaxonomySource(Protocol):
    def groups(
        self,
        db: Session,
        market: str,
    ) -> tuple[str, ...]: ...

    def symbols_for_group(
        self,
        db: Session,
        group: str,
        market: str,
    ) -> tuple[str, ...]: ...


class GroupRankMarketCapSource(Protocol):
    def market_caps(
        self,
        db: Session,
        symbols: Sequence[str],
    ) -> dict[str, float]: ...


@dataclass(frozen=True)
class StockUniverseGroupRankSource:
    service: StockUniverseService

    def active_symbols(
        self,
        db: Session,
        market: str,
    ) -> frozenset[str]:
        return frozenset(
            self.service.get_active_symbols(db, market=market)
        )

    def symbol_names(
        self,
        db: Session,
        symbols: Sequence[str],
    ) -> dict[str, str | None]:
        if not symbols:
            return {}
        return dict(
            db.query(
                StockUniverse.symbol,
                StockUniverse.name,
            )
            .filter(StockUniverse.symbol.in_(symbols))
            .all()
        )


@dataclass(frozen=True)
class IBDIndustryTaxonomySource:
    def groups(
        self,
        db: Session,
        market: str,
    ) -> tuple[str, ...]:
        return tuple(
            IBDIndustryService.get_all_groups(db, market=market)
        )

    def symbols_for_group(
        self,
        db: Session,
        group: str,
        market: str,
    ) -> tuple[str, ...]:
        return tuple(
            IBDIndustryService.get_group_symbols(
                db,
                group,
                market=market,
            )
        )


@dataclass(frozen=True)
class SqlGroupRankMarketCapSource:
    def market_caps(
        self,
        db: Session,
        symbols: Sequence[str],
    ) -> dict[str, float]:
        if not symbols:
            return {}
        rows = db.query(
            StockUniverse.symbol,
            StockUniverse.market_cap,
        ).filter(StockUniverse.symbol.in_(symbols)).all()
        return {
            symbol: market_cap
            for symbol, market_cap in rows
            if market_cap and market_cap > 0
        }
