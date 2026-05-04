"""Bootstrap readiness evaluation for enabled Markets."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..domain.markets.catalog import get_market_catalog
from ..infra.db.models.feature_store import FeatureRun
from ..models.scan_result import SCAN_TRIGGER_SOURCE_AUTO, Scan
from ..models.stock import StockFundamental, StockPrice
from ..models.stock_universe import StockUniverse


@dataclass(frozen=True)
class MarketBootstrapReadiness:
    market: str
    core_ready: bool
    scan_ready: bool

    @property
    def ready(self) -> bool:
        return self.core_ready and self.scan_ready


@dataclass(frozen=True)
class BootstrapReadiness:
    empty_system: bool
    market_results: dict[str, MarketBootstrapReadiness]

    @property
    def ready(self) -> bool:
        return all(result.ready for result in self.market_results.values())

    @property
    def missing_markets(self) -> list[str]:
        return [
            market
            for market, result in self.market_results.items()
            if not result.ready
        ]


class BootstrapReadinessService:
    def normalize_market(self, market: str) -> str:
        return get_market_catalog().get(market).code

    def is_empty_system(self, db: Session) -> bool:
        return not (
            self._has_active_universe_rows(db)
            or self._has_price_rows(db)
            or self._has_fundamental_rows(db)
        )

    def has_core_market_data(self, db: Session, market: str) -> bool:
        return (
            self._has_active_universe_rows(db, market)
            and self._has_price_rows(db, market)
            and self._has_fundamental_rows(db, market)
        )

    def has_completed_auto_scan(self, db: Session, market: str) -> bool:
        normalized_market = self.normalize_market(market)
        return (
            db.query(Scan.id)
            .join(FeatureRun, FeatureRun.id == Scan.feature_run_id)
            .filter(
                Scan.universe_market == normalized_market,
                Scan.status == "completed",
                Scan.trigger_source == SCAN_TRIGGER_SOURCE_AUTO,
                FeatureRun.status == "published",
            )
            .limit(1)
            .first()
            is not None
        )

    def evaluate(
        self,
        db: Session,
        *,
        enabled_markets: list[str],
    ) -> BootstrapReadiness:
        normalized_markets = [
            self.normalize_market(market) for market in enabled_markets
        ]
        return BootstrapReadiness(
            empty_system=self.is_empty_system(db),
            market_results={
                market: MarketBootstrapReadiness(
                    market=market,
                    core_ready=self.has_core_market_data(db, market),
                    scan_ready=self.has_completed_auto_scan(db, market),
                )
                for market in normalized_markets
            },
        )

    def _has_active_universe_rows(
        self,
        db: Session,
        market: str | None = None,
    ) -> bool:
        query = db.query(StockUniverse.id).filter(StockUniverse.is_active.is_(True))
        if market is not None:
            query = query.filter(StockUniverse.market == self.normalize_market(market))
        return query.limit(1).first() is not None

    def _has_price_rows(
        self,
        db: Session,
        market: str | None = None,
    ) -> bool:
        query = (
            db.query(StockPrice.id)
            .join(StockUniverse, StockUniverse.symbol == StockPrice.symbol)
            .filter(StockUniverse.is_active.is_(True))
        )
        if market is not None:
            query = query.filter(StockUniverse.market == self.normalize_market(market))
        return query.limit(1).first() is not None

    def _has_fundamental_rows(
        self,
        db: Session,
        market: str | None = None,
    ) -> bool:
        query = (
            db.query(StockFundamental.id)
            .join(StockUniverse, StockUniverse.symbol == StockFundamental.symbol)
            .filter(StockUniverse.is_active.is_(True))
        )
        if market is not None:
            query = query.filter(StockUniverse.market == self.normalize_market(market))
        return query.limit(1).first() is not None
