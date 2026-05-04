"""Unit tests for Bootstrap readiness evaluation."""

from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.infra.db.models.feature_store import FeatureRun
import app.models.scan_result  # noqa: F401
import app.models.stock  # noqa: F401
import app.models.stock_universe  # noqa: F401
from app.models.scan_result import SCAN_TRIGGER_SOURCE_AUTO, SCAN_TRIGGER_SOURCE_MANUAL, Scan
from app.models.stock import StockFundamental, StockPrice
from app.models.stock_universe import StockUniverse
from app.services.bootstrap_readiness_service import BootstrapReadinessService


class FakeBootstrapReadinessService(BootstrapReadinessService):
    def __init__(
        self,
        *,
        core_ready: dict[str, bool],
        scan_ready: dict[str, bool],
        empty: bool = False,
    ) -> None:
        self.core_ready = core_ready
        self.scan_ready = scan_ready
        self.empty = empty

    def is_empty_system(self, db) -> bool:
        return self.empty

    def has_core_market_data(self, db, market: str) -> bool:
        return self.core_ready.get(market, False)

    def has_completed_auto_scan(self, db, market: str) -> bool:
        return self.scan_ready.get(market, False)


@pytest.fixture
def readiness_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def seed_core_market_data(db, *, symbol: str = "AAPL", market: str = "US", active: bool = True) -> None:
    db.add(
        StockUniverse(
            symbol=symbol,
            name=f"{symbol} Inc.",
            market=market,
            exchange="NYSE",
            currency="USD",
            timezone="America/New_York",
            is_active=active,
        )
    )
    db.add(
        StockPrice(
            symbol=symbol,
            date=date(2026, 5, 1),
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1_000_000,
        )
    )
    db.add(StockFundamental(symbol=symbol, market_cap=1_000_000_000))
    db.commit()


def seed_scan(
    db,
    *,
    market: str = "US",
    scan_status: str = "completed",
    trigger_source: str = SCAN_TRIGGER_SOURCE_AUTO,
    feature_status: str = "published",
) -> None:
    feature_run = FeatureRun(
        as_of_date=date(2026, 5, 1),
        run_type="daily_snapshot",
        status=feature_status,
    )
    db.add(feature_run)
    db.flush()
    db.add(
        Scan(
            scan_id=str(uuid4()),
            criteria={},
            universe="all",
            universe_market=market,
            status=scan_status,
            trigger_source=trigger_source,
            feature_run_id=feature_run.id,
        )
    )
    db.commit()


def test_readiness_requires_core_data_and_auto_scan_for_every_enabled_market() -> None:
    service = FakeBootstrapReadinessService(
        core_ready={"US": True, "HK": True},
        scan_ready={"US": True, "HK": False},
    )

    result = service.evaluate(object(), enabled_markets=["US", "HK"])

    assert result.ready is False
    assert result.missing_markets == ["HK"]
    assert result.market_results["HK"].core_ready is True
    assert result.market_results["HK"].scan_ready is False


def test_readiness_is_ready_when_every_enabled_market_is_complete() -> None:
    service = FakeBootstrapReadinessService(
        core_ready={"US": True, "HK": True},
        scan_ready={"US": True, "HK": True},
    )

    result = service.evaluate(object(), enabled_markets=["US", "HK"])

    assert result.ready is True
    assert result.missing_markets == []


def test_empty_system_is_reported_independently_from_market_readiness() -> None:
    service = FakeBootstrapReadinessService(core_ready={}, scan_ready={}, empty=True)

    result = service.evaluate(object(), enabled_markets=["US"])

    assert result.empty_system is True
    assert result.ready is False


def test_sql_service_reports_empty_system_without_rows(readiness_db) -> None:
    result = BootstrapReadinessService().evaluate(readiness_db, enabled_markets=["US"])

    assert result.empty_system is True
    assert result.ready is False
    assert result.missing_markets == ["US"]


def test_sql_service_reports_ready_with_core_data_and_published_auto_scan(readiness_db) -> None:
    seed_core_market_data(readiness_db)
    seed_scan(readiness_db)

    result = BootstrapReadinessService().evaluate(readiness_db, enabled_markets=["US"])

    assert result.empty_system is False
    assert result.ready is True
    assert result.missing_markets == []
    assert result.market_results["US"].core_ready is True
    assert result.market_results["US"].scan_ready is True


def test_sql_service_ignores_inactive_universe_rows_for_core_readiness(readiness_db) -> None:
    seed_core_market_data(readiness_db, active=False)
    seed_scan(readiness_db)

    result = BootstrapReadinessService().evaluate(readiness_db, enabled_markets=["US"])

    assert result.empty_system is True
    assert result.ready is False
    assert result.missing_markets == ["US"]
    assert result.market_results["US"].core_ready is False
    assert result.market_results["US"].scan_ready is True


@pytest.mark.parametrize(
    "scan_kwargs",
    [
        {"market": "HK"},
        {"scan_status": "running"},
        {"trigger_source": SCAN_TRIGGER_SOURCE_MANUAL},
        {"feature_status": "completed"},
    ],
)
def test_sql_service_requires_published_completed_auto_scan_for_market(
    readiness_db,
    scan_kwargs,
) -> None:
    seed_core_market_data(readiness_db)
    seed_scan(readiness_db, **scan_kwargs)

    result = BootstrapReadinessService().evaluate(readiness_db, enabled_markets=["US"])

    assert result.ready is False
    assert result.missing_markets == ["US"]
    assert result.market_results["US"].core_ready is True
    assert result.market_results["US"].scan_ready is False
