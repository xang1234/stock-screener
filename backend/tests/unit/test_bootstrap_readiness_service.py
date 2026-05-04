"""Unit tests for Bootstrap readiness evaluation."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
import app.models.stock  # noqa: F401
import app.models.stock_universe  # noqa: F401
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


def test_sql_service_reports_empty_system_without_rows() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        result = BootstrapReadinessService().evaluate(db, enabled_markets=["US"])
    finally:
        db.close()
        engine.dispose()

    assert result.empty_system is True
    assert result.ready is False
    assert result.missing_markets == ["US"]
