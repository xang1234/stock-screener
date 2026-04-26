from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models.market_breadth import MarketBreadth
from app.services import server_auth


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine, tables=[MarketBreadth.__table__])
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine, tables=[MarketBreadth.__table__])
        app.dependency_overrides.clear()


def _override_db(session):
    def _yield_db():
        try:
            yield session
        finally:
            pass

    return _yield_db


def _breadth_row(market: str, day: date, *, up: int, down: int) -> MarketBreadth:
    return MarketBreadth(
        market=market,
        date=day,
        stocks_up_4pct=up,
        stocks_down_4pct=down,
        ratio_5day=up / max(down, 1),
        ratio_10day=(up + 1) / max(down, 1),
        stocks_up_25pct_quarter=up + 10,
        stocks_down_25pct_quarter=down + 10,
        stocks_up_25pct_month=up + 5,
        stocks_down_25pct_month=down + 5,
        stocks_up_50pct_month=up + 2,
        stocks_down_50pct_month=down + 2,
        stocks_up_13pct_34days=up + 3,
        stocks_down_13pct_34days=down + 3,
        total_stocks_scanned=up + down,
        calculation_duration_seconds=1.25,
    )


@pytest.mark.asyncio
async def test_current_breadth_filters_by_market(client, session):
    app.dependency_overrides[get_db] = _override_db(session)
    session.add_all([
        _breadth_row("US", date(2026, 4, 24), up=10, down=4),
        _breadth_row("HK", date(2026, 4, 24), up=22, down=8),
    ])
    session.commit()

    response = await client.get("/api/v1/breadth/current", params={"market": "HK"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["market"] == "HK"
    assert payload["stocks_up_4pct"] == 22
    assert payload["stocks_down_4pct"] == 8


@pytest.mark.asyncio
async def test_historical_trend_and_summary_filter_by_market(client, session):
    app.dependency_overrides[get_db] = _override_db(session)
    session.add_all([
        _breadth_row("US", date(2026, 4, 23), up=10, down=4),
        _breadth_row("HK", date(2026, 4, 23), up=20, down=10),
        _breadth_row("HK", date(2026, 4, 24), up=30, down=5),
    ])
    session.commit()

    historical = await client.get(
        "/api/v1/breadth/historical",
        params={
            "start_date": "2026-04-23",
            "end_date": "2026-04-24",
            "market": "HK",
        },
    )
    trend = await client.get(
        "/api/v1/breadth/trend/ratio_5day",
        params={"days": 10, "market": "HK"},
    )
    summary = await client.get("/api/v1/breadth/summary", params={"market": "HK"})

    assert historical.status_code == 200
    assert [row["market"] for row in historical.json()] == ["HK", "HK"]
    assert [row["stocks_up_4pct"] for row in historical.json()] == [30, 20]

    assert trend.status_code == 200
    assert trend.json()["data"] == [
        {"date": "2026-04-23", "value": 2.0},
        {"date": "2026-04-24", "value": 6.0},
    ]

    assert summary.status_code == 200
    assert summary.json() == {
        "market": "HK",
        "latest_date": "2026-04-24",
        "total_records": 2,
        "date_range_start": "2026-04-23",
        "date_range_end": "2026-04-24",
    }


@pytest.mark.asyncio
async def test_manual_calculation_and_backfill_pass_market_to_tasks(client, monkeypatch):
    from app.api.v1 import breadth as breadth_module
    from app.tasks import breadth_tasks

    monkeypatch.setattr(breadth_module.settings, "feature_tasks", True)

    calculation_calls = []

    def fake_calculate(calculation_date=None, market="US"):
        calculation_calls.append((calculation_date, market))

    monkeypatch.setattr(breadth_tasks, "calculate_daily_breadth", fake_calculate)

    backfill_delay = MagicMock(return_value=SimpleNamespace(id="backfill-hk"))
    monkeypatch.setattr(breadth_tasks.backfill_breadth_data, "delay", backfill_delay)

    calculate_response = await client.post(
        "/api/v1/breadth/calculate",
        json={"calculation_date": "2026-04-24", "market": "HK"},
    )
    backfill_response = await client.post(
        "/api/v1/breadth/backfill",
        json={"start_date": "2026-04-23", "end_date": "2026-04-24", "market": "HK"},
    )

    assert calculate_response.status_code == 200
    assert calculation_calls == [("2026-04-24", "HK")]
    assert backfill_response.status_code == 200
    backfill_delay.assert_called_once_with("2026-04-23", "2026-04-24", market="HK")
