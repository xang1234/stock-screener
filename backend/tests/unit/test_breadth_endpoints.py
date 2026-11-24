from __future__ import annotations

from datetime import date, timedelta
from inspect import iscoroutinefunction
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
from app.models.breadth_contributor import (
    MarketBreadthContributor,
    MarketBreadthContributorSnapshot,
)
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
    tables = [
        MarketBreadth.__table__,
        MarketBreadthContributorSnapshot.__table__,
        MarketBreadthContributor.__table__,
    ]
    Base.metadata.create_all(engine, tables=tables)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine, tables=tables)
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
        calculation_revision=3,
    )


def _revision_3_values() -> dict:
    return {
        "advancing_count": 120,
        "declining_count": 80,
        "unchanged_count": 5,
        "new_high_52week_count": 12,
        "new_low_52week_count": 3,
        "t2108_count": 150,
        "t2108_pct": 75.0,
        "atr_10x_extension_count": 7,
        "broad_universe_count": 205,
        "advance_decline_eligible_count": 205,
        "stockbee_daily_eligible_count": 190,
        "stockbee_month_eligible_count": 180,
        "stockbee_34day_eligible_count": 175,
        "stockbee_quarter_eligible_count": 160,
        "t2108_eligible_count": 200,
        "high_low_52week_eligible_count": 150,
        "atr_extension_eligible_count": 195,
        "eligibility_signature": "a" * 64,
        "stockbee_eligibility_signature": "b" * 64,
        "calculation_revision": 3,
    }


def _add_contributor_snapshot(session, row: MarketBreadth):
    session.add(row)
    session.add(
        MarketBreadthContributorSnapshot(
            market=row.market,
            date=row.date,
            calculation_revision=3,
            schema_id="breadth-contributors-v1",
            contributors=[
                MarketBreadthContributor(
                    symbol="AAA",
                    company_name="Alpha",
                    ibd_industry_group="Semiconductors",
                    daily_change_pct=5.5,
                    signals_json={"up_4pct": 5.5},
                )
            ],
        )
    )
    row.stocks_up_4pct = 1
    row.stocks_down_4pct = 0
    row.stocks_up_25pct_quarter = 0
    row.stocks_down_25pct_quarter = 0
    row.stocks_up_25pct_month = 0
    row.stocks_down_25pct_month = 0
    row.stocks_up_50pct_month = 0
    row.stocks_down_50pct_month = 0
    row.stocks_up_13pct_34days = 0
    row.stocks_down_13pct_34days = 0
    row.atr_10x_extension_count = 0
    session.commit()


def test_market_query_description_excludes_unsupported_singapore_breadth():
    from app.api.v1 import breadth as breadth_module

    assert breadth_module.MARKET_QUERY_DESCRIPTION == (
        "Market code: US, HK, IN, JP, KR, TW, CN, CA, DE"
    )
    assert "SG" not in breadth_module.MARKET_QUERY_DESCRIPTION


def test_sync_contributor_queries_run_in_fastapi_threadpool():
    from app.api.v1 import breadth as breadth_module

    assert not iscoroutinefunction(
        breadth_module.get_breadth_contributor_index
    )
    assert not iscoroutinefunction(breadth_module.get_breadth_contributors)


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
    assert payload["broad_universe_count"] is None
    assert payload["calculation_revision"] == 3


@pytest.mark.asyncio
async def test_contributor_index_and_document_are_additive_endpoints(client, session):
    app.dependency_overrides[get_db] = _override_db(session)
    calculation_date = date(2026, 8, 28)
    _add_contributor_snapshot(
        session,
        _breadth_row("US", calculation_date, up=1, down=0),
    )

    index_response = await client.get(
        "/api/v1/breadth/contributors/index",
        params={"market": "US"},
    )
    document_response = await client.get(
        "/api/v1/breadth/contributors",
        params={"market": "US", "date": calculation_date.isoformat()},
    )

    assert index_response.status_code == 200
    assert index_response.json() == {
        "schema": "breadth-contributors-v1",
        "market": "US",
        "calculation_revision": 3,
        "dates": [calculation_date.isoformat()],
    }
    assert document_response.status_code == 200
    payload = document_response.json()
    assert payload["schema"] == "breadth-contributors-v1"
    assert payload["date"] == calculation_date.isoformat()
    assert payload["contributors"] == [
        {
            "symbol": "AAA",
            "company_name": "Alpha",
            "ibd_industry_group": "Semiconductors",
            "daily_change_pct": 5.5,
            "signals": {"up_4pct": 5.5},
        }
    ]

    openapi = app.openapi()
    for model_name in (
        "BreadthContributorIndexResponse",
        "BreadthContributorDocumentResponse",
    ):
        schema = openapi["components"]["schemas"][model_name]
        assert "schema" in schema["properties"]
        assert "schema_id" not in schema["properties"]
        assert "schema" in schema["required"]


@pytest.mark.asyncio
async def test_contributor_document_returns_404_or_409(client, session):
    app.dependency_overrides[get_db] = _override_db(session)
    calculation_date = date(2026, 8, 28)

    unavailable = await client.get(
        "/api/v1/breadth/contributors",
        params={"market": "US", "date": calculation_date.isoformat()},
    )
    assert unavailable.status_code == 404

    row = _breadth_row("US", calculation_date, up=1, down=0)
    _add_contributor_snapshot(session, row)
    row.stocks_up_4pct = 2
    session.commit()
    inconsistent = await client.get(
        "/api/v1/breadth/contributors",
        params={"market": "US", "date": calculation_date.isoformat()},
    )
    assert inconsistent.status_code == 409


@pytest.mark.asyncio
async def test_current_breadth_ignores_newer_legacy_row(client, session):
    app.dependency_overrides[get_db] = _override_db(session)
    corrected = _breadth_row("US", date(2026, 8, 20), up=12, down=3)
    stale = _breadth_row("US", date(2026, 8, 21), up=99, down=1)
    stale.calculation_revision = 2
    session.add_all([corrected, stale])
    session.commit()

    response = await client.get("/api/v1/breadth/current", params={"market": "US"})

    assert response.status_code == 200
    assert response.json()["date"] == "2026-08-20"
    assert response.json()["stocks_up_4pct"] == 12


@pytest.mark.asyncio
async def test_current_breadth_returns_not_found_when_only_legacy_rows_exist(
    client, session
):
    app.dependency_overrides[get_db] = _override_db(session)
    stale = _breadth_row("US", date(2026, 8, 21), up=99, down=1)
    stale.calculation_revision = 2
    session.add(stale)
    session.commit()

    response = await client.get("/api/v1/breadth/current", params={"market": "US"})

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_revision_3_fields_are_flat_and_legacy_fields_are_unchanged(client, session):
    app.dependency_overrides[get_db] = _override_db(session)
    row = _breadth_row("US", date(2026, 8, 21), up=10, down=4)
    for name, value in _revision_3_values().items():
        setattr(row, name, value)
    session.add(row)
    session.commit()

    response = await client.get("/api/v1/breadth/current", params={"market": "US"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["stocks_up_4pct"] == 10
    assert payload["total_stocks_scanned"] == 14
    assert payload["t2108_pct"] == pytest.approx(75.0)
    assert payload["stockbee_daily_eligible_count"] == 190
    assert payload["stockbee_eligibility_signature"] == "b" * 64
    assert payload["calculation_revision"] == 3
    assert "stockbee" not in payload
    assert "context" not in payload
    assert payload["above_sma200_pct"] is None
    assert payload["down_20pct_from_high_pct"] is None


@pytest.mark.asyncio
async def test_current_and_trend_serve_participation(client, session):
    app.dependency_overrides[get_db] = _override_db(session)
    row = _breadth_row("US", date.today(), up=10, down=4)
    for name, value in {
        **_revision_3_values(),
        "above_sma200_count": 68,
        "above_sma200_pct": 34.0,
        "sma200_eligible_count": 200,
        "down_20pct_from_high_count": 90,
        "down_20pct_from_high_pct": 45.0,
        "off_high_eligible_count": 200,
        "up_4pct_near_high_count": 4,
        "above_sma20_count": 120,
        "above_sma20_pct": 60.0,
        "sma20_eligible_count": 200,
        "above_sma50_count": 104,
        "above_sma50_pct": 52.0,
        "sma50_eligible_count": 200,
        "trend_template_entry_count": 7,
        "trend_template_entry_eligible_count": 180,
    }.items():
        setattr(row, name, value)
    session.add(row)
    session.commit()

    current = (
        await client.get("/api/v1/breadth/current", params={"market": "US"})
    ).json()
    trend = await client.get(
        "/api/v1/breadth/trend/above_sma200_pct",
        params={"days": 1, "market": "US"},
    )

    assert current["above_sma200_pct"] == pytest.approx(34.0)
    assert current["down_20pct_from_high_pct"] == pytest.approx(45.0)
    assert current["up_4pct_near_high_count"] == 4
    assert current["off_high_eligible_count"] == 200
    assert trend.status_code == 200
    assert trend.json()["data"][0]["value"] == pytest.approx(34.0)
    assert current["above_sma20_pct"] == pytest.approx(60.0)
    assert current["above_sma50_pct"] == pytest.approx(52.0)
    assert current["sma50_eligible_count"] == 200
    assert current["trend_template_entry_count"] == 7
    assert current["trend_template_entry_eligible_count"] == 180
    for indicator, value in (("above_sma20_pct", 60.0), ("trend_template_entry_count", 7)):
        series = await client.get(
            f"/api/v1/breadth/trend/{indicator}", params={"days": 1, "market": "US"}
        )
        assert series.json()["data"][0]["value"] == pytest.approx(value)


@pytest.mark.asyncio
async def test_trend_accepts_revision_3_indicators_and_denominators(client, session):
    app.dependency_overrides[get_db] = _override_db(session)
    row = _breadth_row("US", date.today(), up=10, down=4)
    for name, value in _revision_3_values().items():
        setattr(row, name, value)
    session.add(row)
    session.commit()

    t2108 = await client.get(
        "/api/v1/breadth/trend/t2108_pct",
        params={"days": 1, "market": "US"},
    )
    coverage = await client.get(
        "/api/v1/breadth/trend/t2108_eligible_count",
        params={"days": 1, "market": "US"},
    )

    assert t2108.status_code == 200
    assert t2108.json()["data"][0]["value"] == pytest.approx(75.0)
    assert coverage.status_code == 200
    assert coverage.json()["data"][0]["value"] == 200


@pytest.mark.asyncio
async def test_historical_trend_and_summary_filter_by_market(client, session):
    app.dependency_overrides[get_db] = _override_db(session)
    today = date.today()
    previous_day = today - timedelta(days=1)
    session.add_all([
        _breadth_row("US", previous_day, up=10, down=4),
        _breadth_row("HK", previous_day, up=20, down=10),
        _breadth_row("HK", today, up=30, down=5),
    ])
    session.commit()

    historical = await client.get(
        "/api/v1/breadth/historical",
        params={
            "start_date": previous_day.isoformat(),
            "end_date": today.isoformat(),
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
        {"date": previous_day.isoformat(), "value": 2.0},
        {"date": today.isoformat(), "value": 6.0},
    ]

    assert summary.status_code == 200
    assert summary.json() == {
        "market": "HK",
        "latest_date": today.isoformat(),
        "total_records": 2,
        "date_range_start": previous_day.isoformat(),
        "date_range_end": today.isoformat(),
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
