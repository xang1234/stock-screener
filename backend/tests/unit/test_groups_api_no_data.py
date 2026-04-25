"""No-data contract tests for groups endpoints."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from app.main import app
from app.wiring.bootstrap import get_ui_snapshot_service


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_get_current_rankings_returns_404_when_no_rankings_exist(monkeypatch, client):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    class _FakeGroupRankService:
        def get_current_rankings(self, db, limit=197):  # noqa: ARG002
            return []

    monkeypatch.setattr(
        "app.api.v1.groups._get_group_rank_service",
        lambda: _FakeGroupRankService(),
    )

    response = await client.get("/api/v1/groups/rankings/current")

    assert response.status_code == 404
    assert response.json()["detail"] == "No ranking data available. Run a calculation first."


@pytest.mark.asyncio
async def test_get_groups_bootstrap_returns_404_when_snapshot_not_published(monkeypatch, client):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    class _FakeSnapshotService:
        def get_groups_bootstrap(self):
            return None

    app.dependency_overrides[get_ui_snapshot_service] = lambda: _FakeSnapshotService()
    try:
        response = await client.get("/api/v1/groups/bootstrap")
    finally:
        app.dependency_overrides.pop(get_ui_snapshot_service, None)

    assert response.status_code == 404
    assert response.json()["detail"] == "No published groups bootstrap snapshot is available"


@pytest.mark.asyncio
async def test_get_current_rankings_supports_non_us_market_scope(monkeypatch, client):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    class _FakeMarketGroupService:
        def get_current_rankings(self, db, *, market, limit=197, calculation_date=None):  # noqa: ARG002
            assert market == "HK"
            return [
                {
                    "industry_group": "Internet Services",
                    "date": "2026-04-18",
                    "rank": 3,
                    "avg_rs_rating": 82.1,
                    "median_rs_rating": 81.5,
                    "weighted_avg_rs_rating": 84.0,
                    "rs_std_dev": 3.2,
                    "num_stocks": 7,
                    "num_stocks_rs_above_80": 4,
                    "pct_rs_above_80": 57.14,
                    "top_symbol": "0700.HK",
                    "top_rs_rating": 97.0,
                    "rank_change_1w": 2,
                    "rank_change_1m": 4,
                    "rank_change_3m": None,
                    "rank_change_6m": None,
                }
            ]

    monkeypatch.setattr(
        "app.api.v1.groups._get_market_group_service",
        lambda: _FakeMarketGroupService(),
    )

    response = await client.get("/api/v1/groups/rankings/current", params={"market": "HK"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["market_scope"] == "HK"
    assert payload["rankings"][0]["industry_group"] == "Internet Services"


@pytest.mark.asyncio
async def test_get_current_rankings_supports_india_market_scope(monkeypatch, client):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    class _FakeMarketGroupService:
        def get_current_rankings(self, db, *, market, limit=197, calculation_date=None):  # noqa: ARG002
            assert market == "IN"
            return [
                {
                    "industry_group": "Banks",
                    "date": "2026-04-18",
                    "rank": 8,
                    "avg_rs_rating": 78.0,
                    "median_rs_rating": 77.5,
                    "weighted_avg_rs_rating": 79.0,
                    "rs_std_dev": 4.2,
                    "num_stocks": 12,
                    "num_stocks_rs_above_80": 5,
                    "pct_rs_above_80": 41.67,
                    "top_symbol": "HDFCBANK.NS",
                    "top_rs_rating": 94.0,
                    "rank_change_1w": -1,
                    "rank_change_1m": 3,
                    "rank_change_3m": None,
                    "rank_change_6m": None,
                }
            ]

    monkeypatch.setattr(
        "app.api.v1.groups._get_market_group_service",
        lambda: _FakeMarketGroupService(),
    )

    response = await client.get("/api/v1/groups/rankings/current", params={"market": "IN"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["market_scope"] == "IN"
    assert payload["rankings"][0]["top_symbol"] == "HDFCBANK.NS"


@pytest.mark.asyncio
async def test_get_rank_movers_supports_non_us_market_scope(monkeypatch, client):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    class _FakeMarketGroupService:
        def get_rank_movers(self, db, *, market, period="1w", limit=20, calculation_date=None):  # noqa: ARG002
            assert market == "JP"
            assert period == "1m"
            return {
                "period": "1m",
                "gainers": [
                    {
                        "industry_group": "Transportation Equipment",
                        "date": "2026-04-18",
                        "rank": 5,
                        "avg_rs_rating": 80.0,
                        "median_rs_rating": 79.0,
                        "weighted_avg_rs_rating": 81.0,
                        "rs_std_dev": 2.1,
                        "num_stocks": 6,
                        "num_stocks_rs_above_80": 3,
                        "pct_rs_above_80": 50.0,
                        "top_symbol": "7203.T",
                        "top_rs_rating": 95.0,
                        "rank_change_1w": 1,
                        "rank_change_1m": 6,
                        "rank_change_3m": None,
                        "rank_change_6m": None,
                    }
                ],
                "losers": [],
            }

    monkeypatch.setattr(
        "app.api.v1.groups._get_market_group_service",
        lambda: _FakeMarketGroupService(),
    )

    response = await client.get("/api/v1/groups/rankings/movers", params={"market": "JP", "period": "1m"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["market_scope"] == "JP"
    assert payload["gainers"][0]["industry_group"] == "Transportation Equipment"


@pytest.mark.asyncio
async def test_get_group_detail_supports_non_us_market_scope(monkeypatch, client):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    class _FakeMarketGroupService:
        def get_group_history(self, db, *, market, industry_group, days=180):  # noqa: ARG002
            assert market == "TW"
            assert industry_group == "Semiconductors"
            return {
                "industry_group": "Semiconductors",
                "current_rank": 1,
                "current_avg_rs": 91.2,
                "current_median_rs": 90.5,
                "current_weighted_avg_rs": 92.0,
                "current_rs_std_dev": 1.5,
                "num_stocks": 4,
                "pct_rs_above_80": 100.0,
                "top_symbol": "2330.TW",
                "top_rs_rating": 98.0,
                "rank_change_1w": 0,
                "rank_change_1m": 1,
                "rank_change_3m": None,
                "rank_change_6m": None,
                "history": [
                    {
                        "date": "2026-04-18",
                        "rank": 1,
                        "avg_rs_rating": 91.2,
                        "num_stocks": 4,
                    }
                ],
                "stocks": [],
            }

    monkeypatch.setattr(
        "app.api.v1.groups._get_market_group_service",
        lambda: _FakeMarketGroupService(),
    )

    response = await client.get(
        "/api/v1/groups/rankings/detail",
        params={"market": "TW", "group": "Semiconductors"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["market_scope"] == "TW"
    assert payload["industry_group"] == "Semiconductors"


@pytest.mark.asyncio
async def test_get_groups_bootstrap_supports_non_us_market_scope(monkeypatch, client):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    class _FakeSnapshotService:
        def get_groups_bootstrap(self):
            return None

    class _FakeMarketGroupService:
        def get_current_rankings(self, db, *, market, limit=197, calculation_date=None):  # noqa: ARG002
            assert market == "HK"
            return [
                {
                    "industry_group": "Internet Services",
                    "date": "2026-04-18",
                    "rank": 3,
                    "avg_rs_rating": 82.1,
                    "median_rs_rating": 81.5,
                    "weighted_avg_rs_rating": 84.0,
                    "rs_std_dev": 3.2,
                    "num_stocks": 7,
                    "num_stocks_rs_above_80": 4,
                    "pct_rs_above_80": 57.14,
                    "top_symbol": "0700.HK",
                    "top_rs_rating": 97.0,
                    "rank_change_1w": 2,
                    "rank_change_1m": 4,
                    "rank_change_3m": None,
                    "rank_change_6m": None,
                }
            ]

        def get_rank_movers(self, db, *, market, period="1w", limit=10, calculation_date=None):  # noqa: ARG002
            assert market == "HK"
            return {"period": "1w", "gainers": [], "losers": []}

    monkeypatch.setattr(
        "app.api.v1.groups._get_market_group_service",
        lambda: _FakeMarketGroupService(),
    )
    app.dependency_overrides[get_ui_snapshot_service] = lambda: _FakeSnapshotService()
    try:
        response = await client.get("/api/v1/groups/bootstrap", params={"market": "HK"})
    finally:
        app.dependency_overrides.pop(get_ui_snapshot_service, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload["snapshot_revision"].startswith("groups:HK:")
    assert payload["payload"]["rankings"]["market_scope"] == "HK"
