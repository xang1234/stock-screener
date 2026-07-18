"""No-data contract tests for groups endpoints."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import pytest_asyncio

from app.main import app
from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)
from app.infra.db.models.relative_strength import MarketRsRun
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
        def get_current_rankings(self, db, limit=197, calculation_date=None, *, market="US"):  # noqa: ARG002
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

    class _FakeGroupRankService:
        def get_current_rankings(
            self,
            db,
            limit=197,
            calculation_date=None,
            *,
            market="US",
        ):  # noqa: ARG002
            return []

        def get_rank_movers(
            self,
            db,
            period="1w",
            limit=20,
            calculation_date=None,
            *,
            market="US",
        ):  # noqa: ARG002
            return {"period": period, "gainers": [], "losers": []}

    monkeypatch.setattr(
        "app.api.v1.groups._get_group_rank_service",
        lambda: _FakeGroupRankService(),
    )

    app.dependency_overrides[get_ui_snapshot_service] = lambda: _FakeSnapshotService()
    try:
        response = await client.get("/api/v1/groups/bootstrap")
    finally:
        app.dependency_overrides.pop(get_ui_snapshot_service, None)

    assert response.status_code == 404
    assert response.json()["detail"] == "No published groups bootstrap snapshot is available"


@pytest.mark.asyncio
async def test_get_current_rankings_supports_non_us_market_scope(
    monkeypatch, client, db_session
):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    run = MarketRsRun(
        market="HK",
        as_of_date=date(2026, 4, 18),
        formula_version=BALANCED_RS_FORMULA_VERSION,
        status="completed",
        benchmark_symbol="^HSI",
        benchmark_as_of_date=date(2026, 4, 18),
        universe_hash="hk-api-test",
        expected_symbol_count=5000,
        eligible_symbol_count=5000,
        excluded_symbol_count=0,
        diagnostics_json={},
    )
    db_session.add(run)
    db_session.commit()

    class _FakeGroupRankService:
        def get_current_rankings(self, db, limit=197, calculation_date=None, *, market="US"):  # noqa: ARG002
            assert market == "HK"
            return [
                {
                    "industry_group": "Internet Services",
                    "date": "2026-04-18",
                    "rank": 3,
                    "avg_rs_rating": 82.1,
                    "avg_rs_rating_1m": 41.5,
                    "avg_rs_rating_3m": 63.2,
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
                    "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
                    "market_rs_run_id": run.id,
                }
            ]

    monkeypatch.setattr(
        "app.api.v1.groups._get_group_rank_service",
        lambda: _FakeGroupRankService(),
    )

    response = await client.get("/api/v1/groups/rankings/current", params={"market": "HK"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["market_scope"] == "HK"
    assert payload["rankings"][0]["industry_group"] == "Internet Services"
    assert payload["rs_formula_version"] == BALANCED_RS_FORMULA_VERSION
    assert payload["rs_as_of_date"] == "2026-04-18"
    assert payload["rs_universe_size"] == 5000
    assert payload["rankings"][0]["avg_rs_rating_1m"] == 41.5
    assert payload["rankings"][0]["avg_rs_rating_3m"] == 63.2


@pytest.mark.asyncio
async def test_get_current_rankings_supports_india_market_scope(monkeypatch, client):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    class _FakeGroupRankService:
        def get_current_rankings(self, db, limit=197, calculation_date=None, *, market="US"):  # noqa: ARG002
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
                    "rs_formula_version": LEGACY_RS_FORMULA_VERSION,
                    "market_rs_run_id": None,
                }
            ]

    monkeypatch.setattr(
        "app.api.v1.groups._get_group_rank_service",
        lambda: _FakeGroupRankService(),
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

    class _FakeGroupRankService:
        def get_rank_movers(self, db, period="1w", limit=20, calculation_date=None, *, market="US"):  # noqa: ARG002
            assert market == "JP"
            assert period == "1m"
            assert calculation_date == date(2026, 4, 18)
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
                        "rs_formula_version": LEGACY_RS_FORMULA_VERSION,
                        "market_rs_run_id": None,
                    }
                ],
                "losers": [],
            }

    monkeypatch.setattr(
        "app.api.v1.groups._get_group_rank_service",
        lambda: _FakeGroupRankService(),
    )

    response = await client.get(
        "/api/v1/groups/rankings/movers",
        params={"market": "JP", "period": "1m", "as_of_date": "2026-04-18"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["market_scope"] == "JP"
    assert payload["gainers"][0]["industry_group"] == "Transportation Equipment"


@pytest.mark.asyncio
async def test_get_group_detail_supports_non_us_market_scope(monkeypatch, client):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    class _FakeGroupRankService:
        def get_group_history(self, db, industry_group, days=180, *, market):  # noqa: ARG002
            assert market == "TW"
            assert industry_group == "Semiconductors"
            return {
                "industry_group": "Semiconductors",
                "current_rank": 1,
                "current_avg_rs": 91.2,
                "current_avg_rs_1m": 88.5,
                "current_avg_rs_3m": 90.1,
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
                "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
                "market_rs_run_id": 42,
            }

    monkeypatch.setattr(
        "app.api.v1.groups._get_group_rank_service",
        lambda: _FakeGroupRankService(),
    )

    response = await client.get(
        "/api/v1/groups/rankings/detail",
        params={"market": "TW", "group": "Semiconductors"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["market_scope"] == "TW"
    assert payload["industry_group"] == "Semiconductors"
    assert payload["current_avg_rs_1m"] == 88.5
    assert payload["current_avg_rs_3m"] == 90.1


@pytest.mark.asyncio
async def test_get_group_detail_accepts_pre_scoped_us_payload(monkeypatch, client):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    class _FakeGroupRankService:
        def get_group_history(self, db, group, days=180, *, market="US"):  # noqa: ARG002
            assert group == "Computer-Data Storage"
            return {
                "market_scope": "HK",
                "scope_reason": "stale pre-scoped service payload",
                "industry_group": "Computer-Data Storage",
                "current_rank": 1,
                "current_avg_rs": 96.9,
                "current_median_rs": 100.0,
                "current_weighted_avg_rs": 98.0,
                "current_rs_std_dev": 7.6,
                "num_stocks": 7,
                "pct_rs_above_80": 85.7,
                "top_symbol": "BLZE",
                "top_rs_rating": 100.0,
                "rank_change_1w": 0,
                "rank_change_1m": 0,
                "rank_change_3m": 19,
                "rank_change_6m": 15,
                "history": [
                    {
                        "date": "2026-06-26",
                        "rank": 1,
                        "avg_rs_rating": 96.9,
                        "num_stocks": 7,
                    }
                ],
                "stocks": [],
            }

    monkeypatch.setattr(
        "app.api.v1.groups._get_group_rank_service",
        lambda: _FakeGroupRankService(),
    )

    response = await client.get(
        "/api/v1/groups/rankings/detail",
        params={"market": "US", "group": "Computer-Data Storage"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["market_scope"] == "US"
    assert payload["scope_reason"] is None
    assert payload["industry_group"] == "Computer-Data Storage"


@pytest.mark.asyncio
async def test_get_groups_bootstrap_supports_non_us_market_scope(monkeypatch, client):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    class _FakeSnapshotService:
        def get_groups_bootstrap(self):
            return None

    class _FakeGroupRankService:
        def get_current_rankings(self, db, limit=197, calculation_date=None, *, market="US"):  # noqa: ARG002
            assert market == "HK"
            return [
                {
                    "industry_group": "Internet Services",
                    "date": "2026-04-18",
                    "rank": 3,
                    "avg_rs_rating": 82.1,
                    "avg_rs_rating_1m": 41.5,
                    "avg_rs_rating_3m": 63.2,
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
                    "rs_formula_version": LEGACY_RS_FORMULA_VERSION,
                    "market_rs_run_id": None,
                }
            ]

        def get_rank_movers(self, db, period="1w", limit=10, calculation_date=None, *, market="US"):  # noqa: ARG002
            assert market == "HK"
            return {"period": "1w", "gainers": [], "losers": []}

    monkeypatch.setattr(
        "app.api.v1.groups._get_group_rank_service",
        lambda: _FakeGroupRankService(),
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
    assert payload["payload"]["rankings"]["rs_formula_version"] == LEGACY_RS_FORMULA_VERSION
    assert payload["payload"]["rankings"]["rs_as_of_date"] == "2026-04-18"
    assert payload["payload"]["rankings"]["rankings"][0]["avg_rs_rating_1m"] == 41.5
    assert payload["payload"]["rankings"]["rankings"][0]["avg_rs_rating_3m"] == 63.2


@pytest.mark.asyncio
async def test_get_rrg_scopes_returns_bundle_with_available_scopes(monkeypatch, client):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    class _FakeRRGService:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

        def available_scopes_for_market(self, market):
            assert market == "HK"
            return ("groups", "sectors")

        def get_rrg_scopes(self, db, *, market, scopes, tail_weeks=8, lookback_days=400, as_of_date=None):  # noqa: ARG002
            assert market == "HK"
            assert scopes == ("groups", "sectors")
            assert as_of_date is None
            return {
                "groups": {
                    "date": "2026-04-18",
                    "market": "HK",
                    "scope": "groups",
                    "groups": [
                        {
                            "industry_group": "Internet Services",
                            "rank": 1,
                            "num_stocks": 9,
                            "avg_rs_rating": 82.0,
                            "quadrant": "Leading",
                            "is_provisional": False,
                            "current": {"date": "2026-04-12", "x": 104.0, "y": 103.0},
                            "tail": [{"date": "2026-04-12", "x": 104.0, "y": 103.0}],
                        }
                    ],
                },
                "sectors": {
                    "date": "2026-04-18",
                    "market": "HK",
                    "scope": "sectors",
                    "groups": [],
                },
            }

    monkeypatch.setattr(
        "app.api.v1.groups._get_rrg_service",
        lambda: _FakeRRGService(),
    )

    response = await client.get("/api/v1/groups/rrg/scopes", params={"market": "HK"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["market"] == "HK"
    assert payload["available_scopes"] == ["groups"]
    assert payload["payload"]["groups"]["total_groups"] == 1
    assert payload["payload"]["sectors"]["total_groups"] == 0
    assert payload["payload"]["groups"]["market_scope"] == "HK"


@pytest.mark.asyncio
async def test_get_rrg_scopes_returns_sector_only_bundle(monkeypatch, client):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    class _FakeRRGService:
        def available_scopes_for_market(self, market):
            assert market == "HK"
            return ("sectors",)

        def get_rrg_scopes(self, db, *, market, scopes, tail_weeks=8, lookback_days=400, as_of_date=None):  # noqa: ARG002
            assert market == "HK"
            assert scopes == ("sectors",)
            assert as_of_date is None
            return {
                "sectors": {
                    "date": "2026-04-18",
                    "market": "HK",
                    "scope": "sectors",
                    "groups": [
                        {
                            "industry_group": "Information Technology",
                            "rank": 1,
                            "num_stocks": 21,
                            "avg_rs_rating": 84.0,
                            "quadrant": "Leading",
                            "is_provisional": False,
                            "current": {"date": "2026-04-12", "x": 104.0, "y": 103.0},
                            "tail": [{"date": "2026-04-12", "x": 104.0, "y": 103.0}],
                        }
                    ],
                },
            }

    monkeypatch.setattr(
        "app.api.v1.groups._get_rrg_service",
        lambda: _FakeRRGService(),
    )

    response = await client.get("/api/v1/groups/rrg/scopes", params={"market": "HK"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["date"] == "2026-04-18"
    assert payload["available_scopes"] == ["sectors"]
    assert set(payload["payload"]) == {"sectors"}
    assert payload["payload"]["sectors"]["total_groups"] == 1


@pytest.mark.asyncio
async def test_get_rrg_scopes_requests_only_market_supported_scopes(monkeypatch, client):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    class _FakeRRGService:
        def available_scopes_for_market(self, market):
            assert market == "TW"
            return ("groups",)

        def get_rrg_scopes(self, db, *, market, scopes, tail_weeks=8, lookback_days=400, as_of_date=None):  # noqa: ARG002
            assert market == "TW"
            assert scopes == ("groups",)
            assert as_of_date is None
            return {
                "groups": {
                    "date": "2026-04-18",
                    "market": "TW",
                    "scope": "groups",
                    "groups": [
                        {
                            "industry_group": "Semiconductors",
                            "rank": 1,
                            "num_stocks": 12,
                            "avg_rs_rating": 82.0,
                            "quadrant": "Leading",
                            "is_provisional": False,
                            "current": {"date": "2026-04-12", "x": 104.0, "y": 103.0},
                            "tail": [{"date": "2026-04-12", "x": 104.0, "y": 103.0}],
                        }
                    ],
                },
            }

    monkeypatch.setattr(
        "app.api.v1.groups._get_rrg_service",
        lambda: _FakeRRGService(),
    )

    response = await client.get("/api/v1/groups/rrg/scopes", params={"market": "TW"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["available_scopes"] == ["groups"]
    assert set(payload["payload"]) == {"groups"}


@pytest.mark.asyncio
async def test_get_rrg_scopes_rejects_group_rank_market_without_rrg_capability(monkeypatch, client):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    class _FakeRRGService:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

        def get_rrg_scopes(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return {"groups": {"groups": []}, "sectors": {"groups": []}}

    monkeypatch.setattr(
        "app.api.v1.groups._get_rrg_service",
        lambda: _FakeRRGService(),
    )

    response = await client.get("/api/v1/groups/rrg/scopes", params={"market": "KR"})

    assert response.status_code == 400
    assert "Unsupported RRG market" in response.json()["detail"]
