"""Unit tests for asia.8.3 backend plumbing.

Verifies that market identity (market/exchange/currency) and FX-normalised
USD values (market_cap_usd/adv_usd) flow from the domain extended_fields
dict into the HTTP ScanResultItem response, and that the new filter params
on GET /api/v1/scans/{id}/results produce the right FilterSpec entries.
"""

from __future__ import annotations

from datetime import datetime

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from app.api.v1.scans import router as scans_router
from app.domain.scanning.models import ScanResultItemDomain
from app.schemas.scanning import ScanResultItem
from app.wiring.bootstrap import get_uow
from tests.unit.use_cases.conftest import (
    FakeScan,
    FakeScanRepository,
    FakeScanResultRepository,
    make_domain_item,
)


app = FastAPI()
app.include_router(scans_router, prefix="/api/v1/scans")


class _FakeUoW:
    def __init__(self, scans=None, scan_results=None):
        self.scans = scans or FakeScanRepository()
        self.scan_results = scan_results or FakeScanResultRepository()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def commit(self):
        pass

    def rollback(self):
        pass


def _scan() -> FakeScan:
    return FakeScan(
        scan_id="scan-1",
        status="completed",
        total_stocks=1,
        passed_stocks=1,
        universe="market:hk",
        universe_type="market",
        universe_market="HK",
        started_at=datetime(2024, 1, 1),
    )


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Domain → HTTP schema mapping
# ---------------------------------------------------------------------------


def _make_item(**extended: object) -> ScanResultItemDomain:
    return ScanResultItemDomain(
        symbol="0700.HK",
        composite_score=80.0,
        rating="Buy",
        current_price=410.0,
        screener_outputs={},
        screeners_run=["minervini"],
        composite_method="weighted_average",
        screeners_passed=1,
        screeners_total=1,
        extended_fields=extended,
    )


class TestScanResultItemMapping:
    def test_market_identity_and_usd_fields_pull_from_extended(self):
        item = _make_item(
            market="HK",
            exchange="HKEX",
            currency="HKD",
            market_cap=3_900_000_000_000,
            market_cap_usd=500_000_000_000,
            adv_usd=12_500_000,
            market_themes=["AI Infrastructure"],
            company_name="Tencent Holdings",
        )

        response = ScanResultItem.from_domain(item)

        assert response.market == "HK"
        assert response.exchange == "HKEX"
        assert response.currency == "HKD"
        assert response.market_cap == 3_900_000_000_000
        assert response.market_cap_usd == 500_000_000_000
        assert response.adv_usd == 12_500_000
        assert response.market_themes == ["AI Infrastructure"]

    def test_market_fields_default_to_none_when_unset(self):
        item = _make_item(company_name="Mystery Corp")
        response = ScanResultItem.from_domain(item)

        assert response.market is None
        assert response.exchange is None
        assert response.currency is None
        assert response.market_cap_usd is None
        assert response.adv_usd is None

    def test_market_themes_scalar_is_coerced_to_single_item_list(self):
        item = _make_item(
            market="HK",
            exchange="HKEX",
            currency="HKD",
            market_themes="AI Infrastructure",
        )

        response = ScanResultItem.from_domain(item)

        assert response.market_themes == ["AI Infrastructure"]


# ---------------------------------------------------------------------------
# Filter param parsing — exercised via the real GET /results endpoint
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def wired_repo():
    scan_repo = FakeScanRepository()
    scan = _scan()
    scan_repo.scans["scan-1"] = scan
    scan_repo.rows = [scan]
    result_repo = FakeScanResultRepository(items=[make_domain_item("0700.HK")])
    uow = _FakeUoW(scans=scan_repo, scan_results=result_repo)
    app.dependency_overrides[get_uow] = lambda: uow
    try:
        yield result_repo
    finally:
        app.dependency_overrides.pop(get_uow, None)


@pytest.mark.asyncio
class TestScanFilterParamsUSDPlumbing:
    async def test_market_cap_usd_min_and_max_become_range_filter(self, client, wired_repo):
        resp = await client.get(
            "/api/v1/scans/scan-1/results",
            params={"min_market_cap_usd": 1_000_000_000, "max_market_cap_usd": 50_000_000_000},
        )
        assert resp.status_code == 200
        spec = wired_repo.last_query_args["spec"]
        ranges = {rf.field: rf for rf in spec.filters.range_filters}
        assert ranges["market_cap_usd"].min_value == 1_000_000_000
        assert ranges["market_cap_usd"].max_value == 50_000_000_000

    async def test_adv_usd_min_and_max_become_range_filter(self, client, wired_repo):
        resp = await client.get(
            "/api/v1/scans/scan-1/results",
            params={"min_adv_usd": 2_500_000, "max_adv_usd": 100_000_000},
        )
        assert resp.status_code == 200
        spec = wired_repo.last_query_args["spec"]
        ranges = {rf.field: rf for rf in spec.filters.range_filters}
        assert ranges["adv_usd"].min_value == 2_500_000
        assert ranges["adv_usd"].max_value == 100_000_000

    async def test_legacy_local_currency_filter_still_works(self, client, wired_repo):
        resp = await client.get(
            "/api/v1/scans/scan-1/results",
            params={"min_market_cap": 2_000_000_000},
        )
        assert resp.status_code == 200
        spec = wired_repo.last_query_args["spec"]
        ranges = {rf.field: rf for rf in spec.filters.range_filters}
        assert ranges["market_cap"].min_value == 2_000_000_000

    async def test_markets_param_becomes_categorical_filter_uppercased(self, client, wired_repo):
        resp = await client.get(
            "/api/v1/scans/scan-1/results",
            params={"markets": "us,hk,JP"},
        )
        assert resp.status_code == 200
        spec = wired_repo.last_query_args["spec"]
        cats = {cf.field: cf for cf in spec.filters.categorical_filters}
        assert cats["market"].values == ("US", "HK", "JP")

    async def test_markets_param_ignores_blank_entries(self, client, wired_repo):
        resp = await client.get(
            "/api/v1/scans/scan-1/results",
            params={"markets": " us , , hk "},
        )
        assert resp.status_code == 200
        spec = wired_repo.last_query_args["spec"]
        cats = {cf.field: cf for cf in spec.filters.categorical_filters}
        assert cats["market"].values == ("US", "HK")

    async def test_no_usd_filters_means_no_usd_range_added(self, client, wired_repo):
        resp = await client.get("/api/v1/scans/scan-1/results")
        assert resp.status_code == 200
        spec = wired_repo.last_query_args["spec"]
        fields = {rf.field for rf in spec.filters.range_filters}
        assert "market_cap_usd" not in fields
        assert "adv_usd" not in fields
