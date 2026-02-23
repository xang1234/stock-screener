"""Unit tests for scan results endpoints (results/symbols/single/setup)."""

from __future__ import annotations

from datetime import datetime

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from app.api.v1.scans import router as scans_router
from app.wiring.bootstrap import get_uow
from tests.unit.use_cases.conftest import (
    FakeFeatureStoreRepository,
    FakeScan,
    FakeScanRepository,
    FakeScanResultRepository,
    make_domain_item,
)


app = FastAPI()
app.include_router(scans_router, prefix="/api/v1/scans")


class _FakeUoW:
    """Minimal UoW for scan result endpoint tests."""

    def __init__(self, scans=None, scan_results=None, feature_store=None):
        self.scans = scans or FakeScanRepository()
        self.scan_results = scan_results or FakeScanResultRepository()
        self.feature_store = feature_store or FakeFeatureStoreRepository()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def commit(self):
        pass

    def rollback(self):
        pass


def _make_scan(scan_id: str, *, feature_run_id: int | None = None) -> FakeScan:
    return FakeScan(
        scan_id=scan_id,
        status="completed",
        total_stocks=2,
        passed_stocks=1,
        universe="all",
        universe_type="exchange",
        universe_exchange="all",
        universe_index=None,
        universe_symbols=None,
        feature_run_id=feature_run_id,
        started_at=datetime(2024, 1, 1),
    )


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
class TestScanResultEndpoints:
    async def test_results_default_table_excludes_setup_payload(self, client):
        scan_results = FakeScanResultRepository(
            items=[
                make_domain_item(
                    "AAPL",
                    se_setup_score=81.5,
                    se_explain={"summary": "heavy payload"},
                    se_candidates=[{"pattern": "vcp"}],
                )
            ]
        )
        scan_repo = FakeScanRepository()
        scan_repo.scans["scan-001"] = _make_scan("scan-001")
        uow = _FakeUoW(scans=scan_repo, scan_results=scan_results)

        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.get("/api/v1/scans/scan-001/results")
            assert resp.status_code == 200
            row = resp.json()["results"][0]
            assert row["se_setup_score"] == 81.5
            assert row["se_explain"] is None
            assert row["se_candidates"] is None
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_results_full_includes_setup_payload(self, client):
        scan_results = FakeScanResultRepository(
            items=[
                make_domain_item(
                    "AAPL",
                    se_explain={"summary": "heavy payload"},
                    se_candidates=[{"pattern": "vcp"}],
                )
            ]
        )
        scan_repo = FakeScanRepository()
        scan_repo.scans["scan-001"] = _make_scan("scan-001")
        uow = _FakeUoW(scans=scan_repo, scan_results=scan_results)

        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.get("/api/v1/scans/scan-001/results?detail_level=full")
            assert resp.status_code == 200
            row = resp.json()["results"][0]
            assert row["se_explain"] == {"summary": "heavy payload"}
            assert row["se_candidates"] == [{"pattern": "vcp"}]
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_single_result_core_vs_full_detail_levels(self, client):
        scan_results = FakeScanResultRepository(
            items=[
                make_domain_item(
                    "MSFT",
                    se_explain={"summary": "single payload"},
                    se_candidates=[{"pattern": "flat_base"}],
                )
            ]
        )
        scan_repo = FakeScanRepository()
        scan_repo.scans["scan-002"] = _make_scan("scan-002")
        uow = _FakeUoW(scans=scan_repo, scan_results=scan_results)

        app.dependency_overrides[get_uow] = lambda: uow
        try:
            core = await client.get("/api/v1/scans/scan-002/result/MSFT")
            assert core.status_code == 200
            assert core.json()["se_explain"] is None

            full = await client.get(
                "/api/v1/scans/scan-002/result/MSFT?detail_level=full"
            )
            assert full.status_code == 200
            assert full.json()["se_explain"] == {"summary": "single payload"}
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_symbols_endpoint_returns_lightweight_symbol_page(self, client):
        scan_results = FakeScanResultRepository(
            items=[make_domain_item("AAPL"), make_domain_item("MSFT")]
        )
        scan_repo = FakeScanRepository()
        scan_repo.scans["scan-003"] = _make_scan("scan-003")
        uow = _FakeUoW(scans=scan_repo, scan_results=scan_results)

        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.get("/api/v1/scans/scan-003/symbols?page=1&per_page=1")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["scan_id"] == "scan-003"
            assert payload["total"] == 2
            assert payload["symbols"] == ["AAPL"]
            assert payload["page"] == 1
            assert payload["per_page"] == 1
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_setup_endpoint_returns_explain_payload_only(self, client):
        scan_results = FakeScanResultRepository(
            items=[
                make_domain_item(
                    "NVDA",
                    se_explain={"summary": "drawer payload"},
                    se_candidates=[{"pattern": "cup_with_handle"}],
                )
            ]
        )
        scan_repo = FakeScanRepository()
        scan_repo.scans["scan-004"] = _make_scan("scan-004")
        uow = _FakeUoW(scans=scan_repo, scan_results=scan_results)

        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.get("/api/v1/scans/scan-004/setup/NVDA")
            assert resp.status_code == 200
            assert resp.json() == {
                "scan_id": "scan-004",
                "symbol": "NVDA",
                "se_explain": {"summary": "drawer payload"},
                "se_candidates": [{"pattern": "cup_with_handle"}],
            }
        finally:
            app.dependency_overrides.pop(get_uow, None)
