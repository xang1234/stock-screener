"""Unit tests for migrated scan CRUD endpoints (UoW pattern).

Tests list_scans, get_scan_status, cancel_scan, and delete_scan endpoints
after the Z1 migration from raw SQLAlchemy sessions to UnitOfWork + Repositories.
Uses httpx.AsyncClient + httpx.ASGITransport per project convention.
"""

from __future__ import annotations

from datetime import datetime

import httpx
import pytest
import pytest_asyncio

from app.main import app
from app.wiring.bootstrap import get_uow
from tests.unit.use_cases.conftest import (
    FakeScan,
    FakeScanRepository,
    FakeScanResultRepository,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeUoW:
    """Minimal UoW that plugs into FastAPI Depends(get_uow)."""

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


def _make_scan(scan_id="scan-001", status="completed", **kwargs) -> FakeScan:
    """Build a FakeScan with sensible defaults."""
    defaults = dict(
        total_stocks=100,
        passed_stocks=42,
        universe="all",
        universe_type="exchange",
        universe_exchange="all",
        universe_index=None,
        universe_symbols=None,
        feature_run_id=None,
        started_at=datetime(2024, 1, 1),
    )
    defaults.update(kwargs)
    return FakeScan(scan_id=scan_id, status=status, **defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# TestListScans
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestListScans:
    async def test_returns_recent_scans(self, client):
        scan_repo = FakeScanRepository()
        s1 = _make_scan("scan-001", started_at=datetime(2024, 1, 2))
        s2 = _make_scan("scan-002", started_at=datetime(2024, 1, 1))
        scan_repo.scans = {"scan-001": s1, "scan-002": s2}
        scan_repo.rows = [s1, s2]

        uow = _FakeUoW(scans=scan_repo)
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.get("/api/v1/scans")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["scans"]) == 2
            assert data["scans"][0]["scan_id"] == "scan-001"
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_respects_limit_param(self, client):
        scan_repo = FakeScanRepository()
        for i in range(5):
            s = _make_scan(f"scan-{i:03d}", started_at=datetime(2024, 1, 5 - i))
            scan_repo.scans[s.scan_id] = s
            scan_repo.rows.append(s)

        uow = _FakeUoW(scans=scan_repo)
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.get("/api/v1/scans?limit=2")
            assert resp.status_code == 200
            assert len(resp.json()["scans"]) == 2
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_includes_source_field(self, client):
        scan_repo = FakeScanRepository()
        s_legacy = _make_scan("scan-legacy", feature_run_id=None, started_at=datetime(2024, 1, 2))
        s_fs = _make_scan("scan-fs", feature_run_id=42, started_at=datetime(2024, 1, 3))
        scan_repo.scans = {"scan-legacy": s_legacy, "scan-fs": s_fs}
        scan_repo.rows = [s_legacy, s_fs]

        uow = _FakeUoW(scans=scan_repo)
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.get("/api/v1/scans")
            assert resp.status_code == 200
            scans = {s["scan_id"]: s for s in resp.json()["scans"]}
            assert scans["scan-legacy"]["source"] == "scan_results"
            assert scans["scan-fs"]["source"] == "feature_store"
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_empty_list(self, client):
        uow = _FakeUoW()
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.get("/api/v1/scans")
            assert resp.status_code == 200
            assert resp.json()["scans"] == []
        finally:
            app.dependency_overrides.pop(get_uow, None)


# ---------------------------------------------------------------------------
# TestGetScanStatus
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetScanStatus:
    async def test_completed_scan_shows_full_progress(self, client):
        scan_repo = FakeScanRepository()
        scan_repo.scans["scan-001"] = _make_scan(
            "scan-001", status="completed",
            total_stocks=100, passed_stocks=42,
            started_at=datetime(2024, 1, 1),
        )

        uow = _FakeUoW(scans=scan_repo)
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.get("/api/v1/scans/scan-001/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "completed"
            assert data["progress"] == 100.0
            assert data["completed_stocks"] == 100
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_cancelled_scan_shows_partial_progress(self, client):
        scan_repo = FakeScanRepository()
        scan_repo.scans["scan-002"] = _make_scan(
            "scan-002", status="cancelled", total_stocks=100,
            started_at=datetime(2024, 1, 1),
        )

        result_repo = FakeScanResultRepository()
        # Simulate 30 results persisted before cancellation
        for i in range(30):
            result_repo._persisted_results.append(("scan-002", f"SYM{i}", {}))

        uow = _FakeUoW(scans=scan_repo, scan_results=result_repo)
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.get("/api/v1/scans/scan-002/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "cancelled"
            assert data["progress"] == 30.0
            assert data["completed_stocks"] == 30
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_404_for_missing_scan(self, client):
        uow = _FakeUoW()
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.get("/api/v1/scans/nonexistent/status")
            assert resp.status_code == 404
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_failed_scan_shows_zero_progress(self, client):
        scan_repo = FakeScanRepository()
        scan_repo.scans["scan-003"] = _make_scan(
            "scan-003", status="failed", total_stocks=100,
            started_at=datetime(2024, 1, 1),
        )

        uow = _FakeUoW(scans=scan_repo)
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.get("/api/v1/scans/scan-003/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "failed"
            assert data["progress"] == 0.0
            assert data["completed_stocks"] == 0
        finally:
            app.dependency_overrides.pop(get_uow, None)


# ---------------------------------------------------------------------------
# TestCancelScan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCancelScan:
    async def test_cancel_running_scan(self, client):
        scan_repo = FakeScanRepository()
        scan_repo.scans["scan-001"] = _make_scan(
            "scan-001", status="running",
            started_at=datetime(2024, 1, 1),
        )

        uow = _FakeUoW(scans=scan_repo)
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.post("/api/v1/scans/scan-001/cancel")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "cancelled"
            assert scan_repo.scans["scan-001"].status == "cancelled"
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_cancel_queued_scan(self, client):
        scan_repo = FakeScanRepository()
        scan_repo.scans["scan-002"] = _make_scan(
            "scan-002", status="queued",
            started_at=datetime(2024, 1, 1),
        )

        uow = _FakeUoW(scans=scan_repo)
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.post("/api/v1/scans/scan-002/cancel")
            assert resp.status_code == 200
            assert scan_repo.scans["scan-002"].status == "cancelled"
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_400_for_non_cancellable_status(self, client):
        scan_repo = FakeScanRepository()
        scan_repo.scans["scan-003"] = _make_scan(
            "scan-003", status="completed",
            started_at=datetime(2024, 1, 1),
        )

        uow = _FakeUoW(scans=scan_repo)
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.post("/api/v1/scans/scan-003/cancel")
            assert resp.status_code == 400
            assert "completed" in resp.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_404_for_missing_scan(self, client):
        uow = _FakeUoW()
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.post("/api/v1/scans/nonexistent/cancel")
            assert resp.status_code == 404
        finally:
            app.dependency_overrides.pop(get_uow, None)


# ---------------------------------------------------------------------------
# TestDeleteScan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDeleteScan:
    async def test_delete_completed_scan(self, client):
        scan_repo = FakeScanRepository()
        scan_repo.scans["scan-001"] = _make_scan(
            "scan-001", status="completed",
            started_at=datetime(2024, 1, 1),
        )

        result_repo = FakeScanResultRepository()
        result_repo._persisted_results = [("scan-001", "AAPL", {})]

        uow = _FakeUoW(scans=scan_repo, scan_results=result_repo)
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.delete("/api/v1/scans/scan-001")
            assert resp.status_code == 200
            assert "scan-001" not in scan_repo.scans
            assert result_repo.count_by_scan_id("scan-001") == 0
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_409_for_running_scan(self, client):
        scan_repo = FakeScanRepository()
        scan_repo.scans["scan-002"] = _make_scan(
            "scan-002", status="running",
            started_at=datetime(2024, 1, 1),
        )

        uow = _FakeUoW(scans=scan_repo)
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.delete("/api/v1/scans/scan-002")
            assert resp.status_code == 409
            assert "running" in resp.json()["detail"]
            # Scan should NOT have been deleted
            assert "scan-002" in scan_repo.scans
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_409_for_queued_scan(self, client):
        scan_repo = FakeScanRepository()
        scan_repo.scans["scan-003"] = _make_scan(
            "scan-003", status="queued",
            started_at=datetime(2024, 1, 1),
        )

        uow = _FakeUoW(scans=scan_repo)
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.delete("/api/v1/scans/scan-003")
            assert resp.status_code == 409
            assert "queued" in resp.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_404_for_missing_scan(self, client):
        uow = _FakeUoW()
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.delete("/api/v1/scans/nonexistent")
            assert resp.status_code == 404
        finally:
            app.dependency_overrides.pop(get_uow, None)

    async def test_delete_cancelled_scan(self, client):
        scan_repo = FakeScanRepository()
        scan_repo.scans["scan-004"] = _make_scan(
            "scan-004", status="cancelled",
            started_at=datetime(2024, 1, 1),
        )

        uow = _FakeUoW(scans=scan_repo)
        app.dependency_overrides[get_uow] = lambda: uow
        try:
            resp = await client.delete("/api/v1/scans/scan-004")
            assert resp.status_code == 200
            assert "scan-004" not in scan_repo.scans
        finally:
            app.dependency_overrides.pop(get_uow, None)
