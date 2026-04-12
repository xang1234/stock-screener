"""Unit tests for the scan create endpoint."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from unittest.mock import patch

from app.main import app
from app.wiring.bootstrap import get_create_scan_use_case, get_uow
from app.use_cases.scanning.create_scan import CreateScanResult


class _FakeUoW:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def commit(self):
        pass

    def rollback(self):
        pass


class _FakeCreateScanUseCase:
    def __init__(self, result: CreateScanResult) -> None:
        self.result = result
        self.received_uow = None
        self.received_cmd = None

    def execute(self, uow, cmd):
        self.received_uow = uow
        self.received_cmd = cmd
        return self.result


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_create_scan_returns_completed_and_publishes_bootstraps(client):
    fake_uow = _FakeUoW()
    fake_use_case = _FakeCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-123",
            status="completed",
            total_stocks=2500,
            is_duplicate=False,
            feature_run_id=17,
        )
    )

    app.dependency_overrides[get_uow] = lambda: fake_uow
    app.dependency_overrides[get_create_scan_use_case] = lambda: fake_use_case
    try:
        with patch("app.services.ui_snapshot_service.safe_publish_scan_bootstrap") as mock_publish:
            response = await client.post(
                "/api/v1/scans",
                json={"universe": "all"},
            )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 200
    assert response.json() == {
        "scan_id": "scan-123",
        "status": "completed",
        "total_stocks": 2500,
        "message": "Scan completed instantly for 2500 stocks",
        "feature_run_id": 17,
    }
    assert response.headers["deprecation"] == "true"
    assert "sunset" in response.headers
    assert response.headers["x-universe-compat-mode"] == "legacy"
    assert response.headers["x-universe-legacy-value"] == "all"
    assert fake_use_case.received_uow is fake_uow
    assert fake_use_case.received_cmd.universe_type == "all"
    assert fake_use_case.received_cmd.universe_market is None
    assert mock_publish.call_count == 2
    assert mock_publish.call_args_list[0].args == ("scan-123",)
    assert mock_publish.call_args_list[1].args == ()


@pytest.mark.asyncio
async def test_create_scan_returns_queued_without_bootstrap_publish(client):
    fake_uow = _FakeUoW()
    fake_use_case = _FakeCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-456",
            status="queued",
            total_stocks=99,
            is_duplicate=False,
            feature_run_id=None,
        )
    )

    app.dependency_overrides[get_uow] = lambda: fake_uow
    app.dependency_overrides[get_create_scan_use_case] = lambda: fake_use_case
    try:
        with patch("app.services.ui_snapshot_service.safe_publish_scan_bootstrap") as mock_publish:
            response = await client.post(
                "/api/v1/scans",
                json={"universe": "all"},
            )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 200
    assert response.json() == {
        "scan_id": "scan-456",
        "status": "queued",
        "total_stocks": 99,
        "message": "Scan queued for 99 stocks",
        "feature_run_id": None,
    }
    assert response.headers["deprecation"] == "true"
    assert response.headers["x-universe-legacy-value"] == "all"
    assert fake_use_case.received_uow is fake_uow
    assert fake_use_case.received_cmd.universe_market is None
    assert mock_publish.call_count == 0


@pytest.mark.asyncio
async def test_create_scan_accepts_market_universe_def(client):
    fake_uow = _FakeUoW()
    fake_use_case = _FakeCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-market",
            status="queued",
            total_stocks=1200,
            is_duplicate=False,
            feature_run_id=None,
        )
    )

    app.dependency_overrides[get_uow] = lambda: fake_uow
    app.dependency_overrides[get_create_scan_use_case] = lambda: fake_use_case
    try:
        response = await client.post(
            "/api/v1/scans",
            json={"universe_def": {"type": "market", "market": "HK"}},
        )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 200
    assert "deprecation" not in response.headers
    assert "x-universe-compat-mode" not in response.headers
    assert fake_use_case.received_cmd.universe_type == "market"
    assert fake_use_case.received_cmd.universe_market == "HK"
    assert fake_use_case.received_cmd.universe_key == "market:HK"
