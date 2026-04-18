"""Unit tests for the scan create endpoint."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from unittest.mock import patch

from app.main import app
from app.services import server_auth
from app.wiring.bootstrap import get_create_scan_use_case, get_uow
from app.use_cases.scanning.create_scan import (
    ActiveScanConflict,
    ActiveScanConflictError,
    CreateScanResult,
)


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


class _ConflictCreateScanUseCase(_FakeCreateScanUseCase):
    def execute(self, uow, cmd):
        self.received_uow = uow
        self.received_cmd = cmd
        raise ActiveScanConflictError(
            ActiveScanConflict(
                scan_id="scan-active",
                status="running",
                trigger_source="manual",
                total_stocks=42,
                started_at=None,
            )
        )


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
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
    payload = response.json()
    assert payload["scan_id"] == "scan-123"
    assert payload["status"] == "completed"
    assert payload["total_stocks"] == 2500
    assert payload["message"] == "Scan completed instantly for 2500 stocks"
    assert payload["feature_run_id"] == 17
    assert payload["universe_def"]["type"] == "all"
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
    payload = response.json()
    assert payload["scan_id"] == "scan-456"
    assert payload["status"] == "queued"
    assert payload["total_stocks"] == 99
    assert payload["message"] == "Scan queued for 99 stocks"
    assert payload["feature_run_id"] is None
    assert payload["universe_def"]["type"] == "all"
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
    payload = response.json()
    assert payload["universe_def"] == {
        "type": "market",
        "market": "HK",
        "exchange": None,
        "index": None,
        "symbols": None,
        "allow_inactive_symbols": False,
    }


@pytest.mark.asyncio
async def test_create_scan_records_legacy_telemetry_counter(client):
    """Legacy-path requests bump the compatibility counter so operators can
    track remaining legacy callers before the sunset date."""
    fake_uow = _FakeUoW()
    fake_use_case = _FakeCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-legacy-counter",
            status="queued",
            total_stocks=10,
            is_duplicate=False,
            feature_run_id=None,
        )
    )

    app.dependency_overrides[get_uow] = lambda: fake_uow
    app.dependency_overrides[get_create_scan_use_case] = lambda: fake_use_case
    try:
        with patch(
            "app.services.universe_compat_metrics.record_legacy_universe_usage"
        ) as mock_record:
            response = await client.post(
                "/api/v1/scans",
                json={"universe": "nyse"},
            )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 200
    mock_record.assert_called_once_with("nyse")


@pytest.mark.asyncio
async def test_create_scan_does_not_record_counter_for_typed_requests(client):
    """Typed universe_def requests must not bump the legacy counter."""
    fake_uow = _FakeUoW()
    fake_use_case = _FakeCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-typed-no-counter",
            status="queued",
            total_stocks=7,
            is_duplicate=False,
            feature_run_id=None,
        )
    )

    app.dependency_overrides[get_uow] = lambda: fake_uow
    app.dependency_overrides[get_create_scan_use_case] = lambda: fake_use_case
    try:
        with patch(
            "app.services.universe_compat_metrics.record_legacy_universe_usage"
        ) as mock_record:
            response = await client.post(
                "/api/v1/scans",
                json={"universe_def": {"type": "all"}},
            )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 200
    mock_record.assert_not_called()


@pytest.mark.asyncio
async def test_create_scan_returns_409_when_another_scan_is_active(client):
    fake_uow = _FakeUoW()
    fake_use_case = _ConflictCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-conflict",
            status="queued",
            total_stocks=0,
            is_duplicate=False,
        )
    )

    app.dependency_overrides[get_uow] = lambda: fake_uow
    app.dependency_overrides[get_create_scan_use_case] = lambda: fake_use_case
    try:
        response = await client.post(
            "/api/v1/scans",
            json={"universe": "all"},
        )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 409
    payload = response.json()
    assert payload["detail"]["code"] == "scan_already_active"
    assert payload["detail"]["active_scan"]["scan_id"] == "scan-active"
    assert payload["detail"]["active_scan"]["status"] == "running"


@pytest.mark.asyncio
async def test_create_scan_returns_409_when_market_prices_refresh_is_active(client):
    fake_uow = _FakeUoW()
    fake_use_case = _FakeCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-hk",
            status="queued",
            total_stocks=1200,
            is_duplicate=False,
            feature_run_id=None,
        )
    )

    app.dependency_overrides[get_uow] = lambda: fake_uow
    app.dependency_overrides[get_create_scan_use_case] = lambda: fake_use_case
    try:
        with patch(
            "app.api.v1.scans.get_runtime_activity_status",
            return_value={
                "bootstrap": {},
                "summary": {"active_market_count": 1, "active_markets": ["HK"], "status": "active"},
                "markets": [
                    {
                        "market": "HK",
                        "lifecycle": "daily_refresh",
                        "stage_key": "prices",
                        "stage_label": "Price Refresh",
                        "status": "running",
                        "progress_mode": "determinate",
                        "percent": 40.0,
                        "current": 400,
                        "total": 1000,
                        "message": "Refreshing prices",
                        "task_name": "smart_refresh_cache",
                        "task_id": "task-hk",
                        "updated_at": "2026-04-18T00:00:00Z",
                    }
                ],
            },
        ):
            response = await client.post(
                "/api/v1/scans",
                json={"universe_def": {"type": "market", "market": "HK"}},
            )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 409
    payload = response.json()
    assert payload["detail"]["code"] == "market_refresh_active"
    assert payload["detail"]["market"] == "HK"
    assert payload["detail"]["active_stages"] == ["prices"]
    assert payload["detail"]["lifecycle"] == "daily_refresh"
    assert "wait for it to finish" in payload["detail"]["message"].lower()
    assert fake_use_case.received_cmd is None


@pytest.mark.asyncio
async def test_create_scan_returns_409_when_market_fundamentals_refresh_is_queued(client):
    fake_uow = _FakeUoW()
    fake_use_case = _FakeCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-us",
            status="queued",
            total_stocks=3200,
            is_duplicate=False,
            feature_run_id=None,
        )
    )

    app.dependency_overrides[get_uow] = lambda: fake_uow
    app.dependency_overrides[get_create_scan_use_case] = lambda: fake_use_case
    try:
        with patch(
            "app.api.v1.scans.get_runtime_activity_status",
            return_value={
                "bootstrap": {},
                "summary": {"active_market_count": 1, "active_markets": ["US"], "status": "active"},
                "markets": [
                    {
                        "market": "US",
                        "lifecycle": "bootstrap",
                        "stage_key": "fundamentals",
                        "stage_label": "Fundamentals Refresh",
                        "status": "queued",
                        "progress_mode": "indeterminate",
                        "percent": None,
                        "current": None,
                        "total": None,
                        "message": "Queued fundamentals refresh",
                        "task_name": "refresh_all_fundamentals",
                        "task_id": "task-us",
                        "updated_at": "2026-04-18T00:00:00Z",
                    }
                ],
            },
        ):
            response = await client.post(
                "/api/v1/scans",
                json={"universe_def": {"type": "market", "market": "US"}},
            )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 409
    payload = response.json()
    assert payload["detail"]["code"] == "market_refresh_active"
    assert payload["detail"]["market"] == "US"
    assert payload["detail"]["active_stages"] == ["fundamentals"]
    assert payload["detail"]["lifecycle"] == "bootstrap"
    assert fake_use_case.received_cmd is None


@pytest.mark.asyncio
async def test_create_scan_allows_non_core_runtime_activity_for_market(client):
    fake_uow = _FakeUoW()
    fake_use_case = _FakeCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-hk-ok",
            status="queued",
            total_stocks=1200,
            is_duplicate=False,
            feature_run_id=None,
        )
    )

    app.dependency_overrides[get_uow] = lambda: fake_uow
    app.dependency_overrides[get_create_scan_use_case] = lambda: fake_use_case
    try:
        with patch(
            "app.api.v1.scans.get_runtime_activity_status",
            return_value={
                "bootstrap": {},
                "summary": {"active_market_count": 1, "active_markets": ["HK"], "status": "active"},
                "markets": [
                    {
                        "market": "HK",
                        "lifecycle": "daily_refresh",
                        "stage_key": "breadth",
                        "stage_label": "Breadth Calculation",
                        "status": "running",
                        "progress_mode": "indeterminate",
                        "percent": None,
                        "current": None,
                        "total": None,
                        "message": "Calculating breadth",
                        "task_name": "refresh_market_breadth",
                        "task_id": "task-hk",
                        "updated_at": "2026-04-18T00:00:00Z",
                    }
                ],
            },
        ):
            response = await client.post(
                "/api/v1/scans",
                json={"universe_def": {"type": "market", "market": "HK"}},
            )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 200
    assert fake_use_case.received_cmd.universe_market == "HK"


@pytest.mark.asyncio
async def test_create_scan_allows_other_market_refresh_activity(client):
    fake_uow = _FakeUoW()
    fake_use_case = _FakeCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-hk-ok",
            status="queued",
            total_stocks=1200,
            is_duplicate=False,
            feature_run_id=None,
        )
    )

    app.dependency_overrides[get_uow] = lambda: fake_uow
    app.dependency_overrides[get_create_scan_use_case] = lambda: fake_use_case
    try:
        with patch(
            "app.api.v1.scans.get_runtime_activity_status",
            return_value={
                "bootstrap": {},
                "summary": {"active_market_count": 1, "active_markets": ["US"], "status": "active"},
                "markets": [
                    {
                        "market": "US",
                        "lifecycle": "daily_refresh",
                        "stage_key": "prices",
                        "stage_label": "Price Refresh",
                        "status": "running",
                        "progress_mode": "determinate",
                        "percent": 55.0,
                        "current": 550,
                        "total": 1000,
                        "message": "Refreshing prices",
                        "task_name": "smart_refresh_cache",
                        "task_id": "task-us",
                        "updated_at": "2026-04-18T00:00:00Z",
                    }
                ],
            },
        ):
            response = await client.post(
                "/api/v1/scans",
                json={"universe_def": {"type": "market", "market": "HK"}},
            )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 200
    assert fake_use_case.received_cmd.universe_market == "HK"


@pytest.mark.asyncio
async def test_create_scan_returns_409_for_us_exchange_when_us_refresh_is_active(client):
    fake_uow = _FakeUoW()
    fake_use_case = _FakeCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-nyse",
            status="queued",
            total_stocks=500,
            is_duplicate=False,
            feature_run_id=None,
        )
    )

    app.dependency_overrides[get_uow] = lambda: fake_uow
    app.dependency_overrides[get_create_scan_use_case] = lambda: fake_use_case
    try:
        with patch(
            "app.api.v1.scans.get_runtime_activity_status",
            return_value={
                "bootstrap": {},
                "summary": {"active_market_count": 1, "active_markets": ["US"], "status": "active"},
                "markets": [
                    {
                        "market": "US",
                        "lifecycle": "daily_refresh",
                        "stage_key": "prices",
                        "stage_label": "Price Refresh",
                        "status": "running",
                        "progress_mode": "determinate",
                        "percent": 55.0,
                        "current": 550,
                        "total": 1000,
                        "message": "Refreshing prices",
                        "task_name": "smart_refresh_cache",
                        "task_id": "task-us",
                        "updated_at": "2026-04-18T00:00:00Z",
                    }
                ],
            },
        ):
            response = await client.post(
                "/api/v1/scans",
                json={"universe_def": {"type": "exchange", "exchange": "NYSE"}},
            )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 409
    payload = response.json()
    assert payload["detail"]["code"] == "market_refresh_active"
    assert payload["detail"]["market"] == "US"
    assert fake_use_case.received_cmd is None


@pytest.mark.parametrize(
    ("index_name", "market"),
    [
        ("HSI", "HK"),
        ("NIKKEI225", "JP"),
        ("TAIEX", "TW"),
    ],
)
@pytest.mark.asyncio
async def test_create_scan_returns_409_for_index_when_mapped_market_refresh_is_active(
    client,
    index_name,
    market,
):
    fake_uow = _FakeUoW()
    fake_use_case = _FakeCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-hsi",
            status="queued",
            total_stocks=80,
            is_duplicate=False,
            feature_run_id=None,
        )
    )

    app.dependency_overrides[get_uow] = lambda: fake_uow
    app.dependency_overrides[get_create_scan_use_case] = lambda: fake_use_case
    try:
        with patch(
            "app.api.v1.scans.get_runtime_activity_status",
            return_value={
                "bootstrap": {},
                "summary": {"active_market_count": 1, "active_markets": [market], "status": "active"},
                "markets": [
                    {
                        "market": market,
                        "lifecycle": "daily_refresh",
                        "stage_key": "prices",
                        "stage_label": "Price Refresh",
                        "status": "running",
                        "progress_mode": "determinate",
                        "percent": 25.0,
                        "current": 20,
                        "total": 80,
                        "message": "Refreshing prices",
                        "task_name": "smart_refresh_cache",
                        "task_id": "task-hk",
                        "updated_at": "2026-04-18T00:00:00Z",
                    }
                ],
            },
        ):
            response = await client.post(
                "/api/v1/scans",
                json={"universe_def": {"type": "index", "index": index_name}},
            )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 409
    payload = response.json()
    assert payload["detail"]["code"] == "market_refresh_active"
    assert payload["detail"]["market"] == market
    assert payload["detail"]["active_stages"] == ["prices"]
    assert fake_use_case.received_cmd is None
