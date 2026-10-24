"""Unit tests for the scan create endpoint."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

from app.main import app
from app.api.v1.scans import _resolve_scan_guard_market
from app.domain.scanning.models import FreshnessOmissionWarning
from app.domain.markets import market_registry, mic_alias_registry
from app.schemas.universe import Exchange, Market, UniverseDefinition, UniverseType
from app.schemas.scanning import ScanCreateResponse
from app.services import server_auth
from app.wiring.bootstrap import get_create_scan_use_case, get_uow
from app.use_cases.scanning.create_scan import (
    ActiveScanConflict,
    ActiveScanConflictError,
    CreateScanResult,
    SnapshotUnavailableError,
    StaleMarketDataError,
)
from app.use_cases.scanning.published_snapshot import SnapshotIneligible
from app.infra.db.repositories.universe_repo import SqlUniverseRepository
from app.models.stock_universe import StockUniverse
from app.use_cases.scanning.create_scan import CreateScanUseCase
from tests.unit.use_cases.conftest import (
    FakeTaskDispatcher,
    FakeUnitOfWork,
    FakeUniverseRepository,
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


def _warning_payload():
    return {
        "code": "market_data_stale_tail_omitted",
        "message": "Omitted 1 stale symbol from this scan (99.00% fresh).",
        "markets": ["US"],
        "omitted_symbols": ["LHSW"],
        "omitted_count": 1,
        "total_symbols": 100,
        "fresh_count": 99,
        "freshness_rate": 0.99,
        "expected_dates": {"US": "2026-06-18"},
        "oldest_last_cached_dates": {"US": "2026-05-13"},
    }


def test_scan_guard_requires_market_context_for_ambiguous_bse_alias():
    cn_universe = UniverseDefinition(type=UniverseType.EXCHANGE, market=Market.CN, exchange=Exchange.BJSE)
    legacy_bse = SimpleNamespace(market=None, exchange=SimpleNamespace(value="BSE"), index=None)
    scoped_bombay = SimpleNamespace(
        market=SimpleNamespace(value="IN"),
        exchange=SimpleNamespace(value="BSE"),
        index=None,
    )

    assert _resolve_scan_guard_market(cn_universe) == "CN"
    assert _resolve_scan_guard_market(scoped_bombay) == "IN"
    assert _resolve_scan_guard_market(legacy_bse) is None


def test_scan_guard_resolves_registry_exchanges_and_indexes():
    for profile in market_registry.profiles():
        for exchange in profile.exchanges:
            if mic_alias_registry.is_ambiguous(exchange):
                continue
            universe = SimpleNamespace(market=None, exchange=SimpleNamespace(value=exchange), index=None)
            assert _resolve_scan_guard_market(universe) == profile.market.code

        for index in profile.indexes:
            universe = SimpleNamespace(market=None, exchange=None, index=SimpleNamespace(value=index))
            assert _resolve_scan_guard_market(universe) == profile.market.code


def test_scan_create_response_warnings_are_typed():
    universe_def = UniverseDefinition(type=UniverseType.ALL)
    payload = ScanCreateResponse(
        scan_id="scan-warning",
        status="queued",
        total_stocks=99,
        message="Scan queued for 99 stocks",
        warnings=[_warning_payload()],
        universe_def=universe_def,
    )

    assert payload.warnings[0].code == "market_data_stale_tail_omitted"

    with pytest.raises(ValueError):
        ScanCreateResponse(
            scan_id="scan-warning",
            status="queued",
            total_stocks=99,
            message="Scan queued for 99 stocks",
            warnings=[{"code": "unknown_warning", "message": "too loose"}],
            universe_def=universe_def,
        )


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def test_http_create_scan_factory_has_no_request_bound_parameters():
    """Round 5 Codex P1: FastAPI exposes dependency-factory kwargs as
    query parameters. The HTTP-bound factory MUST NOT accept kwargs —
    otherwise a client could pass ?with_freshness_gate=false and bypass
    the staleness gate.
    """
    import inspect
    from app.wiring.bootstrap import get_create_scan_use_case

    sig = inspect.signature(get_create_scan_use_case)
    assert list(sig.parameters.keys()) == [], (
        f"get_create_scan_use_case must have no parameters (FastAPI would expose "
        f"them as query params and bypass security); found {list(sig.parameters)}"
    )


@pytest.mark.asyncio
async def test_create_scan_completed_still_returns_when_bootstrap_enqueue_fails(client):
    fake_use_case = _FakeCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-123", status="completed", total_stocks=10, is_duplicate=False, feature_run_id=17
        )
    )
    app.dependency_overrides[get_uow] = lambda: _FakeUoW()
    app.dependency_overrides[get_create_scan_use_case] = lambda: fake_use_case
    try:
        with (
            patch("app.services.ui_snapshot_service.safe_publish_scan_bootstrap") as mock_publish,
            patch(
                "app.tasks.scan_tasks.publish_scan_bootstrap_snapshots.apply_async",
                side_effect=ConnectionError("broker down"),
            ),
        ):
            response = await client.post("/api/v1/scans", json={"universe": "all"})
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    # Regular result endpoints serve the scan; the bootstrap catches up later.
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    mock_publish.assert_not_called()


@pytest.mark.asyncio
async def test_create_scan_returns_completed_and_queues_bootstrap_publish(client):
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
        with (
            patch("app.services.ui_snapshot_service.safe_publish_scan_bootstrap") as mock_publish,
            patch("app.tasks.scan_tasks.publish_scan_bootstrap_snapshots.apply_async") as mock_enqueue,
        ):
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
    # The snapshot rebuild runs on the general queue, not in the request.
    mock_publish.assert_not_called()
    mock_enqueue.assert_called_once_with(args=["scan-123"], retry=False, connection=ANY)


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
            warnings=(FreshnessOmissionWarning.from_dict(_warning_payload()),),
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
    assert payload["warnings"] == [_warning_payload()]
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
        "mic": None,
        "exchange": None,
        "index": None,
        "listing_tier": None,
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


def _price_refresh(market, task_id, *, percent=55.0, current=550, total=1000):
    return {
        "market": market,
        "lifecycle": "daily_refresh",
        "stage_key": "prices",
        "stage_label": "Price Refresh",
        "status": "running",
        "progress_mode": "determinate",
        "percent": percent,
        "current": current,
        "total": total,
        "message": "Refreshing prices",
        "task_name": "smart_refresh_cache",
        "task_id": task_id,
        "updated_at": "2026-04-18T00:00:00Z",
    }


async def _post_scan_during_activity(client, universe_def, activity):
    fake_uow = FakeUnitOfWork(
        universe=FakeUniverseRepository(
            universe_def.get("symbols"),
            markets={"NVDA": "US", "0700.HK": "HK", "ZZZZ.T": "JP"},
        )
    )
    fake_use_case = _FakeCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-during-activity",
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
                "summary": {"active_market_count": 1, "active_markets": [activity["market"]], "status": "active"},
                "markets": [activity],
            },
        ):
            response = await client.post("/api/v1/scans", json={"universe_def": universe_def})
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)
    return response, fake_use_case


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("universe_def", "activity", "expected_detail"),
    [
        pytest.param(
            {"type": "market", "market": "HK"},
            _price_refresh("HK", "task-hk", percent=40.0, current=400),
            {"market": "HK", "active_stages": ["prices"], "lifecycle": "daily_refresh"},
            id="market_prices_refresh_is_active",
        ),
        pytest.param(
            {"type": "market", "market": "US"},
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
            },
            {"market": "US", "active_stages": ["fundamentals"], "lifecycle": "bootstrap"},
            id="market_fundamentals_refresh_is_queued",
        ),
        pytest.param(
            {"type": "exchange", "exchange": "NYSE"},
            _price_refresh("US", "task-us"),
            {"market": "US"},
            id="us_exchange_when_us_refresh_is_active",
        ),
        pytest.param(
            {"type": "exchange", "exchange": "KOSPI"},
            _price_refresh("KR", "task-kr"),
            {"market": "KR"},
            id="kr_exchange_KOSPI_when_kr_refresh_is_active",
        ),
        pytest.param(
            {"type": "exchange", "exchange": "KOSDAQ"},
            _price_refresh("KR", "task-kr"),
            {"market": "KR"},
            id="kr_exchange_KOSDAQ_when_kr_refresh_is_active",
        ),
        *(
            pytest.param(
                {"type": "index", "index": index_name},
                _price_refresh(market, "task-hk", percent=25.0, current=20, total=80),
                {"market": market, "active_stages": ["prices"]},
                id=f"index_{index_name}_when_{market}_refresh_is_active",
            )
            for index_name, market in [("HSI", "HK"), ("NIKKEI225", "JP"), ("TAIEX", "TW")]
        ),
        pytest.param(
            {"type": "custom", "symbols": ["NVDA", "0700.HK"]},
            _price_refresh("HK", "task-hk"),
            {"market": "HK", "active_stages": ["prices"]},
            id="custom_symbols_when_one_of_their_markets_refreshes",
        ),
    ],
)
async def test_create_scan_returns_409_when_scan_market_refresh_is_active(
    client, universe_def, activity, expected_detail
):
    response, fake_use_case = await _post_scan_during_activity(client, universe_def, activity)

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "market_refresh_active"
    assert {key: detail[key] for key in expected_detail} == expected_detail
    assert "wait for it to finish" in detail["message"].lower()
    assert fake_use_case.received_cmd is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("universe_def", "activity", "cmd_field", "expected"),
    [
        pytest.param(
            {"type": "market", "market": "HK"},
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
            },
            "universe_market",
            "HK",
            id="allows_non_core_runtime_activity_for_market",
        ),
        pytest.param(
            {"type": "market", "market": "HK"},
            _price_refresh("US", "task-us"),
            "universe_market",
            "HK",
            id="allows_other_market_refresh_activity",
        ),
        pytest.param(
            {"type": "exchange", "exchange": "KOSPI"},
            _price_refresh("US", "task-us"),
            "universe_exchange",
            "KOSPI",
            id="kr_exchange_ignores_us_refresh_activity",
        ),
        pytest.param(
            {"type": "custom", "symbols": ["NVDA", "0700.HK"]},
            _price_refresh("JP", "task-jp"),
            "universe_symbols",
            ["NVDA", "0700.HK"],
            id="custom_symbols_ignore_other_market_refresh",
        ),
    ],
)
async def test_create_scan_runs_when_no_scan_market_refresh_is_active(
    client, universe_def, activity, cmd_field, expected
):
    response, fake_use_case = await _post_scan_during_activity(client, universe_def, activity)

    assert response.status_code == 200
    assert getattr(fake_use_case.received_cmd, cmd_field) == expected


@pytest.mark.asyncio
async def test_create_scan_returns_409_when_market_price_data_is_stale(client):
    """The staleness gate lives inside CreateScanUseCase (after idempotency
    and symbol resolution) and surfaces as StaleMarketDataError; the route
    maps that to 409 with the detail payload.
    """
    stale_detail = {
        "code": "market_data_stale",
        "message": (
            "Price data is stale for: US (oldest: 2026-04-22, expected: 2026-04-23). "
            "Wait for the next scheduled refresh before starting a scan."
        ),
        "stale_markets": [
            {
                "market": "US",
                "total_symbols": 3,
                "covered_symbols": 2,
                "uncovered_symbols": 1,
                "oldest_last_cached_date": "2026-04-22",
                "expected_date": "2026-04-23",
            }
        ],
    }

    class _StaleCreateScanUseCase(_FakeCreateScanUseCase):
        def execute(self, uow, cmd):
            self.received_uow = uow
            self.received_cmd = cmd
            raise StaleMarketDataError(stale_detail)

    fake_uow = _FakeUoW()
    fake_use_case = _StaleCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-stale",
            status="queued",
            total_stocks=100,
            is_duplicate=False,
            feature_run_id=None,
        )
    )

    app.dependency_overrides[get_uow] = lambda: fake_uow
    app.dependency_overrides[get_create_scan_use_case] = lambda: fake_use_case
    try:
        response = await client.post(
            "/api/v1/scans",
            json={"universe_def": {"type": "market", "market": "US"}},
        )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 409
    payload = response.json()
    assert payload["detail"]["code"] == "market_data_stale"
    assert payload["detail"]["stale_markets"][0]["market"] == "US"
    assert payload["detail"]["stale_markets"][0]["uncovered_symbols"] == 1
    assert payload["detail"]["stale_markets"][0]["oldest_last_cached_date"] == "2026-04-22"


@pytest.mark.asyncio
async def test_refresh_scan_cache_queues_market_specific_manual_refresh(client, monkeypatch):
    from app.api.v1 import cache as cache_module

    mock_lock = MagicMock()
    mock_lock.get_current_task.return_value = None
    monkeypatch.setattr(cache_module, "get_data_fetch_lock", lambda: mock_lock)

    apply_async = MagicMock(return_value=SimpleNamespace(id="refresh-hk"))
    monkeypatch.setattr(cache_module.smart_refresh_cache, "apply_async", apply_async)

    response = await client.post(
        "/api/v1/scans/refresh-cache",
        json={"market": "hk", "mode": "full"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "queued",
        "task_id": "refresh-hk",
        "message": "Smart refresh started for entire universe, force re-fetch (~2 hours)",
    }
    mock_lock.get_current_task.assert_called_once_with(market="HK")
    apply_async.assert_called_once_with(
        kwargs={"mode": "full", "market": "HK"},
        headers={"origin": "manual"},
        queue="data_fetch_hk",
    )


@pytest.mark.asyncio
async def test_refresh_scan_cache_rejects_invalid_market(client, monkeypatch):
    from app.api.v1 import cache as cache_module

    apply_async = MagicMock()
    monkeypatch.setattr(cache_module.smart_refresh_cache, "apply_async", apply_async)

    response = await client.post(
        "/api/v1/scans/refresh-cache",
        json={"market": "ZZ", "mode": "full"},
    )

    assert response.status_code == 400
    assert "Unsupported market" in response.json()["detail"]
    apply_async.assert_not_called()


@pytest.mark.parametrize("market", ["", "  ", "SHARED", "shared"])
@pytest.mark.asyncio
async def test_refresh_scan_cache_rejects_non_explicit_market_scope(client, monkeypatch, market):
    from app.api.v1 import cache as cache_module

    apply_async = MagicMock()
    monkeypatch.setattr(cache_module.smart_refresh_cache, "apply_async", apply_async)

    response = await client.post(
        "/api/v1/scans/refresh-cache",
        json={"market": market, "mode": "full"},
    )

    assert response.status_code == 400
    assert "Unsupported market" in response.json()["detail"]
    apply_async.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_scan_cache_only_blocks_same_market_active_refresh(client, monkeypatch):
    from app.api.v1 import cache as cache_module

    mock_lock = MagicMock()
    mock_lock.get_current_task.return_value = {
        "task_id": "running-hk",
        "task_name": "smart_refresh_cache",
    }
    monkeypatch.setattr(cache_module, "get_data_fetch_lock", lambda: mock_lock)

    apply_async = MagicMock()
    monkeypatch.setattr(cache_module.smart_refresh_cache, "apply_async", apply_async)

    response = await client.post(
        "/api/v1/scans/refresh-cache",
        json={"market": "HK", "mode": "full"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "already_running",
        "task_id": "running-hk",
        "message": "Refresh already in progress (smart_refresh_cache)",
    }
    mock_lock.get_current_task.assert_called_once_with(market="HK")
    apply_async.assert_not_called()


@pytest.mark.asyncio
async def test_scan_bootstrap_passes_normalized_market_to_snapshot_service(client):
    from app.wiring.bootstrap import get_ui_snapshot_service

    snapshot = SimpleNamespace(
        to_dict=lambda: {
            "snapshot_revision": "1",
            "source_revision": "hk-2",
            "published_at": "2026-09-30T12:00:00Z",
            "is_stale": False,
            "payload": {"market": "HK"},
        }
    )
    service = MagicMock()
    service.get_scan_bootstrap.return_value = snapshot
    app.dependency_overrides[get_ui_snapshot_service] = lambda: service
    try:
        response = await client.get("/api/v1/scans/bootstrap", params={"market": " hk "})
    finally:
        app.dependency_overrides.pop(get_ui_snapshot_service, None)

    assert response.status_code == 200
    service.get_scan_bootstrap.assert_called_once_with(None, market="HK")


@pytest.mark.asyncio
async def test_scan_bootstrap_rejects_unsupported_market(client):
    from app.wiring.bootstrap import get_ui_snapshot_service

    service = MagicMock()
    app.dependency_overrides[get_ui_snapshot_service] = lambda: service
    try:
        response = await client.get("/api/v1/scans/bootstrap", params={"market": "ZZ"})
    finally:
        app.dependency_overrides.pop(get_ui_snapshot_service, None)

    assert response.status_code == 400
    service.get_scan_bootstrap.assert_not_called()


_ACTIVE_HK_REFRESH = {
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
}


@pytest.mark.asyncio
async def test_last_published_scan_is_not_blocked_by_active_refresh(client):
    source = {"data_mode": "last_published", "as_of_date": "2026-10-01", "feature_run_id": 9}
    fake_use_case = _FakeCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-hk",
            status="completed",
            total_stocks=50,
            is_duplicate=False,
            published_source=source,
        )
    )
    app.dependency_overrides[get_uow] = lambda: _FakeUoW()
    app.dependency_overrides[get_create_scan_use_case] = lambda: fake_use_case
    try:
        with patch(
            "app.api.v1.scans.get_runtime_activity_status",
            return_value=_ACTIVE_HK_REFRESH,
        ), patch("app.api.v1.scans._queue_scan_bootstrap_publish"):
            response = await client.post(
                "/api/v1/scans",
                json={
                    "universe_def": {"type": "market", "market": "HK"},
                    "data_mode": "last_published",
                },
            )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 200
    assert fake_use_case.received_cmd.data_mode == "last_published"
    assert response.json()["published_source"] == source


@pytest.mark.asyncio
async def test_unavailable_last_published_snapshot_returns_structured_409(client):
    class _Unavailable(_FakeCreateScanUseCase):
        def execute(self, uow, cmd):
            raise SnapshotUnavailableError(
                SnapshotIneligible("incomplete_coverage", {"market": "US"})
            )

    app.dependency_overrides[get_uow] = lambda: _FakeUoW()
    app.dependency_overrides[get_create_scan_use_case] = lambda: _Unavailable(None)
    try:
        response = await client.post(
            "/api/v1/scans",
            json={"universe": "all", "data_mode": "last_published"},
        )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "snapshot_unavailable"
    assert detail["reason"] == "incomplete_coverage"
    assert detail["details"] == {"market": "US"}


@pytest.mark.asyncio
async def test_create_scan_rejects_unknown_data_mode(client):
    response = await client.post(
        "/api/v1/scans",
        json={"universe": "all", "data_mode": "stale_ok"},
    )

    assert response.status_code == 422


@pytest.fixture
def universe_session():
    # The sync endpoint runs in a worker thread, so the in-memory DB must be
    # one connection shared across threads.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    StockUniverse.__table__.create(engine)
    session = sessionmaker(bind=engine)()
    for symbol, market in (("NVDA", "US"), ("ANET", "US"), ("0700.HK", "HK")):
        session.add(StockUniverse(symbol=symbol, name=symbol, market=market, is_active=True))
    session.commit()
    yield session
    session.close()
    engine.dispose()


async def _create_custom_scan_against_universe(client, universe_session, symbols, *, universe_def=None):
    dispatcher = FakeTaskDispatcher()
    app.dependency_overrides[get_uow] = lambda: FakeUnitOfWork(
        universe=SqlUniverseRepository(universe_session)
    )
    app.dependency_overrides[get_create_scan_use_case] = lambda: CreateScanUseCase(dispatcher)
    try:
        response = await client.post(
            "/api/v1/scans",
            json={"universe_def": universe_def or {"type": "custom", "symbols": symbols}},
        )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)
    return response, dispatcher


@pytest.mark.asyncio
async def test_custom_scan_reports_symbols_missing_from_the_universe(client, universe_session):
    response, dispatcher = await _create_custom_scan_against_universe(
        client, universe_session, ["NVDA", "ZZZZ", "ANET", "QQQQ.HK"]
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_stocks"] == 2
    assert payload["warnings"] == [
        {
            "code": "symbols_not_in_universe",
            "message": "Skipped 2 symbols not in the active stock universe: ZZZZ, QQQQ.HK",
            "dropped_symbols": ["ZZZZ", "QQQQ.HK"],
            "dropped_count": 2,
        }
    ]
    assert dispatcher.dispatched[0][1] == ["NVDA", "ANET"]


@pytest.mark.asyncio
async def test_custom_scan_of_known_symbols_has_no_warning(client, universe_session):
    response, _ = await _create_custom_scan_against_universe(
        client, universe_session, ["NVDA", "0700.HK"]
    )

    assert response.status_code == 200
    assert response.json()["total_stocks"] == 2
    assert response.json()["warnings"] == []


async def _create_custom_scan_during_hk_refresh(client, universe_session, universe_def):
    universe_session.query(StockUniverse).filter_by(symbol="0700.HK").update({"is_active": False})
    universe_session.commit()
    with patch(
        "app.api.v1.scans.get_runtime_activity_status",
        return_value={
            "bootstrap": {},
            "summary": {"active_market_count": 1, "active_markets": ["HK"], "status": "active"},
            "markets": [_price_refresh("HK", "task-hk")],
        },
    ):
        return await _create_custom_scan_against_universe(
            client, universe_session, universe_def["symbols"], universe_def=universe_def
        )


@pytest.mark.asyncio
async def test_custom_scan_guard_skips_the_market_of_an_inactive_symbol(client, universe_session):
    response, dispatcher = await _create_custom_scan_during_hk_refresh(
        client, universe_session, {"type": "custom", "symbols": ["NVDA", "0700.HK"]}
    )

    assert response.status_code == 200
    assert response.json()["total_stocks"] == 1
    assert dispatcher.dispatched[0][1] == ["NVDA"]


@pytest.mark.asyncio
async def test_custom_scan_guard_keeps_inactive_symbols_the_scan_will_use(client, universe_session):
    response, dispatcher = await _create_custom_scan_during_hk_refresh(
        client,
        universe_session,
        {"type": "custom", "symbols": ["NVDA", "0700.HK"], "allow_inactive_symbols": True},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "market_refresh_active"
    assert response.json()["detail"]["market"] == "HK"
    assert dispatcher.dispatched == []


class _CountingUniverseRepository(SqlUniverseRepository):
    def __init__(self, session):
        super().__init__(session)
        self.resolve_calls = 0

    def resolve(self, universe_def):
        self.resolve_calls += 1
        return super().resolve(universe_def)


@pytest.mark.asyncio
async def test_custom_scan_resolves_its_universe_once_for_guard_and_scan(
    client, universe_session, caplog
):
    universe = _CountingUniverseRepository(universe_session)
    dispatcher = FakeTaskDispatcher()
    app.dependency_overrides[get_uow] = lambda: FakeUnitOfWork(universe=universe)
    app.dependency_overrides[get_create_scan_use_case] = lambda: CreateScanUseCase(dispatcher)
    try:
        with caplog.at_level("WARNING", logger="app.services.universe_resolver"):
            response = await client.post(
                "/api/v1/scans",
                json={"universe_def": {"type": "custom", "symbols": ["NVDA", "ZZZZ", "0700.HK"]}},
            )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 200
    assert universe.resolve_calls == 1
    assert [record.getMessage() for record in caplog.records if "Dropped" in record.getMessage()] == [
        "Dropped 1 inactive or unknown symbols from custom universe: ZZZZ"
    ]
    assert dispatcher.dispatched[0][1] == ["NVDA", "0700.HK"]
