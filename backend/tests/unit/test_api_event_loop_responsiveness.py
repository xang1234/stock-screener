"""Blocking endpoint work must not stall other requests on the same worker (#494).

Each case swaps the endpoint's synchronous service for a fake that parks on
a gate. While it is parked, ``/livez`` must still answer. A handler that runs
its blocking body on the event loop stalls the loop inside the fake, so
``/livez`` cannot answer until a watchdog thread opens the gate, and the test
fails instead of hanging.
"""

from __future__ import annotations

import asyncio
import inspect
import threading

import httpx
import pandas as pd
import pytest
import pytest_asyncio

from app.api.v1 import cache as cache_module
from app.api.v1 import scans as scans_module
from app.api.v1 import stocks as stocks_module
from app.api.v1 import technical as technical_module
from app.main import app
from app.services import server_auth
from app.wiring import bootstrap
from app.wiring.runtime_context import current_runtime_services

pytestmark = pytest.mark.integration

_WATCHDOG_SECONDS = 5.0


class _Gate:
    """Parks the calling thread until released; the watchdog bounds the wait."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.runtime_seen = None

    def block(self) -> None:
        self.runtime_seen = current_runtime_services()
        self.entered.set()
        self.release.wait(_WATCHDOG_SECONDS * 2)


def _frame(days: int = 30) -> pd.DataFrame:
    index = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=days, freq="D", name="Date")
    return pd.DataFrame(
        {
            "Open": [100.0] * days,
            "High": [101.0] * days,
            "Low": [99.0] * days,
            "Close": [100.5] * days,
            "Volume": [1_000_000] * days,
        },
        index=index,
    )


class _PriceCache:
    def __init__(self, gate: _Gate) -> None:
        self.gate = gate

    def get_cached_only(self, symbol, period="2y"):
        self.gate.block()
        return _frame()

    def get_many_cached_only(self, symbols, period="2y"):
        self.gate.block()
        return {sym: _frame() for sym in symbols}


class _FundamentalsCache:
    def __init__(self, gate: _Gate) -> None:
        self.gate = gate

    def get_many(self, symbols):
        self.gate.block()
        return {sym: {"pe_ratio": 20.0} for sym in symbols}


class _YFinance:
    def __init__(self, gate: _Gate) -> None:
        self.gate = gate

    def get_stock_info(self, symbol):
        self.gate.block()

    def get_historical_data(self, symbol, period="2y"):
        self.gate.block()


class _Orchestrator:
    def __init__(self, gate: _Gate) -> None:
        self.gate = gate

    def scan_stock_multi(self, **kwargs):
        self.gate.block()
        return {"minervini_score": 1.0}


class _SnapshotService:
    def __init__(self, gate: _Gate) -> None:
        self.gate = gate

    def get_scan_bootstrap(self, scan_id, market=None):
        self.gate.block()


def _patch_price_cache(monkeypatch, gate):
    monkeypatch.setattr(stocks_module, "get_price_cache", lambda: _PriceCache(gate))


def _patch_fundamentals(monkeypatch, gate):
    monkeypatch.setattr(stocks_module, "get_fundamentals_cache", lambda: _FundamentalsCache(gate))


def _patch_stock_yfinance(monkeypatch, gate):
    monkeypatch.setattr(stocks_module, "_get_yfinance_service", lambda: _YFinance(gate))


def _patch_technical_yfinance(monkeypatch, gate):
    monkeypatch.setattr(technical_module, "get_yfinance_service", lambda: _YFinance(gate))


def _patch_orchestrator(monkeypatch, gate):
    monkeypatch.setattr(bootstrap, "get_scan_orchestrator", lambda: _Orchestrator(gate))


def _patch_snapshot(monkeypatch, gate):
    app.dependency_overrides[scans_module.get_ui_snapshot_service] = lambda: _SnapshotService(gate)


def _patch_refresh(monkeypatch, gate):
    def _queue(mode, market=None):
        gate.block()
        raise ValueError("refresh rejected by test")

    monkeypatch.setattr(cache_module, "_queue_manual_smart_refresh", _queue)


# (id, patch, method, path, json body, expected status after release)
_CASES = [
    ("stock-info", _patch_stock_yfinance, "GET", "/api/v1/stocks/AAPL/info", None, 404),
    ("history", _patch_price_cache, "GET", "/api/v1/stocks/AAPL/history", None, 200),
    (
        "history-batch",
        _patch_price_cache,
        "POST",
        "/api/v1/stocks/history/batch",
        {"symbols": ["AAPL", "MSFT"]},
        200,
    ),
    (
        "fundamentals-batch",
        _patch_fundamentals,
        "POST",
        "/api/v1/stocks/fundamentals/batch",
        {"symbols": ["AAPL"]},
        200,
    ),
    ("minervini", _patch_orchestrator, "GET", "/api/v1/technical/AAPL/minervini", None, 200),
    ("stage", _patch_technical_yfinance, "GET", "/api/v1/technical/AAPL/stage", None, 200),
    ("ma-analysis", _patch_technical_yfinance, "GET", "/api/v1/technical/AAPL/ma-analysis", None, 200),
    ("vcp", _patch_technical_yfinance, "GET", "/api/v1/technical/AAPL/vcp", None, 200),
    (
        "52-week-position",
        _patch_technical_yfinance,
        "GET",
        "/api/v1/technical/AAPL/52-week-position",
        None,
        200,
    ),
    ("scan-bootstrap", _patch_snapshot, "GET", "/api/v1/scans/bootstrap", None, 404),
    (
        "scan-refresh-cache",
        _patch_refresh,
        "POST",
        "/api/v1/scans/refresh-cache",
        {"market": "US"},
        400,
    ),
]


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _disable_server_auth(monkeypatch):
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("patch", "method", "path", "body", "expected_status"),
    [case[1:] for case in _CASES],
    ids=[case[0] for case in _CASES],
)
async def test_blocking_endpoint_does_not_stall_other_requests(
    client, monkeypatch, patch, method, path, body, expected_status
):
    runtime = object()
    monkeypatch.setattr(app.state, "runtime_services", runtime, raising=False)
    gate = _Gate()
    patch(monkeypatch, gate)

    watchdog = threading.Timer(_WATCHDOG_SECONDS, gate.release.set)
    watchdog.start()
    try:
        slow = asyncio.create_task(client.request(method, path, json=body))
        assert await asyncio.to_thread(gate.entered.wait, _WATCHDOG_SECONDS)

        quick = await client.get("/livez")
        released_before_quick = gate.release.is_set()

        gate.release.set()
        slow_response = await slow
    finally:
        gate.release.set()
        watchdog.cancel()

    assert quick.status_code == 200
    assert not released_before_quick, f"{path} blocked the event loop until the watchdog fired"
    assert slow_response.status_code == expected_status, slow_response.text
    # The offloaded body still sees the request's runtime services.
    assert gate.runtime_seen is runtime


def test_route_modules_have_no_async_endpoints():
    """New handlers in these modules call synchronous services; keep them `def`."""
    for module in (stocks_module, technical_module, scans_module):
        offenders = [
            route.path
            for route in module.router.routes
            if inspect.iscoroutinefunction(getattr(route, "endpoint", None))
        ]
        assert offenders == [], f"{module.__name__} async endpoints: {offenders}"
