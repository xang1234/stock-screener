"""Unit tests for runtime capability endpoints."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from pydantic import ValidationError

from app.database import get_db
from app.domain.markets.catalog import get_market_catalog
from app.domain.scanning.defaults import get_default_scan_profile
from app.main import app
from app.schemas.app_runtime import (
    MarketCatalogEntryResponse,
    RuntimeUniverseSelectionResponse,
)


class _FakeUISnapshotService:
    def ui_snapshot_flags(self) -> dict[str, bool]:
        return {
            "enabled": True,
            "scan": True,
            "breadth": False,
            "groups": True,
            "themes": False,
        }


class _FakeDb:
    pass


class _FakeBootstrapStatus:
    bootstrap_required = True
    empty_system = True
    primary_market = "US"
    enabled_markets = ["US", "HK"]
    bootstrap_state = "running"
    supported_markets = ["US", "HK", "JP", "TW"]


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_app_capabilities_includes_scan_defaults(client, monkeypatch):
    from app.api.v1 import app_runtime as module

    monkeypatch.setattr(
        type(module.settings),
        "capability_flags",
        lambda _self: {"themes": True, "chatbot": True, "tasks": True},
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_ui_snapshot_service",
        lambda: _FakeUISnapshotService(),
    )
    monkeypatch.setattr(module, "get_runtime_bootstrap_status", lambda _db: _FakeBootstrapStatus())
    app.dependency_overrides[get_db] = lambda: _FakeDb()

    try:
        response = await client.get("/api/v1/app-capabilities")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    data = response.json()
    assert data["scan_defaults"] == get_default_scan_profile("US")
    assert data["ui_snapshots"] == _FakeUISnapshotService().ui_snapshot_flags()
    assert data["features"] == {"themes": True, "chatbot": True, "tasks": True}
    assert data["bootstrap_required"] is True
    assert data["primary_market"] == "US"
    assert data["enabled_markets"] == ["US", "HK"]
    assert data["bootstrap_state"] == "running"
    assert data["supported_markets"] == get_market_catalog().supported_market_codes()
    assert data["market_catalog"]["version"]
    market_catalog = {
        market["code"]: market
        for market in data["market_catalog"]["markets"]
    }
    assert market_catalog["US"]["label"] == "United States"
    assert market_catalog["US"]["primary_mic"] == "XNYS"
    assert market_catalog["US"]["supported_currencies"] == ["USD"]
    assert market_catalog["US"]["default_currency"] == "USD"
    assert market_catalog["US"]["mic_facts"][0]["mic"] == "XNYS"
    assert market_catalog["HK"]["capabilities"]["finviz_screening"] is False
    assert data["api_base_path"] == "/api"


@pytest.mark.asyncio
async def test_app_capabilities_exposes_catalog_backed_universe_options(client, monkeypatch):
    from app.api.v1 import app_runtime as module

    class _OnlyHongKongEnabledStatus(_FakeBootstrapStatus):
        primary_market = "HK"
        enabled_markets = ["HK"]

    monkeypatch.setattr(
        type(module.settings),
        "capability_flags",
        lambda _self: {"themes": True, "chatbot": True, "tasks": True},
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_ui_snapshot_service",
        lambda: _FakeUISnapshotService(),
    )
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _OnlyHongKongEnabledStatus(),
    )
    app.dependency_overrides[get_db] = lambda: _FakeDb()

    try:
        response = await client.get("/api/v1/app-capabilities")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    data = response.json()
    options = data["universe_options"]
    catalog = get_market_catalog()
    assert options["version"] == data["market_catalog"]["version"]
    assert options["supported_markets"] == catalog.supported_market_codes()
    assert options["enabled_markets"] == ["HK"]

    markets = {market["code"]: market for market in options["markets"]}
    assert set(markets) == set(catalog.supported_market_codes())
    assert markets["US"]["enabled"] is False
    assert markets["HK"]["enabled"] is True
    assert markets["US"]["market"]["universe_def"] == {
        "type": "market",
        "market": "US",
    }
    assert markets["HK"]["capabilities"]["official_universe"] is True

    us_mics = {mic["mic"]: mic for mic in markets["US"]["mics"]}
    assert us_mics["XNYS"]["universe_def"] == {
        "type": "market",
        "market": "US",
        "mic": "XNYS",
    }
    assert "NYSE" in us_mics["XNYS"]["aliases"]

    us_aliases = {
        (alias["alias"], alias["mic"])
        for alias in markets["US"]["mic_aliases"]
    }
    assert ("NYSE", "XNYS") in us_aliases

    us_indexes = {index["key"]: index for index in markets["US"]["indexes"]}
    assert us_indexes["SP500"]["universe_def"] == {
        "type": "index",
        "index": "SP500",
    }

    hk_listing_tiers = {
        tier["key"]: tier
        for tier in markets["HK"]["listing_tiers"]
    }
    assert hk_listing_tiers["main_board"]["universe_def"] == {
        "type": "market",
        "market": "HK",
        "mic": "XHKG",
        "listing_tier": "main_board",
    }

    assert "readiness" not in options
    for market in options["markets"]:
        assert "scan_ready" not in market
        assert "status" not in market
        assert "bootstrap_state" not in market


def test_runtime_universe_option_rejects_invalid_universe_definition():
    with pytest.raises(ValidationError):
        RuntimeUniverseSelectionResponse.model_validate(
            {
                "value": "market:NOPE",
                "label": "Invalid market",
                "universe_def": {"type": "market", "market": "NOPE"},
            }
        )


def test_market_catalog_runtime_schema_requires_canonical_market_facts():
    payload = get_market_catalog().get("US").as_runtime_payload()
    for key in (
        "primary_mic",
        "mics",
        "supported_currencies",
        "default_currency",
        "mic_facts",
    ):
        missing_payload = dict(payload)
        missing_payload.pop(key)

        with pytest.raises(ValueError):
            MarketCatalogEntryResponse(**missing_payload)


@pytest.mark.asyncio
async def test_runtime_bootstrap_status_endpoint_returns_persisted_state(client, monkeypatch):
    from app.api.v1 import app_runtime as module

    monkeypatch.setattr(module, "get_runtime_bootstrap_status", lambda _db: _FakeBootstrapStatus())
    app.dependency_overrides[get_db] = lambda: _FakeDb()

    try:
        response = await client.get("/api/v1/runtime/bootstrap-status")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload["bootstrap_required"] is True
    assert payload["empty_system"] is True
    assert payload["primary_market"] == "US"
    assert payload["enabled_markets"] == ["US", "HK"]
    assert payload["bootstrap_state"] == "running"


@pytest.mark.asyncio
async def test_runtime_activity_endpoint_returns_runtime_activity_payload(client, monkeypatch):
    from app.api.v1 import app_runtime as module

    monkeypatch.setattr(
        module,
        "get_runtime_activity_status",
        lambda _db: {
            "bootstrap": {
                "state": "running",
                "app_ready": False,
                "primary_market": "US",
                "enabled_markets": ["US", "HK"],
                "current_stage": "Price Refresh",
                "progress_mode": "indeterminate",
                "percent": None,
                "message": "Refreshing prices",
                "background_warning": "Additional data loading continues in the background.",
            },
            "summary": {
                "active_market_count": 1,
                "active_markets": ["US"],
                "status": "active",
            },
            "markets": [
                {
                    "market": "US",
                    "lifecycle": "bootstrap",
                    "stage_key": "prices",
                    "stage_label": "Price Refresh",
                    "status": "running",
                    "progress_mode": "determinate",
                    "percent": 50.0,
                    "current": 50,
                    "total": 100,
                    "message": "Refreshing prices",
                    "task_name": "smart_refresh_cache",
                    "task_id": "task-us",
                    "updated_at": "2026-04-18T12:00:00+00:00",
                }
            ],
        },
    )
    app.dependency_overrides[get_db] = lambda: _FakeDb()

    try:
        response = await client.get("/api/v1/runtime/activity")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload["bootstrap"]["current_stage"] == "Price Refresh"
    assert payload["bootstrap"]["progress_mode"] == "indeterminate"
    assert payload["bootstrap"]["percent"] is None
    assert payload["summary"]["active_markets"] == ["US"]
    assert payload["markets"][0]["progress_mode"] == "determinate"
    assert payload["markets"][0]["task_name"] == "smart_refresh_cache"


@pytest.mark.asyncio
async def test_runtime_bootstrap_start_persists_preferences_and_queues_orchestration(client, monkeypatch):
    from app.api.v1 import app_runtime as module
    from app.services import server_auth

    saved_states = []

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    def _save(_db, *, primary_market, enabled_markets, bootstrap_state):
        saved_states.append(bootstrap_state)
        return type(
            "SavedPrefs",
            (),
            {
                "primary_market": primary_market,
                "enabled_markets": enabled_markets,
                "bootstrap_state": bootstrap_state,
            },
        )()

    statuses = iter([
        type(
            "BootstrapStatus",
            (),
            {
                "bootstrap_required": True,
                "empty_system": True,
                "primary_market": "US",
                "enabled_markets": ["US"],
                "bootstrap_state": "not_started",
                "supported_markets": ["US", "HK", "JP", "TW"],
            },
        )(),
        type(
            "BootstrapStatus",
            (),
            {
                "bootstrap_required": True,
                "empty_system": True,
                "primary_market": "HK",
                "enabled_markets": ["HK", "US"],
                "bootstrap_state": "running",
                "supported_markets": ["US", "HK", "JP", "TW"],
            },
        )(),
    ])

    monkeypatch.setattr(module, "save_runtime_preferences", _save)
    monkeypatch.setattr(module, "queue_local_runtime_bootstrap", lambda **_kwargs: "task-bootstrap-123")
    monkeypatch.setattr(module, "get_runtime_bootstrap_status", lambda _db: next(statuses))
    app.dependency_overrides[get_db] = lambda: _FakeDb()

    try:
        response = await client.post(
            "/api/v1/runtime/bootstrap",
            json={"primary_market": "HK", "enabled_markets": ["HK", "US"]},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    assert saved_states == ["running"]
    payload = response.json()
    assert payload["task_id"] == "task-bootstrap-123"
    assert payload["primary_market"] == "HK"
    assert payload["enabled_markets"] == ["HK", "US"]
    assert payload["bootstrap_state"] == "running"


@pytest.mark.asyncio
async def test_runtime_bootstrap_mutations_require_auth(client, monkeypatch):
    from app.api.v1 import app_runtime as module
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", True)
    monkeypatch.setattr(server_auth.settings, "server_auth_password", "secret-pass")
    monkeypatch.setattr(server_auth.settings, "server_auth_session_secret", "secret-signing-key")
    monkeypatch.setattr(
        module,
        "save_runtime_preferences",
        lambda _db, *, primary_market, enabled_markets, bootstrap_state: type(
            "SavedPrefs",
            (),
            {
                "primary_market": primary_market,
                "enabled_markets": enabled_markets,
                "bootstrap_state": bootstrap_state,
            },
        )(),
    )
    monkeypatch.setattr(module, "queue_local_runtime_bootstrap", lambda **_kwargs: "task-bootstrap-123")
    monkeypatch.setattr(module, "get_runtime_bootstrap_status", lambda _db: _FakeBootstrapStatus())
    app.dependency_overrides[get_db] = lambda: _FakeDb()

    try:
        unauth_bootstrap = await client.post(
            "/api/v1/runtime/bootstrap",
            json={"primary_market": "US", "enabled_markets": ["US"]},
        )
        unauth_markets = await client.patch(
            "/api/v1/runtime/markets",
            json={"primary_market": "US", "enabled_markets": ["US"]},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert unauth_bootstrap.status_code == 401
    assert unauth_markets.status_code == 401


@pytest.mark.asyncio
async def test_runtime_bootstrap_start_does_not_persist_running_state_when_queue_fails(client, monkeypatch):
    from app.api.v1 import app_runtime as module
    from app.services import server_auth

    saved_states = []

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    current_status = type(
        "BootstrapStatus",
        (),
        {
            "bootstrap_required": True,
            "empty_system": True,
            "primary_market": "US",
            "enabled_markets": ["US"],
            "bootstrap_state": "not_started",
            "supported_markets": ["US", "HK", "JP", "TW"],
        },
    )()

    def _save(_db, *, primary_market, enabled_markets, bootstrap_state):
        saved_states.append(bootstrap_state)
        return type(
            "SavedPrefs",
            (),
            {
                "primary_market": primary_market,
                "enabled_markets": enabled_markets,
                "bootstrap_state": bootstrap_state,
            },
        )()

    monkeypatch.setattr(module, "get_runtime_bootstrap_status", lambda _db: current_status)
    monkeypatch.setattr(module, "save_runtime_preferences", _save)
    monkeypatch.setattr(
        module,
        "queue_local_runtime_bootstrap",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("queue unavailable")),
    )
    app.dependency_overrides[get_db] = lambda: _FakeDb()

    try:
        with pytest.raises(RuntimeError, match="queue unavailable"):
            await client.post(
                "/api/v1/runtime/bootstrap",
                json={"primary_market": "US", "enabled_markets": ["US"]},
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert saved_states == ["running", "not_started"]


@pytest.mark.asyncio
async def test_runtime_bootstrap_start_rejects_duplicate_running_bootstrap(client, monkeypatch):
    from app.api.v1 import app_runtime as module
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: type(
            "BootstrapStatus",
            (),
            {
                "bootstrap_required": True,
                "empty_system": True,
                "primary_market": "US",
                "enabled_markets": ["US"],
                "bootstrap_state": "running",
                "supported_markets": ["US", "HK", "JP", "TW"],
            },
        )(),
    )
    app.dependency_overrides[get_db] = lambda: _FakeDb()

    try:
        response = await client.post(
            "/api/v1/runtime/bootstrap",
            json={"primary_market": "US", "enabled_markets": ["US"]},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "bootstrap_already_running"
