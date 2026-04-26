"""Unit tests for local runtime bootstrap status behavior."""

from __future__ import annotations


def test_non_empty_resume_state_does_not_require_bootstrap(monkeypatch):
    from app.services import runtime_preferences_service as module

    prefs = module.RuntimePreferences(
        primary_market="US",
        enabled_markets=["US"],
        bootstrap_state="not_started",
    )

    monkeypatch.setattr(module, "get_runtime_preferences", lambda _db: prefs)
    monkeypatch.setattr(module, "_has_active_universe_rows", lambda _db, market=None: market is None)
    monkeypatch.setattr(module, "_has_price_rows", lambda _db, market=None: False)
    monkeypatch.setattr(module, "_has_fundamental_rows", lambda _db, market=None: False)
    monkeypatch.setattr(module, "_has_core_market_data", lambda _db, market: False)

    status = module.get_runtime_bootstrap_status(object())

    assert status.empty_system is False
    assert status.bootstrap_required is False
    assert status.bootstrap_state == "not_started"


def test_running_bootstrap_stays_required_until_all_enabled_markets_have_scans(monkeypatch):
    from app.services import runtime_preferences_service as module

    prefs = module.RuntimePreferences(
        primary_market="US",
        enabled_markets=["US", "HK"],
        bootstrap_state="running",
    )

    monkeypatch.setattr(module, "get_runtime_preferences", lambda _db: prefs)
    monkeypatch.setattr(module, "_has_active_universe_rows", lambda _db, market=None: True)
    monkeypatch.setattr(module, "_has_price_rows", lambda _db, market=None: False)
    monkeypatch.setattr(module, "_has_fundamental_rows", lambda _db, market=None: False)
    monkeypatch.setattr(module, "_has_core_market_data", lambda _db, market: False)
    monkeypatch.setattr(module, "_has_completed_auto_scan", lambda _db, market: market == "US")

    status = module.get_runtime_bootstrap_status(object())

    assert status.empty_system is False
    assert status.bootstrap_required is True
    assert status.bootstrap_state == "running"


def test_bootstrap_ready_requires_completed_auto_scans_for_every_enabled_market(monkeypatch):
    from app.services import runtime_preferences_service as module

    prefs = module.RuntimePreferences(
        primary_market="US",
        enabled_markets=["US", "HK"],
        bootstrap_state="running",
    )

    monkeypatch.setattr(module, "get_runtime_preferences", lambda _db: prefs)
    monkeypatch.setattr(module, "_has_active_universe_rows", lambda _db, market=None: True)
    monkeypatch.setattr(module, "_has_price_rows", lambda _db, market=None: True)
    monkeypatch.setattr(module, "_has_fundamental_rows", lambda _db, market=None: True)
    monkeypatch.setattr(module, "_has_core_market_data", lambda _db, market: True)
    monkeypatch.setattr(module, "_has_completed_auto_scan", lambda _db, market: True)

    status = module.get_runtime_bootstrap_status(object())

    assert status.bootstrap_required is False
    assert status.bootstrap_state == "ready"
