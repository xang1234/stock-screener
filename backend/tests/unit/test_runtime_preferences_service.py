"""Unit tests for local runtime bootstrap status behavior."""

from __future__ import annotations


class _FakeBootstrapReadinessService:
    def __init__(self, module, *, empty_system, market_results, calls=None):
        self._module = module
        self._empty_system = empty_system
        self._market_results = market_results
        self.calls = calls if calls is not None else []

    def evaluate(self, db, *, enabled_markets):
        self.calls.append((db, list(enabled_markets)))
        return self._module.BootstrapReadiness(
            empty_system=self._empty_system,
            market_results={
                market: self._module.MarketBootstrapReadiness(
                    market=market,
                    core_ready=core_ready,
                    scan_ready=scan_ready,
                )
                for market, core_ready, scan_ready in self._market_results
            },
        )


def test_non_empty_resume_state_does_not_require_bootstrap(monkeypatch):
    from app.services import runtime_preferences_service as module

    prefs = module.RuntimePreferences(
        primary_market="US",
        enabled_markets=["US"],
        bootstrap_state="not_started",
    )

    monkeypatch.setattr(module, "get_runtime_preferences", lambda _db: prefs)
    monkeypatch.setattr(
        module,
        "get_bootstrap_readiness_service",
        lambda: _FakeBootstrapReadinessService(
            module,
            empty_system=False,
            market_results=[("US", False, False)],
        ),
    )

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
    monkeypatch.setattr(
        module,
        "get_bootstrap_readiness_service",
        lambda: _FakeBootstrapReadinessService(
            module,
            empty_system=False,
            market_results=[
                ("US", True, True),
                ("HK", False, False),
            ],
        ),
    )

    status = module.get_runtime_bootstrap_status(object())

    assert status.empty_system is False
    assert status.bootstrap_required is True
    assert status.bootstrap_state == "running"


def test_failed_bootstrap_stays_required_until_all_enabled_markets_have_scans(monkeypatch):
    from app.services import runtime_preferences_service as module

    prefs = module.RuntimePreferences(
        primary_market="US",
        enabled_markets=["US", "HK"],
        bootstrap_state="failed",
    )

    monkeypatch.setattr(module, "get_runtime_preferences", lambda _db: prefs)
    monkeypatch.setattr(
        module,
        "get_bootstrap_readiness_service",
        lambda: _FakeBootstrapReadinessService(
            module,
            empty_system=False,
            market_results=[
                ("US", True, True),
                ("HK", True, False),
            ],
        ),
    )

    status = module.get_runtime_bootstrap_status(object())

    assert status.empty_system is False
    assert status.bootstrap_required is True
    assert status.bootstrap_state == "failed"


def test_bootstrap_ready_requires_completed_auto_scans_for_every_enabled_market(monkeypatch):
    from app.services import runtime_preferences_service as module

    prefs = module.RuntimePreferences(
        primary_market="US",
        enabled_markets=["US", "HK"],
        bootstrap_state="running",
    )

    monkeypatch.setattr(module, "get_runtime_preferences", lambda _db: prefs)
    monkeypatch.setattr(
        module,
        "get_bootstrap_readiness_service",
        lambda: _FakeBootstrapReadinessService(
            module,
            empty_system=False,
            market_results=[
                ("US", True, True),
                ("HK", True, True),
            ],
        ),
    )

    status = module.get_runtime_bootstrap_status(object())

    assert status.bootstrap_required is False
    assert status.bootstrap_state == "ready"


def test_bootstrap_status_uses_readiness_service(monkeypatch):
    from app.services import runtime_preferences_service as module

    prefs = module.RuntimePreferences(
        primary_market="US",
        enabled_markets=["US", "HK"],
        bootstrap_state="ready",
    )
    db = object()
    calls = []

    monkeypatch.setattr(module, "get_runtime_preferences", lambda _db: prefs)
    monkeypatch.setattr(
        module,
        "get_bootstrap_readiness_service",
        lambda: _FakeBootstrapReadinessService(
            module,
            empty_system=False,
            market_results=[
                ("US", True, True),
                ("HK", False, True),
            ],
            calls=calls,
        ),
    )

    status = module.get_runtime_bootstrap_status(db)

    assert calls == [(db, ["US", "HK"])]
    assert status.bootstrap_required is True
    assert status.bootstrap_state == "running"
