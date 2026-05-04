"""Unit tests for local runtime bootstrap status behavior."""

from __future__ import annotations


class _FakeBootstrapReadinessService:
    def __init__(self, module, *, empty_system, market_results, calls=None):
        self._module = module
        self._empty_system = empty_system
        self._market_results = market_results
        self.calls = calls if calls is not None else []

    def evaluate(self, db, *, enabled_markets, bootstrap_started_at=None):
        self.calls.append((db, list(enabled_markets), bootstrap_started_at))
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

    assert calls == [(db, ["US", "HK"], None)]
    assert status.bootstrap_required is True
    assert status.bootstrap_state == "running"


def test_bootstrap_status_passes_bootstrap_start_boundary_to_readiness(monkeypatch):
    from datetime import datetime, timezone

    from app.services import runtime_preferences_service as module

    bootstrap_started_at = datetime(2026, 5, 4, 9, 30, tzinfo=timezone.utc)
    prefs = module.RuntimePreferences(
        primary_market="US",
        enabled_markets=["US"],
        bootstrap_state="running",
        bootstrap_started_at=bootstrap_started_at,
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
            market_results=[("US", True, True)],
            calls=calls,
        ),
    )

    status = module.get_runtime_bootstrap_status(db)

    assert calls == [(db, ["US"], bootstrap_started_at)]
    assert status.bootstrap_state == "ready"


def test_save_runtime_preferences_preserves_running_bootstrap_start_boundary() -> None:
    from datetime import datetime, timezone

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database import Base
    from app.models.app_settings import AppSetting
    from app.services import runtime_preferences_service as module

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(engine)()
    try:
        existing_started_at = datetime(2026, 5, 4, 9, 30, tzinfo=timezone.utc)
        db.add(
            AppSetting(
                key=module.BOOTSTRAP_STARTED_AT_KEY,
                value=existing_started_at.isoformat(),
                category=module.RUNTIME_SETTINGS_CATEGORY,
            )
        )
        db.commit()

        prefs = module.save_runtime_preferences(
            db,
            primary_market="US",
            enabled_markets=["US", "HK"],
            bootstrap_state="running",
        )

        persisted_started_at = (
            db.query(AppSetting)
            .filter(AppSetting.key == module.BOOTSTRAP_STARTED_AT_KEY)
            .one()
        )
        assert prefs.bootstrap_started_at == existing_started_at
        assert persisted_started_at.value == existing_started_at.isoformat()
    finally:
        db.close()
        engine.dispose()
