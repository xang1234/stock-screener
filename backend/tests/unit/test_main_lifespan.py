from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock


def test_initialize_runtime_seeds_declared_social_sources(monkeypatch, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app import main as module
    from app.database import Base
    from app.infra.db.models.social_signals import SocialSourceConfiguration
    from app.services.social_source_admin_service import SEED_SOCIAL_SOURCES

    engine = create_engine(f"sqlite:///{tmp_path / 'startup.sqlite'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine)
    monkeypatch.setattr(module, "engine", engine)
    monkeypatch.setattr(module, "SessionLocal", sessions)
    monkeypatch.setattr(module, "migrate_database_to_head", lambda selected: "current")

    module.initialize_runtime()

    with sessions() as db:
        rows = db.query(SocialSourceConfiguration).order_by(
            SocialSourceConfiguration.x_list_id
        ).all()
        assert [(row.x_list_id, row.lifecycle_state, row.provenance) for row in rows] == sorted(
            (list_id, "enabled", "system_seed")
            for list_id, _name, _url in SEED_SOCIAL_SOURCES
        )
    engine.dispose()


def test_group_history_startup_trigger_uses_shutdown_independent_daemon(monkeypatch):
    from app import main as module

    publisher = Mock(name="publisher-thread")
    thread_factory = Mock(return_value=publisher)
    monkeypatch.setattr(module.threading, "Thread", thread_factory)

    result = module.trigger_group_history_reconciliation_on_startup()

    assert result == {"status": "dispatching"}
    thread_factory.assert_called_once_with(
        target=module._publish_group_history_reconciliation,
        name="group-history-startup-publisher",
        daemon=True,
    )
    publisher.start.assert_called_once_with()


def test_daemon_publisher_can_remain_blocked_without_lifespan_owning_it(monkeypatch):
    from app import main as module

    dispatch_started = threading.Event()
    release_dispatch = threading.Event()
    discovery = Mock()
    discovery.delay.side_effect = lambda: (
        dispatch_started.set(),
        release_dispatch.wait(),
        SimpleNamespace(id="discovery-1"),
    )[-1]
    monkeypatch.setattr(
        "app.tasks.group_history_tasks.discover_group_history_reconciliation",
        discovery,
    )

    monkeypatch.setattr(module, "initialize_runtime", Mock())
    monkeypatch.setattr(
        module,
        "initialize_process_runtime_services",
        Mock(return_value=object()),
    )
    monkeypatch.setattr(module, "clear_runtime_services", Mock())
    monkeypatch.setattr(module.settings, "mcp_http_enabled", False)
    monkeypatch.setattr(module.engine, "dispose", Mock())
    test_app = SimpleNamespace(state=SimpleNamespace())

    async def run_lifespan():
        async with module.lifespan(test_app):
            assert dispatch_started.wait(timeout=0.1)

    try:
        asyncio.run(asyncio.wait_for(run_lifespan(), timeout=0.2))
    finally:
        release_dispatch.set()
