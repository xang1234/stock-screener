"""The stdio entry point must build the process-scoped runtime services.

Only the FastAPI lifespan and the Celery worker built them, so over stdio every
tool that resolves a service failed with "RuntimeServices are not initialized" -
``group_rankings`` did so on main before this change. One call in ``main()``
makes the stdio transport behave like the other two entry points.
"""

from __future__ import annotations

from app.interfaces.mcp import server as server_module


def test_main_initializes_runtime_services_before_serving(monkeypatch):
    """One initialisation call, before the transport starts serving."""
    calls: list[str] = []

    def fake_initialize(*, session_factory=None, force=False):
        calls.append(f"initialize:{force}")
        return object()

    monkeypatch.setattr(server_module, "initialize_process_runtime_services", fake_initialize)
    monkeypatch.setattr(
        server_module.MarketCopilotMcpServer,
        "serve_forever",
        lambda self: calls.append("serve"),
    )

    server_module.main()

    assert calls == ["initialize:False", "serve"]


def test_the_initialised_process_scope_resolves_the_shared_services(tmp_path):
    """After the call the stdio process resolves what the API resolves.

    ``group_rankings`` and ``market_calendar_status`` both go through this
    resolver, and both are the same process-scoped instance on every call - a
    private instance per tool would hide a broken scope instead of reporting it.
    """
    from app.wiring import runtime_context
    from app.wiring.bootstrap import get_group_rank_service, get_market_calendar_service
    from tests.helpers.mcp_fixture import create_mcp_test_session_factory

    session_factory, _engine = create_mcp_test_session_factory(
        f"sqlite:///{tmp_path / 'stdio-scope.sqlite3'}"
    )

    try:
        server_module.initialize_process_runtime_services(session_factory=session_factory)

        assert get_market_calendar_service() is get_market_calendar_service()
        assert get_group_rank_service() is get_group_rank_service()
    finally:
        runtime_context.clear_runtime_services()
