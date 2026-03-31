"""Unit tests for the Hermes Market Copilot service."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.interfaces.mcp.market_copilot import MarketCopilotService
from tests.helpers.mcp_fixture import create_mcp_test_session_factory, seed_market_copilot_data


@pytest.fixture()
def session_factory():
    factory, _engine = create_mcp_test_session_factory()
    seed_market_copilot_data(factory)
    return factory


@pytest.fixture()
def read_only_service(session_factory):
    return MarketCopilotService(
        session_factory,
        SimpleNamespace(
            mcp_watchlist_writes_enabled=False,
            mcp_server_name="stockscreen-market-copilot",
        ),
    )


def _tool_payload(result: dict) -> dict:
    assert result.get("isError") is not True
    return result["structuredContent"]


def test_lists_expected_tools(read_only_service):
    tool_names = {tool["name"] for tool in read_only_service.list_tools()}
    assert tool_names == {
        "market_overview",
        "compare_feature_runs",
        "find_candidates",
        "explain_symbol",
        "watchlist_snapshot",
        "theme_state",
        "task_status",
        "watchlist_add",
    }


def test_market_overview_returns_expected_sections(read_only_service):
    payload = _tool_payload(read_only_service.call_tool("market_overview", {}))

    assert "Published feature run 2" in payload["summary"]
    assert payload["runs"]["selected"]["id"] == 2
    assert payload["breadth"]["date"] == "2026-03-29"
    assert payload["alerts"][0]["title"] == "AI Infrastructure velocity spike"
    assert payload["tasks"][0]["name"] == "daily-group-ranking-calculation"
    assert payload["top_candidates"][0]["symbol"] == "NVDA"


def test_find_candidates_applies_filters(read_only_service):
    payload = _tool_payload(
        read_only_service.call_tool(
            "find_candidates",
            {
                "filters": {
                    "min_score": 83,
                    "stage": 2,
                    "sort_field": "composite_score",
                    "sort_order": "desc",
                },
                "limit": 10,
            },
        )
    )

    symbols = [row["symbol"] for row in payload["symbols"]]
    assert symbols == ["NVDA", "PANW", "AVGO"]


def test_explain_symbol_full_returns_explanation_and_peers(read_only_service):
    payload = _tool_payload(
        read_only_service.call_tool(
            "explain_symbol",
            {"symbol": "nvda", "depth": "full"},
        )
    )

    assert payload["symbol"] == "NVDA"
    assert payload["result"]["rating"] == "Strong Buy"
    assert payload["explanation"]["symbol"] == "NVDA"
    assert payload["peers"][0]["symbol"] == "AVGO"
    assert payload["setup_payload"]["se_explain"]["thesis"].startswith("NVDA")


def test_theme_state_for_named_theme_returns_constituents(read_only_service):
    payload = _tool_payload(
        read_only_service.call_tool(
            "theme_state",
            {"theme_name": "AI", "limit": 5},
        )
    )

    assert payload["theme"]["display_name"] == "AI Infrastructure"
    assert [row["symbol"] for row in payload["constituents"]] == ["NVDA", "AVGO"]
    assert payload["alerts"][0]["alert_type"] == "velocity_spike"


def test_task_status_unknown_task_returns_guidance(read_only_service):
    payload = _tool_payload(
        read_only_service.call_tool(
            "task_status",
            {"task_name": "not-a-real-task"},
        )
    )

    assert "not registered" in payload["summary"]
    assert payload["tasks"] == []
    assert payload["next_actions"]


def test_task_status_handles_tasks_without_history(read_only_service):
    payload = _tool_payload(read_only_service.call_tool("task_status", {}))

    assert payload["tasks"]
    assert any(task["last_run"] is None for task in payload["tasks"])
    assert "scheduled tasks tracked" in payload["summary"]


def test_watchlist_add_respects_write_gate(read_only_service):
    result = read_only_service.call_tool(
        "watchlist_add",
        {"watchlist": "Leaders", "symbols": ["MSFT"], "reason": "Fixture add"},
    )
    payload = _tool_payload(result)

    assert payload["writes_enabled"] is False
    assert payload["added"] == []
    assert payload["skipped"] == ["MSFT"]


def test_watchlist_add_writes_when_enabled(session_factory):
    service = MarketCopilotService(
        session_factory,
        SimpleNamespace(
            mcp_watchlist_writes_enabled=True,
            mcp_server_name="stockscreen-market-copilot",
        ),
    )
    payload = _tool_payload(
        service.call_tool(
            "watchlist_add",
            {"watchlist": "Leaders", "symbols": ["MSFT", "NVDA"], "reason": "Promote"},
        )
    )

    assert payload["writes_enabled"] is True
    assert payload["added"] == [{"symbol": "MSFT", "display_name": "Microsoft Corporation", "reason": "Promote"}]
    assert payload["skipped"] == ["NVDA"]

    snapshot = _tool_payload(service.call_tool("watchlist_snapshot", {"watchlist": "Leaders"}))
    symbols = [row["symbol"] for row in snapshot["items"]]
    assert symbols == ["NVDA", "MSFT"]


def test_invalid_sort_field_returns_tool_error(read_only_service):
    result = read_only_service.call_tool(
        "find_candidates",
        {"filters": {"sort_field": "not_supported"}},
    )

    assert result["isError"] is True
    assert result["structuredContent"]["error"]["code"] == "invalid_arguments"


def test_invalid_sort_field_returns_tool_error_for_watchlist_scope(read_only_service):
    result = read_only_service.call_tool(
        "find_candidates",
        {"universe": "Leaders", "filters": {"sort_field": "not_supported"}},
    )

    assert result["isError"] is True
    assert result["structuredContent"]["error"]["code"] == "invalid_arguments"
