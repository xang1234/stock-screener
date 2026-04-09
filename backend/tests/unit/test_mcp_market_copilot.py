"""Unit tests for the Hermes Market Copilot service."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import delete

from app.interfaces.mcp.market_copilot import MarketCopilotService
from app.infra.db.models.feature_store import FeatureRun
from app.models.theme import ThemeMetrics
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
        "group_rankings",
        "stock_lookup",
        "stock_snapshot",
        "breadth_snapshot",
        "daily_digest",
    }


def test_market_overview_returns_expected_sections(read_only_service):
    payload = _tool_payload(read_only_service.call_tool("market_overview", {}))

    assert "Published feature run 2" in payload["summary"]
    assert payload["runs"]["selected"]["id"] == 2
    assert payload["breadth"]["date"] == "2026-03-29"
    assert payload["alerts"][0]["title"] == "AI Infrastructure velocity spike"
    assert payload["tasks"][0]["name"] == "daily-group-ranking-calculation"
    assert payload["top_candidates"][0]["symbol"] == "NVDA"
    assert payload["citations"][2]["as_of"] == "2026-03-29"
    assert payload["citations"][3]["as_of"] == "2026-03-29"


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


def test_watchlist_snapshot_does_not_treat_like_wildcards_as_fuzzy_matches(read_only_service):
    payload = _tool_payload(read_only_service.call_tool("watchlist_snapshot", {"watchlist": "%"}))

    assert "was not found" in payload["summary"]
    assert payload["watchlist"] is None


def test_theme_state_does_not_treat_like_wildcards_as_fuzzy_matches(read_only_service):
    payload = _tool_payload(read_only_service.call_tool("theme_state", {"theme_name": "%", "limit": 5}))

    assert "No active theme matched" in payload["summary"]
    assert payload["themes"] == []


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


def test_stock_snapshot_only_marks_theme_metrics_fresh_when_metrics_exist(session_factory, read_only_service):
    with session_factory() as db:
        run = db.query(FeatureRun).filter(FeatureRun.status == "published").order_by(FeatureRun.id.desc()).first()
        assert run is not None
        db.execute(delete(ThemeMetrics).where(ThemeMetrics.date == run.as_of_date))
        db.commit()

    payload = _tool_payload(read_only_service.call_tool("stock_snapshot", {"symbol": "NVDA"}))

    assert payload["themes"]
    assert all(theme["momentum_score"] is None for theme in payload["themes"])
    assert "theme_metrics" not in payload["freshness"]["sources"]


def test_stock_lookup_returns_universe_info(read_only_service):
    payload = _tool_payload(read_only_service.call_tool("stock_lookup", {"symbol": "nvda"}))

    assert payload["stock"]["symbol"] == "NVDA"
    assert payload["stock"]["name"] == "NVIDIA Corporation"
    assert payload["stock"]["sector"] == "Information Technology"
    assert payload["technicals"] is None


def test_stock_lookup_with_technicals(read_only_service):
    payload = _tool_payload(
        read_only_service.call_tool(
            "stock_lookup",
            {"symbol": "NVDA", "include_technicals": True},
        )
    )

    assert payload["stock"]["symbol"] == "NVDA"
    assert payload["technicals"] is not None
    assert payload["technicals"]["composite_score"] == 92.0
    assert payload["technicals"]["rs_rating"] == 95


def test_stock_lookup_unknown_symbol(read_only_service):
    payload = _tool_payload(read_only_service.call_tool("stock_lookup", {"symbol": "ZZZZ"}))

    assert payload["stock"] is None
    assert "not found" in payload["summary"]


def test_stock_snapshot_returns_combined_context(read_only_service):
    payload = _tool_payload(read_only_service.call_tool("stock_snapshot", {"symbol": "NVDA"}))

    assert payload["stock"]["symbol"] == "NVDA"
    assert payload["technicals"]["composite_score"] == 92.0
    assert payload["themes"][0]["display_name"] == "AI Infrastructure"
    assert payload["watchlists"][0]["name"] == "Leaders"
    assert payload["breadth"]["date"] == "2026-03-29"


def test_stock_snapshot_without_published_run_does_not_duplicate_theme_rows(session_factory):
    with session_factory() as db:
        db.query(FeatureRun).update({FeatureRun.status: "failed"})
        db.commit()

    service = MarketCopilotService(
        session_factory,
        SimpleNamespace(
            mcp_watchlist_writes_enabled=False,
            mcp_server_name="stockscreen-market-copilot",
        ),
    )

    payload = _tool_payload(service.call_tool("stock_snapshot", {"symbol": "NVDA"}))

    assert payload["technicals"] is None
    assert [theme["display_name"] for theme in payload["themes"]] == ["AI Infrastructure"]


def test_breadth_snapshot_returns_multiple_days(read_only_service):
    payload = _tool_payload(read_only_service.call_tool("breadth_snapshot", {"days": 5}))

    assert len(payload["snapshots"]) == 3  # only 3 seeded
    assert payload["snapshots"][0]["date"] == "2026-03-29"
    assert payload["snapshots"][1]["date"] == "2026-03-28"


def test_breadth_snapshot_default_days(read_only_service):
    payload = _tool_payload(read_only_service.call_tool("breadth_snapshot", {}))

    assert len(payload["snapshots"]) >= 1
    assert payload["facts"][0]["key"] == "latest_date"


def test_group_rankings_returns_rankings(read_only_service):
    payload = _tool_payload(
        read_only_service.call_tool("group_rankings", {"limit": 10, "period": "1w"})
    )

    assert len(payload["rankings"]) == 2
    assert payload["rankings"][0]["industry_group"] == "Semiconductors"
    assert payload["rankings"][0]["rank"] == 1


def test_daily_digest_returns_envelope(read_only_service):
    payload = _tool_payload(read_only_service.call_tool("daily_digest", {}))

    assert "digest" in payload
    assert payload["digest"]["as_of_date"] is not None
    assert payload["digest"]["market"]["stance"] is not None
    assert "Daily digest" in payload["summary"]


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
