"""Golden-style snapshots for stable MCP tool outputs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.interfaces.mcp.market_copilot import MarketCopilotService
from tests.helpers.mcp_fixture import create_mcp_test_session_factory, seed_market_copilot_data

_SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"


@pytest.fixture()
def service():
    factory, _engine = create_mcp_test_session_factory()
    seed_market_copilot_data(factory)
    return MarketCopilotService(
        factory,
        SimpleNamespace(
            mcp_watchlist_writes_enabled=False,
            mcp_server_name="stockscreen-market-copilot",
        ),
    )


def _normalized_payload(service: MarketCopilotService, tool_name: str, arguments: dict) -> dict:
    result = service.call_tool(tool_name, arguments)
    assert result.get("isError") is not True
    payload = result["structuredContent"]
    payload["freshness"]["generated_at"] = "<generated_at>"
    return payload


def _assert_snapshot_matches(snapshot_name: str, payload: dict) -> None:
    expected = json.loads((_SNAPSHOT_DIR / snapshot_name).read_text())
    assert payload == expected


def test_market_overview_snapshot(service):
    payload = _normalized_payload(service, "market_overview", {})
    _assert_snapshot_matches("mcp_market_overview.json", payload)


def test_compare_feature_runs_snapshot(service):
    payload = _normalized_payload(service, "compare_feature_runs", {})
    _assert_snapshot_matches("mcp_compare_feature_runs.json", payload)


def test_watchlist_snapshot_snapshot(service):
    payload = _normalized_payload(service, "watchlist_snapshot", {"watchlist": "Leaders"})
    _assert_snapshot_matches("mcp_watchlist_snapshot.json", payload)
