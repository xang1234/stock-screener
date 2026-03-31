"""Integration test for the stdio MCP server transport."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.helpers.mcp_fixture import create_mcp_test_session_factory, seed_market_copilot_data
from tests.helpers.mcp_stdio import read_mcp_message, write_mcp_message


def test_mcp_server_lists_tools_and_serves_market_overview(tmp_path):
    backend_dir = Path(__file__).resolve().parents[2]
    database_path = tmp_path / "mcp-server.sqlite3"
    database_url = f"sqlite:///{database_path}"

    session_factory, _engine = create_mcp_test_session_factory(database_url)
    seed_market_copilot_data(session_factory)

    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    env["MCP_WATCHLIST_WRITES_ENABLED"] = "false"

    process = subprocess.Popen(
        [sys.executable, "-m", "app.interfaces.mcp.server"],
        cwd=backend_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    try:
        assert process.stdin is not None
        assert process.stdout is not None

        write_mcp_message(
            process.stdin,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "1.0"},
                },
            },
        )
        initialize = read_mcp_message(process.stdout)
        assert initialize["result"]["capabilities"]["tools"]["listChanged"] is False

        write_mcp_message(
            process.stdin,
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            },
        )

        write_mcp_message(
            process.stdin,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            },
        )
        tools_list = read_mcp_message(process.stdout)
        tool_names = {tool["name"] for tool in tools_list["result"]["tools"]}
        assert "market_overview" in tool_names
        assert "watchlist_add" in tool_names

        write_mcp_message(
            process.stdin,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "market_overview", "arguments": {}},
            },
        )
        tool_call = read_mcp_message(process.stdout)
        structured = tool_call["result"]["structuredContent"]
        assert "Published feature run 2" in structured["summary"]
        assert structured["runs"]["selected"]["id"] == 2
    finally:
        process.terminate()
        process.wait(timeout=5)
