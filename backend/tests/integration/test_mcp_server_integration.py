"""Integration tests for the stdio and HTTP MCP server transports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import httpx
import pytest
import pytest_asyncio

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


@pytest.mark.asyncio
class TestMcpHttpTransport:
    """Integration tests for the Streamable HTTP MCP transport."""

    @pytest_asyncio.fixture()
    async def client(self, tmp_path):
        """Create an httpx client with an isolated FastAPI app and seeded test data."""
        from types import SimpleNamespace

        from fastapi import FastAPI

        from app.interfaces.mcp.http_transport import mcp_router
        from app.interfaces.mcp.market_copilot import MarketCopilotService
        from app.interfaces.mcp.server import MarketCopilotMcpServer
        import app.interfaces.mcp.http_transport as http_mod

        database_path = tmp_path / "mcp-http.sqlite3"
        database_url = f"sqlite:///{database_path}"

        session_factory, test_engine = create_mcp_test_session_factory(database_url)
        seed_market_copilot_data(session_factory)

        test_settings = SimpleNamespace(
            mcp_server_name="test-mcp-http",
            mcp_watchlist_writes_enabled=False,
        )
        test_service = MarketCopilotService(session_factory, test_settings)
        test_server = MarketCopilotMcpServer(test_service, transport=None)

        # Override the lazy singleton so the router uses our test server
        original_server = http_mod._server
        http_mod._server = test_server

        # Build an isolated app with only the MCP router — avoids dependency
        # on app.main's import-time MCP_HTTP_ENABLED gate.
        test_app = FastAPI()
        test_app.include_router(mcp_router, prefix="/mcp")

        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

        http_mod._server = original_server

    async def test_initialize_returns_capabilities(self, client):
        resp = await client.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest-http", "version": "1.0"},
                },
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["result"]["capabilities"]["tools"]["listChanged"] is False
        assert "name" in body["result"]["serverInfo"]

    async def test_tools_list_returns_12_tools(self, client):
        resp = await client.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            },
        )
        body = resp.json()
        tool_names = {t["name"] for t in body["result"]["tools"]}
        assert len(tool_names) == 12
        assert "group_rankings" in tool_names
        assert "stock_lookup" in tool_names
        assert "breadth_snapshot" in tool_names
        assert "daily_digest" in tool_names

    async def test_tool_call_market_overview(self, client):
        resp = await client.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "market_overview", "arguments": {}},
            },
        )
        body = resp.json()
        structured = body["result"]["structuredContent"]
        assert "Published feature run 2" in structured["summary"]

    async def test_notification_returns_202(self, client):
        resp = await client.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            },
        )
        assert resp.status_code == 202

    async def test_invalid_json_returns_parse_error(self, client):
        resp = await client.post(
            "/mcp/",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        body = resp.json()
        assert body["error"]["code"] == -32700

    async def test_non_dict_body_returns_invalid_request(self, client):
        resp = await client.post("/mcp/", json=[1, 2, 3])
        body = resp.json()
        assert body["error"]["code"] == -32600
        assert "not array" in body["error"]["message"]

    async def test_delete_returns_200(self, client):
        resp = await client.delete("/mcp/")
        assert resp.status_code == 200

    async def test_get_returns_405(self, client):
        resp = await client.get("/mcp/")
        assert resp.status_code == 405
