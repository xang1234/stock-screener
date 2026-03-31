"""Minimal stdio MCP server for the Hermes Market Copilot integration."""

from __future__ import annotations

import json
import sys
from typing import Any

from app.config import settings
from app.database import SessionLocal

from .market_copilot import MarketCopilotService

SUPPORTED_PROTOCOL_VERSIONS = (
    "2025-11-05",
    "2025-06-18",
    "2024-11-05",
)
SERVER_VERSION = "0.1.0"


class JsonRpcError(Exception):
    """Structured JSON-RPC error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class StdioTransport:
    """Read and write JSON-RPC messages over stdio with Content-Length framing."""

    def __init__(self, instream=None, outstream=None) -> None:
        self._instream = instream or sys.stdin.buffer
        self._outstream = outstream or sys.stdout.buffer

    def read_message(self) -> dict[str, Any] | None:
        headers: dict[str, str] = {}
        while True:
            line = self._instream.readline()
            if not line:
                return None
            if line in (b"\r\n", b"\n"):
                break
            decoded = line.decode("utf-8").strip()
            if ":" not in decoded:
                raise JsonRpcError(-32700, "Malformed header line")
            name, value = decoded.split(":", 1)
            headers[name.lower()] = value.strip()

        try:
            content_length = int(headers["content-length"])
        except (KeyError, ValueError) as exc:
            raise JsonRpcError(-32700, "Missing or invalid Content-Length header") from exc

        body = self._instream.read(content_length)
        if len(body) != content_length:
            return None

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise JsonRpcError(-32700, f"Failed to decode JSON payload: {exc}") from exc

    def write_message(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), default=str).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        self._outstream.write(header)
        self._outstream.write(body)
        self._outstream.flush()


class MarketCopilotMcpServer:
    """Serve StockScreenClaude tools using the MCP tool protocol."""

    def __init__(self, service: MarketCopilotService, transport: StdioTransport | None = None) -> None:
        self._service = service
        self._transport = transport or StdioTransport()

    def serve_forever(self) -> None:
        while True:
            try:
                message = self._transport.read_message()
            except JsonRpcError as exc:
                self._transport.write_message(self._error_response(None, exc.code, exc.message))
                continue

            if message is None:
                return

            response = self._handle_message(message)
            if response is not None:
                self._transport.write_message(response)

    def _handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") or {}

        if method == "initialize":
            protocol_version = params.get("protocolVersion")
            negotiated = (
                protocol_version
                if protocol_version in SUPPORTED_PROTOCOL_VERSIONS
                else SUPPORTED_PROTOCOL_VERSIONS[0]
            )
            return self._result_response(
                request_id,
                {
                    "protocolVersion": negotiated,
                    "capabilities": {
                        "tools": {
                            "listChanged": False,
                        }
                    },
                    "serverInfo": {
                        "name": getattr(settings, "mcp_server_name", "stockscreen-market-copilot"),
                        "version": SERVER_VERSION,
                    },
                    "instructions": (
                        "Use these tools for deterministic market state, feature-run diffs, "
                        "watchlist workflows, theme inspection, and task health."
                    ),
                },
            )

        if method == "notifications/initialized":
            return None

        if method == "ping":
            return self._result_response(request_id, {})

        if method == "tools/list":
            return self._result_response(request_id, {"tools": self._service.list_tools()})

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments")
            if not isinstance(tool_name, str) or not tool_name:
                return self._error_response(request_id, -32602, "tools/call requires a non-empty tool name")
            return self._result_response(request_id, self._service.call_tool(tool_name, arguments))

        if request_id is None:
            return None
        return self._error_response(request_id, -32601, f"Method not found: {method}")

    def _result_response(self, request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _error_response(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }


def main() -> None:
    """Run the stdio MCP server."""

    service = MarketCopilotService(SessionLocal, settings)
    MarketCopilotMcpServer(service).serve_forever()


if __name__ == "__main__":
    main()
