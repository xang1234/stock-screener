"""Simple stdio helpers for MCP integration tests."""

from __future__ import annotations

import json


def write_mcp_message(stream, payload: dict) -> None:
    """Write one MCP/JSON-RPC message with Content-Length framing."""

    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), default=str).encode("utf-8")
    stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8"))
    stream.write(body)
    stream.flush()


def read_mcp_message(stream) -> dict:
    """Read one framed MCP/JSON-RPC message."""

    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            raise EOFError("MCP subprocess closed stdout")
        if line in (b"\r\n", b"\n"):
            break
        name, value = line.decode("utf-8").strip().split(":", 1)
        headers[name.lower()] = value.strip()

    content_length = int(headers["content-length"])
    body = stream.read(content_length)
    if len(body) != content_length:
        raise EOFError("Incomplete MCP message body")
    return json.loads(body)
