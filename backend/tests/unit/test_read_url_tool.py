import socket

import pytest
import httpcore

from app.services.chatbot.tools.read_url import ReadUrlTool, _PinnedPublicIPBackend


@pytest.mark.asyncio
async def test_read_url_rejects_non_http_scheme():
    tool = ReadUrlTool()
    result = await tool.read_url("file:///etc/passwd")

    assert result["success"] is False
    assert result["error"] == "Unsupported URL scheme"


@pytest.mark.asyncio
async def test_read_url_blocks_private_host():
    tool = ReadUrlTool()
    result = await tool.read_url("https://127.0.0.1:8080")

    assert result["success"] is False
    assert result["error"] == "Blocked host"


@pytest.mark.asyncio
async def test_read_url_requires_https():
    tool = ReadUrlTool()
    result = await tool.read_url("http://example.com/article")

    assert result["success"] is False
    assert result["error"] == "Only HTTPS URLs are allowed"


@pytest.mark.asyncio
async def test_pinned_public_ip_backend_retries_alternate_public_ips(monkeypatch):
    backend = _PinnedPublicIPBackend()
    attempts: list[str] = []
    stream = object()

    async def fake_connect_tcp(*, host, port, timeout=None, local_address=None, socket_options=None):
        attempts.append(host)
        if host == "2001:db8::1":
            raise httpcore.ConnectError("ipv6 unreachable")
        return stream

    monkeypatch.setattr(backend, "_resolve_public_ips", lambda host: ("2001:db8::1", "93.184.216.34"))
    monkeypatch.setattr(backend._backend, "connect_tcp", fake_connect_tcp)

    result = await backend.connect_tcp("example.com", 443)

    assert result is stream
    assert attempts == ["2001:db8::1", "93.184.216.34"]


def test_pinned_public_ip_backend_blocks_mixed_public_private_dns_answers(monkeypatch):
    backend = _PinnedPublicIPBackend()

    def fake_getaddrinfo(*args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(httpcore.ConnectError, match="Blocked host"):
        backend._resolve_public_ips("example.com")
