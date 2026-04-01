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


def test_pinned_public_ip_backend_blocks_mixed_public_private_dns_answers(monkeypatch):
    backend = _PinnedPublicIPBackend()

    def fake_getaddrinfo(*args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
        ]

    import socket

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(httpcore.ConnectError, match="Blocked host"):
        backend._resolve_public_ip("example.com")
