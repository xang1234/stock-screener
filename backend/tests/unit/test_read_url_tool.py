import pytest

from app.services.chatbot.tools.read_url import ReadUrlTool


@pytest.mark.asyncio
async def test_read_url_rejects_non_http_scheme():
    tool = ReadUrlTool()
    result = await tool.read_url("file:///etc/passwd")

    assert result["success"] is False
    assert result["error"] == "Unsupported URL scheme"


@pytest.mark.asyncio
async def test_read_url_blocks_private_host():
    tool = ReadUrlTool()
    result = await tool.read_url("http://127.0.0.1:8080")

    assert result["success"] is False
    assert result["error"] == "Blocked host"
