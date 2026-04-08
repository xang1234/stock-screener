"""Unit tests for chatbot web search provider fallback routing."""

from __future__ import annotations

import pytest

from app.config import settings
from app.services.chatbot.tools.web_search import WebSearchTool


@pytest.mark.asyncio
async def test_web_search_uses_tavily_when_available(monkeypatch) -> None:
    monkeypatch.setattr(settings, "tavily_api_key", "tavily-test")
    monkeypatch.setattr(settings, "serper_api_key", "serper-test")
    tool = WebSearchTool()

    async def _fake_tavily(*, query: str, max_results: int, search_type: str):
        assert query == "ai chips"
        assert max_results == 3
        assert search_type == "general"
        return [{"title": "AI Chips Rally", "url": "https://example.com/a", "content": "Semis are up", "score": 0.91}]

    async def _fake_serper(*, query: str, max_results: int, search_type: str):  # pragma: no cover - should not run
        raise AssertionError("Serper should not be called when Tavily succeeds")

    monkeypatch.setattr(tool, "_search_tavily", _fake_tavily)
    monkeypatch.setattr(tool, "_search_serper", _fake_serper)

    result = await tool.search("ai chips", max_results=3)

    assert result["provider"] == "tavily"
    assert result["total_results"] == 1
    assert result["results"][0]["url"] == "https://example.com/a"
    assert result["references"][0]["type"] == "web"


@pytest.mark.asyncio
async def test_web_search_falls_back_to_serper_when_tavily_fails(monkeypatch) -> None:
    monkeypatch.setattr(settings, "tavily_api_key", "tavily-test")
    monkeypatch.setattr(settings, "serper_api_key", "serper-test")
    tool = WebSearchTool()

    async def _failing_tavily(*, query: str, max_results: int, search_type: str):
        raise ValueError("tavily outage")

    async def _fake_serper(*, query: str, max_results: int, search_type: str):
        assert query == "nvda stock finance"
        assert max_results == 5
        assert search_type == "finance"
        return [{"title": "NVDA News", "link": "https://example.com/nvda", "snippet": "record highs"}]

    monkeypatch.setattr(tool, "_search_tavily", _failing_tavily)
    monkeypatch.setattr(tool, "_search_serper", _fake_serper)

    result = await tool.search_finance("nvda")

    assert result["provider"] == "serper"
    assert result["total_results"] == 1
    assert result["results"][0]["url"] == "https://example.com/nvda"


@pytest.mark.asyncio
async def test_web_search_returns_empty_when_no_providers_configured(monkeypatch) -> None:
    monkeypatch.setattr(settings, "tavily_api_key", "")
    monkeypatch.setattr(settings, "serper_api_key", "")
    tool = WebSearchTool()

    result = await tool.search("macro")

    assert result["provider"] == "unavailable"
    assert result["results"] == []
    assert result["references"] == []
