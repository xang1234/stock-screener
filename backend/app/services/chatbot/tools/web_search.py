"""Web search tool with explicit Tavily-primary and Serper-fallback routing."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, TypedDict

import httpx

from ....config import settings

logger = logging.getLogger(__name__)


SearchType = Literal["general", "news", "finance"]


class SearchResultItem(TypedDict):
    title: str
    url: str
    snippet: str
    score: float


class SearchReferenceItem(TypedDict):
    type: str
    title: str
    url: str
    snippet: str


class SearchEnvelope(TypedDict):
    query: str
    answer: str | None
    results: List[SearchResultItem]
    total_results: int
    provider: str
    references: List[SearchReferenceItem]


class WebSearchTool:
    """Web search via Tavily with Serper fallback."""

    def __init__(self) -> None:
        self._tavily_api_key = (getattr(settings, "tavily_api_key", "") or "").strip()
        self._serper_api_key = (getattr(settings, "serper_api_key", "") or "").strip()
        self._timeout = httpx.Timeout(12.0, connect=6.0)

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_type: SearchType = "general",
    ) -> SearchEnvelope:
        """Execute web search with provider fallback."""
        if search_type == "news":
            return await self.search_news(query, max_results)
        if search_type == "finance":
            return await self.search_finance(query, max_results)

        return await self._search_with_provider_fallback(
            query=query,
            normalized_query=query,
            max_results=max_results,
            search_type="general",
        )

    async def search_news(self, query: str, max_results: int = 5) -> SearchEnvelope:
        """Search for recent news articles."""
        return await self._search_with_provider_fallback(
            query=query,
            normalized_query=query,
            max_results=max_results,
            search_type="news",
        )

    async def search_finance(self, query: str, max_results: int = 5) -> SearchEnvelope:
        """Search with finance context bias."""
        finance_query = f"{query} stock finance"
        return await self._search_with_provider_fallback(
            query=query,
            normalized_query=finance_query,
            max_results=max_results,
            search_type="finance",
        )

    async def _search_with_provider_fallback(
        self,
        *,
        query: str,
        normalized_query: str,
        max_results: int,
        search_type: SearchType,
    ) -> SearchEnvelope:
        provider_error: Exception | None = None

        if self._tavily_api_key:
            try:
                tavily_results = await self._search_tavily(
                    query=normalized_query,
                    max_results=max_results,
                    search_type=search_type,
                )
                return self._format_results(
                    query=query,
                    provider="tavily",
                    raw_results=tavily_results,
                    search_type=search_type,
                )
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                provider_error = exc
                logger.warning(
                    "Tavily search failed; trying Serper fallback",
                    extra={
                        "event": "chatbot_web_search_provider_failure",
                        "path": "chatbot.tools.web_search",
                        "pipeline": None,
                        "run_id": None,
                        "symbol": None,
                        "error_code": "web_search_tavily_failed",
                    },
                    exc_info=exc,
                )

        if self._serper_api_key:
            try:
                serper_results = await self._search_serper(
                    query=normalized_query,
                    max_results=max_results,
                    search_type=search_type,
                )
                return self._format_results(
                    query=query,
                    provider="serper",
                    raw_results=serper_results,
                    search_type=search_type,
                )
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                provider_error = exc
                logger.error(
                    "Serper fallback search failed",
                    extra={
                        "event": "chatbot_web_search_provider_failure",
                        "path": "chatbot.tools.web_search",
                        "pipeline": None,
                        "run_id": None,
                        "symbol": None,
                        "error_code": "web_search_serper_failed",
                    },
                    exc_info=exc,
                )

        if provider_error:
            logger.error(
                "Web search providers unavailable for request",
                extra={
                    "event": "chatbot_web_search_unavailable",
                    "path": "chatbot.tools.web_search",
                    "pipeline": None,
                    "run_id": None,
                    "symbol": None,
                    "error_code": "web_search_all_providers_failed",
                },
                exc_info=provider_error,
            )

        return self._format_results(
            query=query,
            provider="unavailable",
            raw_results=[],
            search_type=search_type,
        )

    async def _search_tavily(
        self,
        *,
        query: str,
        max_results: int,
        search_type: SearchType,
    ) -> list[dict[str, Any]]:
        if not self._tavily_api_key:
            return []
        payload: dict[str, Any] = {
            "api_key": self._tavily_api_key,
            "query": query,
            "max_results": max(1, min(max_results, 10)),
            "search_depth": "advanced" if search_type == "news" else "basic",
        }
        if search_type == "news":
            payload["topic"] = "news"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            data = response.json()
        results = data.get("results") if isinstance(data, dict) else []
        return results if isinstance(results, list) else []

    async def _search_serper(
        self,
        *,
        query: str,
        max_results: int,
        search_type: SearchType,
    ) -> list[dict[str, Any]]:
        if not self._serper_api_key:
            return []
        endpoint = "https://google.serper.dev/news" if search_type == "news" else "https://google.serper.dev/search"
        payload: dict[str, Any] = {"q": query, "num": max(1, min(max_results, 10))}
        headers = {"X-API-KEY": self._serper_api_key, "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        if not isinstance(data, dict):
            return []
        if search_type == "news":
            raw_news = data.get("news", [])
            return raw_news if isinstance(raw_news, list) else []
        raw_organic = data.get("organic", [])
        return raw_organic if isinstance(raw_organic, list) else []

    def _format_results(
        self,
        *,
        query: str,
        provider: str,
        raw_results: list[dict[str, Any]],
        search_type: SearchType,
    ) -> SearchEnvelope:
        formatted: List[SearchResultItem] = []
        references: List[SearchReferenceItem] = []

        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("name") or "").strip()
            url = str(
                item.get("url")
                or item.get("link")
                or item.get("href")
                or ""
            ).strip()
            snippet = str(
                item.get("content")
                or item.get("snippet")
                or item.get("description")
                or ""
            ).strip()
            score = float(item.get("score") or 0.0)
            if not url:
                continue

            formatted.append(
                SearchResultItem(
                    title=title,
                    url=url,
                    snippet=snippet,
                    score=score,
                )
            )
            references.append(
                SearchReferenceItem(
                    type="news" if search_type == "news" else "web",
                    title=title or url,
                    url=url,
                    snippet=(snippet[:150] + "...") if len(snippet) > 150 else snippet,
                )
            )

        return SearchEnvelope(
            query=query,
            answer=None,
            results=formatted,
            total_results=len(formatted),
            provider=provider,
            references=references,
        )

    async def close(self) -> None:
        """No persistent resources to close."""
        return None

    def get_tool_description(self) -> Dict[str, Any]:
        """Return tool description for action/research agents."""
        return {
            "name": "web_search",
            "description": "Search the web for information using Tavily with Serper fallback.",
            "parameters": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {"type": "integer", "description": "Max results (default 5)", "default": 5},
                "search_type": {"type": "string", "description": "Type: general, news, finance", "default": "general"},
            },
            "required": ["query"],
        }
