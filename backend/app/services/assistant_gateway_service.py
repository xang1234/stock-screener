"""Backend-owned Hermes assistant gateway and transcript persistence."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from sqlalchemy import desc, func
from sqlalchemy.orm import Session, joinedload, sessionmaker

from ..config import settings
from ..database import SessionLocal
from ..interfaces.mcp.market_copilot import MarketCopilotService
from ..models.chatbot import Conversation, Message
from ..models.stock_universe import StockUniverse
from ..models.user_watchlist import UserWatchlist, WatchlistItem
from ..services.errors import ServiceError
from ..services.watchlist_import_service import split_import_results

logger = logging.getLogger(__name__)

_ASSISTANT_SYSTEM_PROMPT = """You are the StockScreenClaude Assistant.

Use StockScreenClaude's internal market data tools first when the user asks about scans, themes, breadth,
group rankings, watchlists, or symbol-specific setup context. Use broader web research only when internal
data is missing, stale, or insufficient. Clearly separate internal platform signals from external web/news
context. Mention freshness caveats whenever internal data appears stale or incomplete.

Stay in research-assistant mode. Do not imply certainty, guaranteed outcomes, or personalized investment
advice. Prefer concise, evidence-based responses with citations where possible.
"""

_CONTENT_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)(?:\s+\"([^\"]+)\")?\)")
_RAW_URL_PATTERN = re.compile(r"(?<!\()(?P<url>https?://[^\s)]+)")
_MAX_TOOL_ROUND_TRIPS = 3
_TOOL_NAME_SUFFIXES = (
    "market_overview",
    "compare_feature_runs",
    "find_candidates",
    "explain_symbol",
    "watchlist_snapshot",
    "watchlist_add",
    "theme_state",
    "task_status",
    "group_rankings",
    "stock_lookup",
    "stock_snapshot",
    "breadth_snapshot",
    "daily_digest",
)


class AssistantGatewayError(ServiceError):
    """Base class for assistant gateway failures."""

    error_code = "assistant_gateway_error"


class AssistantConversationNotFoundError(AssistantGatewayError):
    """Raised when the requested conversation does not exist."""

    error_code = "assistant_conversation_not_found"


class AssistantWatchlistNotFoundError(AssistantGatewayError):
    """Raised when a watchlist preview request targets an unknown watchlist."""

    error_code = "assistant_watchlist_not_found"


class AssistantUpstreamUnavailableError(AssistantGatewayError):
    """Raised when the Hermes API cannot be reached."""

    error_code = "assistant_upstream_unavailable"


class AssistantUpstreamAuthError(AssistantGatewayError):
    """Raised when the Hermes API rejects credentials."""

    error_code = "assistant_upstream_auth_failed"


class AssistantGatewayService:
    """Application-facing assistant gateway backed by Hermes."""

    def __init__(
        self,
        *,
        app_settings: Any = settings,
        session_factory: sessionmaker = SessionLocal,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
        market_copilot_service: MarketCopilotService | None = None,
    ) -> None:
        self._settings = app_settings
        self._session_factory = session_factory
        self._client_factory = client_factory
        self._market_copilot = market_copilot_service or MarketCopilotService(session_factory, app_settings)

    def create_conversation(self, db: Session, title: str | None = None) -> Conversation:
        conversation = Conversation(
            conversation_id=str(uuid.uuid4()),
            title=(title or "New Conversation").strip()[:200] or "New Conversation",
            is_active=True,
            message_count=0,
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation

    def list_conversations(self, db: Session, *, limit: int, offset: int) -> tuple[list[Conversation], int]:
        total = db.query(Conversation).filter(Conversation.is_active.is_(True)).count()
        conversations = (
            db.query(Conversation)
            .filter(Conversation.is_active.is_(True))
            .order_by(desc(Conversation.updated_at))
            .offset(offset)
            .limit(limit)
            .all()
        )
        return conversations, total

    def get_conversation(self, db: Session, conversation_id: str) -> Conversation | None:
        conversation = (
            db.query(Conversation)
            .options(joinedload(Conversation.messages))
            .filter(Conversation.conversation_id == conversation_id)
            .first()
        )
        if conversation is None:
            return None
        conversation.messages = sorted(
            conversation.messages,
            key=lambda row: (row.created_at, row.id),
        )
        return conversation

    async def health(self) -> dict[str, Any]:
        configured_api_base = self._configured_hermes_api_base()
        if not configured_api_base:
            return {
                "status": "misconfigured",
                "available": False,
                "streaming": True,
                "popup_enabled": False,
                "model": getattr(self._settings, "hermes_model", None),
                "detail": "HERMES_API_BASE is not configured.",
            }

        last_network_error: Exception | None = None
        timed_out = False
        try:
            async with self._open_client() as client:
                for api_base in self._candidate_api_bases():
                    try:
                        response = await client.get(self._models_url(api_base), headers=self._request_headers())
                    except (httpx.ConnectError, httpx.NetworkError) as exc:
                        last_network_error = exc
                        continue
                    except httpx.TimeoutException:
                        timed_out = True
                        continue

                    if response.status_code in {401, 403}:
                        return {
                            "status": "auth_error",
                            "available": False,
                            "streaming": True,
                            "popup_enabled": False,
                            "model": getattr(self._settings, "hermes_model", None),
                            "detail": "Hermes rejected the configured API key.",
                        }

                    if response.is_success:
                        return {
                            "status": "healthy",
                            "available": True,
                            "streaming": True,
                            "popup_enabled": True,
                            "model": getattr(self._settings, "hermes_model", None),
                            "detail": None,
                        }

                    return {
                        "status": "error",
                        "available": False,
                        "streaming": True,
                        "popup_enabled": False,
                        "model": getattr(self._settings, "hermes_model", None),
                        "detail": f"Hermes returned HTTP {response.status_code}.",
                    }
        except AssistantUpstreamAuthError:
            return {
                "status": "auth_error",
                "available": False,
                "streaming": True,
                "popup_enabled": False,
                "model": getattr(self._settings, "hermes_model", None),
                "detail": "Hermes rejected the configured API key.",
            }

        if last_network_error is not None:
            detail = self._unreachable_hermes_detail(last_network_error)
            logger.warning(
                "Hermes health check failed for %s: %s",
                configured_api_base,
                detail,
            )
            return {
                "status": "unavailable",
                "available": False,
                "streaming": True,
                "popup_enabled": False,
                "model": getattr(self._settings, "hermes_model", None),
                "detail": detail,
            }
        if timed_out:
            logger.warning(
                "Hermes health check timed out for %s after %s second(s).",
                configured_api_base,
                self._settings.hermes_request_timeout_seconds,
            )
            return {
                "status": "timeout",
                "available": False,
                "streaming": True,
                "popup_enabled": False,
                "model": getattr(self._settings, "hermes_model", None),
                "detail": "Hermes health check timed out.",
            }

        return {
            "status": "error",
            "available": False,
            "streaming": True,
            "popup_enabled": False,
            "model": getattr(self._settings, "hermes_model", None),
            "detail": f"Hermes returned HTTP {response.status_code}.",
        }

    def preview_watchlist_add(
        self,
        db: Session,
        *,
        watchlist: str,
        symbols: list[str],
        reason: str | None = None,
    ) -> dict[str, Any]:
        resolved_watchlist = self._resolve_watchlist(db, watchlist)
        if resolved_watchlist is None:
            raise AssistantWatchlistNotFoundError(f"Watchlist {watchlist!r} was not found.")

        requested_symbols = self._normalize_symbols(symbols)
        if not requested_symbols:
            raise AssistantGatewayError("At least one non-empty symbol is required.", error_code="assistant_invalid_symbols")
        known_symbols = {
            row[0]
            for row in (
                db.query(StockUniverse.symbol)
                .filter(
                    StockUniverse.symbol.in_(requested_symbols),
                    StockUniverse.is_active.is_(True),
                )
                .all()
            )
        }
        existing_symbols = {
            row[0]
            for row in (
                db.query(WatchlistItem.symbol)
                .filter(WatchlistItem.watchlist_id == resolved_watchlist.id)
                .all()
            )
        }
        addable_symbols, existing_items, invalid_symbols = split_import_results(
            requested_symbols,
            known_symbols,
            existing_symbols,
        )

        return {
            "watchlist": {"id": resolved_watchlist.id, "name": resolved_watchlist.name},
            "requested_symbols": requested_symbols,
            "addable_symbols": addable_symbols,
            "existing_symbols": existing_items,
            "invalid_symbols": invalid_symbols,
            "reason": reason,
            "summary": (
                f"{len(addable_symbols)} symbol(s) can be added to {resolved_watchlist.name}; "
                f"{len(existing_items)} already exist and {len(invalid_symbols)} are invalid."
            ),
        }

    async def stream_message(
        self,
        db: Session,
        *,
        conversation_id: str,
        content: str,
    ) -> AsyncIterator[dict[str, Any]]:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.conversation_id == conversation_id, Conversation.is_active.is_(True))
            .first()
        )
        if conversation is None:
            raise AssistantConversationNotFoundError("Conversation not found")

        user_message = Message(
            conversation_id=conversation_id,
            role="user",
            content=content.strip(),
        )
        db.add(user_message)
        conversation.message_count = int(conversation.message_count or 0) + 1
        if not conversation.title or conversation.title == "New Conversation":
            conversation.title = self._summarize_title(content)
        db.commit()

        history = self._conversation_history(db, conversation_id, limit=20)
        request_messages: list[dict[str, Any]] = list(history)
        combined_tool_calls: list[dict[str, Any]] = []
        source_references: list[dict[str, Any]] = []
        final_content_events: list[dict[str, Any]] = []
        final_content_parts: list[str] = []

        try:
            async with self._open_client() as client:
                api_base = await self._resolve_chat_api_base(client)
                for _ in range(_MAX_TOOL_ROUND_TRIPS):
                    tool_calls = self._initial_tool_call_state()
                    buffered_content_events: list[dict[str, Any]] = []
                    buffered_content_parts: list[str] = []

                    async with client.stream(
                        "POST",
                        self._chat_url(api_base),
                        json=self._build_chat_payload(request_messages),
                        headers=self._request_headers(),
                    ) as response:
                        await self._ensure_upstream_success(response)

                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if not data:
                                continue
                            if data == "[DONE]":
                                break

                            chunk = self._parse_stream_chunk(data)
                            if chunk is None:
                                continue

                            normalized_chunks = self._normalize_stream_chunk(chunk, tool_calls)
                            for normalized in normalized_chunks:
                                if normalized["type"] == "content":
                                    buffered_content_events.append(normalized)
                                    buffered_content_parts.append(normalized["content"])
                                    continue
                                yield normalized

                    finalized_tool_calls = self._finalize_tool_calls(tool_calls)
                    (
                        round_tool_calls,
                        assistant_tool_message,
                        tool_messages,
                        result_chunks,
                        reference_items,
                    ) = self._execute_tool_calls(finalized_tool_calls)
                    combined_tool_calls.extend(round_tool_calls)
                    source_references.extend(reference_items)

                    if assistant_tool_message is None:
                        final_content_events = buffered_content_events
                        final_content_parts = buffered_content_parts
                        break

                    for result_chunk in result_chunks:
                        yield result_chunk
                    request_messages.extend([assistant_tool_message, *tool_messages])
                else:
                    final_content_events = buffered_content_events
                    final_content_parts = buffered_content_parts
        except AssistantGatewayError:
            raise
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            detail = self._unreachable_hermes_detail(exc)
            logger.warning(
                "Hermes request failed for %s during assistant stream: %s",
                self._configured_hermes_api_base(),
                detail,
            )
            raise AssistantUpstreamUnavailableError(detail) from exc
        except httpx.TimeoutException as exc:
            logger.warning(
                "Hermes request timed out for %s after %s second(s).",
                self._configured_hermes_api_base(),
                self._settings.hermes_request_timeout_seconds,
            )
            raise AssistantUpstreamUnavailableError("Hermes response timed out.") from exc

        for content_chunk in final_content_events:
            yield content_chunk

        source_references.extend(
            self._extract_content_references(
                "".join(final_content_parts),
            )
        )
        source_references = self._dedupe_references(source_references)

        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=("".join(final_content_parts)).strip() or "No assistant response returned.",
            agent_type="hermes",
            tool_calls=combined_tool_calls or None,
            source_references=source_references or None,
        )
        db.add(assistant_message)
        conversation.message_count = int(conversation.message_count or 0) + 1
        db.commit()
        db.refresh(assistant_message)

        yield {
            "type": "done",
            "message_id": assistant_message.id,
            "message": self._message_payload(assistant_message),
            "references": source_references,
            "tool_calls": combined_tool_calls,
        }

    def _build_chat_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model": self._settings.hermes_model,
            "stream": True,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": _ASSISTANT_SYSTEM_PROMPT},
                *messages,
            ],
        }

    def _conversation_history(self, db: Session, conversation_id: str, *, limit: int) -> list[dict[str, Any]]:
        rows = (
            db.query(Message)
            .filter(
                Message.conversation_id == conversation_id,
                Message.role.in_(["user", "assistant"]),
            )
            .order_by(desc(Message.created_at), desc(Message.id))
            .limit(limit)
            .all()
        )
        return [
            {"role": row.role, "content": row.content}
            for row in reversed(rows)
        ]

    def _request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._settings.hermes_api_key:
            headers["Authorization"] = f"Bearer {self._settings.hermes_api_key}"
        return headers

    def _open_client(self) -> httpx.AsyncClient:
        if self._client_factory is not None:
            return self._client_factory()
        return httpx.AsyncClient(timeout=self._settings.hermes_request_timeout_seconds)

    def _configured_hermes_api_base(self) -> str:
        return str(self._settings.hermes_api_base or "").strip()

    def _chat_url(self, api_base: str) -> str:
        return f"{api_base.rstrip('/')}/chat/completions"

    def _models_url(self, api_base: str) -> str:
        return f"{api_base.rstrip('/')}/models"

    def _is_running_in_container(self) -> bool:
        return os.path.exists("/.dockerenv")

    def _candidate_api_bases(self) -> list[str]:
        configured_api_base = self._configured_hermes_api_base()
        if not configured_api_base:
            return []

        candidates = [configured_api_base]
        parsed = urlparse(configured_api_base)
        if parsed.hostname == "hermes" and not self._is_running_in_container():
            localhost_netloc = "127.0.0.1"
            if parsed.port is not None:
                localhost_netloc = f"{localhost_netloc}:{parsed.port}"
            localhost_base = parsed._replace(netloc=localhost_netloc).geturl()
            if localhost_base not in candidates:
                candidates.append(localhost_base)
        return candidates

    def _unreachable_hermes_detail(self, exc: Exception) -> str:
        detail = f"Unable to reach Hermes: {exc}"
        parsed = urlparse(self._configured_hermes_api_base())
        if parsed.hostname == "hermes":
            if self._is_running_in_container():
                return (
                    f"{detail}. Start the Hermes sidecar with "
                    "`bash scripts/start_docker_assistant_stack.sh .env.docker`, "
                    "or point HERMES_API_BASE at a reachable host."
                )
            return (
                f"{detail}. The hostname 'hermes' only resolves inside Docker Compose. "
                "For local backend runs, set HERMES_API_BASE=http://127.0.0.1:8642/v1 "
                "or start Hermes locally on that address with `bash scripts/run_local_hermes_gateway.sh`."
            )
        if parsed.hostname in {"127.0.0.1", "localhost"} and parsed.port:
            return (
                f"{detail}. No Hermes API server is listening on {parsed.hostname}:{parsed.port}. "
                "Start Hermes with `bash scripts/run_local_hermes_gateway.sh`, "
                "or point HERMES_API_BASE at the host/port where Hermes is running."
            )
        return detail

    async def _resolve_chat_api_base(self, client: httpx.AsyncClient) -> str:
        candidate_api_bases = self._candidate_api_bases()
        if len(candidate_api_bases) <= 1:
            configured_api_base = self._configured_hermes_api_base()
            if configured_api_base:
                return configured_api_base
            raise AssistantUpstreamUnavailableError("HERMES_API_BASE is not configured.")

        last_network_error: Exception | None = None
        for api_base in candidate_api_bases:
            try:
                response = await client.get(self._models_url(api_base), headers=self._request_headers())
            except (httpx.ConnectError, httpx.NetworkError) as exc:
                last_network_error = exc
                continue
            except httpx.TimeoutException as exc:
                last_network_error = exc
                continue

            if response.status_code in {401, 403}:
                raise AssistantUpstreamAuthError("Hermes rejected the configured API key.")
            if response.is_success:
                if api_base != self._configured_hermes_api_base():
                    logger.warning(
                        "Falling back to local Hermes base %s because configured base %s is unreachable.",
                        api_base,
                        self._configured_hermes_api_base(),
                    )
                return api_base
            raise AssistantUpstreamUnavailableError(f"Hermes returned HTTP {response.status_code}.")

        if isinstance(last_network_error, httpx.TimeoutException):
            raise AssistantUpstreamUnavailableError("Hermes response timed out.") from last_network_error
        if last_network_error is not None:
            raise AssistantUpstreamUnavailableError(self._unreachable_hermes_detail(last_network_error)) from last_network_error
        raise AssistantUpstreamUnavailableError("Unable to resolve a reachable Hermes endpoint.")

    async def _ensure_upstream_success(self, response: httpx.Response) -> None:
        if response.status_code in {401, 403}:
            raise AssistantUpstreamAuthError("Hermes rejected the configured API key.")
        if response.is_success:
            return
        try:
            body = await response.aread()
            detail = body.decode("utf-8")[:500]
        except Exception:  # pragma: no cover - defensive guard
            detail = f"HTTP {response.status_code}"
        raise AssistantUpstreamUnavailableError(f"Hermes returned HTTP {response.status_code}: {detail}")

    def _parse_stream_chunk(self, payload: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            logger.debug("Skipping non-JSON assistant stream chunk: %s", payload)
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

    def _initial_tool_call_state(self) -> dict[int, dict[str, Any]]:
        return {}

    def _normalize_stream_chunk(
        self,
        chunk: dict[str, Any],
        tool_calls: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if chunk.get("type") in {"content", "tool_call", "tool_result", "error"}:
            normalized = [chunk]
            if chunk["type"] == "tool_call":
                index = chunk.get("index")
                if index is None:
                    index = len(tool_calls)
                index = int(index)
                entry = tool_calls.setdefault(index, {"index": index, "arguments_parts": []})
                entry["name"] = chunk.get("tool") or chunk.get("name")
                entry["id"] = chunk.get("id")
                params = chunk.get("params")
                if params is not None:
                    entry["arguments"] = params if isinstance(params, dict) else {"raw": params}
            return normalized

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return []

        delta = (choices[0] or {}).get("delta") or {}
        normalized: list[dict[str, Any]] = []
        content = delta.get("content")
        if isinstance(content, str) and content:
            normalized.append({"type": "content", "content": content})

        for tool_delta in delta.get("tool_calls") or []:
            if not isinstance(tool_delta, dict):
                continue
            index = int(tool_delta.get("index") or 0)
            entry = tool_calls.setdefault(index, {"index": index, "arguments_parts": []})
            entry["id"] = tool_delta.get("id") or entry.get("id")
            function = tool_delta.get("function") or {}
            entry["name"] = function.get("name") or entry.get("name")
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments:
                entry.setdefault("arguments_parts", []).append(arguments)
            if not entry.get("announced") and entry.get("name"):
                entry["announced"] = True
                normalized.append(
                    {
                        "type": "tool_call",
                        "tool": entry["name"],
                        "params": self._safe_json_loads("".join(entry.get("arguments_parts", []))),
                    }
                )
        return normalized

    def _finalize_tool_calls(self, tool_calls: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
        finalized: list[dict[str, Any]] = []
        for index in sorted(tool_calls):
            entry = dict(tool_calls[index])
            arguments = entry.get("arguments")
            if arguments is None:
                arguments = self._safe_json_loads("".join(entry.get("arguments_parts", [])))
            if not isinstance(arguments, dict):
                arguments = {"raw": arguments}
            finalized.append(
                {
                    "id": entry.get("id"),
                    "index": index,
                    "name": entry.get("name"),
                    "arguments": arguments,
                }
            )
        return finalized

    def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, Any] | None,
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        combined_tool_calls: list[dict[str, Any]] = []
        wire_tool_calls: list[dict[str, Any]] = []
        tool_messages: list[dict[str, Any]] = []
        result_chunks: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []

        for tool_call in tool_calls:
            raw_tool_name = tool_call.get("name") or "tool"
            normalized_tool_name = self._match_market_copilot_tool(raw_tool_name)
            arguments = tool_call.get("arguments") or {}
            tool_call_id = str(tool_call.get("id") or f"call_{tool_call.get('index', len(combined_tool_calls))}")
            combined = {
                "tool": normalized_tool_name or raw_tool_name,
                "args": arguments,
                "result": None,
            }

            if normalized_tool_name is not None:
                result = self._market_copilot.call_tool(normalized_tool_name, arguments)
                payload = result.get("structuredContent")
                combined["result"] = payload
                references.extend(self._references_from_tool_payload(normalized_tool_name, payload))
                result_chunks.append(
                    {
                        "type": "tool_result",
                        "tool": normalized_tool_name,
                        "status": "success" if result.get("isError") is not True else "error",
                        "result": payload,
                    }
                )
                wire_tool_calls.append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": raw_tool_name,
                            "arguments": self._json_dumps(arguments),
                        },
                    }
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": self._json_dumps(payload or {}),
                    }
                )

            combined_tool_calls.append(combined)

        assistant_tool_message = None
        if wire_tool_calls:
            assistant_tool_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": wire_tool_calls,
            }

        return combined_tool_calls, assistant_tool_message, tool_messages, result_chunks, references

    def _match_market_copilot_tool(self, raw_name: str | None) -> str | None:
        if not raw_name:
            return None
        if raw_name in _TOOL_NAME_SUFFIXES:
            return raw_name
        for suffix in _TOOL_NAME_SUFFIXES:
            if raw_name.endswith(f"_{suffix}") or raw_name.endswith(f".{suffix}"):
                return suffix
        return None

    def _references_from_tool_payload(
        self,
        tool_name: str,
        payload: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        summary = str(payload.get("summary") or "").strip()
        references: list[dict[str, Any]] = []
        citations = payload.get("citations") or []
        for index, citation in enumerate(citations, start=1):
            if not isinstance(citation, dict):
                continue
            references.append(
                {
                    "type": "internal",
                    "title": str(citation.get("label") or citation.get("source") or tool_name),
                    "url": self._citation_link(citation.get("source"), citation.get("reference")),
                    "section": str(citation.get("as_of") or "") or None,
                    "snippet": summary or None,
                    "reference_number": index,
                }
            )
        return references

    def _citation_link(self, source: Any, reference: Any) -> str:
        reference_str = str(reference or "")
        source_str = str(source or "")
        if source_str in {"stock_universe", "feature_runs"} and ":" in reference_str:
            _, symbol = reference_str.split(":", 1)
            return f"/stocks/{symbol}"
        if source_str == "theme_clusters":
            return "/themes"
        if source_str == "market_breadth":
            return "/breadth"
        if source_str == "user_watchlists":
            return "/market-scan"
        if source_str == "digest":
            return "/digest"
        if source_str == "task_execution":
            return "/"
        return "/"

    def _extract_content_references(self, content: str) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        for match in _CONTENT_LINK_PATTERN.finditer(content or ""):
            title = match.group(1).strip() or self._domain_label(match.group(2))
            references.append(
                {
                    "type": "web",
                    "title": title,
                    "url": match.group(2),
                    "section": None,
                    "snippet": match.group(3) or None,
                    "reference_number": None,
                }
            )
        for match in _RAW_URL_PATTERN.finditer(content or ""):
            url = match.group("url")
            if any(existing.get("url") == url for existing in references):
                continue
            references.append(
                {
                    "type": "web",
                    "title": self._domain_label(url),
                    "url": url,
                    "section": None,
                    "snippet": None,
                    "reference_number": None,
                }
            )
        return references

    def _dedupe_references(self, references: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for ref in references:
            key = (str(ref.get("title") or ""), str(ref.get("url") or ""))
            if key in seen or not ref.get("url"):
                continue
            seen.add(key)
            deduped.append(ref)
        next_reference_number = 1
        for ref in deduped:
            if ref.get("reference_number") is None:
                continue
            ref["reference_number"] = next_reference_number
            next_reference_number += 1
        return deduped

    def _domain_label(self, url: str) -> str:
        parsed = urlparse(url)
        return parsed.netloc or url

    def _message_payload(self, message: Message) -> dict[str, Any]:
        return {
            "id": message.id,
            "conversation_id": message.conversation_id,
            "role": message.role,
            "content": message.content,
            "agent_type": message.agent_type,
            "tool_name": message.tool_name,
            "tool_input": message.tool_input,
            "tool_output": message.tool_output,
            "tool_calls": message.tool_calls,
            "source_references": message.source_references,
            "created_at": (
                message.created_at.isoformat()
                if getattr(message, "created_at", None) is not None
                else datetime.now(UTC).isoformat()
            ),
        }

    def _resolve_watchlist(self, db: Session, watchlist: str) -> UserWatchlist | None:
        needle = str(watchlist or "").strip()
        if not needle:
            return None

        query = db.query(UserWatchlist)
        if needle.isdigit():
            row = query.filter(UserWatchlist.id == int(needle)).first()
            if row is not None:
                return row
        exact = query.filter(func.lower(UserWatchlist.name) == needle.lower()).first()
        if exact is not None:
            return exact
        escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return (
            query.filter(UserWatchlist.name.ilike(f"%{escaped}%", escape="\\"))
            .order_by(UserWatchlist.position.asc())
            .first()
        )

    def _normalize_symbols(self, symbols: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for symbol in symbols:
            clean = str(symbol or "").strip().upper()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            normalized.append(clean)
        return normalized

    def _summarize_title(self, content: str) -> str:
        cleaned = " ".join(content.strip().split())
        if len(cleaned) <= 60:
            return cleaned
        return f"{cleaned[:57].rstrip()}..."

    def _safe_json_loads(self, raw: Any) -> Any:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return raw
        candidate = raw.strip()
        if not candidate:
            return {}
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return candidate

    def _json_dumps(self, value: Any) -> str:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
