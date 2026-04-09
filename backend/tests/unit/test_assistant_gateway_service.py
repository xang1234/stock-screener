from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

from app.models.chatbot import Conversation, Message
from app.services.assistant_gateway_service import AssistantGatewayService
from tests.helpers.mcp_fixture import create_mcp_test_session_factory, seed_market_copilot_data


@pytest.fixture()
def session_factory():
    factory, engine = create_mcp_test_session_factory()
    Conversation.__table__.create(bind=engine)
    Message.__table__.create(bind=engine)
    seed_market_copilot_data(factory)
    return factory


@pytest.fixture()
def assistant_settings():
    return SimpleNamespace(
        hermes_api_base="http://hermes.test/v1",
        hermes_api_key="test-key",
        hermes_model="hermes-agent",
        hermes_request_timeout_seconds=30,
        mcp_watchlist_writes_enabled=False,
        mcp_server_name="stockscreen-market-copilot",
    )


@pytest.mark.asyncio
async def test_health_reports_available(session_factory, assistant_settings):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "hermes-agent"}]})

    service = AssistantGatewayService(
        app_settings=assistant_settings,
        session_factory=session_factory,
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    payload = await service.health()

    assert payload["available"] is True
    assert payload["status"] == "healthy"


def test_preview_watchlist_add_classifies_symbols(session_factory, assistant_settings):
    service = AssistantGatewayService(
        app_settings=assistant_settings,
        session_factory=session_factory,
    )

    with session_factory() as db:
        preview = service.preview_watchlist_add(
            db,
            watchlist="Leaders",
            symbols=["MSFT", "NVDA", "ZZZZ"],
            reason="Promote leadership names",
        )

    assert preview["watchlist"]["name"] == "Leaders"
    assert preview["addable_symbols"] == ["MSFT"]
    assert preview["existing_symbols"] == ["NVDA"]
    assert preview["invalid_symbols"] == ["ZZZZ"]


def test_get_conversation_orders_tied_timestamps_by_id(session_factory, assistant_settings):
    service = AssistantGatewayService(
        app_settings=assistant_settings,
        session_factory=session_factory,
    )

    with session_factory() as db:
        conversation = service.create_conversation(db, "Ordering check")
        created_at = datetime(2026, 4, 9, tzinfo=UTC)
        db.add_all(
            [
                Message(
                    conversation_id=conversation.conversation_id,
                    role="assistant",
                    content="second",
                    created_at=created_at,
                ),
                Message(
                    conversation_id=conversation.conversation_id,
                    role="user",
                    content="first",
                    created_at=created_at,
                ),
            ]
        )
        db.commit()

        loaded = service.get_conversation(db, conversation.conversation_id)

    assert loaded is not None
    assert [message.id for message in loaded.messages] == sorted(message.id for message in loaded.messages)


def test_normalize_stream_chunk_keeps_explicit_custom_tool_index_zero(session_factory, assistant_settings):
    service = AssistantGatewayService(
        app_settings=assistant_settings,
        session_factory=session_factory,
    )
    tool_calls = {
        0: {
            "index": 0,
            "name": "stock_snapshot",
            "arguments_parts": [],
        }
    }

    service._normalize_stream_chunk(
        {
            "type": "tool_call",
            "index": 0,
            "tool": "stock_snapshot",
            "params": {"symbol": "NVDA"},
        },
        tool_calls,
    )

    assert set(tool_calls) == {0}
    assert tool_calls[0]["arguments"] == {"symbol": "NVDA"}


@pytest.mark.asyncio
async def test_stream_message_persists_transcript_and_tool_results(session_factory, assistant_settings):
    stream_responses = [
        b"""data: {\"choices\": [{\"delta\": {\"tool_calls\": [{\"index\": 0, \"id\": \"call_1\", \"function\": {\"name\": \"mcp_stockscreen_market_stock_snapshot\", \"arguments\": \"{\\\"symbol\\\":\\\"NVDA\\\"}\"}}]}}]}\n\ndata: [DONE]\n\n""",
        b"""data: {\"choices\": [{\"delta\": {\"content\": \"Use [1] for the internal snapshot. See [Reuters](https://example.com) for broader context.\"}}]}\n\ndata: [DONE]\n\n""",
    ]
    request_payloads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = request.content.decode("utf-8")
        request_payloads.append(payload)
        assert "StockScreenClaude Assistant" in payload
        assert "\"role\":\"user\"" in payload
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=stream_responses[len(request_payloads) - 1],
        )

    service = AssistantGatewayService(
        app_settings=assistant_settings,
        session_factory=session_factory,
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with session_factory() as db:
        conversation = service.create_conversation(db, "NVDA review")
        chunks = [
            chunk
            async for chunk in service.stream_message(
                db,
                conversation_id=conversation.conversation_id,
                content="Review NVDA using internal data first.",
            )
        ]

    assert any(chunk["type"] == "tool_result" for chunk in chunks)
    assert any(chunk["type"] == "content" for chunk in chunks)
    done_chunk = next(chunk for chunk in chunks if chunk["type"] == "done")
    assert done_chunk["message"]["role"] == "assistant"
    assert done_chunk["message"]["content"] == "Use [1] for the internal snapshot. See [Reuters](https://example.com) for broader context."
    assert done_chunk["tool_calls"][0]["tool"] == "stock_snapshot"
    assert done_chunk["references"][0]["reference_number"] == 1
    assert done_chunk["references"][0]["url"] == "/stocks/NVDA"
    assert any(reference["url"] == "https://example.com" and reference["reference_number"] is None for reference in done_chunk["references"])
    assert len(request_payloads) == 2
    assert "\"role\":\"tool\"" in request_payloads[1]
    assert "\"tool_call_id\":\"call_1\"" in request_payloads[1]

    with session_factory() as db:
        persisted_messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation.conversation_id)
            .order_by(Message.id.asc())
            .all()
        )
        assert [message.role for message in persisted_messages] == ["user", "assistant"]
        assert persisted_messages[1].agent_type == "hermes"
        assert persisted_messages[1].tool_calls[0]["tool"] == "stock_snapshot"
        assert persisted_messages[1].content.startswith("Use [1] for the internal snapshot.")


@pytest.mark.asyncio
async def test_stream_message_preserves_partial_content_when_tool_round_trips_hit_limit(session_factory, assistant_settings):
    stream_responses = [
        b"""data: {\"choices\": [{\"delta\": {\"content\": \"Round one. \", \"tool_calls\": [{\"index\": 0, \"id\": \"call_1\", \"function\": {\"name\": \"mcp_stockscreen_market_stock_snapshot\", \"arguments\": \"{\\\"symbol\\\":\\\"NVDA\\\"}\"}}]}}]}\n\ndata: [DONE]\n\n""",
        b"""data: {\"choices\": [{\"delta\": {\"content\": \"Round two. \", \"tool_calls\": [{\"index\": 0, \"id\": \"call_2\", \"function\": {\"name\": \"mcp_stockscreen_market_stock_snapshot\", \"arguments\": \"{\\\"symbol\\\":\\\"MSFT\\\"}\"}}]}}]}\n\ndata: [DONE]\n\n""",
        b"""data: {\"choices\": [{\"delta\": {\"content\": \"Round three partial. \", \"tool_calls\": [{\"index\": 0, \"id\": \"call_3\", \"function\": {\"name\": \"mcp_stockscreen_market_stock_snapshot\", \"arguments\": \"{\\\"symbol\\\":\\\"AVGO\\\"}\"}}]}}]}\n\ndata: [DONE]\n\n""",
    ]
    request_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        response = httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=stream_responses[request_count],
        )
        request_count += 1
        return response

    service = AssistantGatewayService(
        app_settings=assistant_settings,
        session_factory=session_factory,
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with session_factory() as db:
        conversation = service.create_conversation(db, "Round trip limit")
        chunks = [
            chunk
            async for chunk in service.stream_message(
                db,
                conversation_id=conversation.conversation_id,
                content="Keep calling tools until the limit is hit.",
            )
        ]

    done_chunk = next(chunk for chunk in chunks if chunk["type"] == "done")
    assert done_chunk["message"]["content"] == "Round three partial."
    assert request_count == 3
