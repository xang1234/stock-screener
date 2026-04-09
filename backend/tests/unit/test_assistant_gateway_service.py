from __future__ import annotations

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


@pytest.mark.asyncio
async def test_stream_message_persists_transcript_and_tool_results(session_factory, assistant_settings):
    stream_bytes = b"""data: {\"choices\": [{\"delta\": {\"tool_calls\": [{\"index\": 0, \"id\": \"call_1\", \"function\": {\"name\": \"mcp_stockscreen_market_stock_snapshot\", \"arguments\": \"{\\\"symbol\\\":\\\"NVDA\\\"}\"}}]}}]}\n\ndata: {\"choices\": [{\"delta\": {\"content\": \"NVDA remains one of the stronger symbols in the current leadership set.\"}}]}\n\ndata: [DONE]\n\n"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = request.content.decode("utf-8")
        assert "StockScreenClaude Assistant" in payload
        assert "\"role\":\"user\"" in payload
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=stream_bytes,
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
    done_chunk = next(chunk for chunk in chunks if chunk["type"] == "done")
    assert done_chunk["message"]["role"] == "assistant"
    assert done_chunk["tool_calls"][0]["tool"] == "stock_snapshot"
    assert done_chunk["references"]

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
