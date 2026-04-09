from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
import httpx
import pytest
import pytest_asyncio

from app.api.v1 import assistant as assistant_api
from app.models.chatbot import Conversation, Message
from app.services.assistant_gateway_service import AssistantGatewayService
from tests.helpers.mcp_fixture import create_mcp_test_session_factory, seed_market_copilot_data


@pytest_asyncio.fixture()
async def client():
    session_factory, engine = create_mcp_test_session_factory()
    Conversation.__table__.create(bind=engine)
    Message.__table__.create(bind=engine)
    seed_market_copilot_data(session_factory)

    stream_bytes = b"""data: {\"choices\": [{\"delta\": {\"content\": \"Breadth still looks constructive.\"}}]}\n\ndata: [DONE]\n\n"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "hermes-agent"}]})
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=stream_bytes,
        )

    gateway = AssistantGatewayService(
        app_settings=SimpleNamespace(
            hermes_api_base="http://hermes.test/v1",
            hermes_api_key="test-key",
            hermes_model="hermes-agent",
            hermes_request_timeout_seconds=30,
            mcp_watchlist_writes_enabled=False,
            mcp_server_name="stockscreen-market-copilot",
        ),
        session_factory=session_factory,
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    test_app = FastAPI()
    test_app.include_router(assistant_api.router, prefix="/api/v1/assistant")
    test_app.dependency_overrides[assistant_api.get_db] = override_get_db
    test_app.dependency_overrides[assistant_api._get_assistant_gateway_service] = lambda: gateway

    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client

    engine.dispose()


@pytest.mark.asyncio
async def test_assistant_conversation_crud_and_stream(client):
    create_response = await client.post("/api/v1/assistant/conversations", json={"title": "Breadth review"})
    assert create_response.status_code == 200
    conversation = create_response.json()

    list_response = await client.get("/api/v1/assistant/conversations")
    assert list_response.status_code == 200
    assert list_response.json()["total"] == 1

    stream_response = await client.post(
        f"/api/v1/assistant/conversations/{conversation['conversation_id']}/messages",
        json={"content": "How does breadth look?"},
    )
    assert stream_response.status_code == 200
    assert '"type": "done"' in stream_response.text

    detail_response = await client.get(f"/api/v1/assistant/conversations/{conversation['conversation_id']}")
    assert detail_response.status_code == 200
    assert detail_response.json()["message_count"] == 2


@pytest.mark.asyncio
async def test_assistant_watchlist_preview_endpoint(client):
    response = await client.post(
        "/api/v1/assistant/watchlist-add-preview",
        json={"watchlist": "Leaders", "symbols": ["MSFT", "NVDA", "ZZZZ"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["addable_symbols"] == ["MSFT"]
    assert body["existing_symbols"] == ["NVDA"]
    assert body["invalid_symbols"] == ["ZZZZ"]


@pytest.mark.asyncio
async def test_send_message_returns_http_404_for_inactive_conversation():
    session_factory, engine = create_mcp_test_session_factory()
    Conversation.__table__.create(bind=engine)
    Message.__table__.create(bind=engine)
    seed_market_copilot_data(session_factory)

    gateway = AssistantGatewayService(
        app_settings=SimpleNamespace(
            hermes_api_base="http://hermes.test/v1",
            hermes_api_key="test-key",
            hermes_model="hermes-agent",
            hermes_request_timeout_seconds=30,
            mcp_watchlist_writes_enabled=False,
            mcp_server_name="stockscreen-market-copilot",
        ),
        session_factory=session_factory,
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )

    with session_factory() as db:
        conversation = gateway.create_conversation(db, "Inactive")
        conversation.is_active = False
        db.commit()
        inactive_conversation_id = conversation.conversation_id

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    test_app = FastAPI()
    test_app.include_router(assistant_api.router, prefix="/api/v1/assistant")
    test_app.dependency_overrides[assistant_api.get_db] = override_get_db
    test_app.dependency_overrides[assistant_api._get_assistant_gateway_service] = lambda: gateway

    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        response = await async_client.post(
            f"/api/v1/assistant/conversations/{inactive_conversation_id}/messages",
            json={"content": "Hello"},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found"}
    engine.dispose()
