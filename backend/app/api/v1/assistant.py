"""Hermes-backed assistant API endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ...database import get_db
from ...schemas.assistant import (
    AssistantConversationCreate,
    AssistantConversationDetailResponse,
    AssistantConversationListResponse,
    AssistantConversationResponse,
    AssistantHealthResponse,
    AssistantMessageCreate,
    AssistantWatchlistAddPreviewRequest,
    AssistantWatchlistAddPreviewResponse,
)
from ...services.assistant_gateway_service import (
    AssistantConversationNotFoundError,
    AssistantGatewayError,
    AssistantGatewayService,
    AssistantUpstreamAuthError,
    AssistantUpstreamUnavailableError,
    AssistantWatchlistNotFoundError,
)

router = APIRouter()


def _get_assistant_gateway_service() -> AssistantGatewayService:
    return AssistantGatewayService()


@router.post("/conversations", response_model=AssistantConversationResponse)
def create_conversation(
    request: AssistantConversationCreate | None = None,
    db: Session = Depends(get_db),
    gateway: AssistantGatewayService = Depends(_get_assistant_gateway_service),
):
    return gateway.create_conversation(db, request.title if request else None)


@router.get("/conversations", response_model=AssistantConversationListResponse)
def list_conversations(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    gateway: AssistantGatewayService = Depends(_get_assistant_gateway_service),
):
    conversations, total = gateway.list_conversations(db, limit=limit, offset=offset)
    return AssistantConversationListResponse(conversations=conversations, total=total)


@router.get("/conversations/{conversation_id}", response_model=AssistantConversationDetailResponse)
def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    gateway: AssistantGatewayService = Depends(_get_assistant_gateway_service),
):
    conversation = gateway.get_conversation(db, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.post("/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    request: AssistantMessageCreate,
    db: Session = Depends(get_db),
    gateway: AssistantGatewayService = Depends(_get_assistant_gateway_service),
):
    if gateway.get_conversation(db, conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    async def generate():
        try:
            async for chunk in gateway.stream_message(
                db,
                conversation_id=conversation_id,
                content=request.content,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        except AssistantConversationNotFoundError as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc), 'error_code': exc.error_code})}\n\n"
        except AssistantUpstreamAuthError as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc), 'error_code': exc.error_code})}\n\n"
        except AssistantUpstreamUnavailableError as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc), 'error_code': exc.error_code})}\n\n"
        except AssistantGatewayError as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc), 'error_code': exc.error_code})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health", response_model=AssistantHealthResponse)
async def assistant_health(gateway: AssistantGatewayService = Depends(_get_assistant_gateway_service)):
    return AssistantHealthResponse(**(await gateway.health()))


@router.post("/watchlist-add-preview", response_model=AssistantWatchlistAddPreviewResponse)
def watchlist_add_preview(
    request: AssistantWatchlistAddPreviewRequest,
    db: Session = Depends(get_db),
    gateway: AssistantGatewayService = Depends(_get_assistant_gateway_service),
):
    try:
        preview = gateway.preview_watchlist_add(
            db,
            watchlist=request.watchlist,
            symbols=request.symbols,
            reason=request.reason,
        )
    except AssistantWatchlistNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AssistantWatchlistAddPreviewResponse(**preview)
