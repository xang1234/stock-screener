"""Tests for LLM config defaults and model registry exposure."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.v1 import config as config_api
from app.api.v1.config import get_llm_config, update_llm_model
from app.database import Base
from app.models.app_settings import AppSetting
from app.schemas.config import LLMModelUpdate


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.mark.asyncio
async def test_get_llm_config_defaults_to_minimax(monkeypatch, db_session) -> None:
    async def _fake_check_ollama_status(_api_base: str) -> str:
        return "disconnected"

    monkeypatch.setattr(config_api, "check_ollama_status", _fake_check_ollama_status)

    response = await get_llm_config(db=db_session, _auth=True)

    assert response.extraction["current_model"] == "minimax/MiniMax-M2.7"
    assert response.merge["current_model"] == "minimax/MiniMax-M2.7"
    assert any(model["id"] == "openai/glm-4.7-flash" and model["provider"] == "zai" for model in response.available_models)
    assert not any(model["provider"] in {"deepseek", "together_ai", "openrouter"} for model in response.available_models)


@pytest.mark.asyncio
async def test_update_llm_model_persists_zai_selection(db_session) -> None:
    payload = LLMModelUpdate(model_id="openai/glm-4.7-flash", use_case="extraction")

    response = await update_llm_model(request=payload, db=db_session, _auth=True)

    persisted = db_session.query(AppSetting).filter(AppSetting.key == "llm_extraction_model").first()
    assert response["status"] == "success"
    assert persisted is not None
    assert persisted.value == "openai/glm-4.7-flash"


@pytest.mark.asyncio
async def test_update_llm_model_rejects_unsupported_provider_for_extraction(db_session) -> None:
    payload = LLMModelUpdate(model_id="groq/qwen/qwen3-32b", use_case="extraction")

    with pytest.raises(HTTPException) as exc_info:
        await update_llm_model(request=payload, db=db_session, _auth=True)

    assert exc_info.value.status_code == 400
    assert "not supported for use_case 'extraction'" in str(exc_info.value)
