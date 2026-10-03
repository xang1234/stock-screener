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
from app.services.llm.config import OLLAMA_API_BASE_DEFAULT


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
        """Answer without reaching the network; this test is about the base URL."""
        return "disconnected"

    monkeypatch.setattr(config_api, "check_ollama_status", _fake_check_ollama_status)

    response = await get_llm_config(db=db_session, _auth=True)

    assert response.extraction["current_model"] == "minimax/MiniMax-M2.7"
    assert response.merge["current_model"] == "minimax/MiniMax-M2.7"
    assert any(model["id"] == "openai/glm-4.7-flash" and model["provider"] == "zai" for model in response.available_models)
    assert not any(model["provider"] in {"deepseek", "together_ai", "openrouter"} for model in response.available_models)


def test_update_llm_model_persists_zai_selection(db_session) -> None:
    payload = LLMModelUpdate(model_id="openai/glm-4.7-flash", use_case="extraction")

    response = update_llm_model(request=payload, db=db_session, _auth=True)

    persisted = db_session.query(AppSetting).filter(AppSetting.key == "llm_extraction_model").first()
    assert response["status"] == "success"
    assert persisted is not None
    assert persisted.value == "openai/glm-4.7-flash"


def test_update_llm_model_rejects_unsupported_provider_for_extraction(db_session) -> None:
    payload = LLMModelUpdate(model_id="groq/qwen/qwen3-32b", use_case="extraction")

    with pytest.raises(HTTPException) as exc_info:
        update_llm_model(request=payload, db=db_session, _auth=True)

    assert exc_info.value.status_code == 400
    assert "not supported for use_case 'extraction'" in str(exc_info.value)


# --- Ollama destination precedence -------------------------------------------------
#
# Both readers used to overwrite the saved row with the process environment, so a
# redeploy that reset ``OLLAMA_API_BASE`` made the API report the cloud host while the
# workers still sent extraction to the saved daemon. The tests below pin the order for
# the readers; ``test_ollama_llm_routing`` pins it for ``LLMService``.


class _RecordingOllamaClient:
    """Stand-in for ``httpx.AsyncClient`` that records the URL it was asked for."""

    def __init__(self, seen: dict, **kwargs) -> None:
        """Keep the caller's dict so every request URL lands in one place."""
        self._seen = seen

    async def __aenter__(self):
        """Return self, so ``async with`` yields this recorder."""
        return self

    async def __aexit__(self, *_exc) -> bool:
        """Swallow nothing: close the block without suppressing exceptions."""
        return False

    async def get(self, url: str):
        """Record the requested URL and answer with an empty model list."""
        self._seen["url"] = url
        return _FakeOllamaResponse()


class _FakeOllamaResponse:
    """Minimal 200 response carrying no models."""

    status_code = 200

    def json(self) -> dict:
        """Return the payload shape ``get_ollama_models`` reads."""
        return {"models": []}


def _record_ollama_requests(monkeypatch) -> dict:
    """Swap in the recording client and return the dict it fills."""
    seen: dict = {}
    monkeypatch.setattr(
        config_api.httpx,
        "AsyncClient",
        lambda **kwargs: _RecordingOllamaClient(seen, **kwargs),
    )
    return seen


@pytest.mark.asyncio
async def test_llm_config_reports_the_saved_row_over_the_environment(
    monkeypatch, db_session
) -> None:
    """The restart shape: the deployment environment returns, the saved row survives.

    ``OLLAMA_API_BASE`` is set here the way a redeploy would set it. The row already
    holds the daemon an admin selected, and that is what the endpoint has to report.
    """
    config_api.set_setting(db_session, "ollama_api_base", "http://ollama:11434")
    monkeypatch.setenv("OLLAMA_API_BASE", "https://ollama.com")

    async def _fake_check_ollama_status(_api_base: str) -> str:
        """Answer without reaching the network; this test is about the base URL."""
        return "connected"

    monkeypatch.setattr(config_api, "check_ollama_status", _fake_check_ollama_status)

    response = await get_llm_config(db=db_session, _auth=True)

    assert response.ollama_api_base == "http://ollama:11434", (
        "the reader reported the environment instead of the admin's saved selection"
    )


@pytest.mark.asyncio
async def test_ollama_models_reads_the_saved_row_not_the_environment(
    monkeypatch, db_session
) -> None:
    """The second reader has to agree with the first, or the model list lies."""
    config_api.set_setting(db_session, "ollama_api_base", "http://ollama:11434")
    monkeypatch.setenv("OLLAMA_API_BASE", "https://ollama.com")
    seen = _record_ollama_requests(monkeypatch)

    await config_api.get_ollama_models(db=db_session, _auth=True)

    assert seen["url"] == "http://ollama:11434/api/tags", (
        f"the model list queried {seen['url']!r} instead of the saved daemon"
    )


@pytest.mark.asyncio
async def test_both_readers_fall_back_to_the_environment_when_nothing_is_saved(
    monkeypatch, db_session
) -> None:
    """With no row the environment still applies -- the fix must not drop it."""
    monkeypatch.setenv("OLLAMA_API_BASE", "http://env-only:11434")
    seen = _record_ollama_requests(monkeypatch)

    async def _fake_check_ollama_status(_api_base: str) -> str:
        """Answer without reaching the network; this test is about the base URL."""
        return "disconnected"

    monkeypatch.setattr(config_api, "check_ollama_status", _fake_check_ollama_status)

    response = await get_llm_config(db=db_session, _auth=True)
    await config_api.get_ollama_models(db=db_session, _auth=True)

    assert response.ollama_api_base == "http://env-only:11434"
    assert seen["url"] == "http://env-only:11434/api/tags"


@pytest.mark.asyncio
async def test_both_readers_fall_back_to_the_cloud_default_when_nothing_is_set(
    monkeypatch, db_session
) -> None:
    """Nothing saved and nothing exported is the documented cloud default."""
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    seen = _record_ollama_requests(monkeypatch)

    async def _fake_check_ollama_status(_api_base: str) -> str:
        """Answer without reaching the network; this test is about the base URL."""
        return "disconnected"

    monkeypatch.setattr(config_api, "check_ollama_status", _fake_check_ollama_status)

    response = await get_llm_config(db=db_session, _auth=True)
    await config_api.get_ollama_models(db=db_session, _auth=True)

    assert response.ollama_api_base == OLLAMA_API_BASE_DEFAULT
    assert seen["url"] == f"{OLLAMA_API_BASE_DEFAULT}/api/tags"
