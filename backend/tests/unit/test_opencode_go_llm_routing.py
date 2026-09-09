"""OpenCode Go is a named, extraction-only OpenAI-compatible provider."""

import asyncio

import pytest

from app.services.llm.config import (
    DEFAULT_MODEL_BY_USE_CASE,
    get_model_by_id,
    is_model_supported_for_use_case,
)
from app.services.llm.groq_key_manager import GroqKeyManager
from app.services.llm.llm_service import LLMError, LLMPreDispatchError, LLMService
from app.services.llm.zai_key_manager import ZAIKeyManager


MODEL = "opencode-go/deepseek-v4-flash"


def _service():
    service = LLMService.__new__(LLMService)
    service._groq_key_manager = GroqKeyManager(keys=[])
    service._zai_key_manager = ZAIKeyManager(keys=[])
    service._minimax_api_key = ""
    service._minimax_api_base = "https://api.minimax.io/v1"
    service._opencode_go_api_key = "go-secret"
    service._opencode_go_api_base = "https://opencode.ai/zen/go/v1"
    return service


def test_opencode_go_deepseek_flash_is_a_sanctioned_extraction_model():
    assert DEFAULT_MODEL_BY_USE_CASE["extraction"] == "minimax/MiniMax-M2.7"
    assert get_model_by_id(MODEL) == {
        "id": MODEL,
        "name": "DeepSeek V4 Flash (OpenCode Go)",
        "provider": "opencode-go",
        "category": "cloud",
    }
    assert is_model_supported_for_use_case(model_id=MODEL, use_case="extraction")
    assert not is_model_supported_for_use_case(model_id=MODEL, use_case="chatbot")


def test_opencode_go_routes_through_its_openai_compatible_endpoint():
    params = {
        "model": MODEL,
        "reasoning_effort": "high",
        "extra_body": {"fixture": True},
    }

    _service()._apply_provider_overrides(params)

    assert params["model"] == "openai/deepseek-v4-flash"
    assert params["api_key"] == "go-secret"
    assert params["api_base"] == "https://opencode.ai/zen/go/v1"
    assert "reasoning_effort" not in params
    assert params["extra_body"] == {
        "fixture": True,
        "reasoning_effort": "none",
    }
    assert params["extra_headers"]["User-Agent"].startswith("StockScreen/")
    assert params["extra_headers"]["x-opencode-session"].startswith("social-")


def test_opencode_go_never_falls_back_to_an_unrelated_openai_key():
    service = _service()
    service._opencode_go_api_key = ""

    with pytest.raises(LLMError, match="opencode_go_api_key_not_configured"):
        service._apply_provider_overrides({"model": MODEL})


def test_metered_missing_key_is_classified_before_provider_dispatch(monkeypatch):
    from app.services.llm import llm_service as llm_module

    service = LLMService(use_case="extraction")
    service._opencode_go_api_key = ""
    calls = []

    async def provider_call(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(llm_module, "acompletion", provider_call)

    with pytest.raises(LLMPreDispatchError):
        asyncio.run(service.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "fixture"}],
            max_tokens=20,
            allow_fallbacks=False,
            num_retries=0,
            metered=True,
        ))

    assert calls == []
