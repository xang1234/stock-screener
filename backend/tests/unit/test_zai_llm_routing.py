"""Tests for Z.AI model routing and theme defaults."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.config import settings
from app.services.llm.config import get_preset_for_use_case
from app.services.llm.llm_service import LLMService
from app.services.theme_extraction_service import ThemeExtractionService


def _llm_json_response(payload: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=payload),
            )
        ]
    )


def test_extraction_preset_defaults_to_zai_glm_flash() -> None:
    preset = get_preset_for_use_case("extraction")

    assert preset.primary.model_id == "openai/glm-4.7-flash"
    assert preset.fallbacks == []


def test_apply_provider_overrides_injects_zai_api_key_and_base(monkeypatch) -> None:
    monkeypatch.setattr(settings, "zai_api_key", "test-zai-key")
    monkeypatch.setattr(settings, "zai_api_base", "https://api.z.ai/api/paas/v4")

    service = LLMService.__new__(LLMService)
    params = {"model": "openai/glm-4.7-flash"}

    service._apply_provider_overrides(params)

    assert params["api_key"] == "test-zai-key"
    assert params["api_base"] == "https://api.z.ai/api/paas/v4"


def test_apply_provider_overrides_clears_zai_overrides_for_non_zai_models() -> None:
    service = LLMService.__new__(LLMService)
    params = {
        "model": "groq/qwen/qwen3-32b",
        "api_key": "stale-key",
        "api_base": "https://stale.invalid",
    }

    service._apply_provider_overrides(params)

    assert "api_key" not in params
    assert "api_base" not in params


def test_try_generate_litellm_disables_fallbacks_for_configured_zai_model() -> None:
    service = ThemeExtractionService.__new__(ThemeExtractionService)
    service.llm = SimpleNamespace(completion=AsyncMock(return_value=_llm_json_response("[]")))
    service.pipeline_config = None
    service.configured_model = "openai/glm-4.7-flash"

    result = service._try_generate_litellm("prompt")

    assert result == "[]"
    kwargs = service.llm.completion.await_args.kwargs
    assert kwargs["model"] == "openai/glm-4.7-flash"
    assert kwargs["allow_fallbacks"] is False
