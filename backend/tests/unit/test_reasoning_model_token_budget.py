"""Tests for the token budget given to a reasoning model.

A reasoning model writes its preamble to the same ``max_tokens`` budget as the
answer. When the budget is sized for a plain completion, the preamble can
consume all of it and leave the answer empty -- the claim review then rejects
the item as ``claim_review_invalid``.

The budget test keys on the provider rather than on a per-model list, because
the family is not one model: over the seventeen ids the Ollama Cloud endpoint
serves, four exhaust 2,000 tokens on the preamble and answer inside 4,000.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.services.llm.llm_service import LLMService
from app.services.theme_extraction_service import ThemeExtractionService


def _llm_json_response(payload: str):
    """A minimal stand-in for a LiteLLM completion response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=payload),
            )
        ]
    )


def _service_with_configured_model(model_id: str) -> ThemeExtractionService:
    """A service whose configured model and LLM client are stubbed."""
    service = ThemeExtractionService.__new__(ThemeExtractionService)
    service.llm = SimpleNamespace(
        preset=SimpleNamespace(primary=SimpleNamespace(model_id="minimax/MiniMax-M2.7")),
        completion=AsyncMock(return_value=_llm_json_response("[]")),
    )
    service.pipeline_config = None
    service.configured_model = model_id
    return service


def test_is_ollama_model_detects_the_cloud_destination() -> None:
    """The bare name reaches ollama.com."""
    assert LLMService._is_ollama_model("ollama/deepseek-v4.1-flash") is True


def test_is_ollama_model_detects_the_local_daemon_form() -> None:
    """The ``:cloud`` tag names the local daemon, but it is the same family."""
    assert LLMService._is_ollama_model("ollama/deepseek-v4.1-flash:cloud") is True


def test_is_ollama_model_detects_another_family_member() -> None:
    """A newly admitted entry must not have to be added to a second list."""
    assert LLMService._is_ollama_model("ollama/gpt-oss:20b") is True
    assert LLMService._is_ollama_model("ollama/kimi-k2.7-code") is True


def test_is_ollama_model_rejects_other_sanctioned_models() -> None:
    """Sanctioned providers outside the Ollama family must not inherit the room."""
    assert LLMService._is_ollama_model("minimax/MiniMax-M2.7") is False
    assert LLMService._is_ollama_model("openai/glm-4.7-flash") is False
    assert LLMService._is_ollama_model("groq/qwen/qwen3-32b") is False


def test_try_generate_litellm_gives_the_cloud_reasoning_model_the_high_budget() -> None:
    """The bare form reaches ollama.com and reasons before it answers."""
    service = _service_with_configured_model("ollama/deepseek-v4.1-flash")

    result = service._try_generate_litellm("prompt")

    assert result == "[]"
    kwargs = service.llm.completion.await_args.kwargs
    assert kwargs["model"] == "ollama/deepseek-v4.1-flash"
    assert kwargs["allow_fallbacks"] is True
    assert kwargs["max_tokens"] == ThemeExtractionService.HIGH_EXTRACTION_MAX_TOKENS


def test_try_generate_litellm_gives_the_local_daemon_form_the_high_budget() -> None:
    """The ``:cloud`` tag names the same family, so it gets the same room."""
    service = _service_with_configured_model("ollama/deepseek-v4.1-flash:cloud")

    result = service._try_generate_litellm("prompt")

    assert result == "[]"
    kwargs = service.llm.completion.await_args.kwargs
    assert kwargs["model"] == "ollama/deepseek-v4.1-flash:cloud"
    assert kwargs["max_tokens"] == ThemeExtractionService.HIGH_EXTRACTION_MAX_TOKENS


def test_try_generate_litellm_gives_a_measured_family_member_the_high_budget() -> None:
    """``glm-5.3-flash`` exhausted 2,000 tokens across 2 of 3 measured runs."""
    service = _service_with_configured_model("ollama/glm-5.3-flash")

    result = service._try_generate_litellm("prompt")

    assert result == "[]"
    kwargs = service.llm.completion.await_args.kwargs
    assert kwargs["max_tokens"] == ThemeExtractionService.HIGH_EXTRACTION_MAX_TOKENS


def test_try_generate_litellm_keeps_the_default_budget_for_a_plain_model() -> None:
    """A model outside the reasoning providers still gets the default budget."""
    service = _service_with_configured_model("groq/qwen/qwen3-32b")

    result = service._try_generate_litellm("prompt")

    assert result == "[]"
    kwargs = service.llm.completion.await_args.kwargs
    assert kwargs["max_tokens"] == ThemeExtractionService.DEFAULT_EXTRACTION_MAX_TOKENS
