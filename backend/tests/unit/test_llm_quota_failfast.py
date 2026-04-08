"""Quota fail-fast tests for LLM fallback behavior."""

from __future__ import annotations

import pytest

from app.services.llm.llm_service import LLMQuotaExceededError, LLMService


@pytest.mark.asyncio
async def test_call_with_fallbacks_fails_fast_when_only_groq_models_remain() -> None:
    service = LLMService.__new__(LLMService)

    async def _always_quota(_params, _num_retries):
        raise LLMQuotaExceededError("tokens per day limit reached")

    service._call_with_retry = _always_quota

    with pytest.raises(LLMQuotaExceededError, match="tokens per day"):
        await service._call_with_fallbacks(
            params={"model": "groq/qwen/qwen3-32b"},
            fallbacks=["groq/llama-3.3-70b-versatile"],
            num_retries=0,
        )


@pytest.mark.asyncio
async def test_call_with_fallbacks_continues_to_non_groq_after_groq_quota() -> None:
    service = LLMService.__new__(LLMService)
    calls: list[str] = []

    async def _quota_then_success(params, _num_retries):
        model = str(params["model"])
        calls.append(model)
        if model.startswith("groq/"):
            raise LLMQuotaExceededError("tokens per day limit reached")
        return {"model": model, "status": "ok"}

    service._call_with_retry = _quota_then_success

    result = await service._call_with_fallbacks(
        params={"model": "groq/qwen/qwen3-32b"},
        fallbacks=["openai/glm-4.7-flash"],
        num_retries=0,
    )

    assert result["model"] == "openai/glm-4.7-flash"
    assert calls == ["groq/qwen/qwen3-32b", "openai/glm-4.7-flash"]
