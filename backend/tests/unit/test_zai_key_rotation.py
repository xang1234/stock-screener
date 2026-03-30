"""Tests for Z.AI managed key rotation in the shared LLM service."""

from __future__ import annotations

import pytest

from app.services.llm.config import get_preset_for_use_case
from app.services.llm.groq_key_manager import GroqKeyManager
from app.services.llm.llm_service import LLMError, LLMQuotaExceededError, LLMService
from app.services.llm.zai_key_manager import ZAIKeyManager
from app.services.llm import llm_service as llm_service_module
from app.services.llm import zai_key_manager as zai_key_manager_module


class FakeRateLimitError(Exception):
    """Test double for LiteLLM rate-limit errors."""


class FakeAPIError(Exception):
    """Test double for LiteLLM API errors."""


def _build_service() -> LLMService:
    service = LLMService.__new__(LLMService)
    service.preset = get_preset_for_use_case("extraction")
    service._groq_key_manager = GroqKeyManager(keys=[])
    service._zai_key_manager = ZAIKeyManager(keys=["zai-key-1", "zai-key-2"])
    return service


@pytest.mark.asyncio
async def test_call_with_retry_reuses_same_zai_key_across_retries(monkeypatch) -> None:
    service = _build_service()
    calls: list[str] = []
    reports: list[tuple[str, float | None]] = []

    monkeypatch.setattr(llm_service_module, "RateLimitError", FakeRateLimitError)

    async def fake_acompletion(**params):
        calls.append(params["api_key"])
        if len(calls) == 1:
            raise FakeRateLimitError("429 retry after 2 s")
        return {"status": "ok"}

    def track_report(key: str, retry_after: float | None) -> None:
        reports.append((key, retry_after))

    monkeypatch.setattr(llm_service_module, "acompletion", fake_acompletion)
    monkeypatch.setattr(service._zai_key_manager, "report_rate_limit", track_report)

    result = await service._call_with_retry(
        {"model": "openai/glm-4.7-flash"},
        num_retries=1,
        base_delay=0.0,
        max_delay=0.0,
    )

    assert result == {"status": "ok"}
    assert calls == ["zai-key-1", "zai-key-1"]
    assert reports == [("zai-key-1", 2.0)]


def test_zai_key_rotates_on_later_request_after_rate_limit(monkeypatch) -> None:
    service = _build_service()
    first_params = {"model": "openai/glm-4.7-flash"}
    second_params = {"model": "openai/glm-4.7-flash"}

    monkeypatch.setattr(zai_key_manager_module.random, "random", lambda: 0.0)

    service._apply_provider_overrides(first_params)
    service._zai_key_manager.report_rate_limit("zai-key-1", retry_after=30.0)
    service._apply_provider_overrides(second_params)

    assert first_params["api_key"] == "zai-key-1"
    assert second_params["api_key"] == "zai-key-2"


@pytest.mark.asyncio
async def test_completion_stream_reports_zai_api_error_rate_limit(monkeypatch) -> None:
    service = _build_service()
    reports: list[tuple[str, float | None]] = []

    monkeypatch.setattr(llm_service_module, "APIError", FakeAPIError)

    async def fake_acompletion(**_params):
        raise FakeAPIError("429 retry after 3 s")

    def track_report(key: str, retry_after: float | None) -> None:
        reports.append((key, retry_after))

    monkeypatch.setattr(llm_service_module, "acompletion", fake_acompletion)
    monkeypatch.setattr(service._zai_key_manager, "report_rate_limit", track_report)

    stream = service._completion_stream(
        messages=[{"role": "user", "content": "hello"}],
        model="openai/glm-4.7-flash",
        allow_fallbacks=False,
    )

    with pytest.raises(LLMError, match="All streaming models failed"):
        async for _chunk in stream:
            pass

    assert reports == [("zai-key-1", 3.0)]


def test_completion_sync_reports_zai_api_error_rate_limit(monkeypatch) -> None:
    service = _build_service()
    reports: list[tuple[str, float | None]] = []

    monkeypatch.setattr(llm_service_module, "APIError", FakeAPIError)

    def fake_completion(**_params):
        raise FakeAPIError("429 retry after 4 s")

    def track_report(key: str, retry_after: float | None) -> None:
        reports.append((key, retry_after))

    monkeypatch.setattr(llm_service_module, "completion", fake_completion)
    monkeypatch.setattr(service._zai_key_manager, "report_rate_limit", track_report)

    with pytest.raises(LLMError, match="All models failed"):
        service.completion_sync(
            messages=[{"role": "user", "content": "hello"}],
            model="openai/glm-4.7-flash",
            allow_fallbacks=False,
            num_retries=0,
        )

    assert reports == [("zai-key-1", 4.0)]


def test_completion_sync_raises_quota_exhausted_for_zai_api_error(monkeypatch) -> None:
    service = _build_service()
    reports: list[tuple[str, float | None]] = []

    monkeypatch.setattr(llm_service_module, "APIError", FakeAPIError)

    def fake_completion(**_params):
        raise FakeAPIError("429 quota exceeded retry after 5 s")

    def track_report(key: str, retry_after: float | None) -> None:
        reports.append((key, retry_after))

    monkeypatch.setattr(llm_service_module, "completion", fake_completion)
    monkeypatch.setattr(service._zai_key_manager, "report_rate_limit", track_report)

    with pytest.raises(LLMQuotaExceededError, match="quota exhausted"):
        service.completion_sync(
            messages=[{"role": "user", "content": "hello"}],
            model="openai/glm-4.7-flash",
            allow_fallbacks=False,
            num_retries=0,
        )

    assert reports == [("zai-key-1", 5.0)]
