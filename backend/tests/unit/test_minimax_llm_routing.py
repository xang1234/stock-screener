"""Tests for Minimax model routing and provider overrides."""

from __future__ import annotations

from app.services.llm.groq_key_manager import GroqKeyManager
from app.services.llm.zai_key_manager import ZAIKeyManager
from app.services.llm.llm_service import LLMService


def test_is_minimax_model_detects_minimax_prefix() -> None:
    assert LLMService._is_minimax_model("minimax/MiniMax-M2.7") is True
    assert LLMService._is_minimax_model("minimax/MiniMax-M2.5") is True


def test_is_minimax_model_rejects_non_minimax() -> None:
    assert LLMService._is_minimax_model("groq/qwen/qwen3-32b") is False
    assert LLMService._is_minimax_model("openai/glm-4.7-flash") is False
    assert LLMService._is_minimax_model("deepseek/deepseek-chat") is False


def test_apply_provider_overrides_injects_minimax_api_key_and_base() -> None:
    service = LLMService.__new__(LLMService)
    service._groq_key_manager = GroqKeyManager(keys=[])
    service._zai_key_manager = ZAIKeyManager(keys=[])
    service._minimax_api_key = "test-minimax-key"
    service._minimax_api_base = "https://api.minimax.io/v1"
    params = {"model": "minimax/MiniMax-M2.7"}

    service._apply_provider_overrides(params)

    assert params["api_key"] == "test-minimax-key"
    assert params["api_base"] == "https://api.minimax.io/v1"


def test_apply_provider_overrides_does_not_affect_non_minimax_models() -> None:
    service = LLMService.__new__(LLMService)
    service._groq_key_manager = GroqKeyManager(keys=[])
    service._zai_key_manager = ZAIKeyManager(keys=[])
    service._minimax_api_key = "test-minimax-key"
    service._minimax_api_base = "https://api.minimax.io/v1"
    params = {"model": "groq/qwen/qwen3-32b"}

    service._apply_provider_overrides(params)

    assert "api_key" not in params
    assert "api_base" not in params
