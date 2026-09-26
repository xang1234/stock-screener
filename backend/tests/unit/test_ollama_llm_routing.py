"""Tests for Ollama model routing and provider overrides."""

from __future__ import annotations

from app.services.llm.groq_key_manager import GroqKeyManager
from app.services.llm.zai_key_manager import ZAIKeyManager
from app.services.llm.llm_service import LLMService


def _service(*, key: str = "", base: str = "https://ollama.com") -> LLMService:
    service = LLMService.__new__(LLMService)
    service._groq_key_manager = GroqKeyManager(keys=[])
    service._zai_key_manager = ZAIKeyManager(keys=[])
    service._minimax_api_key = ""
    service._minimax_api_base = "https://api.minimax.io/v1"
    service._opencode_go_api_key = ""
    service._opencode_go_api_base = "https://opencode.ai/zen/go/v1"
    service._ollama_api_key = key
    service._ollama_api_base = base
    return service


def test_is_ollama_model_detects_both_prefixes() -> None:
    assert LLMService._is_ollama_model("ollama/deepseek-v4.1-flash") is True
    assert LLMService._is_ollama_model("ollama_chat/deepseek-v4.1-flash") is True


def test_is_ollama_model_rejects_other_providers() -> None:
    assert LLMService._is_ollama_model("minimax/MiniMax-M2.7") is False
    assert LLMService._is_ollama_model("openai/glm-4.7-flash") is False
    assert LLMService._is_ollama_model("groq/qwen/qwen3-32b") is False
    assert LLMService._is_ollama_model("opencode-go/deepseek-v4-flash") is False


def test_apply_provider_overrides_routes_ollama_to_chat_endpoint() -> None:
    service = _service(key="test-ollama-key")
    params = {"model": "ollama/deepseek-v4.1-flash"}

    provider_name, provider_key, _ = service._apply_provider_overrides(params)

    assert params["model"] == "ollama_chat/deepseek-v4.1-flash"
    assert params["api_base"] == "https://ollama.com"
    assert params["api_key"] == "test-ollama-key"
    assert provider_name == "ollama"
    assert provider_key == "test-ollama-key"


def test_apply_provider_overrides_omits_key_for_local_daemon() -> None:
    """A local Ollama daemon needs no key, so none must be injected."""
    service = _service(key="", base="http://ollama:11434")
    params = {"model": "ollama/qwen3:8b"}

    service._apply_provider_overrides(params)

    assert params["api_base"] == "http://ollama:11434"
    assert "api_key" not in params


def test_apply_provider_overrides_keeps_model_id_with_colon_tag() -> None:
    """Local model tags use a colon; the tag must survive the rewrite untouched."""
    service = _service(base="http://ollama:11434")
    params = {"model": "ollama/llama3.1:8b-instruct-q4_K_M"}

    service._apply_provider_overrides(params)

    assert params["model"] == "ollama_chat/llama3.1:8b-instruct-q4_K_M"


def test_apply_provider_overrides_is_idempotent_for_ollama() -> None:
    """Re-applying must not produce ``ollama_chat/ollama_chat/...``."""
    service = _service(key="test-ollama-key")
    params = {"model": "ollama/deepseek-v4.1-flash"}

    service._apply_provider_overrides(params)
    service._apply_provider_overrides(params)

    assert params["model"] == "ollama_chat/deepseek-v4.1-flash"


def test_apply_provider_overrides_does_not_affect_other_providers() -> None:
    service = _service(key="test-ollama-key")
    params = {"model": "groq/qwen/qwen3-32b"}

    service._apply_provider_overrides(params)

    assert "api_key" not in params
    assert "api_base" not in params
    assert params["model"] == "groq/qwen/qwen3-32b"


def test_ollama_model_is_sanctioned_for_extraction() -> None:
    from app.services.llm.config import is_model_supported_for_use_case

    assert is_model_supported_for_use_case(
        model_id="ollama/deepseek-v4.1-flash", use_case="extraction"
    ) is True


def test_ollama_model_is_not_sanctioned_for_chatbot() -> None:
    """Ollama is enabled for extraction only; chatbot stays on its existing models."""
    from app.services.llm.config import is_model_supported_for_use_case

    assert is_model_supported_for_use_case(
        model_id="ollama/deepseek-v4.1-flash", use_case="chatbot"
    ) is False


def test_ollama_provider_has_env_var_mapping() -> None:
    from app.services.llm.config import PROVIDER_ENV_VARS

    assert PROVIDER_ENV_VARS["ollama"] == "OLLAMA_API_KEY"


def test_ollama_model_is_listed_as_available() -> None:
    from app.services.llm.config import AVAILABLE_MODELS

    entry = next(
        (m for m in AVAILABLE_MODELS if m["id"] == "ollama/deepseek-v4.1-flash"), None
    )
    assert entry is not None
    assert entry["provider"] == "ollama"


def test_settings_expose_ollama_fields() -> None:
    """The settings fields must exist, otherwise pydantic drops the env vars silently."""
    from app.config.settings import Settings

    configured = Settings(ollama_api_key="k", ollama_api_base="http://ollama:11434")

    assert configured.ollama_api_key == "k"
    assert configured.ollama_api_base == "http://ollama:11434"


def test_settings_default_ollama_base_points_at_cloud() -> None:
    from app.config.settings import Settings

    assert Settings().ollama_api_base == "https://ollama.com"


def test_setup_api_keys_reads_ollama_settings(monkeypatch) -> None:
    """Exercise the real wiring path from settings into the cached instance attributes."""
    from app.config.settings import settings as live_settings
    from app.services.llm.llm_service import LLMService
    from app.services.llm.config import get_preset_for_use_case

    monkeypatch.setattr(live_settings, "ollama_api_key", "wired-key", raising=False)
    monkeypatch.setattr(live_settings, "ollama_api_base", "http://ollama:11434", raising=False)

    service = LLMService.__new__(LLMService)
    service.preset = get_preset_for_use_case("extraction")
    service._setup_api_keys()

    assert service._ollama_api_key == "wired-key"
    assert service._ollama_api_base == "http://ollama:11434"


def test_setup_api_keys_falls_back_to_cloud_default(monkeypatch) -> None:
    """With nothing configured the base must still resolve to Ollama Cloud."""
    from app.config.settings import settings as live_settings
    from app.services.llm.llm_service import LLMService
    from app.services.llm.config import get_preset_for_use_case

    monkeypatch.setattr(live_settings, "ollama_api_key", "", raising=False)
    monkeypatch.setattr(live_settings, "ollama_api_base", "", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)

    service = LLMService.__new__(LLMService)
    service.preset = get_preset_for_use_case("extraction")
    service._setup_api_keys()

    assert service._ollama_api_key == ""
    assert service._ollama_api_base == "https://ollama.com"
