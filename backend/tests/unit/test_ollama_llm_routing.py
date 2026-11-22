"""Tests for Ollama model routing and provider overrides."""

from __future__ import annotations

import os

import pytest

from app.services.llm.groq_key_manager import GroqKeyManager
from app.services.llm.zai_key_manager import ZAIKeyManager
from app.services.llm.llm_service import LLMService


def _settings_with_ollama_key(key: str, base: str = "https://ollama.com"):
    """A stand-in that still carries the cloud base, as a Settings instance does.

    ``base`` mirrors the real field default on purpose. A stub without it would
    let the "saved selection wins" test pass even if the code went back to
    reading Settings first, because ``getattr`` would find nothing to prefer.
    """

    class _Settings:
        """Carries both Ollama fields, key included."""

        ollama_api_key = key
        ollama_api_base = base

    return _Settings()


def _service(*, key: str = "", base: str = "https://ollama.com") -> LLMService:
    """An ``LLMService`` with the Ollama attributes set and nothing else wired.

    Built via ``__new__`` so the provider overrides can be exercised without a
    real client; every attribute ``_apply_provider_overrides`` reads is set here
    explicitly, so a missing one fails loudly instead of falling back.
    """
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
    """Both spellings count as Ollama; ``ollama/`` is the one users configure."""
    assert LLMService._is_ollama_model("ollama/deepseek-v4.1-flash") is True
    assert LLMService._is_ollama_model("ollama_chat/deepseek-v4.1-flash") is True


def test_is_ollama_model_rejects_other_providers() -> None:
    """A substring match would catch these; the check must be prefix-based."""
    assert LLMService._is_ollama_model("minimax/MiniMax-M2.7") is False
    assert LLMService._is_ollama_model("openai/glm-4.7-flash") is False
    assert LLMService._is_ollama_model("groq/qwen/qwen3-32b") is False
    assert LLMService._is_ollama_model("opencode-go/deepseek-v4-flash") is False


def test_apply_provider_overrides_routes_ollama_to_chat_endpoint() -> None:
    """``ollama/`` is rewritten to LiteLLM's ``ollama_chat/`` route.

    ``ollama_chat`` makes LiteLLM append ``/api/chat`` itself, which is what the
    Ollama Cloud endpoint expects. The key is injected only when one is set.
    """
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
    """An unrelated provider must come out of the override untouched.

    Guards against a rewritten branch matching too broadly and attaching the
    Ollama key or base URL to another provider's request.
    """
    service = _service(key="test-ollama-key")
    params = {"model": "groq/qwen/qwen3-32b"}

    service._apply_provider_overrides(params)

    assert "api_key" not in params
    assert "api_base" not in params
    assert params["model"] == "groq/qwen/qwen3-32b"


def test_ollama_model_is_sanctioned_for_extraction() -> None:
    """The allowlist must accept it, or the config API rejects the selection."""
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
    """Without a mapping the provider never receives its key."""
    from app.services.llm.config import PROVIDER_ENV_VARS

    assert PROVIDER_ENV_VARS["ollama"] == "OLLAMA_API_KEY"


def test_ollama_model_is_listed_as_available() -> None:
    """It must appear in the catalogue the UI and the config API read."""
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
    """With nothing configured, the default must reach Ollama Cloud."""
    from app.config.settings import Settings

    assert Settings().ollama_api_base == "https://ollama.com"


def test_setup_api_keys_reads_ollama_settings(monkeypatch) -> None:
    """Exercise the real wiring path from the saved selection into the attributes."""
    from app.services.llm.config import get_preset_for_use_case
    from app.services.llm.llm_service import LLMService

    monkeypatch.setattr(
        "app.services.llm.llm_service._load_ollama_api_base",
        lambda: "http://ollama:11434",
    )
    monkeypatch.setattr(
        "app.services.llm.llm_service.settings", _settings_with_ollama_key("wired-key")
    )

    service = LLMService.__new__(LLMService)
    service.preset = get_preset_for_use_case("extraction")
    service._setup_api_keys("http://ollama:11434")

    assert service._ollama_api_key == "wired-key"
    assert service._ollama_api_base == "http://ollama:11434"


def test_setup_api_keys_falls_back_to_cloud_default(monkeypatch) -> None:
    """With nothing configured the base must still resolve to Ollama Cloud."""
    from app.services.llm.config import get_preset_for_use_case
    from app.services.llm.llm_service import LLMService

    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    monkeypatch.setattr(
        "app.services.llm.llm_service.settings", _settings_with_ollama_key("")
    )

    service = LLMService.__new__(LLMService)
    service.preset = get_preset_for_use_case("extraction")
    service._setup_api_keys(None)

    assert service._ollama_api_key == ""
    assert service._ollama_api_base == "https://ollama.com"


def test_saved_selection_beats_the_settings_default(monkeypatch) -> None:
    """The admin's saved destination wins over the cloud default in Settings.

    Raised by @xang1234 as P1. ``POST /config/ollama`` writes the
    ``ollama_api_base`` row and ``OLLAMA_API_BASE``, but it cannot reach a
    Settings instance built from the environment -- and that instance carries a
    truthy default. Reading Settings first therefore returned the cloud host for
    a deployment an admin had pointed at a local daemon:

        saved base http://127.0.0.1:9  ->  resolved https://ollama.com
        (reproduced before the fix)

    The saved value is authoritative now, whatever Settings says.
    """
    from app.config.settings import Settings
    from app.services.llm.config import get_preset_for_use_case
    from app.services.llm.llm_service import LLMService

    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    # The default a Settings instance carries when nothing is in the environment.
    assert Settings.model_fields["ollama_api_base"].default == "https://ollama.com"
    monkeypatch.setattr(
        "app.services.llm.llm_service.settings",
        _settings_with_ollama_key("cloud-key", base="https://ollama.com"),
    )

    service = LLMService.__new__(LLMService)
    service.preset = get_preset_for_use_case("extraction")
    service._setup_api_keys("http://127.0.0.1:9")

    assert service._ollama_api_base == "http://127.0.0.1:9", (
        "a saved local-daemon base was overridden; extraction prompts would go "
        "to ollama.com"
    )


def test_environment_applies_when_nothing_is_saved(monkeypatch) -> None:
    """``OLLAMA_API_BASE`` still configures a deployment with no saved row."""
    from app.services.llm.config import get_preset_for_use_case
    from app.services.llm.llm_service import LLMService

    monkeypatch.setenv("OLLAMA_API_BASE", "http://ollama:11434")
    monkeypatch.setattr(
        "app.services.llm.llm_service.settings", _settings_with_ollama_key("")
    )

    service = LLMService.__new__(LLMService)
    service.preset = get_preset_for_use_case("extraction")
    service._setup_api_keys(None)

    assert service._ollama_api_base == "http://ollama:11434"


def test_a_fresh_service_resolves_the_saved_selection(monkeypatch) -> None:
    """A newly constructed LLMService must pick up the saved value by itself.

    The point of the fix: the admin's choice is not a constructor argument that
    a route has to remember, and it is not pushed into the process environment
    of the API container only. A worker building its own service reads the row.
    """
    import app.services.llm.llm_service as mod

    monkeypatch.setattr(mod, "_load_ollama_api_base", lambda: "http://127.0.0.1:9")
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)

    service = mod.LLMService(use_case="extraction")

    assert service._ollama_api_base == "http://127.0.0.1:9"


def test_saved_selection_is_read_from_the_app_settings_row(monkeypatch) -> None:
    """``_load_ollama_api_base`` reads the row ``POST /config/ollama`` writes."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import app.database as db_module
    import app.services.llm.llm_service as mod
    from app.database import Base
    from app.models.app_settings import AppSetting

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as session:
        session.add(
            AppSetting(
                key="ollama_api_base",
                value="http://127.0.0.1:9",
                category="llm",
                description="Ollama API base URL",
            )
        )
        session.commit()

    monkeypatch.setattr(db_module, "SessionLocal", factory)
    assert mod._load_ollama_api_base() == "http://127.0.0.1:9"


def test_missing_saved_selection_yields_none(monkeypatch) -> None:
    """An empty table means "nothing saved"."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import app.database as db_module
    import app.services.llm.llm_service as mod
    from app.database import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(db_module, "SessionLocal", sessionmaker(bind=engine))
    assert mod._load_ollama_api_base() is None


def _break_session_local(monkeypatch) -> None:
    import app.database as db_module

    def _boom(*_args, **_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(db_module, "SessionLocal", _boom)


def test_unreadable_saved_selection_never_falls_back_to_the_cloud(
    monkeypatch, caplog
) -> None:
    """A failed read of the saved row must not send Ollama prompts to ollama.com.

    The admin may have saved a local daemon. A transient DB error in a worker
    used to read as "nothing saved", so the resolver fell through to the cloud
    default and extraction prompts left the deployment.
    """
    import logging

    import app.services.llm.llm_service as mod

    _break_session_local(monkeypatch)
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)

    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        service = mod.LLMService(use_case="extraction")
    assert any(
        r.levelno == logging.WARNING and "Ollama" in r.getMessage()
        for r in caplog.records
    )

    with pytest.raises(mod.LLMPreDispatchError):
        service._apply_provider_overrides({"model": "ollama/deepseek-v4.1-flash"})

    # Other providers do not depend on the Ollama row.
    params: dict = {"model": "minimax/MiniMax-M2.7"}
    service._apply_provider_overrides(params)
    assert params["api_base"] == service._minimax_api_base


def test_unreadable_saved_selection_still_honours_the_environment(monkeypatch) -> None:
    """An explicit ``OLLAMA_API_BASE`` is a destination the operator chose."""
    import app.services.llm.llm_service as mod

    _break_session_local(monkeypatch)
    monkeypatch.setenv("OLLAMA_API_BASE", "http://ollama:11434")

    params: dict = {"model": "ollama/deepseek-v4.1-flash"}
    mod.LLMService(use_case="extraction")._apply_provider_overrides(params)

    assert params["api_base"] == "http://ollama:11434"


@pytest.mark.parametrize(
    "model_id",
    [
        "ollama/deepseek-v4.1-flash",
        "ollama/deepseek-v4.1-flash:cloud",
    ],
)
def test_both_ollama_destinations_are_sanctioned_for_extraction(model_id: str) -> None:
    """Cloud and local-daemon identifiers must both pass the extraction allowlist.

    Raised by @xang1234 as P1 on the model list. Ollama's documentation uses the
    bare name for a direct request to ollama.com and the ``:cloud`` tag for a
    model reached through a local daemon's ``/api/chat``. Only the bare form was
    registered, so selecting a local daemon left the allowlist rejecting the one
    sanctioned model:

        ollama/deepseek-v4.1-flash        extraction -> True
        ollama/deepseek-v4.1-flash:cloud  extraction -> False   (before the fix)
    """
    from app.services.llm.config import is_model_supported_for_use_case

    assert is_model_supported_for_use_case(model_id=model_id, use_case="extraction")


def test_both_ollama_destinations_are_offered_in_the_registry() -> None:
    """A sanctioned model must also be selectable, or the allowlist is unusable."""
    from app.services.llm.config import AVAILABLE_MODELS

    ids = {model["id"] for model in AVAILABLE_MODELS}
    assert "ollama/deepseek-v4.1-flash" in ids
    assert "ollama/deepseek-v4.1-flash:cloud" in ids


def test_local_daemon_model_still_reaches_the_native_endpoint() -> None:
    """The ``:cloud`` tag must survive into LiteLLM's model parameter.

    The routing strips the ``ollama/`` prefix; the tag is part of the model the
    daemon is asked for and must not be dropped on the way.
    """
    params: dict = {"model": "ollama/deepseek-v4.1-flash:cloud"}

    provider, _key, _manager = _service()._apply_provider_overrides(params)

    assert provider == "ollama"
    assert params["model"] == "ollama_chat/deepseek-v4.1-flash:cloud"
    assert params["api_base"] == "https://ollama.com"


@pytest.mark.asyncio
async def test_saved_endpoint_through_the_api_reaches_a_new_service(monkeypatch) -> None:
    """The endpoint-to-service path @xang1234 asked for, end to end.

    Reproduces his report through the real API route rather than by writing the
    row by hand: ``POST /config/ollama`` persists the admin's choice, and a
    freshly constructed ``LLMService`` must then resolve *that* destination --
    not the cloud host the settings default carries.

    Before the fix this resolved ``https://ollama.com`` for a saved
    ``http://127.0.0.1:9``, because the route cannot reach a Settings instance
    built from the environment and the truthy default won over ``OLLAMA_API_BASE``.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import app.database as db_module
    import app.services.llm.llm_service as mod
    from app.api.v1 import config as config_api
    from app.api.v1.config import update_ollama_settings
    from app.database import Base
    from app.models.app_settings import AppSetting
    from app.schemas.config import OllamaSettings

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(db_module, "SessionLocal", factory)
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)

    async def _fake_check_ollama_status(_api_base: str) -> str:
        return "connected"

    monkeypatch.setattr(config_api, "check_ollama_status", _fake_check_ollama_status)

    with factory() as db:
        await update_ollama_settings(
            request=OllamaSettings(api_base="http://127.0.0.1:9"),
            db=db,
            _auth=True,
        )

    # The route wrote the row the service reads.
    with factory() as db:
        saved = db.query(AppSetting).filter(AppSetting.key == "ollama_api_base").first()
    assert saved is not None
    assert saved.value == "http://127.0.0.1:9"

    # The row is the only runtime source. Exporting it into this one process'
    # environment would make the API diverge from the workers.
    assert "OLLAMA_API_BASE" not in os.environ

    # A service constructed afterwards must still resolve the saved destination.
    assert mod.LLMService(use_case="extraction")._ollama_api_base == "http://127.0.0.1:9"


def test_cloud_key_is_not_forwarded_to_a_local_daemon() -> None:
    """A key configured for Ollama Cloud must not reach another host.

    Raised by @coderabbitai. ``ollama_chat`` forwards whatever key it is given
    as an ``Authorization`` header, so a deployment with ``OLLAMA_API_KEY`` set
    was sending that secret to a local or third-party base as soon as the admin
    pointed it at one -- and a local daemon ignores it anyway.

    The key is now attached only when the destination is the cloud host. The
    pre-existing local test could not catch this: it passes ``key=""``, so
    ``api_key`` is absent either way.
    """
    params: dict = {"model": "ollama/deepseek-v4.1-flash:cloud"}

    _service(key="test-ollama-key", base="http://ollama:11434")._apply_provider_overrides(
        params
    )

    assert "api_key" not in params, (
        "the cloud API key was forwarded to a local daemon base: "
        f"{params.get('api_key')!r}"
    )


def test_cloud_key_is_not_forwarded_to_a_third_party_base() -> None:
    """Any non-cloud base is treated the same way, not just a known local host."""
    params: dict = {"model": "ollama/deepseek-v4.1-flash"}

    _service(
        key="test-ollama-key", base="https://ollama-proxy.internal.example"
    )._apply_provider_overrides(params)

    assert "api_key" not in params


def test_cloud_key_is_attached_at_the_cloud_destination() -> None:
    """The cloud host still authenticates; the fix must not disable that."""
    params: dict = {"model": "ollama/deepseek-v4.1-flash"}

    _service(key="test-ollama-key", base="https://ollama.com")._apply_provider_overrides(
        params
    )

    assert params["api_key"] == "test-ollama-key"
    assert params["api_base"] == "https://ollama.com"


def test_a_trailing_slash_does_not_defeat_the_cloud_check() -> None:
    """``POST /config/ollama`` strips the slash, but env configs may not."""
    params: dict = {"model": "ollama/deepseek-v4.1-flash"}

    _service(key="test-ollama-key", base="https://ollama.com/")._apply_provider_overrides(
        params
    )

    assert params["api_key"] == "test-ollama-key"


def test_a_local_daemon_needs_no_key_to_route() -> None:
    """Without a key the local path must still resolve, as it always did."""
    params: dict = {"model": "ollama/deepseek-v4.1-flash:cloud"}

    provider, key, _manager = _service(base="http://ollama:11434")._apply_provider_overrides(
        params
    )

    assert provider == "ollama"
    assert key is None
    assert params["api_base"] == "http://ollama:11434"
