from types import SimpleNamespace

from app.config import settings
from app.services import llm as llm_module
from app.services import theme_taxonomy_service as taxonomy_module
from app.services.llm import LLMError
from app.services.llm import llm_service as llm_service_module
from app.services.llm.groq_key_manager import GroqKeyManager
from app.services.llm.zai_key_manager import ZAIKeyManager
from app.services.theme_taxonomy_service import ThemeTaxonomyService


def _naming_service(monkeypatch) -> ThemeTaxonomyService:
    service = ThemeTaxonomyService.__new__(ThemeTaxonomyService)
    monkeypatch.setattr(
        service,
        "_fallback_name_cluster",
        lambda group: {"l1_name": group[0].display_name, "category": "other", "_fallback": True},
    )
    return service


def _configure_llm_keys(monkeypatch, *, zai_keys: list[str]) -> None:
    """Build a real LLMService whose only keys are ``zai_keys``."""
    for name in ("GROQ_API_KEY", "GROQ_API_KEYS", "ZAI_API_KEY", "ZAI_API_KEYS",
                 "MINIMAX_API_KEY", "OPENCODE_GO_API_KEY", "OLLAMA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name in ("groq_api_key", "groq_api_keys", "zai_api_key", "zai_api_keys",
                 "minimax_api_key", "opencode_go_api_key"):
        monkeypatch.setattr(settings, name, "")
    monkeypatch.setattr(llm_service_module, "_load_ollama_api_base", lambda: None)
    monkeypatch.setattr(llm_service_module, "get_groq_key_manager", lambda: GroqKeyManager(keys=[]))
    monkeypatch.setattr(llm_service_module, "get_zai_key_manager", lambda: ZAIKeyManager(keys=zai_keys))


def test_unavailable_llm_names_every_cluster_by_fallback_without_sleeping(monkeypatch):
    def broken_llm(**_kwargs):
        raise RuntimeError("no provider configured")

    sleeps: list[float] = []
    monkeypatch.setattr(llm_module, "LLMService", broken_llm)
    monkeypatch.setattr(taxonomy_module.time, "sleep", sleeps.append)
    service = _naming_service(monkeypatch)
    clusters = [[SimpleNamespace(display_name=f"Theme {i}")] for i in range(5)]

    assignments = service._phase3_llm_naming(clusters, dry_run=True)

    assert [a["l1_name"] for a in assignments] == [f"Theme {i}" for i in range(5)]
    assert sleeps == []


def test_llm_without_provider_keys_names_by_fallback_without_sleeping(monkeypatch):
    _configure_llm_keys(monkeypatch, zai_keys=[])

    def keyless_completion(self, **_kwargs):
        raise LLMError("no api key")

    sleeps: list[float] = []
    monkeypatch.setattr(llm_service_module.LLMService, "completion_sync", keyless_completion)
    monkeypatch.setattr(taxonomy_module.time, "sleep", sleeps.append)
    service = _naming_service(monkeypatch)
    clusters = [[SimpleNamespace(display_name=f"Theme {i}")] for i in range(5)]

    assignments = service._phase3_llm_naming(clusters, dry_run=True)

    assert [a["l1_name"] for a in assignments] == [f"Theme {i}" for i in range(5)]
    assert sleeps == []


def test_fallback_provider_key_alone_still_names_clusters_by_llm(monkeypatch):
    _configure_llm_keys(monkeypatch, zai_keys=["zai-key"])
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content='{"l1_name": "AI Infrastructure", "category": "technology"}'),
    )])
    monkeypatch.setattr(llm_service_module.LLMService, "completion_sync", lambda self, **_kw: response)
    monkeypatch.setattr(taxonomy_module.time, "sleep", lambda _seconds: None)
    service = _naming_service(monkeypatch)
    clusters = [[SimpleNamespace(display_name=f"Theme {i}")] for i in range(2)]

    assignments = service._phase3_llm_naming(clusters, dry_run=True)

    assert [a["l1_name"] for a in assignments] == ["AI Infrastructure", "AI Infrastructure"]
