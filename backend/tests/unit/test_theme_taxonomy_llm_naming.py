from types import SimpleNamespace

from app.services import llm as llm_module
from app.services import theme_taxonomy_service as taxonomy_module
from app.services.theme_taxonomy_service import ThemeTaxonomyService


def test_unavailable_llm_names_every_cluster_by_fallback_without_sleeping(monkeypatch):
    def broken_llm(**_kwargs):
        raise RuntimeError("no provider configured")

    sleeps: list[float] = []
    monkeypatch.setattr(llm_module, "LLMService", broken_llm)
    monkeypatch.setattr(taxonomy_module.time, "sleep", sleeps.append)
    service = ThemeTaxonomyService.__new__(ThemeTaxonomyService)
    monkeypatch.setattr(
        service,
        "_fallback_name_cluster",
        lambda group: {"l1_name": group[0].display_name, "category": "other", "_fallback": True},
    )
    clusters = [[SimpleNamespace(display_name=f"Theme {i}")] for i in range(5)]

    assignments = service._phase3_llm_naming(clusters, dry_run=True)

    assert [a["l1_name"] for a in assignments] == [f"Theme {i}" for i in range(5)]
    assert sleeps == []
