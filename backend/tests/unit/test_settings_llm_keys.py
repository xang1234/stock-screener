import importlib

import app.services.ibd_industry_service as ibd_industry_service
from app.config.settings import Settings
from app.models.industry import IBDIndustryGroup
from app.scripts import export_static_site
from app.tasks import industry_tasks

settings_module = importlib.import_module("app.config.settings")


def test_zai_api_keys_list_prefers_multi_key_field() -> None:
    settings = Settings(zai_api_keys=" key-a, key-b ", zai_api_key="single-key")

    assert settings.zai_api_keys_list == ["key-a", "key-b"]


def test_zai_api_keys_list_falls_back_to_single_key() -> None:
    settings = Settings(zai_api_key="single-key")

    assert settings.zai_api_keys_list == ["single-key"]


def test_universe_source_timeout_seconds_must_be_positive() -> None:
    try:
        Settings(universe_source_timeout_seconds=0)
    except ValueError as exc:
        assert "universe_source_timeout_seconds must be > 0" in str(exc)
    else:
        raise AssertionError("Expected invalid universe_source_timeout_seconds to fail")


def test_ibd_industry_csv_path_is_configurable(tmp_path, monkeypatch) -> None:
    override = tmp_path / "custom" / "ibd.csv"
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_text("AAPL,Software\n", encoding="utf-8")

    monkeypatch.setattr(industry_tasks.settings, "ibd_industry_csv_path", str(override))
    monkeypatch.setattr(export_static_site.settings, "ibd_industry_csv_path", str(override))

    assert industry_tasks._tracked_ibd_csv_path() == override
    assert export_static_site._tracked_ibd_csv_path() == override


def test_ibd_industry_csv_path_falls_back_to_project_data_when_override_is_missing(
    tmp_path,
    monkeypatch,
) -> None:
    fallback = tmp_path / "data" / "IBD_industry_group.csv"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_text("AAPL,Software\n", encoding="utf-8")
    missing_override = tmp_path / "missing" / "ibd.csv"

    monkeypatch.setattr(ibd_industry_service, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(industry_tasks.settings, "ibd_industry_csv_path", str(missing_override))
    monkeypatch.setattr(export_static_site.settings, "ibd_industry_csv_path", str(missing_override))

    assert industry_tasks._tracked_ibd_csv_path() == fallback
    assert export_static_site._tracked_ibd_csv_path() == fallback


def test_ibd_industry_loader_falls_back_to_project_data_when_override_is_missing(
    tmp_path,
    monkeypatch,
) -> None:
    fallback = tmp_path / "data" / "IBD_industry_group.csv"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_text("aapl,Software\nmsft,Software\n", encoding="utf-8")
    missing_override = tmp_path / "missing" / "ibd.csv"

    class _FakeDb:
        def __init__(self) -> None:
            self.inserted: list[dict[str, str]] = []
            self.execute_calls = 0
            self.commit_calls = 0
            self.rollback_calls = 0

        def execute(self, _statement) -> None:
            self.execute_calls += 1

        def commit(self) -> None:
            self.commit_calls += 1

        def bulk_insert_mappings(self, model, rows) -> None:
            assert model is IBDIndustryGroup
            self.inserted.extend(rows)

        def rollback(self) -> None:
            self.rollback_calls += 1

    monkeypatch.setattr(ibd_industry_service, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        ibd_industry_service.settings,
        "ibd_industry_csv_path",
        str(missing_override),
    )

    fake_db = _FakeDb()
    loaded = ibd_industry_service.IBDIndustryService.load_from_csv(fake_db)

    assert loaded == 2
    assert fake_db.execute_calls == 1
    assert fake_db.commit_calls == 2
    assert fake_db.rollback_calls == 0
    assert fake_db.inserted == [
        {"symbol": "AAPL", "industry_group": "Software"},
        {"symbol": "MSFT", "industry_group": "Software"},
    ]


def test_ibd_industry_loader_does_not_fallback_for_explicit_missing_path(
    tmp_path,
    monkeypatch,
) -> None:
    fallback = tmp_path / "data" / "IBD_industry_group.csv"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_text("aapl,Software\nmsft,Software\n", encoding="utf-8")
    missing_override = tmp_path / "missing" / "ibd.csv"
    explicit_missing_path = tmp_path / "manual" / "ibd.csv"

    class _FakeDb:
        def __init__(self) -> None:
            self.execute_calls = 0
            self.commit_calls = 0
            self.rollback_calls = 0

        def execute(self, _statement) -> None:
            self.execute_calls += 1

        def commit(self) -> None:
            self.commit_calls += 1

        def bulk_insert_mappings(self, model, rows) -> None:
            raise AssertionError(f"Unexpected bulk insert for {model}: {rows}")

        def rollback(self) -> None:
            self.rollback_calls += 1

    monkeypatch.setattr(ibd_industry_service, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        ibd_industry_service.settings,
        "ibd_industry_csv_path",
        str(missing_override),
    )

    try:
        ibd_industry_service.IBDIndustryService.resolve_tracked_csv_path(
            explicit_missing_path,
        )
    except FileNotFoundError as exc:
        assert str(explicit_missing_path) in str(exc)
    else:
        raise AssertionError("Expected explicit missing path resolution to raise FileNotFoundError")

    fake_db = _FakeDb()
    try:
        ibd_industry_service.IBDIndustryService.load_from_csv(
            fake_db,
            csv_path=explicit_missing_path,
        )
    except FileNotFoundError as exc:
        assert str(explicit_missing_path) in str(exc)
    else:
        raise AssertionError("Expected explicit missing CSV path to raise FileNotFoundError")

    assert fake_db.execute_calls == 0
    assert fake_db.commit_calls == 0
    assert fake_db.rollback_calls == 0


def test_get_project_root_detects_container_style_runtime_layout(tmp_path) -> None:
    settings_path = tmp_path / "app" / "app" / "config" / "settings.py"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text("# test", encoding="utf-8")
    (tmp_path / "app" / "data").mkdir(parents=True, exist_ok=True)

    assert settings_module._get_project_root(settings_path) == tmp_path / "app"


def test_get_project_root_prefers_repo_root_over_backend_subdir_with_matching_markers(tmp_path) -> None:
    repo_root = tmp_path / "stock-screener"
    settings_path = repo_root / "backend" / "app" / "config" / "settings.py"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text("# test", encoding="utf-8")

    (repo_root / "data").mkdir(parents=True, exist_ok=True)
    (repo_root / "frontend").mkdir(parents=True, exist_ok=True)
    (repo_root / "backend" / "data").mkdir(parents=True, exist_ok=True)

    assert settings_module._get_project_root(settings_path) == repo_root


def test_get_project_root_prefers_nearest_strong_match_in_nested_workspace(tmp_path) -> None:
    outer_workspace = tmp_path / "workspace"
    inner_repo = outer_workspace / "projects" / "stock-screener"
    settings_path = inner_repo / "backend" / "app" / "config" / "settings.py"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text("# test", encoding="utf-8")

    # Outer workspace looks more "complete", but should not win over the
    # nearer repo root for this checkout.
    (outer_workspace / "data").mkdir(parents=True, exist_ok=True)
    (outer_workspace / "frontend").mkdir(parents=True, exist_ok=True)
    (outer_workspace / "backend").mkdir(parents=True, exist_ok=True)
    (outer_workspace / "docker-compose.yml").write_text("version: '3'", encoding="utf-8")

    (inner_repo / "data").mkdir(parents=True, exist_ok=True)
    (inner_repo / "backend").mkdir(parents=True, exist_ok=True)

    assert settings_module._get_project_root(settings_path) == inner_repo
