"""Tests for the static-site export CLI bootstrap behavior."""

from __future__ import annotations

from types import SimpleNamespace

import app.scripts.export_static_site as export_script
import app.tasks.breadth_tasks as breadth_tasks
import app.tasks.cache_tasks as cache_tasks
import app.tasks.fundamentals_tasks as fundamentals_tasks
import app.tasks.group_rank_tasks as group_rank_tasks
import app.tasks.universe_tasks as universe_tasks
from app.interfaces.tasks import feature_store_tasks


def test_ensure_database_path_ready_creates_parent_directory(tmp_path, monkeypatch):
    database_path = tmp_path / "nested" / "data" / "stockscanner.db"
    monkeypatch.setattr(
        export_script.settings,
        "database_url",
        f"sqlite:///{database_path}",
    )

    export_script._ensure_database_path_ready()  # noqa: SLF001 - intentional unit test coverage

    assert database_path.parent.exists()


def test_run_daily_refresh_bootstraps_universe_before_other_tasks(monkeypatch):
    calls: list[str] = []

    def make_task(name: str):
        return SimpleNamespace(
            run=lambda **kwargs: calls.append(name) or {"task": name, "kwargs": kwargs},
        )

    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", make_task("universe_refresh"))
    monkeypatch.setattr(cache_tasks, "smart_refresh_cache", make_task("cache_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(breadth_tasks, "calculate_daily_breadth_with_gapfill", make_task("breadth_refresh"))
    monkeypatch.setattr(group_rank_tasks, "calculate_daily_group_rankings", make_task("groups_refresh"))
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))

    results, warnings = export_script._run_daily_refresh(  # noqa: SLF001 - intentional unit test coverage
        refresh_themes_best_effort=False,
    )

    assert warnings == []
    assert calls == [
        "universe_refresh",
        "cache_refresh",
        "fundamentals_refresh",
        "breadth_refresh",
        "groups_refresh",
        "feature_snapshot",
    ]
    assert results["universe_refresh"]["task"] == "universe_refresh"
    assert results["cache_refresh"]["kwargs"] == {"mode": "full"}
