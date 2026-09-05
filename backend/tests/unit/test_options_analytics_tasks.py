from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.celery_app import celery_app
from app.services.runtime_activity_contract import stage_label
from app.services.task_registry_service import SCHEDULED_TASKS


def test_disabled_options_task_exits_before_database_or_lock(monkeypatch) -> None:
    from app.interfaces.tasks import options_analytics_tasks as module

    monkeypatch.setattr(module.settings, "options_analytics_enabled", False)
    monkeypatch.setattr(
        module,
        "SessionLocal",
        lambda: (_ for _ in ()).throw(AssertionError("database touched")),
    )

    result = module.refresh_options_analytics.run()

    assert result == {
        "status": "skipped",
        "reason_codes": ["options_analytics_disabled"],
    }


def test_options_task_runs_us_use_case_in_process(monkeypatch) -> None:
    from app.interfaces.tasks import options_analytics_tasks as module
    from app.tasks.data_fetch_lock import disable_serialized_data_fetch_lock

    db = SimpleNamespace(close=lambda: None)
    executed = []
    activities = []
    monkeypatch.setattr(module.settings, "options_analytics_enabled", True)
    monkeypatch.setattr(module, "SessionLocal", lambda: db)
    monkeypatch.setattr(
        module,
        "get_refresh_options_analytics_use_case",
        lambda session: SimpleNamespace(
            execute=lambda command: executed.append((session, command))
            or {
                "status": "published",
                "run_id": 7,
                "expected_count": 40,
                "completed_count": 38,
                "core_valid_current_count": 37,
                "failed_count": 2,
                "retried_count": 3,
                "coverage": 0.925,
            }
        ),
    )
    monkeypatch.setattr(
        module,
        "mark_market_activity_started",
        lambda _db, **values: activities.append(("started", values)),
    )
    monkeypatch.setattr(
        module,
        "mark_market_activity_completed",
        lambda _db, **values: activities.append(("completed", values)),
    )

    with disable_serialized_data_fetch_lock():
        result = module.refresh_options_analytics.run(source_run_id=33, market="US")

    assert result["status"] == "published"
    assert result["run_id"] == 7
    assert executed[0][1].source_run_id == 33
    completed = activities[-1]
    assert completed[0] == "completed"
    assert completed[1]["current"] == 38
    assert completed[1]["total"] == 40
    assert completed[1]["stage_key"] == "options"
    assert "core_valid=37" in completed[1]["message"]
    assert "failed=2" in completed[1]["message"]
    assert "retried=3" in completed[1]["message"]
    assert "coverage=92.5%" in completed[1]["message"]


def test_failed_optional_activity_write_rolls_back_session() -> None:
    from app.interfaces.tasks import options_analytics_tasks as module

    db = MagicMock()

    def fail(_db, **_values):
        raise RuntimeError("activity write failed")

    module._mark_activity_safely(fail, db, market="US")

    db.rollback.assert_called_once_with()


def test_options_task_resolves_published_us_pointer_when_run_is_omitted(
    monkeypatch,
) -> None:
    from app.interfaces.tasks import options_analytics_tasks as module
    from app.tasks.data_fetch_lock import disable_serialized_data_fetch_lock

    executed = []
    db = SimpleNamespace(
        get=lambda _model, key: SimpleNamespace(run_id=91)
        if key == "latest_published_market:US"
        else None,
        close=lambda: None,
    )
    monkeypatch.setattr(module.settings, "options_analytics_enabled", True)
    monkeypatch.setattr(module, "SessionLocal", lambda: db)
    monkeypatch.setattr(
        module,
        "get_refresh_options_analytics_use_case",
        lambda _session: SimpleNamespace(
            execute=lambda command: executed.append(command)
            or {"status": "published", "run_id": 8}
        ),
    )
    monkeypatch.setattr(module, "_mark_activity_safely", lambda *_args, **_kwargs: None)

    with disable_serialized_data_fetch_lock():
        result = module.refresh_options_analytics.run()

    assert result["status"] == "published"
    assert executed[0].source_run_id == 91


def test_options_task_rejects_non_us_market_before_work(monkeypatch) -> None:
    from app.interfaces.tasks import options_analytics_tasks as module
    from app.tasks.data_fetch_lock import disable_serialized_data_fetch_lock

    monkeypatch.setattr(module.settings, "options_analytics_enabled", True)

    with disable_serialized_data_fetch_lock():
        assert module.refresh_options_analytics.run(market="HK") == {
            "status": "skipped",
            "reason_codes": ["market_unsupported"],
        }


def test_options_task_is_registered_on_existing_us_data_fetch_queue() -> None:
    task_name = (
        "app.interfaces.tasks.options_analytics_tasks.refresh_options_analytics"
    )
    assert "app.interfaces.tasks.options_analytics_tasks" in celery_app.conf.include
    assert celery_app.conf.task_routes[task_name] == {"queue": "data_fetch_us"}
    assert SCHEDULED_TASKS["daily-us-options-analytics"]["task_function"] == task_name
    assert stage_label("options") == "Options Analytics"


def test_options_registry_entry_follows_its_own_feature_flag(monkeypatch) -> None:
    from app.services import task_registry_service as module

    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.first.return_value = (
        None
    )
    monkeypatch.setattr(module.settings, "cache_warmup_enabled", True)
    monkeypatch.setattr(module.settings, "options_analytics_enabled", False)

    tasks = {
        task["name"]: task
        for task in module.TaskRegistryService().get_all_scheduled_tasks(db)
    }

    assert tasks["daily-market-pipeline-us"]["is_enabled"] is True
    assert tasks["daily-us-options-analytics"]["is_enabled"] is False
