"""Theme handlers queue the themes bootstrap rebuild instead of building it inline (#526)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

_API_DIR = Path(__file__).resolve().parents[2] / "app" / "api" / "v1"


@pytest.mark.parametrize(
    "module",
    ["themes_review_merge.py", "themes_taxonomy.py", "themes_content_sources.py", "themes_queries.py"],
)
def test_theme_handlers_do_not_rebuild_the_bootstrap_inline(module):
    source = (_API_DIR / module).read_text(encoding="utf-8")

    assert "safe_publish_themes_bootstrap_variants" not in source
    assert "queue_themes_bootstrap_publish" in source


def test_marking_an_alert_read_queues_a_bootstrap_rebuild():
    # The unread count is part of the themes source revision, so a read alert
    # makes every snapshot stale until something republishes it.
    from unittest.mock import MagicMock

    from app.api.v1 import themes_review_merge

    alert = MagicMock()
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = alert

    with patch("app.tasks.theme_discovery_tasks.queue_themes_bootstrap_publish") as queue:
        themes_review_merge.mark_alert_read(7, db=db)

    db.commit.assert_called_once()
    queue.assert_called_once_with()


@pytest.mark.parametrize("pipeline", ["technical", None])
def test_publish_themes_bootstrap_snapshots_rebuilds_the_pipeline_variants(pipeline):
    from app.tasks.theme_discovery_tasks import publish_themes_bootstrap_snapshots

    with patch("app.services.ui_snapshot_service.safe_publish_themes_bootstrap_variants") as publish:
        publish_themes_bootstrap_snapshots.run(pipeline)

    publish.assert_called_once_with(pipeline)


def test_queue_themes_bootstrap_publish_uses_a_short_timeout_connection():
    from app.tasks import theme_discovery_tasks

    with (
        patch.object(theme_discovery_tasks.celery_app, "connection_for_write") as connection_for_write,
        patch.object(
            theme_discovery_tasks.publish_themes_bootstrap_snapshots, "apply_async"
        ) as apply_async,
    ):
        theme_discovery_tasks.queue_themes_bootstrap_publish("fundamental")

    options = connection_for_write.call_args.kwargs["transport_options"]
    assert options["socket_connect_timeout"] <= 1.0
    assert options["max_retries"] == 0
    connection = connection_for_write.return_value.__enter__.return_value
    apply_async.assert_called_once_with(args=["fundamental"], retry=False, connection=connection)


def test_queue_themes_bootstrap_publish_logs_and_returns_when_the_broker_is_down():
    from app.tasks import theme_discovery_tasks

    with (
        patch.object(theme_discovery_tasks.celery_app, "connection_for_write") as connection_for_write,
        patch.object(theme_discovery_tasks.logger, "warning") as warning,
    ):
        connection_for_write.return_value.__enter__.side_effect = ConnectionError("down")
        theme_discovery_tasks.queue_themes_bootstrap_publish(None)

    warning.assert_called_once()
    assert warning.call_args.kwargs.get("exc_info") is True


@pytest.mark.parametrize(
    "task_path",
    [
        "app.tasks.theme_discovery_tasks.publish_themes_bootstrap_snapshots",
        "app.tasks.scan_tasks.publish_scan_bootstrap_snapshots",
    ],
)
def test_bootstrap_publish_tasks_skip_the_result_backend(task_path):
    # apply_async subscribes to the result backend before sending unless the
    # task ignores its result, and that subscription retries for minutes when
    # Redis is down, outside the enqueue's short broker timeout. It hung CI.
    import importlib

    module_name, task_name = task_path.rsplit(".", 1)
    task = getattr(importlib.import_module(module_name), task_name)

    assert task.ignore_result is True


def test_changing_feed_pipelines_queues_a_bootstrap_rebuild(monkeypatch):
    # Reconciliation rewrites item pipeline state, which is part of the themes
    # source revision; without a rebuild the snapshot stays stale.
    from unittest.mock import MagicMock

    from app.api.v1 import themes_content_sources as api
    from app.schemas.theme import ContentSourceUpdate

    existing = MagicMock(pipelines=["technical"])
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = existing
    monkeypatch.setattr(api, "_reject_social_owned", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api, "reconcile_source_pipeline_change", lambda **_kwargs: {})
    monkeypatch.setattr(api.ContentSourceResponse, "model_validate", lambda _obj: None)

    with patch("app.tasks.theme_discovery_tasks.queue_themes_bootstrap_publish") as queue:
        api.update_content_source(
            3, ContentSourceUpdate(pipelines=["technical", "fundamental"]), db=db
        )

    db.commit.assert_called_once()
    queue.assert_called_once_with()


def test_deactivating_a_feed_queues_a_bootstrap_rebuild(monkeypatch):
    from unittest.mock import MagicMock

    from app.api.v1 import themes_content_sources as api

    db = MagicMock()
    monkeypatch.setattr(api, "_reject_social_owned", lambda *_args, **_kwargs: None)

    with patch("app.tasks.theme_discovery_tasks.queue_themes_bootstrap_publish") as queue:
        api.delete_content_source(3, db=db)

    db.commit.assert_called_once()
    queue.assert_called_once_with()
