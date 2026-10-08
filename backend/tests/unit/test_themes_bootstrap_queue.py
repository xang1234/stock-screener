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
