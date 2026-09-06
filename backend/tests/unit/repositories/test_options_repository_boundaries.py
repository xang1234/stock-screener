from __future__ import annotations

from app.infra.db.repositories.options_history_repository import (
    SqlOptionsHistoryRepository,
)
from app.infra.db.repositories.options_retention import SqlOptionsRetentionRepository
from app.infra.db.repositories.options_run_writer import SqlOptionsRunWriter
from app.infra.db.repositories.published_options_reader import (
    SqlPublishedOptionsReader,
)


def test_options_repositories_expose_only_their_owned_responsibilities() -> None:
    assert not hasattr(SqlOptionsRunWriter, "get_published_run")
    assert not hasattr(SqlOptionsRunWriter, "export_history_observations")
    assert not hasattr(SqlOptionsRunWriter, "prune")
    assert not hasattr(SqlPublishedOptionsReader, "publish")
    assert not hasattr(SqlOptionsHistoryRepository, "publish")
    assert not hasattr(SqlOptionsRetentionRepository, "get_published_run")
