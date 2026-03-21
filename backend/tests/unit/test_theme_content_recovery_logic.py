"""Focused tests for theme content corruption recovery helpers."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import DatabaseError

import app.api.v1.themes as themes_api


def test_reset_corrupt_theme_content_storage_compacts_before_recreate(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    class _DummyConn:
        pass

    class _DummyBegin:
        def __enter__(self):
            return _DummyConn()

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyEngine:
        def begin(self):
            return _DummyBegin()

    monkeypatch.setattr(themes_api, "engine", _DummyEngine())
    monkeypatch.setattr(themes_api, "_drop_theme_content_tables", lambda conn: calls.append("drop"))
    monkeypatch.setattr(
        themes_api,
        "_rewind_theme_content_source_cursors",
        lambda conn: calls.append("rewind"),
    )
    monkeypatch.setattr(
        themes_api,
        "_attempt_compact_theme_content_storage",
        lambda: calls.append("compact") or True,
    )
    monkeypatch.setattr(themes_api, "_recreate_theme_content_tables", lambda: calls.append("recreate"))

    themes_api._reset_corrupt_theme_content_storage(
        DatabaseError(
            "SELECT * FROM content_items",
            {},
            Exception("database disk image is malformed"),
        )
    )

    assert calls == ["drop", "rewind", "compact", "recreate"]


def test_attempt_compact_theme_content_storage_skips_vacuum_when_checkpoint_is_busy(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    class _Result:
        def fetchone(self):
            return (1, 12, 0)

    class _DummyConn:
        def execution_options(self, **kwargs):
            assert kwargs == {"isolation_level": "AUTOCOMMIT"}
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, statement):
            sql = str(statement)
            calls.append(sql)
            if sql == "PRAGMA wal_checkpoint(TRUNCATE)":
                return _Result()
            pytest.fail("VACUUM should not run when wal_checkpoint reports busy readers")

    class _DummyEngine:
        def connect(self):
            return _DummyConn()

    monkeypatch.setattr(themes_api, "engine", _DummyEngine())

    assert not themes_api._attempt_compact_theme_content_storage()
    assert calls == ["PRAGMA wal_checkpoint(TRUNCATE)"]
