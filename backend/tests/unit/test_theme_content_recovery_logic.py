"""Focused tests for theme content corruption recovery helpers."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import DatabaseError

import app.api.v1.themes as themes_api


def test_reset_corrupt_theme_content_storage_recreates_immediately_after_rewind(
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
    monkeypatch.setattr(themes_api, "_recreate_theme_content_tables", lambda: calls.append("recreate"))

    themes_api._reset_corrupt_theme_content_storage(
        DatabaseError(
            "SELECT * FROM content_items",
            {},
            Exception("database disk image is malformed"),
        )
    )

    assert calls == ["drop", "rewind", "recreate"]
