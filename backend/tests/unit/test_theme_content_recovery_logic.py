"""Focused tests for theme content corruption recovery helpers."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import DatabaseError

import app.api.v1.themes as themes_api
import app.services.theme_content_recovery_service as recovery_service


def test_reset_corrupt_theme_content_storage_recreates_immediately_after_rewind(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []
    begin_calls = {"count": 0}

    class _DummyConn:
        pass

    class _DummyBegin:
        def __enter__(self):
            begin_calls["count"] += 1
            return _DummyConn()

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyEngine:
        def begin(self):
            return _DummyBegin()

    monkeypatch.setattr(recovery_service, "engine", _DummyEngine())
    monkeypatch.setattr(
        recovery_service,
        "_acquire_theme_content_reset_lock",
        lambda conn: calls.append("lock"),
    )
    monkeypatch.setattr(recovery_service, "drop_theme_content_tables", lambda conn: calls.append("drop"))
    monkeypatch.setattr(
        recovery_service,
        "rewind_theme_content_source_cursors",
        lambda conn: calls.append("rewind"),
    )
    monkeypatch.setattr(
        recovery_service,
        "recreate_theme_content_tables",
        lambda conn: calls.append("recreate"),
    )

    themes_api._reset_corrupt_theme_content_storage(
        DatabaseError(
            "SELECT * FROM content_items",
            {},
            Exception("database disk image is malformed"),
        )
    )

    assert begin_calls["count"] == 1
    assert calls == ["lock", "drop", "rewind", "recreate"]
