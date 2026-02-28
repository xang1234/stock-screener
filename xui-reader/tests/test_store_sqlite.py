"""SQLite store migration and persistence behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import pytest

from xui_reader.errors import StoreError
from xui_reader.models import Checkpoint, SourceKind, SourceRef, TweetItem
from xui_reader.store.sqlite import SQLiteStore


def test_sqlite_store_bootstrap_creates_tables_and_tracks_migrations(tmp_path: Path) -> None:
    db_path = tmp_path / "reader.db"
    store = SQLiteStore(db_path)
    try:
        assert store.migration_versions() == ("0001_initial_store_schema",)
    finally:
        store.close()

    conn = sqlite3.connect(str(db_path))
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "schema_migrations" in tables
        assert "sources" in tables
        assert "checkpoints" in tables
        assert "tweets" in tables
        assert "runs" in tables

        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()
        assert journal_mode is not None
        assert str(journal_mode[0]).lower() == "wal"
    finally:
        conn.close()


def test_sqlite_store_migrates_existing_db_without_data_loss(tmp_path: Path) -> None:
    db_path = tmp_path / "existing.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE sources (
                source_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                value TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE tweets (
                source_id TEXT NOT NULL,
                tweet_id TEXT NOT NULL,
                created_at TEXT,
                author_handle TEXT,
                text TEXT,
                is_reply INTEGER NOT NULL DEFAULT 0,
                is_repost INTEGER NOT NULL DEFAULT 0,
                is_pinned INTEGER NOT NULL DEFAULT 0,
                has_quote INTEGER NOT NULL DEFAULT 0,
                quote_tweet_id TEXT,
                inserted_at TEXT NOT NULL,
                PRIMARY KEY (source_id, tweet_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sources(source_id, kind, value, enabled, updated_at)
            VALUES('src:1', 'list', '123', 1, '2026-01-01T00:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO tweets(
                source_id, tweet_id, created_at, author_handle, text,
                is_reply, is_repost, is_pinned, has_quote, quote_tweet_id, inserted_at
            )
            VALUES(
                'src:1', '111', '2026-01-02T00:00:00+00:00', '@alice', 'hello',
                0, 0, 0, 0, NULL, '2026-01-02T00:00:00+00:00'
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    store = SQLiteStore(db_path)
    try:
        # Bootstrap is idempotent and leaves existing data intact.
        assert store.migration_versions() == ("0001_initial_store_schema",)
        items = store.load_new_since(datetime(2026, 1, 1, tzinfo=timezone.utc))
        assert [item.tweet_id for item in items] == ["111"]
    finally:
        store.close()

    # Re-opening should not reapply or fail with duplicate DDL.
    reopened = SQLiteStore(db_path)
    try:
        assert reopened.migration_versions() == ("0001_initial_store_schema",)
    finally:
        reopened.close()


def test_sqlite_store_save_items_idempotent_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "reader.db")
    try:
        created_at = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
        saved = store.save_items(
            "user:alice",
            (
                TweetItem(
                    tweet_id="9001",
                    created_at=created_at,
                    author_handle="@alice",
                    text="hello",
                    source_id="user:alice",
                    is_reply=False,
                    is_repost=False,
                    is_pinned=False,
                    has_quote=False,
                    quote_tweet_id=None,
                ),
            ),
        )
        assert saved == 1

        saved_again = store.save_items(
            "user:alice",
            (
                TweetItem(
                    tweet_id="9001",
                    created_at=created_at,
                    author_handle="@alice",
                    text="hello",
                    source_id="user:alice",
                    is_reply=False,
                    is_repost=False,
                    is_pinned=False,
                    has_quote=False,
                    quote_tweet_id=None,
                ),
            ),
        )
        assert saved_again == 0

        checkpoint = Checkpoint(
            source_id="user:alice",
            last_seen_id="9001",
            last_seen_time=created_at,
            updated_at=datetime(2026, 2, 1, 12, 5, tzinfo=timezone.utc),
        )
        store.save_checkpoint(checkpoint)
        loaded = store.load_checkpoint("user:alice")
        assert loaded is not None
        assert loaded.source_id == checkpoint.source_id
        assert loaded.last_seen_id == checkpoint.last_seen_id
        assert loaded.last_seen_time == checkpoint.last_seen_time
    finally:
        store.close()


def test_sqlite_store_upsert_source_and_run_lifecycle(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "reader.db")
    try:
        source = SourceRef(
            source_id="list:demo",
            kind=SourceKind.LIST,
            value="12345",
            enabled=True,
        )
        store.upsert_source(source)
        run_id = store.begin_run(source.source_id, started_at=datetime(2026, 2, 2, tzinfo=timezone.utc))
        store.finish_run(
            run_id,
            status="success",
            observed_count=10,
            saved_count=8,
            finished_at=datetime(2026, 2, 2, 0, 1, tzinfo=timezone.utc),
        )
    finally:
        store.close()

    conn = sqlite3.connect(str(tmp_path / "reader.db"))
    try:
        source_row = conn.execute(
            "SELECT source_id, kind, value, enabled FROM sources WHERE source_id = ?",
            ("list:demo",),
        ).fetchone()
        assert source_row == ("list:demo", "list", "12345", 1)

        run_row = conn.execute(
            """
            SELECT source_id, status, observed_count, saved_count, finished_at
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        assert run_row is not None
        assert run_row[0] == "list:demo"
        assert run_row[1] == "success"
        assert run_row[2] == 10
        assert run_row[3] == 8
        assert run_row[4] == "2026-02-02T00:01:00+00:00"
    finally:
        conn.close()


def test_sqlite_store_load_new_since_filters_by_created_time(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "reader.db")
    try:
        source_id = "user:bob"
        store.save_items(
            source_id,
            (
                TweetItem(
                    tweet_id="10",
                    created_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
                    author_handle="@bob",
                    text="older",
                    source_id=source_id,
                ),
                TweetItem(
                    tweet_id="11",
                    created_at=datetime(2026, 2, 3, tzinfo=timezone.utc),
                    author_handle="@bob",
                    text="newer",
                    source_id=source_id,
                ),
                TweetItem(
                    tweet_id="12",
                    created_at=None,
                    author_handle="@bob",
                    text="no time",
                    source_id=source_id,
                ),
            ),
        )

        loaded = store.load_new_since(datetime(2026, 2, 2, tzinfo=timezone.utc))
        assert [item.tweet_id for item in loaded] == ["11"]
    finally:
        store.close()


def test_sqlite_store_save_items_rejects_source_mismatch(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "reader.db")
    try:
        with pytest.raises(StoreError, match="Tweet source mismatch in save_items"):
            store.save_items(
                "source:a",
                (
                    TweetItem(
                        tweet_id="1",
                        created_at=datetime(2026, 2, 3, tzinfo=timezone.utc),
                        author_handle="@alice",
                        text="bad batch",
                        source_id="source:b",
                    ),
                ),
            )
    finally:
        store.close()


def test_sqlite_store_load_checkpoint_fails_closed_on_invalid_timestamp(tmp_path: Path) -> None:
    db_path = tmp_path / "reader.db"
    store = SQLiteStore(db_path)
    try:
        checkpoint = Checkpoint(
            source_id="user:alice",
            last_seen_id="1",
            last_seen_time=datetime(2026, 2, 1, tzinfo=timezone.utc),
            updated_at=datetime(2026, 2, 1, 0, 1, tzinfo=timezone.utc),
        )
        store.save_checkpoint(checkpoint)
    finally:
        store.close()

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE checkpoints SET updated_at = ? WHERE source_id = ?",
            ("invalid-datetime", "user:alice"),
        )
        conn.commit()
    finally:
        conn.close()

    reopened = SQLiteStore(db_path)
    try:
        with pytest.raises(StoreError, match="invalid 'updated_at'"):
            reopened.load_checkpoint("user:alice")
    finally:
        reopened.close()
