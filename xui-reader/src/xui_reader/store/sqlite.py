"""SQLite-backed store with deterministic migration bootstrap."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from xui_reader.errors import StoreError
from xui_reader.models import Checkpoint, SourceRef, TweetItem

_SOURCE_KIND_UNKNOWN = "unknown"


@dataclass(frozen=True)
class Migration:
    version: str
    statements: tuple[str, ...]


DEFAULT_MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version="0001_initial_store_schema",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                value TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                source_id TEXT PRIMARY KEY,
                last_seen_id TEXT,
                last_seen_time TEXT,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS tweets (
                source_id TEXT NOT NULL,
                tweet_id TEXT NOT NULL,
                created_at TEXT,
                author_handle TEXT,
                text TEXT,
                is_reply INTEGER NOT NULL DEFAULT 0 CHECK (is_reply IN (0, 1)),
                is_repost INTEGER NOT NULL DEFAULT 0 CHECK (is_repost IN (0, 1)),
                is_pinned INTEGER NOT NULL DEFAULT 0 CHECK (is_pinned IN (0, 1)),
                has_quote INTEGER NOT NULL DEFAULT 0 CHECK (has_quote IN (0, 1)),
                quote_tweet_id TEXT,
                inserted_at TEXT NOT NULL,
                PRIMARY KEY (source_id, tweet_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                observed_count INTEGER NOT NULL DEFAULT 0,
                saved_count INTEGER NOT NULL DEFAULT 0,
                error TEXT
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_tweets_created_at ON tweets(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_tweets_source_created ON tweets(source_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_runs_source_started ON runs(source_id, started_at)",
        ),
    ),
)


class SQLiteMigrationRunner:
    """Apply ordered migrations and enforce base pragmas."""

    def __init__(self, migrations: Sequence[Migration] | None = None) -> None:
        self._migrations = tuple(migrations or DEFAULT_MIGRATIONS)
        versions = [migration.version for migration in self._migrations]
        if versions != sorted(versions):
            raise StoreError("Migrations must be in ascending version order.")
        if len(set(versions)) != len(versions):
            raise StoreError("Migration versions must be unique.")

    def bootstrap(self, conn: sqlite3.Connection) -> None:
        self._apply_pragmas(conn)
        self._ensure_migration_table(conn)
        self._apply_pending_migrations(conn)

    def _apply_pragmas(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")

    def _ensure_migration_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """
        )
        conn.commit()

    def applied_versions(self, conn: sqlite3.Connection) -> tuple[str, ...]:
        rows = conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
        return tuple(str(row[0]) for row in rows)

    def _apply_pending_migrations(self, conn: sqlite3.Connection) -> None:
        applied = set(self.applied_versions(conn))
        for migration in self._migrations:
            if migration.version in applied:
                continue
            try:
                conn.execute("BEGIN")
                for statement in migration.statements:
                    conn.execute(statement)
                conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES(?, ?)",
                    (migration.version, _dt_to_db(_utcnow())),
                )
                conn.commit()
            except sqlite3.Error as exc:
                conn.rollback()
                raise StoreError(
                    f"Failed applying migration '{migration.version}': {exc}."
                ) from exc


class SQLiteStore:
    """SQLite implementation of source/checkpoint/item/run persistence."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        migration_runner: SQLiteMigrationRunner | None = None,
    ) -> None:
        self._db_path = Path(db_path).expanduser()
        self._migration_runner = migration_runner or SQLiteMigrationRunner()

        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
        except sqlite3.Error as exc:
            raise StoreError(f"Could not open SQLite database '{self._db_path}': {exc}.") from exc
        except OSError as exc:
            raise StoreError(f"Could not prepare database path '{self._db_path}': {exc}.") from exc

        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._migration_runner.bootstrap(self._conn)

    @property
    def db_path(self) -> Path:
        return self._db_path

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error as exc:
            raise StoreError(f"Could not close SQLite database '{self._db_path}': {exc}.") from exc

    def upsert_source(self, source: SourceRef) -> None:
        now = _dt_to_db(_utcnow())
        try:
            self._conn.execute(
                """
                INSERT INTO sources(source_id, kind, value, enabled, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    kind = excluded.kind,
                    value = excluded.value,
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at
                """,
                (
                    source.source_id,
                    source.kind.value,
                    source.value,
                    int(source.enabled),
                    now,
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(f"Could not upsert source '{source.source_id}': {exc}.") from exc

    def save_items(self, source_id: str, items: tuple[TweetItem, ...]) -> int:
        if not items:
            return 0
        self._ensure_source_placeholder(source_id)

        rows = [
            (
                source_id,
                item.tweet_id,
                _dt_to_db(item.created_at),
                item.author_handle,
                item.text,
                int(bool(item.is_reply)),
                int(bool(item.is_repost)),
                int(bool(item.is_pinned)),
                int(bool(item.has_quote)),
                item.quote_tweet_id,
                _dt_to_db(_utcnow()),
            )
            for item in items
        ]
        before = self._conn.total_changes
        try:
            self._conn.executemany(
                """
                INSERT OR IGNORE INTO tweets(
                    source_id,
                    tweet_id,
                    created_at,
                    author_handle,
                    text,
                    is_reply,
                    is_repost,
                    is_pinned,
                    has_quote,
                    quote_tweet_id,
                    inserted_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(f"Could not persist tweet items for '{source_id}': {exc}.") from exc
        return self._conn.total_changes - before

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self._ensure_source_placeholder(checkpoint.source_id)
        try:
            self._conn.execute(
                """
                INSERT INTO checkpoints(source_id, last_seen_id, last_seen_time, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    last_seen_id = excluded.last_seen_id,
                    last_seen_time = excluded.last_seen_time,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.source_id,
                    checkpoint.last_seen_id,
                    _dt_to_db(checkpoint.last_seen_time),
                    _dt_to_db(checkpoint.updated_at),
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(
                f"Could not save checkpoint for source '{checkpoint.source_id}': {exc}."
            ) from exc

    def load_checkpoint(self, source_id: str) -> Checkpoint | None:
        try:
            row = self._conn.execute(
                """
                SELECT source_id, last_seen_id, last_seen_time, updated_at
                FROM checkpoints
                WHERE source_id = ?
                """,
                (source_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StoreError(f"Could not load checkpoint for source '{source_id}': {exc}.") from exc

        if row is None:
            return None
        return Checkpoint(
            source_id=str(row["source_id"]),
            last_seen_id=str(row["last_seen_id"]) if row["last_seen_id"] is not None else None,
            last_seen_time=_db_to_dt(row["last_seen_time"]),
            updated_at=_db_to_dt(row["updated_at"]) or _utcnow(),
        )

    def load_new_since(self, since: datetime) -> tuple[TweetItem, ...]:
        since_value = _dt_to_db(_normalize_datetime(since))
        try:
            rows = self._conn.execute(
                """
                SELECT tweet_id, source_id, created_at, author_handle, text,
                       is_reply, is_repost, is_pinned, has_quote, quote_tweet_id
                FROM tweets
                WHERE created_at IS NOT NULL AND created_at > ?
                ORDER BY created_at DESC, tweet_id DESC
                """,
                (since_value,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise StoreError(f"Could not load new items since '{since_value}': {exc}.") from exc

        return tuple(
            TweetItem(
                tweet_id=str(row["tweet_id"]),
                created_at=_db_to_dt(row["created_at"]),
                author_handle=str(row["author_handle"]) if row["author_handle"] is not None else None,
                text=str(row["text"]) if row["text"] is not None else None,
                source_id=str(row["source_id"]),
                is_reply=bool(row["is_reply"]),
                is_repost=bool(row["is_repost"]),
                is_pinned=bool(row["is_pinned"]),
                has_quote=bool(row["has_quote"]),
                quote_tweet_id=str(row["quote_tweet_id"]) if row["quote_tweet_id"] is not None else None,
            )
            for row in rows
        )

    def begin_run(self, source_id: str, started_at: datetime | None = None) -> int:
        self._ensure_source_placeholder(source_id)
        started = _dt_to_db(_normalize_datetime(started_at or _utcnow()))
        try:
            cursor = self._conn.execute(
                """
                INSERT INTO runs(source_id, started_at, status)
                VALUES(?, ?, 'running')
                """,
                (source_id, started),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(f"Could not begin run for source '{source_id}': {exc}.") from exc
        run_id = cursor.lastrowid
        if run_id is None:
            raise StoreError("SQLite did not return run id for inserted run.")
        return int(run_id)

    def finish_run(
        self,
        run_id: int,
        *,
        status: str,
        observed_count: int = 0,
        saved_count: int = 0,
        error: str | None = None,
        finished_at: datetime | None = None,
    ) -> None:
        finished = _dt_to_db(_normalize_datetime(finished_at or _utcnow()))
        try:
            cursor = self._conn.execute(
                """
                UPDATE runs
                SET finished_at = ?,
                    status = ?,
                    observed_count = ?,
                    saved_count = ?,
                    error = ?
                WHERE run_id = ?
                """,
                (finished, status, observed_count, saved_count, error, int(run_id)),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(f"Could not finish run '{run_id}': {exc}.") from exc
        if cursor.rowcount == 0:
            raise StoreError(f"Run id '{run_id}' was not found.")

    def migration_versions(self) -> tuple[str, ...]:
        return self._migration_runner.applied_versions(self._conn)

    def _ensure_source_placeholder(self, source_id: str) -> None:
        try:
            self._conn.execute(
                """
                INSERT INTO sources(source_id, kind, value, enabled, updated_at)
                VALUES(?, ?, ?, 1, ?)
                ON CONFLICT(source_id) DO NOTHING
                """,
                (source_id, _SOURCE_KIND_UNKNOWN, source_id, _dt_to_db(_utcnow())),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(
                f"Could not ensure source placeholder for '{source_id}': {exc}."
            ) from exc


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _dt_to_db(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _normalize_datetime(value).isoformat()


def _db_to_dt(raw: object) -> datetime | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return _normalize_datetime(parsed)

