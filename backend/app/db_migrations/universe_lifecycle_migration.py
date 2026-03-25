"""Idempotent migration for stock universe lifecycle fields and audit tables."""

from __future__ import annotations

from contextlib import nullcontext
import logging
from datetime import datetime

from sqlalchemy import text

from ..database import is_corruption_error
from ..infra.db.portability import column_names, index_defs, is_sqlite, sql_timestamp_type, table_names
from ..models.stock_universe import StockUniverseStatusEvent

logger = logging.getLogger(__name__)

_ROW_UPDATE_BATCH_SIZE = 250


def _get_columns(conn, table_name: str) -> set[str]:
    return column_names(conn, table_name)


def _get_index_columns(conn, table_name: str) -> dict[str, tuple[str, ...]]:
    """Return existing index column tuples keyed by index name."""
    return {
        index["name"]: tuple(index["columns"])
        for index in index_defs(conn, table_name)
        if index.get("name")
    }


def _normalize_text(value) -> str | None:
    """Normalize legacy text values without using SQLite string functions."""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _build_select_expr(columns: set[str], column_name: str) -> str:
    if column_name in columns:
        return column_name
    return f"NULL AS {column_name}"


def _legacy_is_active_value(value) -> int:
    return 1 if bool(value) else 0


def _nested_savepoint(conn):
    if hasattr(conn, "begin_nested"):
        return conn.begin_nested()
    return nullcontext()


def _ensure_index(
    conn,
    *,
    table_name: str,
    index_name: str,
    columns: tuple[str, ...],
    ddl: str,
) -> None:
    """Create an index only when no equivalent index already exists."""
    try:
        existing_indexes = _get_index_columns(conn, table_name)
    except Exception as exc:
        if not is_corruption_error(exc):
            raise
        logger.warning(
            "Universe lifecycle migration skipped optional index %s on %s while "
            "inspecting existing indexes due to SQLite corruption signature: %s",
            index_name,
            table_name,
            exc,
        )
        return

    if index_name in existing_indexes:
        return
    matching_name = next(
        (
            name
            for name, existing_columns in existing_indexes.items()
            if existing_columns == columns
        ),
        None,
    )
    if matching_name is not None:
        logger.info(
            "Universe lifecycle migration reusing existing index %s on %s for columns %s",
            matching_name,
            table_name,
            columns,
        )
        return

    try:
        with _nested_savepoint(conn):
            conn.execute(text(ddl))
    except Exception as exc:
        if not is_corruption_error(exc):
            raise
        logger.warning(
            "Universe lifecycle migration skipped optional index %s on %s due to SQLite "
            "corruption signature: %s",
            index_name,
            table_name,
            exc,
        )


def _derive_lifecycle_status(
    raw_status: str | None,
    status_reason: str | None,
    legacy_is_active: int,
) -> str:
    """Derive lifecycle status, handling partially migrated rows safely."""
    if raw_status is None:
        return "active" if legacy_is_active else "inactive_manual"

    # If a prior startup added the column but crashed before row backfill,
    # SQLite surfaces the default 'active' for existing rows. When the row
    # still lacks any backfill reason, fall back to the legacy active flag.
    if raw_status == "active" and not legacy_is_active and status_reason is None:
        return "inactive_manual"

    return raw_status


def _backfill_stock_universe_rows(conn, columns: set[str]) -> None:
    """Backfill lifecycle data row-by-row to avoid brittle full-table text scans."""
    full_metadata_backfill = "status" not in columns
    select_columns = [
        "id",
        "COALESCE(is_active, 1) AS legacy_is_active",
        _build_select_expr(columns, "status"),
        _build_select_expr(columns, "status_reason"),
        _build_select_expr(columns, "first_seen_at"),
        _build_select_expr(columns, "last_seen_in_source_at"),
        _build_select_expr(columns, "deactivated_at"),
        _build_select_expr(columns, "added_at"),
        _build_select_expr(columns, "updated_at"),
    ]
    rows = conn.execute(
        text(f"SELECT {', '.join(select_columns)} FROM stock_universe")
    ).mappings().all()

    now = datetime.utcnow()
    updates: list[dict[str, object]] = []
    for row in rows:
        raw_status = _normalize_text(row["status"])
        status_reason = _normalize_text(row["status_reason"])
        status = _derive_lifecycle_status(
            raw_status,
            status_reason,
            _legacy_is_active_value(row["legacy_is_active"]),
        )
        if (
            not full_metadata_backfill
            and raw_status == "active"
            and not _legacy_is_active_value(row["legacy_is_active"])
            and status_reason is None
        ):
            continue
        if status_reason is None:
            status_reason = (
                "Existing active symbol"
                if status == "active"
                else "Backfilled from legacy inactive flag"
            )

        first_seen_at = row["first_seen_at"] or row["added_at"] or row["updated_at"] or now
        last_seen_in_source_at = row["last_seen_in_source_at"]
        if status == "active" and last_seen_in_source_at is None:
            last_seen_in_source_at = row["updated_at"] or row["added_at"] or now

        deactivated_at = row["deactivated_at"]
        if status != "active" and deactivated_at is None:
            deactivated_at = row["updated_at"] or now
        elif status == "active":
            deactivated_at = None

        target_is_active = 1 if status == "active" else 0
        core_fields_need_repair = (
            raw_status != status
            or _legacy_is_active_value(row["legacy_is_active"]) != target_is_active
            or (status != "active" and row["deactivated_at"] != deactivated_at)
        )
        metadata_fields_need_repair = (
            row["status_reason"] != status_reason
            or row["first_seen_at"] != first_seen_at
            or row["last_seen_in_source_at"] != last_seen_in_source_at
            or row["deactivated_at"] != deactivated_at
        )

        if not core_fields_need_repair and not (
            full_metadata_backfill and metadata_fields_need_repair
        ):
            continue

        updates.append(
            {
                "id": row["id"],
                "status": status,
                "status_reason": status_reason,
                "first_seen_at": first_seen_at,
                "last_seen_in_source_at": last_seen_in_source_at,
                "deactivated_at": deactivated_at,
                "is_active": target_is_active,
            }
        )

    if not updates:
        return

    update_stmt = text(
        """
        UPDATE stock_universe
        SET status = :status,
            status_reason = :status_reason,
            first_seen_at = :first_seen_at,
            last_seen_in_source_at = :last_seen_in_source_at,
            deactivated_at = :deactivated_at,
            is_active = :is_active
        WHERE id = :id
        """
    )

    skipped_ids: list[int] = []
    for start in range(0, len(updates), _ROW_UPDATE_BATCH_SIZE):
        batch = updates[start:start + _ROW_UPDATE_BATCH_SIZE]
        try:
            with _nested_savepoint(conn):
                conn.execute(update_stmt, batch)
            continue
        except Exception as exc:
            if not is_corruption_error(exc):
                raise
            logger.warning(
                "Universe lifecycle batch backfill hit SQLite corruption signature; "
                "retrying row-by-row for ids %s-%s: %s",
                batch[0]["id"],
                batch[-1]["id"],
                exc,
            )

        for params in batch:
            try:
                with _nested_savepoint(conn):
                    conn.execute(update_stmt, params)
            except Exception as row_exc:
                if not is_corruption_error(row_exc):
                    raise
                skipped_ids.append(params["id"])

    if skipped_ids:
        logger.warning(
            "Universe lifecycle migration skipped %d stock_universe rows during backfill "
            "because SQLite reported corruption signatures. Sample ids: %s",
            len(skipped_ids),
            skipped_ids[:20],
        )


def migrate_universe_lifecycle(engine) -> None:
    """Add lifecycle columns to stock_universe and create audit tables."""
    with engine.connect() as conn:
        existing_tables = table_names(conn)

        if "stock_universe" not in existing_tables:
            logger.info("stock_universe table not present yet; lifecycle migration skipped")
            return

        legacy_columns = _get_columns(conn, "stock_universe")

        timestamp_type = sql_timestamp_type(conn)
        add_columns = {
            "status": "ALTER TABLE stock_universe ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
            "status_reason": "ALTER TABLE stock_universe ADD COLUMN status_reason TEXT",
            "first_seen_at": f"ALTER TABLE stock_universe ADD COLUMN first_seen_at {timestamp_type}",
            "last_seen_in_source_at": f"ALTER TABLE stock_universe ADD COLUMN last_seen_in_source_at {timestamp_type}",
            "deactivated_at": f"ALTER TABLE stock_universe ADD COLUMN deactivated_at {timestamp_type}",
            "consecutive_fetch_failures": (
                "ALTER TABLE stock_universe ADD COLUMN consecutive_fetch_failures INTEGER NOT NULL DEFAULT 0"
            ),
            "last_fetch_success_at": f"ALTER TABLE stock_universe ADD COLUMN last_fetch_success_at {timestamp_type}",
            "last_fetch_failure_at": f"ALTER TABLE stock_universe ADD COLUMN last_fetch_failure_at {timestamp_type}",
        }

        for name, ddl in add_columns.items():
            if name not in legacy_columns:
                conn.execute(text(ddl))

        _backfill_stock_universe_rows(conn, legacy_columns)

        if is_sqlite(conn):
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS stock_universe_status_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        old_status TEXT,
                        new_status TEXT NOT NULL,
                        trigger_source TEXT NOT NULL,
                        reason TEXT,
                        payload_json TEXT,
                        created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP)
                    )
                    """
                )
            )
        else:
            StockUniverseStatusEvent.__table__.create(bind=conn, checkfirst=True)
        _ensure_index(
            conn,
            table_name="stock_universe_status_events",
            index_name="idx_stock_universe_status_events_symbol_created",
            columns=("symbol", "created_at"),
            ddl="""
                CREATE INDEX IF NOT EXISTS idx_stock_universe_status_events_symbol_created
                ON stock_universe_status_events(symbol, created_at)
            """,
        )
        _ensure_index(
            conn,
            table_name="stock_universe_status_events",
            index_name="idx_stock_universe_status_events_status_created",
            columns=("new_status", "created_at"),
            ddl="""
                CREATE INDEX IF NOT EXISTS idx_stock_universe_status_events_status_created
                ON stock_universe_status_events(new_status, created_at)
            """,
        )
        _ensure_index(
            conn,
            table_name="stock_universe",
            index_name="idx_stock_universe_exchange_status",
            columns=("exchange", "status"),
            ddl="""
                CREATE INDEX IF NOT EXISTS idx_stock_universe_exchange_status
                ON stock_universe(exchange, status)
            """,
        )
        _ensure_index(
            conn,
            table_name="stock_universe",
            index_name="idx_stock_universe_status_active",
            columns=("status", "is_active"),
            ddl="""
                CREATE INDEX IF NOT EXISTS idx_stock_universe_status_active
                ON stock_universe(status, is_active)
            """,
        )
        conn.commit()

    logger.info("Universe lifecycle migration completed")
