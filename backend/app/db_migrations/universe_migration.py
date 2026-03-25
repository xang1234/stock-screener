"""
Idempotent migration: add structured universe columns to scans table and backfill.

Replaces the destructive cleanup_invalid_universe_scans() that deleted
scans with universe values like 'nyse', 'nasdaq', 'sp500'. Instead, this
migration preserves those scans by backfilling them with proper typed fields.

Safe to run on every startup — detects existing columns and skips if present.
"""
import hashlib
import json
import logging
from sqlalchemy import text

from ..infra.db.portability import column_names

logger = logging.getLogger(__name__)

# Columns to add (name, type_spec)
NEW_COLUMNS = [
    ("universe_key", "VARCHAR(128)"),
    ("universe_type", "VARCHAR(20)"),
    ("universe_exchange", "VARCHAR(20)"),
    ("universe_index", "VARCHAR(20)"),
    ("universe_symbols", "TEXT"),  # JSON stored as text in SQLite
]

# Mapping from legacy universe string to structured fields
LEGACY_MAP = {
    "all":    {"type": "all",      "key": "all"},
    "nyse":   {"type": "exchange", "exchange": "NYSE",   "key": "exchange:NYSE"},
    "nasdaq": {"type": "exchange", "exchange": "NASDAQ", "key": "exchange:NASDAQ"},
    "amex":   {"type": "exchange", "exchange": "AMEX",   "key": "exchange:AMEX"},
    "sp500":  {"type": "index",    "index": "SP500",     "key": "index:SP500"},
}


def migrate_scan_universe_schema_and_backfill(engine) -> None:
    """
    Add structured universe columns to scans table and backfill existing rows.

    This migration is idempotent:
    1. Schema patch: adds missing columns (skips if already present)
    2. Backfill: updates rows where universe_key IS NULL

    Args:
        engine: SQLAlchemy engine instance
    """
    import time
    start = time.time()

    with engine.connect() as conn:
        # Step 1: Detect existing columns
        existing_columns = _get_existing_columns(conn)
        columns_added = _add_missing_columns(conn, existing_columns)

        if columns_added:
            conn.commit()
            logger.info(f"Added {columns_added} new columns to scans table")

        # Step 2: Backfill rows where universe_key IS NULL
        backfilled = _backfill_universe_fields(conn)
        if backfilled > 0:
            conn.commit()
            logger.info(f"Backfilled {backfilled} scan rows with structured universe fields")

    elapsed = time.time() - start
    logger.info(f"Universe migration completed in {elapsed:.2f}s")


def _get_existing_columns(conn) -> set:
    """Get set of column names currently on the scans table."""
    return column_names(conn, "scans")


def _add_missing_columns(conn, existing_columns: set) -> int:
    """Add any missing structured universe columns. Returns count of columns added."""
    added = 0
    for col_name, col_type in NEW_COLUMNS:
        if col_name not in existing_columns:
            conn.execute(text(f"ALTER TABLE scans ADD COLUMN {col_name} {col_type}"))
            logger.info(f"Added column scans.{col_name} ({col_type})")
            added += 1

    # Add indexes for new columns (CREATE INDEX IF NOT EXISTS is safe)
    if added > 0:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_scans_universe_key ON scans(universe_key)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_scans_universe_type ON scans(universe_type)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_scans_universe_exchange ON scans(universe_exchange)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_scans_universe_index ON scans(universe_index)"
        ))

    return added


def _backfill_universe_fields(conn) -> int:
    """
    Backfill structured universe fields for rows where universe_key IS NULL.

    Returns count of rows updated.
    """
    # Find rows needing backfill
    result = conn.execute(text(
        "SELECT id, scan_id, universe FROM scans WHERE universe_key IS NULL"
    ))
    rows = result.fetchall()

    if not rows:
        return 0

    total = 0

    for row_id, scan_id, universe in rows:
        universe_lower = (universe or "").strip().lower()
        mapped = LEGACY_MAP.get(universe_lower)

        if mapped:
            # Known legacy value — direct mapping
            conn.execute(text(
                "UPDATE scans SET universe_key = :key, universe_type = :type, "
                "universe_exchange = :exchange, universe_index = :index "
                "WHERE id = :id"
            ), {
                "key": mapped["key"],
                "type": mapped["type"],
                "exchange": mapped.get("exchange"),
                "index": mapped.get("index"),
                "id": row_id,
            })
            total += 1

        elif universe_lower in ("custom", "test"):
            # Derive symbols from scan_results join
            sym_result = conn.execute(text(
                "SELECT symbol FROM scan_results WHERE scan_id = :scan_id ORDER BY symbol"
            ), {"scan_id": scan_id})
            symbols = sorted(r[0] for r in sym_result.fetchall())

            if symbols:
                joined = ",".join(symbols)
                digest = hashlib.sha256(joined.encode()).hexdigest()[:12]
                key = f"{universe_lower}:{digest}"
            else:
                key = f"{universe_lower}:empty"

            conn.execute(text(
                "UPDATE scans SET universe_key = :key, universe_type = :type, "
                "universe_symbols = :symbols WHERE id = :id"
            ), {
                "key": key,
                "type": universe_lower,
                "symbols": json.dumps(symbols) if symbols else None,
                "id": row_id,
            })
            total += 1

        else:
            # Unknown value — safe fallback to preserve data
            key = f"legacy:{universe_lower}" if universe_lower else "legacy:unknown"
            conn.execute(text(
                "UPDATE scans SET universe_key = :key, universe_type = 'custom' "
                "WHERE id = :id"
            ), {"key": key, "id": row_id})
            total += 1
            logger.warning(
                f"Scan {scan_id} has unknown universe '{universe}', "
                f"backfilled as legacy:{universe_lower}"
            )

    logger.info(f"Backfilled {total} scan rows")
    return total
