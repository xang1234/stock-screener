"""
Idempotent migration: create Feature Store tables.

Creates 4 tables for the pre-computed daily screening snapshot store:
  - feature_runs: screening computation metadata
  - feature_run_universe_symbols: symbols in each run
  - stock_feature_daily: per-symbol scores
  - feature_run_pointers: atomic publish pointers

Safe to run on every startup — uses CREATE TABLE IF NOT EXISTS and
CREATE INDEX IF NOT EXISTS throughout.
"""
import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)


def migrate_feature_store_tables(engine) -> None:
    """
    Create Feature Store tables and indexes idempotently.

    Each table is created independently with IF NOT EXISTS, so partial
    previous runs or manual table drops are handled gracefully.

    Args:
        engine: SQLAlchemy engine instance
    """
    with engine.connect() as conn:
        existing = _get_existing_tables(conn)

        _create_feature_runs(conn, existing)
        _create_feature_run_universe_symbols(conn, existing)
        _create_stock_feature_daily(conn, existing)
        _create_feature_run_pointers(conn, existing)

        conn.commit()

    logger.info("Feature Store migration completed")


def _get_existing_tables(conn) -> set:
    """Return set of table names currently in the database."""
    result = conn.execute(text(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ))
    return {row[0] for row in result.fetchall()}


def _create_feature_runs(conn, existing: set) -> None:
    """Create the feature_runs table."""
    if "feature_runs" in existing:
        logger.info("Table feature_runs already exists, skipping")
        return

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS feature_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            as_of_date  DATE       NOT NULL,
            run_type    TEXT       NOT NULL,
            status      TEXT       NOT NULL,
            created_at  DATETIME   DEFAULT (CURRENT_TIMESTAMP),
            updated_at  DATETIME   DEFAULT (CURRENT_TIMESTAMP),
            completed_at DATETIME,
            published_at DATETIME,
            code_version TEXT,
            universe_hash TEXT,
            input_hash  TEXT,
            config_json TEXT,
            correlation_id TEXT,
            stats_json  TEXT,
            warnings_json TEXT
        )
    """))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_feature_runs_as_of_date "
        "ON feature_runs(as_of_date)"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_feature_runs_status "
        "ON feature_runs(status)"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_feature_runs_correlation_id "
        "ON feature_runs(correlation_id)"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_feature_runs_date_status "
        "ON feature_runs(as_of_date, status)"
    ))
    logger.info("Created table feature_runs")


def _create_feature_run_universe_symbols(conn, existing: set) -> None:
    """Create the feature_run_universe_symbols table."""
    if "feature_run_universe_symbols" in existing:
        logger.info("Table feature_run_universe_symbols already exists, skipping")
        return

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS feature_run_universe_symbols (
            run_id  INTEGER NOT NULL REFERENCES feature_runs(id) ON DELETE CASCADE,
            symbol  TEXT    NOT NULL,
            PRIMARY KEY (run_id, symbol)
        )
    """))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_frus_run_id "
        "ON feature_run_universe_symbols(run_id)"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_frus_symbol "
        "ON feature_run_universe_symbols(symbol)"
    ))
    logger.info("Created table feature_run_universe_symbols")


def _create_stock_feature_daily(conn, existing: set) -> None:
    """Create the stock_feature_daily table."""
    if "stock_feature_daily" in existing:
        logger.info("Table stock_feature_daily already exists, skipping")
        return

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS stock_feature_daily (
            run_id          INTEGER NOT NULL REFERENCES feature_runs(id) ON DELETE CASCADE,
            symbol          TEXT    NOT NULL,
            as_of_date      DATE    NOT NULL,
            composite_score REAL,
            overall_rating  INTEGER,
            passes_count    INTEGER,
            details_json    TEXT,
            PRIMARY KEY (run_id, symbol)
        )
    """))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_sfd_run_id "
        "ON stock_feature_daily(run_id)"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_sfd_symbol "
        "ON stock_feature_daily(symbol)"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_sfd_as_of_date "
        "ON stock_feature_daily(as_of_date)"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_sfd_composite_score "
        "ON stock_feature_daily(composite_score)"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_sfd_overall_rating "
        "ON stock_feature_daily(overall_rating)"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_stock_feature_daily_run_score "
        "ON stock_feature_daily(run_id, composite_score)"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_stock_feature_daily_run_rating "
        "ON stock_feature_daily(run_id, overall_rating)"
    ))
    logger.info("Created table stock_feature_daily")


def _create_feature_run_pointers(conn, existing: set) -> None:
    """Create the feature_run_pointers table."""
    if "feature_run_pointers" in existing:
        logger.info("Table feature_run_pointers already exists, skipping")
        return

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS feature_run_pointers (
            key        TEXT    PRIMARY KEY,
            run_id     INTEGER NOT NULL REFERENCES feature_runs(id) ON DELETE CASCADE,
            updated_at DATETIME DEFAULT (CURRENT_TIMESTAMP)
        )
    """))
    logger.info("Created table feature_run_pointers")
