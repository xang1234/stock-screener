"""Expand symbol-related column widths for ASIA multi-market support.

This migration widens core symbol-bearing columns from VARCHAR(10) to VARCHAR(20)
so suffixed and exchange-local identifiers (e.g. 0700.HK, 2330.TW, 3008.TWO)
fit without truncation.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260411_0003"
down_revision = "20260410_0002"
branch_labels = None
depends_on = None

OLD_LEN = 10
NEW_LEN = 20


def _lock_tables_for_downgrade() -> None:
    """Acquire write-blocking locks before downgrade preflight checks."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    tables = [
        "watchlist",
        "provider_snapshot_rows",
        "theme_constituents",
        "ibd_group_ranks",
        "ibd_group_peer_cache",
        "ibd_industry_groups",
        "stock_universe_status_events",
        "stock_universe",
        "scan_results",
        "stock_industry",
        "stock_technicals",
        "stock_fundamentals",
        "stock_prices",
    ]
    for table_name in tables:
        bind.execute(sa.text(f"LOCK TABLE {table_name} IN ACCESS EXCLUSIVE MODE"))


def _alter_length(table_name: str, column_name: str, from_len: int, to_len: int) -> None:
    """Alter VARCHAR length with batch mode for cross-dialect safety."""
    with op.batch_alter_table(table_name) as batch_op:
        batch_op.alter_column(
            column_name,
            existing_type=sa.String(length=from_len),
            type_=sa.String(length=to_len),
        )


def _assert_max_length(table_name: str, column_name: str, max_len: int) -> None:
    """
    Ensure downgrade safety before shrinking column width.

    We fail fast with a deterministic error instead of partially applying
    a downgrade that can break once long symbols exist in production data.
    """
    bind = op.get_bind()
    stmt = sa.text(
        f"SELECT 1 FROM {table_name} "
        f"WHERE {column_name} IS NOT NULL AND LENGTH({column_name}) > :max_len "
        "LIMIT 1"
    )
    row = bind.execute(stmt, {"max_len": max_len}).first()
    if row is not None:
        raise RuntimeError(
            f"Cannot shrink {table_name}.{column_name} to {max_len}: "
            "found existing values exceeding target length."
        )


def upgrade() -> None:
    # Core stock/fundamental/scan persistence paths
    _alter_length("stock_prices", "symbol", OLD_LEN, NEW_LEN)
    _alter_length("stock_fundamentals", "symbol", OLD_LEN, NEW_LEN)
    _alter_length("stock_technicals", "symbol", OLD_LEN, NEW_LEN)
    _alter_length("stock_industry", "symbol", OLD_LEN, NEW_LEN)
    _alter_length("scan_results", "symbol", OLD_LEN, NEW_LEN)

    # Universe + lifecycle audit paths
    _alter_length("stock_universe", "symbol", OLD_LEN, NEW_LEN)
    _alter_length("stock_universe_status_events", "symbol", OLD_LEN, NEW_LEN)

    # Group ranking / constituent symbol-bearing paths
    _alter_length("ibd_industry_groups", "symbol", OLD_LEN, NEW_LEN)
    _alter_length("ibd_group_peer_cache", "top_symbol", OLD_LEN, NEW_LEN)
    _alter_length("ibd_group_ranks", "top_symbol", OLD_LEN, NEW_LEN)
    _alter_length("theme_constituents", "symbol", OLD_LEN, NEW_LEN)

    # Provider snapshot + legacy watchlist table
    _alter_length("provider_snapshot_rows", "symbol", OLD_LEN, NEW_LEN)
    _alter_length("watchlist", "symbol", OLD_LEN, NEW_LEN)


def downgrade() -> None:
    # Prevent concurrent writes between preflight and shrink operations.
    _lock_tables_for_downgrade()

    # Preflight safety checks before any shrinking operation.
    _assert_max_length("watchlist", "symbol", OLD_LEN)
    _assert_max_length("provider_snapshot_rows", "symbol", OLD_LEN)
    _assert_max_length("theme_constituents", "symbol", OLD_LEN)
    _assert_max_length("ibd_group_ranks", "top_symbol", OLD_LEN)
    _assert_max_length("ibd_group_peer_cache", "top_symbol", OLD_LEN)
    _assert_max_length("ibd_industry_groups", "symbol", OLD_LEN)
    _assert_max_length("stock_universe_status_events", "symbol", OLD_LEN)
    _assert_max_length("stock_universe", "symbol", OLD_LEN)
    _assert_max_length("scan_results", "symbol", OLD_LEN)
    _assert_max_length("stock_industry", "symbol", OLD_LEN)
    _assert_max_length("stock_technicals", "symbol", OLD_LEN)
    _assert_max_length("stock_fundamentals", "symbol", OLD_LEN)
    _assert_max_length("stock_prices", "symbol", OLD_LEN)

    # Revert in reverse conceptual order.
    _alter_length("watchlist", "symbol", NEW_LEN, OLD_LEN)
    _alter_length("provider_snapshot_rows", "symbol", NEW_LEN, OLD_LEN)

    _alter_length("theme_constituents", "symbol", NEW_LEN, OLD_LEN)
    _alter_length("ibd_group_ranks", "top_symbol", NEW_LEN, OLD_LEN)
    _alter_length("ibd_group_peer_cache", "top_symbol", NEW_LEN, OLD_LEN)
    _alter_length("ibd_industry_groups", "symbol", NEW_LEN, OLD_LEN)

    _alter_length("stock_universe_status_events", "symbol", NEW_LEN, OLD_LEN)
    _alter_length("stock_universe", "symbol", NEW_LEN, OLD_LEN)

    _alter_length("scan_results", "symbol", NEW_LEN, OLD_LEN)
    _alter_length("stock_industry", "symbol", NEW_LEN, OLD_LEN)
    _alter_length("stock_technicals", "symbol", NEW_LEN, OLD_LEN)
    _alter_length("stock_fundamentals", "symbol", NEW_LEN, OLD_LEN)
    _alter_length("stock_prices", "symbol", NEW_LEN, OLD_LEN)
