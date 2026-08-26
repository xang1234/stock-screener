"""Add trading_date to max_pain_snapshots and gex_snapshots.

fetched_at is stored as naive UTC (datetime.utcnow()) and the daily batches
run at 4:30/4:40pm ET, after close. Bucketing history queries by
date_trunc() on that raw UTC timestamp is fragile around DST and the
UTC-midnight boundary -- a run that lands after 8pm ET can drift onto the
next UTC calendar date. trading_date is a plain (non-timezone) Date holding
the correct US/Eastern trading day, safe to group/filter on directly for
the weekly/monthly/quarterly/yearly history endpoints.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260805_0028"
down_revision = "20260730_0027"
branch_labels = None
depends_on = None

_TABLES = ("max_pain_snapshots", "gex_snapshots")


def upgrade() -> None:
    for table in _TABLES:
        op.add_column(table, sa.Column("trading_date", sa.Date(), nullable=True))

        # Backfill from fetched_at, converting UTC -> US/Eastern before taking
        # the date so DST transitions are handled by Postgres's tz database
        # rather than a fixed offset.
        op.execute(
            f"""
            UPDATE {table}
            SET trading_date = (fetched_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York')::date
            WHERE trading_date IS NULL
            """
        )

        op.create_index(
            f"ix_{table}_ticker_trading_date",
            table,
            ["ticker", "trading_date"],
            unique=False,
        )


def downgrade() -> None:
    for table in _TABLES:
        op.drop_index(f"ix_{table}_ticker_trading_date", table)
        op.drop_column(table, "trading_date")
