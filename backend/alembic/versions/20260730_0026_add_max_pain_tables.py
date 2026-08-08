"""Add max pain tables for options analysis."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260730_0026"
down_revision = "20260718_0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create max_pain_batches table
    op.create_table(
        "max_pain_batches",
        sa.Column("id", sa.String(50), nullable=False, primary_key=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tickers_ok", sa.Integer(), nullable=True),
        sa.Column("tickers_failed", sa.Integer(), nullable=True),
        sa.Column("avg_put_call_ratio", sa.Float(), nullable=True),
        sa.Column("closest_to_max_pain", sa.String(10), nullable=True),
        sa.Column("strike_range_pct", sa.Float(), nullable=True),
        sa.Column("max_strikes", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Create max_pain_snapshots table
    op.create_table(
        "max_pain_snapshots",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.String(10), nullable=False),
        sa.Column("company_name", sa.String(200), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("max_pain_strike", sa.Float(), nullable=True),
        sa.Column("expiration", sa.Date(), nullable=True),
        sa.Column("call_oi", sa.Integer(), nullable=True),
        sa.Column("put_oi", sa.Integer(), nullable=True),
        sa.Column("put_call_ratio", sa.Float(), nullable=True),
        sa.Column("last_price", sa.Float(), nullable=True),
        sa.Column("distance_pct", sa.Float(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("batch_id", sa.String(50), sa.ForeignKey("max_pain_batches.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Create indexes for efficient queries
    op.create_index(
        "ix_max_pain_ticker_fetched",
        "max_pain_snapshots",
        ["ticker", "fetched_at"],
        unique=False,
    )

    op.create_index(
        "ix_max_pain_batch_fetched",
        "max_pain_snapshots",
        ["batch_id", "fetched_at"],
        unique=False,
    )

    op.create_index(
        "ix_max_pain_batch_status",
        "max_pain_batches",
        ["status"],
        unique=False,
    )

    op.create_index(
        "ix_max_pain_batch_completed",
        "max_pain_batches",
        ["completed_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_max_pain_batch_completed", "max_pain_batches")
    op.drop_index("ix_max_pain_batch_status", "max_pain_batches")
    op.drop_index("ix_max_pain_batch_fetched", "max_pain_snapshots")
    op.drop_index("ix_max_pain_ticker_fetched", "max_pain_snapshots")
    op.drop_table("max_pain_snapshots")
    op.drop_table("max_pain_batches")
