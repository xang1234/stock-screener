"""Add GEX tables for gamma exposure analysis."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260730_0027"
down_revision = "20260730_0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "gex_batches",
        sa.Column("id", sa.String(50), nullable=False, primary_key=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tickers_total", sa.Integer(), nullable=True),
        sa.Column("tickers_ok", sa.Integer(), nullable=True),
        sa.Column("tickers_failed", sa.Integer(), nullable=True),
        sa.Column("avg_total_gex", sa.Float(), nullable=True),
        sa.Column("closest_to_flip", sa.String(20), nullable=True),
        sa.Column("strike_range_pct", sa.Float(), nullable=True),
        sa.Column("max_strikes", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "gex_snapshots",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("company_name", sa.String(255), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("expiration", sa.Date(), nullable=True),
        sa.Column("spot_price", sa.Float(), nullable=True),
        sa.Column("call_oi", sa.Integer(), nullable=True),
        sa.Column("put_oi", sa.Integer(), nullable=True),
        sa.Column("call_gex", sa.Float(), nullable=True),
        sa.Column("put_gex", sa.Float(), nullable=True),
        sa.Column("total_gex", sa.Float(), nullable=True),
        sa.Column("flip_level", sa.Float(), nullable=True),
        sa.Column("distance_to_flip_pct", sa.Float(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "batch_id",
            sa.String(50),
            sa.ForeignKey("gex_batches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_index("ix_gex_ticker_fetched", "gex_snapshots", ["ticker", "fetched_at"], unique=False)
    op.create_index("ix_gex_batch_fetched", "gex_snapshots", ["batch_id", "fetched_at"], unique=False)
    op.create_index("ix_gex_batch_status", "gex_batches", ["status"], unique=False)
    op.create_index("ix_gex_batch_completed", "gex_batches", ["completed_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_gex_batch_completed", "gex_batches")
    op.drop_index("ix_gex_batch_status", "gex_batches")
    op.drop_index("ix_gex_batch_fetched", "gex_snapshots")
    op.drop_index("ix_gex_ticker_fetched", "gex_snapshots")
    op.drop_table("gex_snapshots")
    op.drop_table("gex_batches")
