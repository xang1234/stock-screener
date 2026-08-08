"""Add iv_history table for durable IV Rank history."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260805_0030"
down_revision = "20260805_0029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "iv_history",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("trading_date", sa.Date(), nullable=False),
        sa.Column("atm_iv", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("ticker", "trading_date", name="uq_iv_history_ticker_trading_date"),
    )

    op.create_index(
        "ix_iv_history_ticker_trading_date",
        "iv_history",
        ["ticker", "trading_date"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_iv_history_ticker_trading_date", "iv_history")
    op.drop_table("iv_history")
