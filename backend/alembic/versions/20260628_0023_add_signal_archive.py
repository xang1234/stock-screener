"""add signal_archive table

Revision ID: 20260628_0023
Revises: 20260618_0022
Create Date: 2026-06-28
"""
from alembic import op
import sqlalchemy as sa

revision = "20260628_0023"
down_revision = "20260618_0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "signal_archive",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("symbol", sa.String(20), nullable=False, index=True),
        sa.Column("signal_date", sa.Date(), nullable=False),
        sa.Column("entry_price", sa.Float(), nullable=True),
        sa.Column("stop_loss", sa.Float(), nullable=True),
        sa.Column("target_price", sa.Float(), nullable=True),
        sa.Column("screener", sa.String(50), nullable=True),
        sa.Column("composite_score", sa.Float(), nullable=True),
        sa.Column("sector", sa.String(100), nullable=True),
        sa.Column("stage", sa.Integer(), nullable=True),
        sa.Column("outcome", sa.String(20), nullable=True),
        sa.Column("outcome_date", sa.Date(), nullable=True),
        sa.Column("outcome_price", sa.Float(), nullable=True),
        sa.Column("pct_return", sa.Float(), nullable=True),
        sa.Column("days_held", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("signal_archive")
