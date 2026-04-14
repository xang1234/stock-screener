"""Add stock_universe_index_membership table.

Asia indices (HSI, Nikkei 225, TAIEX) route through a dedicated membership
table instead of adding boolean columns per index. SP500 keeps using
``stock_universe.is_sp500`` for backward compatibility.

Revision ID: 20260414_0011
Revises: 20260413_0010
Create Date: 2026-04-14
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260414_0011"
down_revision = "20260413_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "stock_universe_index_membership",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("index_name", sa.String(length=32), nullable=False),
        sa.Column("as_of_date", sa.String(length=10), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=False, server_default="manual"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.UniqueConstraint(
            "symbol",
            "index_name",
            name="uq_universe_index_membership_symbol_index",
        ),
    )
    # Two indexes cover every query shape today:
    #   - standalone ``symbol`` serves "which indices contain this symbol?"
    #     (filter on symbol alone — not the leftmost of the composite).
    #   - composite ``(index_name, symbol)`` serves the dominant hot path,
    #     "list constituents of index X" (``WHERE index_name = ?``), via
    #     its leftmost prefix, and also answers "is symbol Y in index X?".
    #     No standalone index_name index needed because the composite's
    #     leftmost prefix already covers it.
    op.create_index(
        "ix_stock_universe_index_membership_symbol",
        "stock_universe_index_membership",
        ["symbol"],
    )
    op.create_index(
        "idx_universe_index_membership_index_symbol",
        "stock_universe_index_membership",
        ["index_name", "symbol"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_universe_index_membership_index_symbol",
        table_name="stock_universe_index_membership",
    )
    op.drop_index(
        "ix_stock_universe_index_membership_symbol",
        table_name="stock_universe_index_membership",
    )
    op.drop_table("stock_universe_index_membership")
