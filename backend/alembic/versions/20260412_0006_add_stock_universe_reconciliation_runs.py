"""Add stock universe reconciliation run artifacts table."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260412_0006"
down_revision = "20260411_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "stock_universe_reconciliation_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("market", sa.String(length=8), nullable=False),
        sa.Column("source_name", sa.String(length=64), nullable=False),
        sa.Column("snapshot_id", sa.String(length=128), nullable=False),
        sa.Column("previous_snapshot_id", sa.String(length=128), nullable=True),
        sa.Column("total_current", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("total_previous", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("added_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("removed_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("changed_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("unchanged_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("artifact_hash", sa.String(length=64), nullable=False),
        sa.Column("artifact_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("market", "snapshot_id", name="uq_universe_reconciliation_market_snapshot"),
    )
    op.create_index(
        "ix_stock_universe_reconciliation_runs_id",
        "stock_universe_reconciliation_runs",
        ["id"],
        unique=False,
    )
    op.create_index(
        "ix_stock_universe_reconciliation_runs_market",
        "stock_universe_reconciliation_runs",
        ["market"],
        unique=False,
    )
    op.create_index(
        "ix_stock_universe_reconciliation_runs_snapshot_id",
        "stock_universe_reconciliation_runs",
        ["snapshot_id"],
        unique=False,
    )
    op.create_index(
        "ix_stock_universe_reconciliation_runs_artifact_hash",
        "stock_universe_reconciliation_runs",
        ["artifact_hash"],
        unique=False,
    )
    op.create_index(
        "ix_stock_universe_reconciliation_runs_created_at",
        "stock_universe_reconciliation_runs",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "idx_universe_reconciliation_market_created",
        "stock_universe_reconciliation_runs",
        ["market", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_universe_reconciliation_market_created", table_name="stock_universe_reconciliation_runs")
    op.drop_index("ix_stock_universe_reconciliation_runs_created_at", table_name="stock_universe_reconciliation_runs")
    op.drop_index("ix_stock_universe_reconciliation_runs_artifact_hash", table_name="stock_universe_reconciliation_runs")
    op.drop_index("ix_stock_universe_reconciliation_runs_snapshot_id", table_name="stock_universe_reconciliation_runs")
    op.drop_index("ix_stock_universe_reconciliation_runs_market", table_name="stock_universe_reconciliation_runs")
    op.drop_index("ix_stock_universe_reconciliation_runs_id", table_name="stock_universe_reconciliation_runs")
    op.drop_table("stock_universe_reconciliation_runs")
