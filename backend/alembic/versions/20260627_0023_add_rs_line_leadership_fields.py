"""Add RS line leadership fields."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260627_0023"
down_revision = "20260618_0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "scan_results",
        sa.Column("rs_line_new_high", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "scan_results",
        sa.Column(
            "rs_line_new_high_before_price",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "scan_results",
        sa.Column(
            "rs_line_blue_dot_recent",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "scan_results",
        sa.Column("rs_line_new_high_date", sa.String(length=10), nullable=True),
    )
    op.create_index("idx_scan_rs_line_new_high", "scan_results", ["scan_id", "rs_line_new_high"])
    op.create_index(
        "idx_scan_rs_line_new_high_before_price",
        "scan_results",
        ["scan_id", "rs_line_new_high_before_price"],
    )
    op.create_index(
        "idx_scan_rs_line_blue_dot_recent",
        "scan_results",
        ["scan_id", "rs_line_blue_dot_recent"],
    )
    op.create_index(
        "idx_scan_rs_line_new_high_date",
        "scan_results",
        ["scan_id", "rs_line_new_high_date"],
    )

    op.add_column(
        "stock_feature_daily",
        sa.Column("rs_line_new_high", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "stock_feature_daily",
        sa.Column(
            "rs_line_new_high_before_price",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "stock_feature_daily",
        sa.Column(
            "rs_line_blue_dot_recent",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "stock_feature_daily",
        sa.Column("rs_line_new_high_date", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_sfd_run_rs_line_new_high",
        "stock_feature_daily",
        ["run_id", "rs_line_new_high"],
    )
    op.create_index(
        "ix_sfd_run_rs_line_new_high_before_price",
        "stock_feature_daily",
        ["run_id", "rs_line_new_high_before_price"],
    )
    op.create_index(
        "ix_sfd_run_rs_line_blue_dot_recent",
        "stock_feature_daily",
        ["run_id", "rs_line_blue_dot_recent"],
    )
    op.create_index(
        "ix_sfd_run_rs_line_new_high_date",
        "stock_feature_daily",
        ["run_id", "rs_line_new_high_date"],
    )


def downgrade() -> None:
    op.drop_index("ix_sfd_run_rs_line_new_high_date", table_name="stock_feature_daily")
    op.drop_index("ix_sfd_run_rs_line_blue_dot_recent", table_name="stock_feature_daily")
    op.drop_index("ix_sfd_run_rs_line_new_high_before_price", table_name="stock_feature_daily")
    op.drop_index("ix_sfd_run_rs_line_new_high", table_name="stock_feature_daily")
    op.drop_column("stock_feature_daily", "rs_line_new_high_date")
    op.drop_column("stock_feature_daily", "rs_line_blue_dot_recent")
    op.drop_column("stock_feature_daily", "rs_line_new_high_before_price")
    op.drop_column("stock_feature_daily", "rs_line_new_high")

    op.drop_index("idx_scan_rs_line_new_high_date", table_name="scan_results")
    op.drop_index("idx_scan_rs_line_blue_dot_recent", table_name="scan_results")
    op.drop_index("idx_scan_rs_line_new_high_before_price", table_name="scan_results")
    op.drop_index("idx_scan_rs_line_new_high", table_name="scan_results")
    op.drop_column("scan_results", "rs_line_new_high_date")
    op.drop_column("scan_results", "rs_line_blue_dot_recent")
    op.drop_column("scan_results", "rs_line_new_high_before_price")
    op.drop_column("scan_results", "rs_line_new_high")
