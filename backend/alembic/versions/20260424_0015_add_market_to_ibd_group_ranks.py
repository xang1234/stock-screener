"""Add market column to ibd_group_ranks for multi-market rankings."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260424_0015"
down_revision = "20260417_0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    # 1. Add the column with a default of 'US' so existing rows backfill cleanly.
    op.add_column(
        "ibd_group_ranks",
        sa.Column("market", sa.String(length=8), nullable=False, server_default="US"),
    )
    op.create_index(
        "idx_ibd_group_rank_market_date",
        "ibd_group_ranks",
        ["market", "date"],
    )

    # 2. Swap the uniqueness constraint to include market.
    # SQLite needs batch mode for constraint changes; Postgres can do it directly.
    if dialect == "sqlite":
        with op.batch_alter_table("ibd_group_ranks") as batch_op:
            batch_op.drop_constraint("uix_ibd_group_rank_date", type_="unique")
            batch_op.create_unique_constraint(
                "uix_ibd_group_rank_market_date",
                ["industry_group", "date", "market"],
            )
    else:
        op.drop_constraint("uix_ibd_group_rank_date", "ibd_group_ranks", type_="unique")
        op.create_unique_constraint(
            "uix_ibd_group_rank_market_date",
            "ibd_group_ranks",
            ["industry_group", "date", "market"],
        )

    # 3. Now that every row has a value, drop the server_default — new rows
    #    must set market explicitly via the application layer so unexpected
    #    writes don't silently land in the 'US' bucket.
    with op.batch_alter_table("ibd_group_ranks") as batch_op:
        batch_op.alter_column("market", server_default=None)


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "sqlite":
        with op.batch_alter_table("ibd_group_ranks") as batch_op:
            batch_op.drop_constraint("uix_ibd_group_rank_market_date", type_="unique")
            batch_op.create_unique_constraint(
                "uix_ibd_group_rank_date",
                ["industry_group", "date"],
            )
    else:
        op.drop_constraint(
            "uix_ibd_group_rank_market_date", "ibd_group_ranks", type_="unique"
        )
        op.create_unique_constraint(
            "uix_ibd_group_rank_date",
            "ibd_group_ranks",
            ["industry_group", "date"],
        )

    op.drop_index("idx_ibd_group_rank_market_date", table_name="ibd_group_ranks")
    op.drop_column("ibd_group_ranks", "market")
