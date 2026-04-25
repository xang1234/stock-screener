"""Add market column to ibd_group_ranks for multi-market rankings."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260424_0015"
down_revision = "20260417_0014"
branch_labels = None
depends_on = None


SQLITE_BATCH_NAMING_CONVENTION = {
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
}


def _unique_constraint_names_for_columns(
    bind,
    table_name: str,
    columns: tuple[str, ...],
    *,
    unnamed_sqlite_name: str,
) -> list[str]:
    inspector = sa.inspect(bind)
    names: list[str] = []
    for constraint in inspector.get_unique_constraints(table_name):
        if tuple(constraint.get("column_names") or ()) != columns:
            continue
        name = constraint.get("name") or unnamed_sqlite_name
        names.append(name)
    return names


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
        unique_names = _unique_constraint_names_for_columns(
            bind,
            "ibd_group_ranks",
            ("industry_group", "date"),
            unnamed_sqlite_name="uq_ibd_group_ranks_industry_group_date",
        )
        with op.batch_alter_table(
            "ibd_group_ranks",
            naming_convention=SQLITE_BATCH_NAMING_CONVENTION,
        ) as batch_op:
            for name in unique_names:
                batch_op.drop_constraint(name, type_="unique")
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

    # (industry_group, date) is about to become unique again; non-US rows
    # written after upgrade would collide. Drop them first rather than leave
    # the schema half-reverted.
    bind.execute(sa.text("DELETE FROM ibd_group_ranks WHERE market <> 'US'"))

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
