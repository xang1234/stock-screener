"""Add market column to market_breadth for multi-market breadth."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260425_0016"
down_revision = "20260424_0015"
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

    op.add_column(
        "market_breadth",
        sa.Column("market", sa.String(length=8), nullable=False, server_default="US"),
    )
    op.create_index(
        "idx_breadth_market_date",
        "market_breadth",
        ["market", "date"],
    )

    # Drop the date-only UNIQUE and replace with (date, market).
    # The baseline migration (20260408_0001) created market_breadth without a
    # named date unique — only ORM-side models declared `unique=True`. So the
    # constraint may or may not exist depending on how the DB was built. We
    # inspect first to avoid failing mid-migration on production DBs.
    date_unique_names = _unique_constraint_names_for_columns(
        bind,
        "market_breadth",
        ("date",),
        unnamed_sqlite_name="uq_market_breadth_date",
    )

    if dialect == "sqlite":
        with op.batch_alter_table(
            "market_breadth",
            naming_convention=SQLITE_BATCH_NAMING_CONVENTION,
        ) as batch_op:
            for name in date_unique_names:
                batch_op.drop_constraint(name, type_="unique")
            batch_op.create_unique_constraint(
                "uix_breadth_date_market",
                ["date", "market"],
            )
    else:
        # Postgres: drop each known implicit/explicit name ONLY if it exists.
        for name in date_unique_names:
            op.drop_constraint(name, "market_breadth", type_="unique")
        op.create_unique_constraint(
            "uix_breadth_date_market",
            "market_breadth",
            ["date", "market"],
        )

    # Drop the server default now that all rows are backfilled; writers must
    # set market explicitly.
    with op.batch_alter_table("market_breadth") as batch_op:
        batch_op.alter_column("market", server_default=None)


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    # Non-US rows would collide with a date-only UNIQUE; drop them first.
    bind.execute(sa.text("DELETE FROM market_breadth WHERE market <> 'US'"))

    if dialect == "sqlite":
        with op.batch_alter_table("market_breadth") as batch_op:
            try:
                batch_op.drop_constraint("uix_breadth_date_market", type_="unique")
            except Exception:
                pass
            # Don't re-create the date-only unique: the baseline migration
            # never materialized it as a named constraint, so recreating here
            # could leave production in a state it never had before.
    else:
        inspector = sa.inspect(bind)
        existing = {
            uc["name"] for uc in inspector.get_unique_constraints("market_breadth")
        }
        if "uix_breadth_date_market" in existing:
            op.drop_constraint(
                "uix_breadth_date_market", "market_breadth", type_="unique"
            )

    op.drop_index("idx_breadth_market_date", table_name="market_breadth")
    op.drop_column("market_breadth", "market")
