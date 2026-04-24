"""Add market column to market_breadth for multi-market breadth."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260425_0016"
down_revision = "20260424_0015"
branch_labels = None
depends_on = None


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
    inspector = sa.inspect(bind)
    existing_uniques = {uc["name"] for uc in inspector.get_unique_constraints("market_breadth")}

    if dialect == "sqlite":
        with op.batch_alter_table("market_breadth") as batch_op:
            # Try all plausible names for the date-only unique; ignore if none exist.
            for name in ("market_breadth_date_key", "uq_market_breadth_date"):
                try:
                    batch_op.drop_constraint(name, type_="unique")
                    break
                except Exception:
                    continue
            batch_op.create_unique_constraint(
                "uix_breadth_date_market",
                ["date", "market"],
            )
    else:
        # Postgres: drop each known implicit/explicit name ONLY if it exists.
        for name in ("market_breadth_date_key", "uq_market_breadth_date"):
            if name in existing_uniques:
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
