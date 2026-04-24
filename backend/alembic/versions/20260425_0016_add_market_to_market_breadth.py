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
    if dialect == "sqlite":
        with op.batch_alter_table("market_breadth") as batch_op:
            # The previous Column(..., unique=True) emits an anonymous unique
            # constraint; SQLite batch-copy tables can sometimes expose it as
            # the implicit "uq_*". Try both names; ignore if neither exists.
            try:
                batch_op.drop_constraint("market_breadth_date_key", type_="unique")
            except Exception:
                pass
            batch_op.create_unique_constraint(
                "uix_breadth_date_market",
                ["date", "market"],
            )
    else:
        # On Postgres the implicit unique is named <table>_<col>_key.
        op.drop_constraint("market_breadth_date_key", "market_breadth", type_="unique")
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
            batch_op.drop_constraint("uix_breadth_date_market", type_="unique")
            batch_op.create_unique_constraint(
                "market_breadth_date_key",
                ["date"],
            )
    else:
        op.drop_constraint("uix_breadth_date_market", "market_breadth", type_="unique")
        op.create_unique_constraint(
            "market_breadth_date_key",
            "market_breadth",
            ["date"],
        )

    op.drop_index("idx_breadth_market_date", table_name="market_breadth")
    op.drop_column("market_breadth", "market")
