"""Add explicit market identity fields to stock_universe and backfill US baseline."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260411_0004"
down_revision = "20260411_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("stock_universe") as batch_op:
        batch_op.add_column(sa.Column("market", sa.String(length=8), nullable=True))
        batch_op.add_column(sa.Column("currency", sa.String(length=8), nullable=True))
        batch_op.add_column(sa.Column("timezone", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("local_code", sa.String(length=32), nullable=True))

    # Backfill existing US baseline rows.
    op.execute(
        sa.text(
            """
            UPDATE stock_universe
            SET
                market = COALESCE(NULLIF(market, ''), 'US'),
                currency = COALESCE(NULLIF(currency, ''), 'USD'),
                timezone = COALESCE(NULLIF(timezone, ''), 'America/New_York'),
                local_code = COALESCE(NULLIF(local_code, ''), symbol)
            """
        )
    )

    with op.batch_alter_table("stock_universe") as batch_op:
        batch_op.alter_column(
            "market",
            existing_type=sa.String(length=8),
            nullable=False,
            server_default="US",
        )
        batch_op.alter_column(
            "currency",
            existing_type=sa.String(length=8),
            nullable=False,
            server_default="USD",
        )
        batch_op.alter_column(
            "timezone",
            existing_type=sa.String(length=64),
            nullable=False,
            server_default="America/New_York",
        )
        batch_op.create_index("ix_stock_universe_market", ["market"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("stock_universe") as batch_op:
        batch_op.drop_index("ix_stock_universe_market")
        batch_op.drop_column("local_code")
        batch_op.drop_column("timezone")
        batch_op.drop_column("currency")
        batch_op.drop_column("market")
