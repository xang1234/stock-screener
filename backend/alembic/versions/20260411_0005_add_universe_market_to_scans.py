"""Add universe_market to scans for market-scoped universe contracts."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260411_0005"
down_revision = "20260411_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    with op.batch_alter_table("scans") as batch_op:
        batch_op.add_column(sa.Column("universe_market", sa.String(length=8), nullable=True))
        batch_op.create_index("ix_scans_universe_market", ["universe_market"], unique=False)

    if dialect == "postgresql":
        market_from_key_expr = "UPPER(SPLIT_PART(universe_key, ':', 2))"
        market_key_condition = "POSITION(':' IN COALESCE(universe_key, '')) > 0"
    elif dialect == "sqlite":
        market_from_key_expr = "UPPER(SUBSTR(universe_key, INSTR(universe_key, ':') + 1))"
        market_key_condition = "INSTR(COALESCE(universe_key, ''), ':') > 0"
    else:
        market_from_key_expr = "NULL"
        market_key_condition = "0 = 1"

    # Backfill legacy scans where the market scope is deterministic.
    op.execute(
        sa.text(
            f"""
            UPDATE scans
            SET universe_market = CASE
                WHEN LOWER(COALESCE(universe_type, '')) = 'market'
                    AND {market_key_condition}
                    THEN {market_from_key_expr}
                WHEN LOWER(COALESCE(universe_type, '')) IN ('all', 'exchange', 'index')
                    THEN 'US'
                ELSE NULL
            END
            WHERE universe_market IS NULL OR TRIM(universe_market) = ''
            """
        )
    )


def downgrade() -> None:
    with op.batch_alter_table("scans") as batch_op:
        batch_op.drop_index("ix_scans_universe_market")
        batch_op.drop_column("universe_market")
