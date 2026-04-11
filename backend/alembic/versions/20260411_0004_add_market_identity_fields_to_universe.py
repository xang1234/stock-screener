"""Add explicit market identity fields to stock_universe with market-aware backfill."""

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

    # Backfill existing rows with market-aware inference.
    # If no non-US signal is present, default to US baseline.
    op.execute(
        sa.text(
            """
            WITH inferred AS (
                SELECT
                    id,
                    CASE
                        WHEN UPPER(COALESCE(exchange, '')) IN ('HKEX', 'SEHK')
                            OR symbol LIKE '%.HK' THEN 'HK'
                        WHEN UPPER(COALESCE(exchange, '')) IN ('TSE', 'JPX', 'XTKS')
                            OR symbol LIKE '%.T' THEN 'JP'
                        WHEN UPPER(COALESCE(exchange, '')) IN ('TWSE', 'TPEX', 'XTAI')
                            OR symbol LIKE '%.TW'
                            OR symbol LIKE '%.TWO' THEN 'TW'
                        ELSE 'US'
                    END AS inferred_market
                FROM stock_universe
            )
            UPDATE stock_universe AS su
            SET
                market = COALESCE(NULLIF(su.market, ''), inferred.inferred_market),
                currency = COALESCE(
                    NULLIF(su.currency, ''),
                    CASE inferred.inferred_market
                        WHEN 'HK' THEN 'HKD'
                        WHEN 'JP' THEN 'JPY'
                        WHEN 'TW' THEN 'TWD'
                        ELSE 'USD'
                    END
                ),
                timezone = COALESCE(
                    NULLIF(su.timezone, ''),
                    CASE inferred.inferred_market
                        WHEN 'HK' THEN 'Asia/Hong_Kong'
                        WHEN 'JP' THEN 'Asia/Tokyo'
                        WHEN 'TW' THEN 'Asia/Taipei'
                        ELSE 'America/New_York'
                    END
                ),
                local_code = COALESCE(
                    NULLIF(su.local_code, ''),
                    CASE
                        WHEN inferred.inferred_market IN ('HK', 'JP', 'TW') AND su.symbol LIKE '%.%'
                            THEN split_part(su.symbol, '.', 1)
                        ELSE su.symbol
                    END
                )
            FROM inferred
            WHERE su.id = inferred.id
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
