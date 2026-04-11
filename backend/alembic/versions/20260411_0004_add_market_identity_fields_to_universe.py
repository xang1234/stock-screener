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
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        local_code_from_symbol_expr = (
            "CASE "
            "WHEN inferred.inferred_market IN ('HK', 'JP', 'TW') AND POSITION('.' IN su.symbol) > 0 "
            "THEN SPLIT_PART(su.symbol, '.', 1) "
            "ELSE su.symbol END"
        )
    elif dialect == "sqlite":
        local_code_from_symbol_expr = (
            "CASE "
            "WHEN inferred.inferred_market IN ('HK', 'JP', 'TW') AND INSTR(su.symbol, '.') > 0 "
            "THEN SUBSTR(su.symbol, 1, INSTR(su.symbol, '.') - 1) "
            "ELSE su.symbol END"
        )
    else:
        # Fallback for untested dialects: preserve canonical symbol.
        local_code_from_symbol_expr = "su.symbol"

    with op.batch_alter_table("stock_universe") as batch_op:
        batch_op.add_column(
            sa.Column("market", sa.String(length=8), nullable=True, server_default=sa.text("'US'"))
        )
        batch_op.add_column(
            sa.Column("currency", sa.String(length=8), nullable=True, server_default=sa.text("'USD'"))
        )
        batch_op.add_column(
            sa.Column(
                "timezone",
                sa.String(length=64),
                nullable=True,
                server_default=sa.text("'America/New_York'"),
            )
        )
        batch_op.add_column(sa.Column("local_code", sa.String(length=32), nullable=True))

    # Backfill existing rows with market-aware inference.
    # If no non-US signal is present, default to US baseline.
    op.execute(
        sa.text(
            f"""
            WITH inferred AS (
                SELECT
                    id,
                    CASE
                        WHEN UPPER(COALESCE(exchange, '')) IN ('HKEX', 'SEHK')
                            OR UPPER(TRIM(COALESCE(symbol, ''))) LIKE '%.HK' THEN 'HK'
                        WHEN UPPER(COALESCE(exchange, '')) IN ('TSE', 'JPX', 'XTKS')
                            OR UPPER(TRIM(COALESCE(symbol, ''))) LIKE '%.T' THEN 'JP'
                        WHEN UPPER(COALESCE(exchange, '')) IN ('TWSE', 'TPEX', 'XTAI')
                            OR UPPER(TRIM(COALESCE(symbol, ''))) LIKE '%.TW'
                            OR UPPER(TRIM(COALESCE(symbol, ''))) LIKE '%.TWO' THEN 'TW'
                        ELSE 'US'
                    END AS inferred_market
                FROM stock_universe
            )
            UPDATE stock_universe AS su
            SET
                market = CASE
                    WHEN NULLIF(su.market, '') IS NULL THEN inferred.inferred_market
                    WHEN su.market = 'US' AND inferred.inferred_market <> 'US' THEN inferred.inferred_market
                    ELSE su.market
                END,
                currency = CASE
                    WHEN NULLIF(su.currency, '') IS NULL THEN
                    CASE inferred.inferred_market
                        WHEN 'HK' THEN 'HKD'
                        WHEN 'JP' THEN 'JPY'
                        WHEN 'TW' THEN 'TWD'
                        ELSE 'USD'
                    END
                    WHEN su.currency = 'USD' AND inferred.inferred_market <> 'US' THEN
                    CASE inferred.inferred_market
                        WHEN 'HK' THEN 'HKD'
                        WHEN 'JP' THEN 'JPY'
                        WHEN 'TW' THEN 'TWD'
                        ELSE 'USD'
                    END
                    ELSE su.currency
                END,
                timezone = CASE
                    WHEN NULLIF(su.timezone, '') IS NULL THEN
                        CASE inferred.inferred_market
                            WHEN 'HK' THEN 'Asia/Hong_Kong'
                            WHEN 'JP' THEN 'Asia/Tokyo'
                            WHEN 'TW' THEN 'Asia/Taipei'
                            ELSE 'America/New_York'
                        END
                    WHEN su.timezone = 'America/New_York' AND inferred.inferred_market <> 'US' THEN
                        CASE inferred.inferred_market
                            WHEN 'HK' THEN 'Asia/Hong_Kong'
                            WHEN 'JP' THEN 'Asia/Tokyo'
                            WHEN 'TW' THEN 'Asia/Taipei'
                            ELSE 'America/New_York'
                        END
                    ELSE su.timezone
                END,
                local_code = COALESCE(
                    NULLIF(su.local_code, ''),
                    {local_code_from_symbol_expr}
                )
            FROM inferred
            WHERE su.id = inferred.id
            """
        )
    )

    # Close any residual null/blank window from concurrent writes before NOT NULL enforcement.
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
