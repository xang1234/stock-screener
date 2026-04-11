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

    market_inference_expr = (
        "CASE "
        "WHEN UPPER(COALESCE(exchange, '')) IN ('HKEX', 'SEHK', 'XHKG') "
        "OR UPPER(TRIM(COALESCE(symbol, ''))) LIKE '%.HK' THEN 'HK' "
        "WHEN UPPER(COALESCE(exchange, '')) IN ('TSE', 'JPX', 'XTKS') "
        "OR UPPER(TRIM(COALESCE(symbol, ''))) LIKE '%.T' THEN 'JP' "
        "WHEN UPPER(COALESCE(exchange, '')) IN ('TWSE', 'TPEX', 'XTAI') "
        "OR UPPER(TRIM(COALESCE(symbol, ''))) LIKE '%.TW' "
        "OR UPPER(TRIM(COALESCE(symbol, ''))) LIKE '%.TWO' THEN 'TW' "
        "ELSE 'US' END"
    )
    non_us_inference_condition = (
        "UPPER(COALESCE(exchange, '')) IN ('HKEX', 'SEHK', 'XHKG', 'TSE', 'JPX', 'XTKS', 'TWSE', 'TPEX', 'XTAI') "
        "OR UPPER(TRIM(COALESCE(symbol, ''))) LIKE '%.HK' "
        "OR UPPER(TRIM(COALESCE(symbol, ''))) LIKE '%.T' "
        "OR UPPER(TRIM(COALESCE(symbol, ''))) LIKE '%.TW' "
        "OR UPPER(TRIM(COALESCE(symbol, ''))) LIKE '%.TWO'"
    )
    inferred_currency_expr = (
        f"CASE {market_inference_expr} "
        "WHEN 'HK' THEN 'HKD' "
        "WHEN 'JP' THEN 'JPY' "
        "WHEN 'TW' THEN 'TWD' "
        "ELSE 'USD' END"
    )
    inferred_timezone_expr = (
        f"CASE {market_inference_expr} "
        "WHEN 'HK' THEN 'Asia/Hong_Kong' "
        "WHEN 'JP' THEN 'Asia/Tokyo' "
        "WHEN 'TW' THEN 'Asia/Taipei' "
        "ELSE 'America/New_York' END"
    )
    needs_backfill_condition = (
        "(NULLIF(market, '') IS NULL OR (market = 'US' AND "
        f"{market_inference_expr} <> 'US')) "
        "OR (NULLIF(currency, '') IS NULL OR (currency = 'USD' AND "
        f"{market_inference_expr} <> 'US')) "
        "OR (NULLIF(timezone, '') IS NULL OR (timezone = 'America/New_York' AND "
        f"{market_inference_expr} <> 'US')) "
        "OR NULLIF(local_code, '') IS NULL"
    )
    needs_null_fill_condition = (
        "NULLIF(market, '') IS NULL "
        "OR NULLIF(currency, '') IS NULL "
        "OR NULLIF(timezone, '') IS NULL "
        "OR NULLIF(local_code, '') IS NULL"
    )

    if dialect == "postgresql":
        local_code_from_symbol_expr = (
            "CASE "
            f"WHEN ({non_us_inference_condition}) AND POSITION('.' IN symbol) > 0 "
            "THEN SPLIT_PART(symbol, '.', 1) "
            "ELSE symbol END"
        )
    elif dialect == "sqlite":
        local_code_from_symbol_expr = (
            "CASE "
            f"WHEN ({non_us_inference_condition}) AND INSTR(symbol, '.') > 0 "
            "THEN SUBSTR(symbol, 1, INSTR(symbol, '.') - 1) "
            "ELSE symbol END"
        )
    else:
        # Fallback for untested dialects: preserve canonical symbol.
        local_code_from_symbol_expr = "symbol"

    with op.batch_alter_table("stock_universe") as batch_op:
        batch_op.add_column(
            sa.Column("market", sa.String(length=8), nullable=True)
        )
        batch_op.add_column(
            sa.Column("currency", sa.String(length=8), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "timezone",
                sa.String(length=64),
                nullable=True,
            )
        )
        batch_op.add_column(sa.Column("local_code", sa.String(length=32), nullable=True))

    # Backfill existing rows with market-aware inference.
    # Rows are updated only when fields are empty, or when US defaults must be corrected for inferred non-US rows.
    op.execute(
        sa.text(
            f"""
            UPDATE stock_universe
            SET
                market = CASE
                    WHEN NULLIF(market, '') IS NULL THEN {market_inference_expr}
                    WHEN market = 'US' AND {market_inference_expr} <> 'US' THEN {market_inference_expr}
                    ELSE market
                END,
                currency = CASE
                    WHEN NULLIF(currency, '') IS NULL THEN
                    {inferred_currency_expr}
                    WHEN currency = 'USD' AND {market_inference_expr} <> 'US' THEN
                    {inferred_currency_expr}
                    ELSE currency
                END,
                timezone = CASE
                    WHEN NULLIF(timezone, '') IS NULL THEN
                        {inferred_timezone_expr}
                    WHEN timezone = 'America/New_York' AND {market_inference_expr} <> 'US' THEN
                        {inferred_timezone_expr}
                    ELSE timezone
                END,
                local_code = COALESCE(
                    NULLIF(local_code, ''),
                    {local_code_from_symbol_expr}
                )
            WHERE {needs_backfill_condition}
            """
        )
    )

    # Close any residual null/blank window from concurrent writes before NOT NULL enforcement.
    op.execute(
        sa.text(
            f"""
            UPDATE stock_universe
            SET
                market = CASE
                    WHEN NULLIF(market, '') IS NULL THEN {market_inference_expr}
                    WHEN market = 'US' AND {market_inference_expr} <> 'US' THEN {market_inference_expr}
                    ELSE market
                END,
                currency = CASE
                    WHEN NULLIF(currency, '') IS NULL THEN
                        {inferred_currency_expr}
                    WHEN currency = 'USD' AND {market_inference_expr} <> 'US' THEN
                        {inferred_currency_expr}
                    ELSE currency
                END,
                timezone = CASE
                    WHEN NULLIF(timezone, '') IS NULL THEN
                        {inferred_timezone_expr}
                    WHEN timezone = 'America/New_York' AND {market_inference_expr} <> 'US' THEN
                        {inferred_timezone_expr}
                    ELSE timezone
                END,
                local_code = COALESCE(
                    NULLIF(local_code, ''),
                    {local_code_from_symbol_expr}
                )
            WHERE {needs_null_fill_condition}
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
