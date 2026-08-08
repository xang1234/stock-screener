"""Add options_metrics_snapshots table for durable daily options-metrics history."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260806_0031"
down_revision = "20260805_0030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "options_metrics_snapshots",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="OK"),
        sa.Column("error", sa.String(1000), nullable=True),
        sa.Column("source", sa.String(20), nullable=True),
        sa.Column("schema_version", sa.Integer(), nullable=True),
        sa.Column("expiration", sa.Date(), nullable=True),
        sa.Column("underlying_price", sa.Float(), nullable=True),
        sa.Column("call_wall", sa.Float(), nullable=True),
        sa.Column("call_wall_gex", sa.Float(), nullable=True),
        sa.Column("put_wall", sa.Float(), nullable=True),
        sa.Column("put_wall_gex", sa.Float(), nullable=True),
        sa.Column("zero_gamma", sa.Float(), nullable=True),
        sa.Column("total_call_gex", sa.Float(), nullable=True),
        sa.Column("total_put_gex", sa.Float(), nullable=True),
        sa.Column("total_gex", sa.Float(), nullable=True),
        sa.Column("net_dex", sa.Float(), nullable=True),
        sa.Column("net_vex", sa.Float(), nullable=True),
        sa.Column("net_cex", sa.Float(), nullable=True),
        sa.Column("ivr", sa.Float(), nullable=True),
        sa.Column("skew", sa.Float(), nullable=True),
        sa.Column("historical_volatility", sa.Float(), nullable=True),
        sa.Column("current_atm_iv", sa.Float(), nullable=True),
        sa.Column("volatility_risk_premium", sa.Float(), nullable=True),
        sa.Column("expected_move", sa.Float(), nullable=True),
        sa.Column("atm_strike", sa.Float(), nullable=True),
        sa.Column("volume_put_call_ratio", sa.Float(), nullable=True),
        sa.Column("open_interest_put_call_ratio", sa.Float(), nullable=True),
        sa.Column("total_call_oi", sa.Integer(), nullable=True),
        sa.Column("total_put_oi", sa.Integer(), nullable=True),
        sa.Column("call_premium_notional", sa.Float(), nullable=True),
        sa.Column("put_premium_notional", sa.Float(), nullable=True),
        sa.Column("max_pain_strike", sa.Float(), nullable=True),
        sa.Column("max_pain_distance_pct", sa.Float(), nullable=True),
        sa.Column("next_earnings_date", sa.Date(), nullable=True),
        sa.Column("greeks_methodology", sa.String(50), nullable=True),
        sa.Column("strikes_json", sa.JSON(), nullable=True),
        sa.Column("iv_smile_json", sa.JSON(), nullable=True),
        sa.Column("unusual_volume_json", sa.JSON(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("trading_date", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("batch_id", sa.String(50), nullable=True),
        sa.UniqueConstraint(
            "ticker", "trading_date", "expiration",
            name="uq_options_metrics_snapshots_ticker_date_expiration",
        ),
    )

    op.create_index(
        "ix_options_metrics_snapshots_ticker",
        "options_metrics_snapshots",
        ["ticker"],
    )
    op.create_index(
        "ix_options_metrics_snapshots_ticker_trading_date",
        "options_metrics_snapshots",
        ["ticker", "trading_date"],
    )
    op.create_index(
        "ix_options_metrics_snapshots_batch_fetched",
        "options_metrics_snapshots",
        ["batch_id", "fetched_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_options_metrics_snapshots_batch_fetched", "options_metrics_snapshots")
    op.drop_index("ix_options_metrics_snapshots_ticker_trading_date", "options_metrics_snapshots")
    op.drop_index("ix_options_metrics_snapshots_ticker", "options_metrics_snapshots")
    op.drop_table("options_metrics_snapshots")
