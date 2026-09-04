"""Add relational Options Command Center persistence.

Revision ID: 20260904_0034
Revises: 20260829_0033
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260904_0034"
down_revision = "20260829_0033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "options_analytics_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("market", sa.String(length=8), nullable=False),
        sa.Column("origin", sa.String(length=32), nullable=False),
        sa.Column("source_feature_run_id", sa.Integer(), nullable=True),
        sa.Column("external_source_feature_run_key", sa.String(length=255), nullable=True),
        sa.Column("calculation_version", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("input_signature", sa.String(length=64), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("risk_free_rate", sa.Float(), nullable=True),
        sa.Column("expected_count", sa.Integer(), nullable=False),
        sa.Column("current_count", sa.Integer(), nullable=False),
        sa.Column("continuity_count", sa.Integer(), nullable=False),
        sa.Column("completed_count", sa.Integer(), nullable=False),
        sa.Column("core_valid_current_count", sa.Integer(), nullable=False),
        sa.Column("failed_count", sa.Integer(), nullable=False),
        sa.Column("retried_count", sa.Integer(), nullable=False),
        sa.Column("coverage", sa.Float(), nullable=False),
        sa.Column("assumptions_json", sa.JSON(), nullable=True),
        sa.Column("warnings_json", sa.JSON(), nullable=True),
        sa.Column("diagnostics_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(origin = 'history_transfer' AND external_source_feature_run_key IS NOT NULL) "
            "OR (origin <> 'history_transfer' AND source_feature_run_id IS NOT NULL)",
            name="ck_options_run_source_identity",
        ),
        sa.ForeignKeyConstraint(
            ["source_feature_run_id"], ["feature_runs.id"], ondelete="RESTRICT"
        ),
        sa.UniqueConstraint(
            "input_signature", "attempt_number", name="uq_options_run_signature_attempt"
        ),
    )
    op.create_index("ix_options_runs_market", "options_analytics_runs", ["market"])
    op.create_index("ix_options_runs_status", "options_analytics_runs", ["status"])
    op.create_index("ix_options_runs_as_of_date", "options_analytics_runs", ["as_of_date"])
    op.create_index(
        "ix_options_runs_market_version_status",
        "options_analytics_runs",
        ["market", "calculation_version", "status"],
    )

    op.create_table(
        "options_analytics_run_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("security_symbol", sa.Text(), nullable=False),
        sa.Column("candidate_kind", sa.String(length=16), nullable=False),
        sa.Column("candidate_rank", sa.Integer(), nullable=True),
        sa.Column("leader_rank", sa.Integer(), nullable=True),
        sa.Column("spot_price", sa.Float(), nullable=True),
        sa.Column("expiration", sa.Date(), nullable=True),
        sa.Column("observation_state", sa.String(length=32), nullable=False),
        sa.Column("observation_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("max_pain", sa.Float(), nullable=True),
        sa.Column("net_gex", sa.Float(), nullable=True),
        sa.Column("gamma_flip", sa.Float(), nullable=True),
        sa.Column("call_wall", sa.Float(), nullable=True),
        sa.Column("put_wall", sa.Float(), nullable=True),
        sa.Column("atm_iv", sa.Float(), nullable=True),
        sa.Column("skew_25_delta", sa.Float(), nullable=True),
        sa.Column("realized_volatility", sa.Float(), nullable=True),
        sa.Column("vrp", sa.Float(), nullable=True),
        sa.Column("activity_intensity", sa.Float(), nullable=True),
        sa.Column("activity_rank", sa.Integer(), nullable=True),
        sa.Column("short_history_observation_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("iv_history_observation_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("lifetime_observation_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("evidence_json", sa.JSON(), nullable=True),
        sa.Column("assumptions_json", sa.JSON(), nullable=True),
        sa.Column("warnings_json", sa.JSON(), nullable=True),
        sa.Column("reasons_json", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["options_analytics_runs.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("run_id", "security_symbol", name="uq_options_run_item_symbol"),
    )
    op.create_index("ix_options_items_run_id", "options_analytics_run_items", ["run_id"])
    op.create_index("ix_options_items_symbol", "options_analytics_run_items", ["security_symbol"])
    op.create_index(
        "ix_options_items_run_kind_activity",
        "options_analytics_run_items",
        ["run_id", "candidate_kind", "activity_rank"],
    )

    op.create_table(
        "options_analytics_strike_points",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("strike", sa.Float(), nullable=False),
        sa.Column("call_open_interest", sa.Integer(), nullable=True),
        sa.Column("put_open_interest", sa.Integer(), nullable=True),
        sa.Column("call_volume", sa.Integer(), nullable=True),
        sa.Column("put_volume", sa.Integer(), nullable=True),
        sa.Column("call_iv", sa.Float(), nullable=True),
        sa.Column("put_iv", sa.Float(), nullable=True),
        sa.Column("estimated_call_gex", sa.Float(), nullable=True),
        sa.Column("estimated_put_gex", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["item_id"], ["options_analytics_run_items.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("item_id", "strike", name="uq_options_strike_item_strike"),
    )
    op.create_index("ix_options_strikes_item_id", "options_analytics_strike_points", ["item_id"])

    op.create_table(
        "options_analytics_pointers",
        sa.Column("market", sa.String(length=8), primary_key=True),
        sa.Column("calculation_version", sa.String(length=64), primary_key=True),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["options_analytics_runs.id"], ondelete="RESTRICT"),
    )


def downgrade() -> None:
    op.drop_table("options_analytics_pointers")
    op.drop_index("ix_options_strikes_item_id", table_name="options_analytics_strike_points")
    op.drop_table("options_analytics_strike_points")
    op.drop_index("ix_options_items_run_kind_activity", table_name="options_analytics_run_items")
    op.drop_index("ix_options_items_symbol", table_name="options_analytics_run_items")
    op.drop_index("ix_options_items_run_id", table_name="options_analytics_run_items")
    op.drop_table("options_analytics_run_items")
    op.drop_index("ix_options_runs_market_version_status", table_name="options_analytics_runs")
    op.drop_index("ix_options_runs_as_of_date", table_name="options_analytics_runs")
    op.drop_index("ix_options_runs_status", table_name="options_analytics_runs")
    op.drop_index("ix_options_runs_market", table_name="options_analytics_runs")
    op.drop_table("options_analytics_runs")
