"""Durable social extraction and installation-wide daily dollar reservations."""
from alembic import op
import sqlalchemy as sa

revision = "20260907_0036"
down_revision = "20260906_0035"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("social_extraction_work",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("content_item_id", sa.Integer, sa.ForeignKey("content_items.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("input_hash", sa.Text, nullable=False),
        sa.Column("prompt_version", sa.Text, nullable=False),
        sa.Column("schema_version", sa.Text, nullable=False),
        sa.Column("selected_model", sa.Text, nullable=False),
        sa.Column("input_snapshot_json", sa.JSON, nullable=False),
        sa.Column("actual_provider", sa.Text),
        sa.Column("actual_model", sa.Text),
        sa.Column("result_json", sa.JSON),
        sa.Column("error_code", sa.Text),
        sa.Column("state", sa.Text, nullable=False),
        sa.Column("claim_token", sa.Text),
        sa.Column("claim_expires_at", sa.DateTime(timezone=True)),
        sa.Column("requested_by_admin", sa.Boolean, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("content_item_id", "input_hash", "prompt_version", "schema_version", "selected_model", name="uq_social_extraction_identity"),
        sa.CheckConstraint("state IN ('pending','running','waiting_budget','succeeded','failed_retryable','failed_terminal','outside_window')", name="ck_social_work_state"))
    op.create_table("social_run_work",
        sa.Column("run_id", sa.Text, sa.ForeignKey("social_signal_runs.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("work_id", sa.Integer, sa.ForeignKey("social_extraction_work.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("input_hash", sa.Text, nullable=False),
        sa.Column("included_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "work_id", name="uq_social_run_work"))
    op.create_table("social_llm_budget_days",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("budget_date", sa.Date, nullable=False),
        sa.Column("timezone", sa.Text, nullable=False),
        sa.Column("period_start_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("limit_usd", sa.Numeric(24, 12), nullable=False),
        sa.Column("reserved_usd", sa.Numeric(24, 12), nullable=False),
        sa.Column("actual_usd", sa.Numeric(24, 12), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.UniqueConstraint("period_start_utc", "period_end_utc", name="uq_social_llm_period"),
        sa.CheckConstraint("limit_usd >= 0 AND reserved_usd >= 0 AND actual_usd >= 0", name="ck_social_budget_money"))
    op.create_table("social_llm_attempts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("idempotency_key", sa.Text, nullable=False, unique=True),
        sa.Column("budget_day_id", sa.Integer, sa.ForeignKey("social_llm_budget_days.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("work_ids", sa.JSON, nullable=False),
        sa.Column("estimated_usd", sa.Numeric(24, 12), nullable=False),
        sa.Column("actual_usd", sa.Numeric(24, 12)),
        sa.Column("pricing_version", sa.Text, nullable=False),
        sa.Column("input_token_limit", sa.Integer, nullable=False),
        sa.Column("output_token_limit", sa.Integer, nullable=False),
        sa.Column("actual_input_tokens", sa.Integer),
        sa.Column("actual_output_tokens", sa.Integer),
        sa.Column("state", sa.Text, nullable=False),
        sa.Column("provider_request_id", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("state IN ('reserved','dispatched','reconciled','uncertain','released')", name="ck_social_attempt_state"),
        sa.CheckConstraint("estimated_usd >= 0 AND (actual_usd IS NULL OR actual_usd >= 0)", name="ck_social_attempt_money"))
    settings = sa.table("app_settings", sa.column("key"), sa.column("value"), sa.column("category"))
    for key, value in (("social_llm_daily_limit_usd", "2"), ("social_llm_budget_timezone", "Asia/Singapore")):
        if op.get_bind().execute(sa.select(settings.c.key).where(settings.c.key == key)).first() is None:
            op.get_bind().execute(settings.insert().values(key=key, value=value, category="social"))


def downgrade():
    for name in ("social_llm_attempts", "social_run_work", "social_llm_budget_days", "social_extraction_work"):
        op.drop_table(name)
    # Preserve administrator settings across a rollback/re-upgrade.
