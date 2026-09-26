"""Add company-exposure research work, provider attempts and resource ledger (Task 02)."""

import sqlalchemy as sa

from alembic import op

revision = "20260925_0059"
down_revision = "20260925_0058"
branch_labels = None
depends_on = None

APPEND_ONLY_TABLES = (
    "company_exposure_runtime_policy_revisions",
    "company_exposure_research_requests",
    "company_exposure_research_events",
    "company_exposure_research_candidates",
    "company_exposure_input_manifests",
    "company_exposure_reservations",
    "company_exposure_reservation_events",
    "company_exposure_provider_attempts",
    "company_exposure_provider_results",
    "company_exposure_artifacts",
    "company_exposure_coverage_items",
)
SEALABLE_TABLES = ()

_REJECT_MUTATION = """
CREATE OR REPLACE FUNCTION company_exposure_reject_mutation()
RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'company_exposure_payload_immutable';
END;
$$ LANGUAGE plpgsql;
"""

_SEAL_ONCE = """
CREATE OR REPLACE FUNCTION company_exposure_seal_once()
RETURNS trigger AS $$
BEGIN
  IF TG_OP = 'DELETE' OR OLD.status = 'sealed' THEN
    RAISE EXCEPTION 'company_exposure_sealed_payload_immutable';
  END IF;
  IF NEW.status <> 'sealed'
     OR NEW.semantic_hash IS NULL
     OR NEW.sealed_at IS NULL
     OR (to_jsonb(NEW) - ARRAY['status','semantic_hash','sealed_at']::text[])
        IS DISTINCT FROM
        (to_jsonb(OLD) - ARRAY['status','semantic_hash','sealed_at']::text[]) THEN
    RAISE EXCEPTION 'company_exposure_payload_immutable';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""


def upgrade():
    op.create_table(
        "company_exposure_resource_pools",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("pool_key", sa.String(length=160), nullable=False),
        sa.Column("unit", sa.String(length=24), nullable=False),
        sa.Column("period", sa.String(length=32), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("capacity", sa.BigInteger(), nullable=True),
        sa.Column("reserved_amount", sa.BigInteger(), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "unit IN ('requests','reported_tokens','currency_amount','blob_bytes')",
            name="ck_cx_pool_unit",
        ),
        sa.CheckConstraint("reserved_amount >= 0", name="ck_cx_pool_reserved"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pool_key", "unit", "period", name="uq_cx_pool_period"),
    )
    op.create_table(
        "company_exposure_runtime_policy_revisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("namespace", sa.String(length=40), nullable=False),
        sa.Column("subject_key", sa.String(length=200), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column("parent_revision_id", sa.Uuid(), nullable=True),
        sa.Column("approval_state", sa.String(length=16), nullable=False),
        sa.Column("principal", sa.String(length=200), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "approval_state IN ('proposed','approved','rejected')",
            name="ck_cx_policy_approval",
        ),
        sa.CheckConstraint(
            "namespace IN ('research_limits','subscription_route','search_cost','acquisition_permission','theme_research','feature_stage')",
            name="ck_cx_policy_namespace",
        ),
        sa.CheckConstraint("revision_number > 0", name="ck_cx_policy_revision"),
        sa.ForeignKeyConstraint(
            ["parent_revision_id"],
            ["company_exposure_runtime_policy_revisions.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "namespace", "subject_key", "revision_number", name="uq_cx_policy_revision"
        ),
    )
    op.create_table(
        "company_exposure_research_requests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("root_request_id", sa.Uuid(), nullable=True),
        sa.Column("parent_request_id", sa.Uuid(), nullable=True),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("requester_principal", sa.String(length=200), nullable=False),
        sa.Column("idempotency_namespace", sa.String(length=200), nullable=False),
        sa.Column("idempotency_key", sa.String(length=200), nullable=False),
        sa.Column("security_id", sa.Integer(), nullable=True),
        sa.Column("issuer_id", sa.Uuid(), nullable=True),
        sa.Column("economic_theme_id", sa.Uuid(), nullable=False),
        sa.Column("market", sa.String(length=8), nullable=True),
        sa.Column("supplied_links", sa.JSON(), nullable=False),
        sa.Column("requested_limits", sa.JSON(), nullable=False),
        sa.Column("policy_revision_ids", sa.JSON(), nullable=False),
        sa.Column("trigger_origin", sa.String(length=80), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind IN ('verify','refresh','discover')", name="ck_cx_request_kind"
        ),
        sa.CheckConstraint(
            "security_id IS NOT NULL OR issuer_id IS NOT NULL OR kind = 'discover'",
            name="ck_cx_request_subject",
        ),
        sa.CheckConstraint(
            "(parent_request_id IS NULL) = (root_request_id IS NULL)",
            name="ck_cx_request_root_parent",
        ),
        sa.ForeignKeyConstraint(
            ["economic_theme_id"], ["economic_themes.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["issuer_id"], ["company_exposure_issuers.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["parent_request_id"],
            ["company_exposure_research_requests.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["root_request_id"],
            ["company_exposure_research_requests.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["security_id"], ["stock_universe.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "idempotency_namespace", "idempotency_key", name="uq_cx_request_idempotency"
        ),
    )
    op.create_index(
        "ix_cx_request_root",
        "company_exposure_research_requests",
        ["root_request_id"],
        unique=False,
    )
    op.create_table(
        "company_exposure_coverage_items",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("stage", sa.String(length=40), nullable=False),
        sa.Column("route", sa.String(length=120), nullable=False),
        sa.Column("subject", sa.String(length=512), nullable=True),
        sa.Column("outcome", sa.String(length=40), nullable=False),
        sa.Column("reason", sa.String(length=120), nullable=True),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["company_exposure_research_requests.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_cx_coverage_request",
        "company_exposure_coverage_items",
        ["request_id"],
        unique=False,
    )
    op.create_table(
        "company_exposure_input_manifests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("stage", sa.String(length=40), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["company_exposure_research_requests.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "request_id", "stage", "input_hash", name="uq_cx_input_manifest"
        ),
    )
    op.create_table(
        "company_exposure_research_candidates",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("root_request_id", sa.Uuid(), nullable=False),
        sa.Column("economic_theme_id", sa.Uuid(), nullable=False),
        sa.Column("issuer_id", sa.Uuid(), nullable=True),
        sa.Column("security_id", sa.Integer(), nullable=True),
        sa.Column("seed_provenance", sa.JSON(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("state", sa.String(length=40), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["root_request_id"],
            ["company_exposure_research_requests.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "company_exposure_research_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(length=40), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("sequence > 0", name="ck_cx_event_sequence"),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["company_exposure_research_requests.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("request_id", "sequence", name="uq_cx_event_sequence"),
    )
    op.create_table(
        "company_exposure_reservations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("pool_id", sa.Uuid(), nullable=False),
        sa.Column("root_request_id", sa.Uuid(), nullable=True),
        sa.Column("root_budget_key", sa.String(length=80), nullable=True),
        sa.Column("unit", sa.String(length=24), nullable=False),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=True),
        sa.Column("period", sa.String(length=32), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("purpose", sa.String(length=80), nullable=False),
        sa.Column("logical_operation_key", sa.String(length=200), nullable=False),
        sa.Column("policy_revision_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(unit = 'currency_amount') = (currency IS NOT NULL)",
            name="ck_cx_reservation_currency",
        ),
        sa.CheckConstraint(
            "unit IN ('requests','reported_tokens','currency_amount','blob_bytes')",
            name="ck_cx_reservation_unit",
        ),
        sa.CheckConstraint("amount >= 0", name="ck_cx_reservation_amount"),
        sa.ForeignKeyConstraint(
            ["pool_id"], ["company_exposure_resource_pools.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["root_request_id"],
            ["company_exposure_research_requests.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_cx_reservation_pool",
        "company_exposure_reservations",
        ["pool_id"],
        unique=False,
    )
    op.create_table(
        "company_exposure_root_budgets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("root_request_id", sa.Uuid(), nullable=False),
        sa.Column("budget_key", sa.String(length=80), nullable=False),
        sa.Column("limit_amount", sa.BigInteger(), nullable=False),
        sa.Column("used_amount", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("used_amount <= limit_amount", name="ck_cx_root_limit"),
        sa.CheckConstraint("used_amount >= 0", name="ck_cx_root_used"),
        sa.ForeignKeyConstraint(
            ["root_request_id"],
            ["company_exposure_research_requests.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("root_request_id", "budget_key", name="uq_cx_root_budget"),
    )
    op.create_table(
        "company_exposure_work_items",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("root_request_id", sa.Uuid(), nullable=False),
        sa.Column("stage", sa.String(length=40), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("policy_bundle_version", sa.String(length=80), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("pause_reason", sa.String(length=80), nullable=True),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_token", sa.Uuid(), nullable=True),
        sa.Column("lease_owner", sa.String(length=200), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claim_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('pending','leased','retryable','paused','completed','cancelled','failed')",
            name="ck_cx_work_status",
        ),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["company_exposure_research_requests.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "request_id",
            "stage",
            "input_hash",
            "policy_bundle_version",
            name="uq_cx_work_key",
        ),
    )
    op.create_index(
        "ix_cx_work_claim",
        "company_exposure_work_items",
        ["status", "available_at"],
        unique=False,
    )
    op.create_table(
        "company_exposure_provider_attempts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("logical_operation_key", sa.String(length=200), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=True),
        sa.Column("root_request_id", sa.Uuid(), nullable=True),
        sa.Column("operation", sa.String(length=80), nullable=False),
        sa.Column("route", sa.String(length=80), nullable=False),
        sa.Column("model", sa.String(length=120), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("policy_hash", sa.String(length=64), nullable=False),
        sa.Column("reservation_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("attempt_number > 0", name="ck_cx_attempt_number"),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["company_exposure_research_requests.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["reservation_id"],
            ["company_exposure_reservations.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "logical_operation_key", "attempt_number", name="uq_cx_attempt_number"
        ),
    )
    op.create_table(
        "company_exposure_reservation_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("reservation_id", sa.Uuid(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(length=24), nullable=False),
        sa.Column("dispatch_phase", sa.String(length=16), nullable=True),
        sa.Column("actual_amount", sa.BigInteger(), nullable=True),
        sa.Column("actual_known", sa.Boolean(), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "dispatch_phase IS NULL OR dispatch_phase IN ('pre_dispatch','dispatched','uncertain')",
            name="ck_cx_reservation_phase",
        ),
        sa.CheckConstraint(
            "state IN ('reserved','dispatched','released','reconciled','uncertain','expired_uncertain','paused_allowance')",
            name="ck_cx_reservation_state",
        ),
        sa.CheckConstraint("sequence > 0", name="ck_cx_reservation_event_sequence"),
        sa.ForeignKeyConstraint(
            ["reservation_id"],
            ["company_exposure_reservations.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "reservation_id", "sequence", name="uq_cx_reservation_event_sequence"
        ),
    )
    op.create_table(
        "company_exposure_provider_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("attempt_id", sa.Uuid(), nullable=False),
        sa.Column("outcome", sa.String(length=24), nullable=False),
        sa.Column("dispatch_phase", sa.String(length=16), nullable=False),
        sa.Column("failure_code", sa.String(length=80), nullable=True),
        sa.Column("provider_request_id", sa.String(length=200), nullable=True),
        sa.Column("reported_usage", sa.JSON(), nullable=False),
        sa.Column("usage_known", sa.Boolean(), nullable=False),
        sa.Column("response_hash", sa.String(length=64), nullable=True),
        sa.Column("retry_after_seconds", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "dispatch_phase IN ('pre_dispatch','dispatched','uncertain')",
            name="ck_cx_result_phase",
        ),
        sa.CheckConstraint(
            "outcome IN ('success','retryable_failure','uncertain','terminal_failure')",
            name="ck_cx_result_outcome",
        ),
        sa.ForeignKeyConstraint(
            ["attempt_id"],
            ["company_exposure_provider_attempts.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("attempt_id"),
    )
    op.create_table(
        "company_exposure_artifacts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("operation", sa.String(length=80), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("policy_hash", sa.String(length=64), nullable=False),
        sa.Column("model_identity", sa.String(length=160), nullable=False),
        sa.Column("result_id", sa.Uuid(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["result_id"], ["company_exposure_provider_results.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "operation",
            "input_hash",
            "policy_hash",
            "model_identity",
            name="uq_cx_artifact_key",
        ),
    )

    with op.batch_alter_table("company_exposure_passage_derivatives") as batch:
        batch.create_foreign_key(
            "fk_cx_derivative_attempt",
            "company_exposure_provider_attempts",
            ["provider_attempt_id"],
            ["id"],
            ondelete="RESTRICT",
        )

    if op.get_bind().dialect.name == "postgresql":
        op.execute(sa.text(_REJECT_MUTATION))
        op.execute(sa.text(_SEAL_ONCE))
        for table in APPEND_ONLY_TABLES:
            op.execute(
                sa.text(
                    f"CREATE TRIGGER trg_{table}_append_only "
                    f"BEFORE UPDATE OR DELETE ON {table} "
                    "FOR EACH ROW EXECUTE FUNCTION company_exposure_reject_mutation()"
                )
            )
        for table in SEALABLE_TABLES:
            op.execute(
                sa.text(
                    f"CREATE TRIGGER trg_{table}_seal_once "
                    f"BEFORE UPDATE OR DELETE ON {table} "
                    "FOR EACH ROW EXECUTE FUNCTION company_exposure_seal_once()"
                )
            )


def downgrade():
    # Disposable/rehearsal databases only: production rollback retains evidence.
    with op.batch_alter_table("company_exposure_passage_derivatives") as batch:
        batch.drop_constraint("fk_cx_derivative_attempt", type_="foreignkey")
    op.drop_table("company_exposure_artifacts")
    op.drop_table("company_exposure_provider_results")
    op.drop_table("company_exposure_reservation_events")
    op.drop_table("company_exposure_provider_attempts")
    op.drop_table("company_exposure_work_items")
    op.drop_table("company_exposure_root_budgets")
    op.drop_table("company_exposure_reservations")
    op.drop_table("company_exposure_research_events")
    op.drop_table("company_exposure_research_candidates")
    op.drop_table("company_exposure_input_manifests")
    op.drop_table("company_exposure_coverage_items")
    op.drop_table("company_exposure_research_requests")
    op.drop_table("company_exposure_runtime_policy_revisions")
    op.drop_table("company_exposure_resource_pools")
