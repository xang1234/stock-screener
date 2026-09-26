"""Add company-exposure claims, assessments and use holds (Task 03)."""

import sqlalchemy as sa

from alembic import op

revision = "20260926_0060"
down_revision = "20260925_0059"
branch_labels = None
depends_on = None

APPEND_ONLY_TABLES = (
    "company_exposure_claims",
    "company_exposure_claim_evidence_links",
    "company_exposure_materiality_measures",
    "company_exposure_assessments",
    "company_exposure_assessment_claim_selections",
    "company_exposure_use_holds",
)
SEALABLE_TABLES = (
    "company_exposure_claim_revisions",
    "company_exposure_assessment_revisions",
)
# (child table, sealable parent table, foreign-key column)
SEALED_CHILDREN = (
    (
        "company_exposure_claim_evidence_links",
        "company_exposure_claim_revisions",
        "claim_revision_id",
    ),
    (
        "company_exposure_materiality_measures",
        "company_exposure_claim_revisions",
        "claim_revision_id",
    ),
    (
        "company_exposure_assessment_claim_selections",
        "company_exposure_assessment_revisions",
        "assessment_revision_id",
    ),
)

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

_GUARD_SEALED_PARENT = """
CREATE OR REPLACE FUNCTION company_exposure_guard_sealed_parent()
RETURNS trigger AS $$
DECLARE
  parent_id uuid;
  parent_sealed boolean;
BEGIN
  parent_id := (to_jsonb(NEW) ->> TG_ARGV[1])::uuid;
  EXECUTE format(
    'SELECT EXISTS (SELECT 1 FROM %I WHERE id = $1 AND status = ''sealed'' FOR SHARE)',
    TG_ARGV[0]
  ) INTO parent_sealed USING parent_id;
  IF parent_sealed THEN
    RAISE EXCEPTION 'company_exposure_sealed_payload_immutable';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""


def upgrade():
    op.create_table(
        "company_exposure_use_holds",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("subject_kind", sa.String(length=32), nullable=False),
        sa.Column("subject_id", sa.String(length=64), nullable=False),
        sa.Column("hold_kind", sa.String(length=40), nullable=False),
        sa.Column("stream_key", sa.String(length=200), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=8), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("actor", sa.String(length=200), nullable=False),
        sa.Column("lifted_hold_id", sa.Uuid(), nullable=True),
        sa.Column("lift_support", sa.JSON(), nullable=True),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "action = 'apply' OR (lifted_hold_id IS NOT NULL AND lift_support IS NOT NULL)",
            name="ck_cx_hold_lift_requires_support",
        ),
        sa.CheckConstraint("action IN ('apply','lift')", name="ck_cx_hold_action"),
        sa.CheckConstraint(
            "hold_kind IN ('stale','undated','disputed','conflict','identity','policy','language_ambiguity','source_integrity','exposure_end','manual')",
            name="ck_cx_hold_kind",
        ),
        sa.CheckConstraint(
            "subject_kind IN ('claim','claim_revision','issuer','issuer_link','assessment','security')",
            name="ck_cx_hold_subject",
        ),
        sa.CheckConstraint("revision_number > 0", name="ck_cx_hold_revision"),
        sa.ForeignKeyConstraint(
            ["lifted_hold_id"], ["company_exposure_use_holds.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "stream_key", "revision_number", name="uq_cx_hold_revision"
        ),
    )
    op.create_index(
        "ix_cx_hold_subject",
        "company_exposure_use_holds",
        ["subject_kind", "subject_id"],
        unique=False,
    )
    op.create_table(
        "company_exposure_assessments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("issuer_id", sa.Uuid(), nullable=False),
        sa.Column("economic_theme_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["economic_theme_id"], ["economic_themes.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["issuer_id"], ["company_exposure_issuers.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "id", "issuer_id", "economic_theme_id", name="uq_cx_assessment_scope"
        ),
        sa.UniqueConstraint(
            "issuer_id", "economic_theme_id", name="uq_cx_assessment_pair"
        ),
    )
    op.create_table(
        "company_exposure_claims",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("proposition_key", sa.String(length=64), nullable=False),
        sa.Column("issuer_id", sa.Uuid(), nullable=False),
        sa.Column("economic_theme_id", sa.Uuid(), nullable=False),
        sa.Column("claim_kind", sa.String(length=32), nullable=False),
        sa.Column("product_or_activity_key", sa.String(length=200), nullable=False),
        sa.Column("reporting_scope", sa.String(length=32), nullable=False),
        sa.Column("scope_label", sa.String(length=200), nullable=True),
        sa.Column("normalized_proposition", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(reporting_scope = 'segment_or_subsidiary') = (scope_label IS NOT NULL)",
            name="ck_cx_claim_scope_label",
        ),
        sa.CheckConstraint(
            "claim_kind IN ('participation','role','product_application','customer_relationship','commercial_status','materiality','exposure_end')",
            name="ck_cx_claim_kind",
        ),
        sa.CheckConstraint(
            "reporting_scope IN ('issuer_consolidated','issuer_standalone','segment_or_subsidiary')",
            name="ck_cx_claim_scope",
        ),
        sa.ForeignKeyConstraint(
            ["economic_theme_id"], ["economic_themes.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["issuer_id"], ["company_exposure_issuers.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "id", "issuer_id", "economic_theme_id", name="uq_cx_claim_scope"
        ),
        sa.UniqueConstraint("proposition_key"),
    )
    op.create_index(
        "ix_cx_claim_pair",
        "company_exposure_claims",
        ["issuer_id", "economic_theme_id"],
        unique=False,
    )
    op.create_table(
        "company_exposure_assessment_revisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("assessment_id", sa.Uuid(), nullable=False),
        sa.Column("issuer_id", sa.Uuid(), nullable=False),
        sa.Column("economic_theme_id", sa.Uuid(), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("input_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("input_manifest", sa.JSON(), nullable=False),
        sa.Column("prior_revision_id", sa.Uuid(), nullable=True),
        sa.Column("request_id", sa.Uuid(), nullable=True),
        sa.Column("coverage", sa.JSON(), nullable=False),
        sa.Column("unresolved_questions", sa.JSON(), nullable=False),
        sa.Column("conflicts", sa.JSON(), nullable=False),
        sa.Column("assessed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("semantic_hash", sa.String(length=64), nullable=True),
        sa.Column("sealed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('unsealed','sealed')", name="ck_cx_assessment_status"
        ),
        sa.CheckConstraint("revision_number > 0", name="ck_cx_assessment_revision"),
        sa.ForeignKeyConstraint(
            ["assessment_id", "issuer_id", "economic_theme_id"],
            [
                "company_exposure_assessments.id",
                "company_exposure_assessments.issuer_id",
                "company_exposure_assessments.economic_theme_id",
            ],
            name="fk_cx_assessment_revision_scope",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["prior_revision_id"],
            ["company_exposure_assessment_revisions.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["company_exposure_research_requests.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "assessment_id", "input_manifest_hash", name="uq_cx_assessment_replay"
        ),
        sa.UniqueConstraint(
            "assessment_id", "revision_number", name="uq_cx_assessment_revision"
        ),
        sa.UniqueConstraint(
            "id",
            "issuer_id",
            "economic_theme_id",
            name="uq_cx_assessment_revision_scope",
        ),
    )
    op.create_table(
        "company_exposure_claim_revisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("claim_id", sa.Uuid(), nullable=False),
        sa.Column("issuer_id", sa.Uuid(), nullable=False),
        sa.Column("economic_theme_id", sa.Uuid(), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("evaluated_theme_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=80), nullable=True),
        sa.Column("commercial_status", sa.String(length=32), nullable=False),
        sa.Column("support_basis", sa.String(length=32), nullable=False),
        sa.Column("conclusion", sa.String(length=16), nullable=False),
        sa.Column("freshness_state", sa.String(length=16), nullable=False),
        sa.Column("hold_reasons", sa.JSON(), nullable=False),
        sa.Column("effective_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("effective_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reporting_period", sa.String(length=64), nullable=True),
        sa.Column("source_publication_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "first_available_to_app_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column("assessed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("supported_as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fresh_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("verification_policy_version", sa.String(length=80), nullable=False),
        sa.Column("model_attempt_refs", sa.JSON(), nullable=False),
        sa.Column("supersedes_revision_id", sa.Uuid(), nullable=True),
        sa.Column("supersession_kind", sa.String(length=24), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("semantic_hash", sa.String(length=64), nullable=True),
        sa.Column("sealed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "commercial_status IN ('research','announced','qualification','commercially_available','shipping_or_operating','discontinued','unknown')",
            name="ck_cx_claim_commercial",
        ),
        sa.CheckConstraint(
            "conclusion IN ('supported','contradicted','disputed','unknown')",
            name="ck_cx_claim_conclusion",
        ),
        sa.CheckConstraint(
            "freshness_state IN ('current','stale','undated')",
            name="ck_cx_claim_freshness",
        ),
        sa.CheckConstraint(
            "status IN ('unsealed','sealed')", name="ck_cx_claim_revision_status"
        ),
        sa.CheckConstraint(
            "supersession_kind IS NULL OR supersession_kind IN ('correction','supersession')",
            name="ck_cx_claim_supersession",
        ),
        sa.CheckConstraint(
            "support_basis IN ('primary_explicit','primary_synthesis','secondary_reported','inferred_unverified','unresolved')",
            name="ck_cx_claim_support",
        ),
        sa.CheckConstraint(
            "(supersedes_revision_id IS NULL) = (supersession_kind IS NULL)",
            name="ck_cx_claim_supersession_pair",
        ),
        sa.CheckConstraint("revision_number > 0", name="ck_cx_claim_revision_number"),
        sa.ForeignKeyConstraint(
            ["claim_id", "issuer_id", "economic_theme_id"],
            [
                "company_exposure_claims.id",
                "company_exposure_claims.issuer_id",
                "company_exposure_claims.economic_theme_id",
            ],
            name="fk_cx_claim_revision_scope",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["supersedes_revision_id"],
            ["company_exposure_claim_revisions.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("claim_id", "revision_number", name="uq_cx_claim_revision"),
        sa.UniqueConstraint(
            "id",
            "claim_id",
            "issuer_id",
            "economic_theme_id",
            name="uq_cx_claim_revision_scope",
        ),
    )
    op.create_table(
        "company_exposure_assessment_claim_selections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("assessment_revision_id", sa.Uuid(), nullable=False),
        sa.Column("claim_id", sa.Uuid(), nullable=False),
        sa.Column("claim_revision_id", sa.Uuid(), nullable=False),
        sa.Column("issuer_id", sa.Uuid(), nullable=False),
        sa.Column("economic_theme_id", sa.Uuid(), nullable=False),
        sa.Column("carried_forward", sa.Boolean(), nullable=False),
        sa.Column("selection_reason", sa.String(length=80), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["assessment_revision_id", "issuer_id", "economic_theme_id"],
            [
                "company_exposure_assessment_revisions.id",
                "company_exposure_assessment_revisions.issuer_id",
                "company_exposure_assessment_revisions.economic_theme_id",
            ],
            name="fk_cx_selection_assessment_scope",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["claim_revision_id", "claim_id", "issuer_id", "economic_theme_id"],
            [
                "company_exposure_claim_revisions.id",
                "company_exposure_claim_revisions.claim_id",
                "company_exposure_claim_revisions.issuer_id",
                "company_exposure_claim_revisions.economic_theme_id",
            ],
            name="fk_cx_selection_claim_scope",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "assessment_revision_id", "claim_id", name="uq_cx_selection_claim"
        ),
    )
    op.create_table(
        "company_exposure_materiality_measures",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("claim_revision_id", sa.Uuid(), nullable=False),
        sa.Column("basis", sa.String(length=16), nullable=False),
        sa.Column("metric", sa.String(length=40), nullable=True),
        sa.Column("value_low", sa.String(length=64), nullable=True),
        sa.Column("value_high", sa.String(length=64), nullable=True),
        sa.Column("qualitative_label", sa.String(length=32), nullable=True),
        sa.Column("unit", sa.String(length=40), nullable=True),
        sa.Column("currency", sa.String(length=8), nullable=True),
        sa.Column("denominator_definition", sa.Text(), nullable=True),
        sa.Column("reporting_scope", sa.String(length=32), nullable=False),
        sa.Column("scope_label", sa.String(length=200), nullable=True),
        sa.Column("period", sa.String(length=64), nullable=True),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("original_precision", sa.String(length=40), nullable=True),
        sa.Column("formula", sa.JSON(), nullable=False),
        sa.Column("operand_refs", sa.JSON(), nullable=False),
        sa.Column("supporting_passage_ids", sa.JSON(), nullable=False),
        sa.Column("raw_reported", sa.JSON(), nullable=False),
        sa.Column("hold_reasons", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "basis <> 'qualitative' OR qualitative_label IS NOT NULL",
            name="ck_cx_materiality_qualitative",
        ),
        sa.CheckConstraint(
            "basis IN ('disclosed','calculated','qualitative','unknown')",
            name="ck_cx_materiality_basis",
        ),
        sa.CheckConstraint(
            "basis NOT IN ('disclosed','calculated') OR (metric IS NOT NULL AND value_low IS NOT NULL AND period IS NOT NULL)",
            name="ck_cx_materiality_numeric",
        ),
        sa.CheckConstraint(
            "qualitative_label IS NULL OR qualitative_label IN ('core_business','explicitly_material','explicitly_limited','unknown')",
            name="ck_cx_materiality_label",
        ),
        sa.CheckConstraint(
            "reporting_scope IN ('issuer_consolidated','issuer_standalone','segment_or_subsidiary')",
            name="ck_cx_materiality_scope",
        ),
        sa.ForeignKeyConstraint(
            ["claim_revision_id"],
            ["company_exposure_claim_revisions.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("claim_revision_id"),
    )
    op.create_table(
        "company_exposure_claim_evidence_links",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("claim_revision_id", sa.Uuid(), nullable=False),
        sa.Column("direction", sa.String(length=16), nullable=False),
        sa.Column("evidence_role", sa.String(length=48), nullable=False),
        sa.Column("passage_id", sa.Uuid(), nullable=True),
        sa.Column("derivative_id", sa.Uuid(), nullable=True),
        sa.Column("premise_claim_revision_id", sa.Uuid(), nullable=True),
        sa.Column("quote", sa.Text(), nullable=True),
        sa.Column("locator", sa.JSON(), nullable=False),
        sa.Column("attribution", sa.JSON(), nullable=False),
        sa.Column("join_scope", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "derivative_id IS NULL OR evidence_role = 'derivative_not_independent_source'",
            name="ck_cx_link_derivative_role",
        ),
        sa.CheckConstraint(
            "direction IN ('supporting','conflicting')", name="ck_cx_link_direction"
        ),
        sa.CheckConstraint(
            "evidence_role <> 'original_primary' OR passage_id IS NOT NULL",
            name="ck_cx_link_primary_is_passage",
        ),
        sa.CheckConstraint(
            "evidence_role IN ('original_primary','original_secondary','derivative_not_independent_source','retrieval_aid_only')",
            name="ck_cx_link_role",
        ),
        sa.CheckConstraint(
            "(CASE WHEN passage_id IS NULL THEN 0 ELSE 1 END) + (CASE WHEN derivative_id IS NULL THEN 0 ELSE 1 END) + (CASE WHEN premise_claim_revision_id IS NULL THEN 0 ELSE 1 END) = 1",
            name="ck_cx_link_one_target",
        ),
        sa.CheckConstraint(
            "premise_claim_revision_id IS NULL OR premise_claim_revision_id <> claim_revision_id",
            name="ck_cx_link_no_self_premise",
        ),
        sa.ForeignKeyConstraint(
            ["claim_revision_id"],
            ["company_exposure_claim_revisions.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["derivative_id"],
            ["company_exposure_passage_derivatives.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["passage_id"], ["company_exposure_passages.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["premise_claim_revision_id"],
            ["company_exposure_claim_revisions.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_cx_link_revision",
        "company_exposure_claim_evidence_links",
        ["claim_revision_id"],
        unique=False,
    )

    if op.get_bind().dialect.name == "postgresql":
        op.execute(sa.text(_REJECT_MUTATION))
        op.execute(sa.text(_SEAL_ONCE))
        op.execute(sa.text(_GUARD_SEALED_PARENT))
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
        for child, parent, column in SEALED_CHILDREN:
            op.execute(
                sa.text(
                    f"CREATE TRIGGER trg_{child}_parent_open "
                    f"BEFORE INSERT ON {child} "
                    "FOR EACH ROW EXECUTE FUNCTION "
                    f"company_exposure_guard_sealed_parent('{parent}', '{column}')"
                )
            )


def downgrade():
    # Disposable/rehearsal databases only: production rollback retains evidence.
    op.drop_table("company_exposure_claim_evidence_links")
    op.drop_table("company_exposure_materiality_measures")
    op.drop_table("company_exposure_assessment_claim_selections")
    op.drop_table("company_exposure_claim_revisions")
    op.drop_table("company_exposure_assessment_revisions")
    op.drop_table("company_exposure_claims")
    op.drop_table("company_exposure_assessments")
    op.drop_table("company_exposure_use_holds")
