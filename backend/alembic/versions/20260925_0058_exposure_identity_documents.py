"""Add company-exposure issuer registry and original-document evidence (Task 01)."""

import sqlalchemy as sa

from alembic import op

revision = "20260925_0058"
down_revision = "20260925_0057"
branch_labels = None
depends_on = None

APPEND_ONLY_TABLES = (
    "company_exposure_issuers",
    "company_exposure_issuer_identifier_revisions",
    "company_exposure_issuer_security_links",
    "company_exposure_legacy_attestation_bridges",
    "company_exposure_documents",
    "company_exposure_document_revisions",
    "company_exposure_document_captures",
    "company_exposure_document_relations",
    "company_exposure_passages",
    "company_exposure_passage_derivatives",
    "company_exposure_evidence_tombstones",
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
        "company_exposure_issuers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provenance", sa.JSON(), nullable=False),
        sa.Column("created_by", sa.String(length=200), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "company_exposure_documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("identity_key", sa.String(length=512), nullable=False),
        sa.Column("provider", sa.String(length=80), nullable=False),
        sa.Column("provider_document_id", sa.String(length=256), nullable=True),
        sa.Column("canonical_url", sa.Text(), nullable=True),
        sa.Column("verified_origin", sa.String(length=255), nullable=True),
        sa.Column("publisher", sa.String(length=255), nullable=True),
        sa.Column("market", sa.String(length=8), nullable=True),
        sa.Column("source_kind", sa.String(length=80), nullable=False),
        sa.Column("issuer_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["issuer_id"], ["company_exposure_issuers.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("identity_key"),
    )
    op.create_index(
        "ix_cx_document_issuer",
        "company_exposure_documents",
        ["issuer_id"],
        unique=False,
    )
    op.create_table(
        "company_exposure_issuer_identifier_revisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("issuer_id", sa.Uuid(), nullable=False),
        sa.Column("market", sa.String(length=8), nullable=False),
        sa.Column("scheme", sa.String(length=40), nullable=False),
        sa.Column("value", sa.String(length=120), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(length=24), nullable=False),
        sa.Column("acceptance_policy", sa.String(length=48), nullable=True),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("actor", sa.String(length=200), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "acceptance_policy IS NULL OR acceptance_policy IN ('administrator_reviewed','official_registry_single_listing','legacy_attestation_import')",
            name="ck_cx_identifier_policy",
        ),
        sa.CheckConstraint(
            "state IN ('proposed','accepted','rejected','review_required')",
            name="ck_cx_identifier_state",
        ),
        sa.CheckConstraint("revision_number > 0", name="ck_cx_identifier_revision"),
        sa.ForeignKeyConstraint(
            ["issuer_id"], ["company_exposure_issuers.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "market",
            "scheme",
            "value",
            "revision_number",
            name="uq_cx_identifier_revision",
        ),
    )
    op.create_index(
        "ix_cx_identifier_issuer",
        "company_exposure_issuer_identifier_revisions",
        ["issuer_id"],
        unique=False,
    )
    op.create_table(
        "company_exposure_issuer_security_links",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("security_id", sa.Integer(), nullable=False),
        sa.Column("issuer_id", sa.Uuid(), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(length=24), nullable=False),
        sa.Column("acceptance_policy", sa.String(length=48), nullable=False),
        sa.Column("link_scope", sa.String(length=40), nullable=False),
        sa.Column("actor", sa.String(length=200), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("prior_revision_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "acceptance_policy IN ('administrator_reviewed','official_registry_single_listing','legacy_attestation_import')",
            name="ck_cx_link_policy",
        ),
        sa.CheckConstraint(
            "state IN ('proposed','accepted','rejected','review_required')",
            name="ck_cx_link_state",
        ),
        sa.CheckConstraint("revision_number > 0", name="ck_cx_link_revision"),
        sa.ForeignKeyConstraint(
            ["issuer_id"], ["company_exposure_issuers.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["prior_revision_id"],
            ["company_exposure_issuer_security_links.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["security_id"], ["stock_universe.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "security_id", "revision_number", name="uq_cx_link_security_revision"
        ),
    )
    op.create_index(
        "ix_cx_link_issuer",
        "company_exposure_issuer_security_links",
        ["issuer_id"],
        unique=False,
    )
    op.create_table(
        "company_exposure_document_relations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("from_document_id", sa.Uuid(), nullable=False),
        sa.Column("to_document_id", sa.Uuid(), nullable=False),
        sa.Column("relation", sa.String(length=40), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("scope", sa.JSON(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("actor", sa.String(length=200), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "relation IN ('translation','exact_mirror','correction','supersession','amendment','unknown_duplicate')",
            name="ck_cx_relation_kind",
        ),
        sa.CheckConstraint(
            "from_document_id <> to_document_id", name="ck_cx_relation_distinct"
        ),
        sa.CheckConstraint("revision_number > 0", name="ck_cx_relation_revision"),
        sa.ForeignKeyConstraint(
            ["from_document_id"], ["company_exposure_documents.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["to_document_id"], ["company_exposure_documents.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "from_document_id",
            "to_document_id",
            "relation",
            "revision_number",
            name="uq_cx_relation_revision",
        ),
    )
    op.create_table(
        "company_exposure_document_revisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("media_type", sa.String(length=120), nullable=False),
        sa.Column("byte_length", sa.BigInteger(), nullable=False),
        sa.Column("blob_key", sa.String(length=255), nullable=False),
        sa.Column("language", sa.String(length=16), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reporting_period", sa.String(length=64), nullable=True),
        sa.Column("effective_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("first_available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("correction_identity", sa.JSON(), nullable=False),
        sa.Column("document_metadata", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("byte_length >= 0", name="ck_cx_revision_bytes"),
        sa.ForeignKeyConstraint(
            ["document_id"], ["company_exposure_documents.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id", "content_hash", name="uq_cx_document_revision_content"
        ),
        sa.UniqueConstraint("id", "content_hash", name="uq_cx_document_revision_hash"),
        sa.UniqueConstraint("id", "document_id", name="uq_cx_document_revision_owner"),
    )
    op.create_table(
        "company_exposure_legacy_attestation_bridges",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("configuration_version", sa.Integer(), nullable=False),
        sa.Column("configuration_hash", sa.String(length=64), nullable=False),
        sa.Column("policy_version", sa.String(length=80), nullable=False),
        sa.Column("legacy_company_id", sa.String(length=500), nullable=False),
        sa.Column("legacy_symbol", sa.String(length=20), nullable=False),
        sa.Column("verified_at", sa.String(length=64), nullable=False),
        sa.Column("security_id", sa.Integer(), nullable=False),
        sa.Column("verification_reference", sa.Text(), nullable=True),
        sa.Column("issuer_id", sa.Uuid(), nullable=False),
        sa.Column("link_revision_id", sa.Uuid(), nullable=False),
        sa.Column("audit_provenance", sa.JSON(), nullable=False),
        sa.Column("actor", sa.String(length=200), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["issuer_id"], ["company_exposure_issuers.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["link_revision_id"],
            ["company_exposure_issuer_security_links.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["security_id"], ["stock_universe.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "configuration_version",
            "configuration_hash",
            "legacy_company_id",
            "security_id",
            name="uq_cx_legacy_bridge_import",
        ),
    )
    op.create_table(
        "company_exposure_document_captures",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("revision_id", sa.Uuid(), nullable=True),
        sa.Column("outcome", sa.String(length=40), nullable=False),
        sa.Column("changed", sa.Boolean(), nullable=False),
        sa.Column("sanitized_url", sa.Text(), nullable=True),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("byte_length", sa.BigInteger(), nullable=True),
        sa.Column("observed_metadata", sa.JSON(), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["company_exposure_documents.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["revision_id", "document_id"],
            [
                "company_exposure_document_revisions.id",
                "company_exposure_document_revisions.document_id",
            ],
            name="fk_cx_capture_own_revision",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_cx_capture_document",
        "company_exposure_document_captures",
        ["document_id", "retrieved_at"],
        unique=False,
    )
    op.create_table(
        "company_exposure_evidence_tombstones",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_revision_id", sa.Uuid(), nullable=True),
        sa.Column("blob_key", sa.String(length=255), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("byte_length", sa.BigInteger(), nullable=False),
        sa.Column("reason", sa.String(length=80), nullable=False),
        sa.Column("authority", sa.String(length=200), nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "reason IN ('legal_removal','unreferenced_gc','abandoned_scratch','operator_archive')",
            name="ck_cx_tombstone_reason",
        ),
        sa.ForeignKeyConstraint(
            ["document_revision_id"],
            ["company_exposure_document_revisions.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "company_exposure_passages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_revision_id", sa.Uuid(), nullable=False),
        sa.Column("revision_content_hash", sa.String(length=64), nullable=False),
        sa.Column("preparation_policy", sa.String(length=80), nullable=False),
        sa.Column("extractor_version", sa.String(length=80), nullable=False),
        sa.Column("locator", sa.JSON(), nullable=False),
        sa.Column("locator_hash", sa.String(length=64), nullable=False),
        sa.Column("original_text", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("context", sa.JSON(), nullable=False),
        sa.Column("language", sa.String(length=16), nullable=True),
        sa.Column("page_index", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_revision_id", "revision_content_hash"],
            [
                "company_exposure_document_revisions.id",
                "company_exposure_document_revisions.content_hash",
            ],
            name="fk_cx_passage_revision_hash",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_revision_id",
            "preparation_policy",
            "locator_hash",
            name="uq_cx_passage_locator",
        ),
    )
    op.create_table(
        "company_exposure_passage_derivatives",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("passage_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("policy_version", sa.String(length=80), nullable=False),
        sa.Column("model_identity", sa.String(length=160), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("parent_text_hash", sa.String(length=64), nullable=False),
        sa.Column("output_text", sa.Text(), nullable=False),
        sa.Column("output", sa.JSON(), nullable=False),
        sa.Column("uncertainty", sa.JSON(), nullable=False),
        sa.Column("evidence_role", sa.String(length=48), nullable=False),
        sa.Column("provider_attempt_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "evidence_role = 'derivative_not_independent_source'",
            name="ck_cx_derivative_role",
        ),
        sa.CheckConstraint(
            "kind IN ('translation','page_image_reading')", name="ck_cx_derivative_kind"
        ),
        sa.ForeignKeyConstraint(
            ["passage_id"], ["company_exposure_passages.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "passage_id",
            "kind",
            "policy_version",
            "model_identity",
            "input_hash",
            name="uq_cx_derivative_input",
        ),
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
    op.drop_table("company_exposure_passage_derivatives")
    op.drop_table("company_exposure_passages")
    op.drop_table("company_exposure_evidence_tombstones")
    op.drop_table("company_exposure_document_captures")
    op.drop_table("company_exposure_legacy_attestation_bridges")
    op.drop_table("company_exposure_document_revisions")
    op.drop_table("company_exposure_document_relations")
    op.drop_table("company_exposure_issuer_security_links")
    op.drop_table("company_exposure_issuer_identifier_revisions")
    op.drop_table("company_exposure_documents")
    op.drop_table("company_exposure_issuers")
