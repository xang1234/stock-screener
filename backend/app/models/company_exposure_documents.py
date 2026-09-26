"""Original documents, immutable content revisions and evidence locators.

* A document is a stable original-source identity (filing accession,
  provider document ID, or a canonical URL bound to its verified origin).
* A revision is unique by ``(document_id, content_hash)``: identical bytes
  retrieved again add a capture event, never another revision.
* Capture order is audit order, not precedence; corrections, translations
  and mirrors are explicit relation revisions.
* Passages bind a revision hash, preparation policy and canonical locator.
  Derivatives (translations, page-image readings) point at their passage and
  can never become primary evidence.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)

from app.database import Base
from app.models.company_exposure_common import append_only, created_at, uuid_pk

_RELATIONS = (
    "'translation','exact_mirror','correction','supersession',"
    "'amendment','unknown_duplicate'"
)
_DERIVATIVE_KINDS = "'translation','page_image_reading'"


@append_only
class ExposureDocument(Base):
    __tablename__ = "company_exposure_documents"

    id = uuid_pk()
    identity_key = Column(String(512), nullable=False, unique=True)
    provider = Column(String(80), nullable=False)
    provider_document_id = Column(String(256), nullable=True)
    canonical_url = Column(Text, nullable=True)
    verified_origin = Column(String(255), nullable=True)
    publisher = Column(String(255), nullable=True)
    market = Column(String(8), nullable=True)
    source_kind = Column(String(80), nullable=False)
    issuer_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_issuers.id", ondelete="RESTRICT"),
        nullable=True,
    )
    created_at = created_at()

    __table_args__ = (Index("ix_cx_document_issuer", "issuer_id"),)


@append_only
class ExposureDocumentRevision(Base):
    __tablename__ = "company_exposure_document_revisions"

    id = uuid_pk()
    document_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_documents.id", ondelete="RESTRICT"),
        nullable=False,
    )
    content_hash = Column(String(64), nullable=False)
    media_type = Column(String(120), nullable=False)
    byte_length = Column(BigInteger, nullable=False)
    blob_key = Column(String(255), nullable=False)
    language = Column(String(16), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    reporting_period = Column(String(64), nullable=True)
    effective_at = Column(DateTime(timezone=True), nullable=True)
    first_available_at = Column(DateTime(timezone=True), nullable=False)
    correction_identity = Column(JSON, nullable=False)
    document_metadata = Column(JSON, nullable=False)
    created_at = created_at()

    __table_args__ = (
        CheckConstraint("byte_length >= 0", name="ck_cx_revision_bytes"),
        UniqueConstraint(
            "document_id", "content_hash", name="uq_cx_document_revision_content"
        ),
        # Composite targets: a capture can only cite a revision of its own
        # document, and a passage only the exact content hash it was cut from.
        UniqueConstraint("id", "document_id", name="uq_cx_document_revision_owner"),
        UniqueConstraint("id", "content_hash", name="uq_cx_document_revision_hash"),
    )


@append_only
class DocumentCaptureEvent(Base):
    """One retrieval/check attempt. Never advances business-evidence dates."""

    __tablename__ = "company_exposure_document_captures"

    id = uuid_pk()
    document_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_documents.id", ondelete="RESTRICT"),
        nullable=False,
    )
    revision_id = Column(Uuid(as_uuid=True), nullable=True)
    outcome = Column(String(40), nullable=False)
    changed = Column(Boolean, nullable=False, default=False)
    sanitized_url = Column(Text, nullable=True)
    http_status = Column(Integer, nullable=True)
    content_hash = Column(String(64), nullable=True)
    byte_length = Column(BigInteger, nullable=True)
    observed_metadata = Column(JSON, nullable=False)
    retrieved_at = Column(DateTime(timezone=True), nullable=False)
    created_at = created_at()

    __table_args__ = (
        ForeignKeyConstraint(
            ["revision_id", "document_id"],
            [
                "company_exposure_document_revisions.id",
                "company_exposure_document_revisions.document_id",
            ],
            name="fk_cx_capture_own_revision",
            ondelete="RESTRICT",
        ),
        Index("ix_cx_capture_document", "document_id", "retrieved_at"),
    )


@append_only
class DocumentRelationRevision(Base):
    __tablename__ = "company_exposure_document_relations"

    id = uuid_pk()
    from_document_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_documents.id", ondelete="RESTRICT"),
        nullable=False,
    )
    to_document_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_documents.id", ondelete="RESTRICT"),
        nullable=False,
    )
    relation = Column(String(40), nullable=False)
    revision_number = Column(Integer, nullable=False)
    scope = Column(JSON, nullable=False)
    evidence = Column(JSON, nullable=False)
    actor = Column(String(200), nullable=False)
    created_at = created_at()

    __table_args__ = (
        CheckConstraint(f"relation IN ({_RELATIONS})", name="ck_cx_relation_kind"),
        CheckConstraint(
            "from_document_id <> to_document_id", name="ck_cx_relation_distinct"
        ),
        CheckConstraint("revision_number > 0", name="ck_cx_relation_revision"),
        UniqueConstraint(
            "from_document_id",
            "to_document_id",
            "relation",
            "revision_number",
            name="uq_cx_relation_revision",
        ),
    )


@append_only
class ExposurePassage(Base):
    __tablename__ = "company_exposure_passages"

    id = uuid_pk()
    document_revision_id = Column(Uuid(as_uuid=True), nullable=False)
    revision_content_hash = Column(String(64), nullable=False)
    preparation_policy = Column(String(80), nullable=False)
    extractor_version = Column(String(80), nullable=False)
    locator = Column(JSON, nullable=False)
    locator_hash = Column(String(64), nullable=False)
    original_text = Column(Text, nullable=False)
    text_hash = Column(String(64), nullable=False)
    context = Column(JSON, nullable=False)
    language = Column(String(16), nullable=True)
    page_index = Column(Integer, nullable=True)
    created_at = created_at()

    __table_args__ = (
        ForeignKeyConstraint(
            ["document_revision_id", "revision_content_hash"],
            [
                "company_exposure_document_revisions.id",
                "company_exposure_document_revisions.content_hash",
            ],
            name="fk_cx_passage_revision_hash",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "document_revision_id",
            "preparation_policy",
            "locator_hash",
            name="uq_cx_passage_locator",
        ),
    )


@append_only
class PassageDerivative(Base):
    """Translation or page-image reading of a passage; never primary evidence."""

    __tablename__ = "company_exposure_passage_derivatives"

    id = uuid_pk()
    passage_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_passages.id", ondelete="RESTRICT"),
        nullable=False,
    )
    kind = Column(String(40), nullable=False)
    policy_version = Column(String(80), nullable=False)
    model_identity = Column(String(160), nullable=False)
    input_hash = Column(String(64), nullable=False)
    parent_text_hash = Column(String(64), nullable=False)
    output_text = Column(Text, nullable=False)
    output = Column(JSON, nullable=False)
    uncertainty = Column(JSON, nullable=False)
    evidence_role = Column(
        String(48), nullable=False, default="derivative_not_independent_source"
    )
    provider_attempt_id = Column(
        Uuid(as_uuid=True),
        ForeignKey(
            "company_exposure_provider_attempts.id",
            ondelete="RESTRICT",
            name="fk_cx_derivative_attempt",
        ),
        nullable=True,
    )
    created_at = created_at()

    __table_args__ = (
        CheckConstraint(f"kind IN ({_DERIVATIVE_KINDS})", name="ck_cx_derivative_kind"),
        CheckConstraint(
            "evidence_role = 'derivative_not_independent_source'",
            name="ck_cx_derivative_role",
        ),
        UniqueConstraint(
            "passage_id",
            "kind",
            "policy_version",
            "model_identity",
            "input_hash",
            name="uq_cx_derivative_input",
        ),
    )


@append_only
class EvidenceTombstoneEvent(Base):
    """A retained blob was removed; its hash and reason stay on record."""

    __tablename__ = "company_exposure_evidence_tombstones"

    id = uuid_pk()
    document_revision_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_document_revisions.id", ondelete="RESTRICT"),
        nullable=True,
    )
    blob_key = Column(String(255), nullable=False)
    content_hash = Column(String(64), nullable=False)
    byte_length = Column(BigInteger, nullable=False)
    reason = Column(String(80), nullable=False)
    authority = Column(String(200), nullable=False)
    detail = Column(Text, nullable=True)
    created_at = created_at()

    __table_args__ = (
        CheckConstraint(
            "reason IN ('legal_removal','unreferenced_gc','abandoned_scratch','operator_archive')",
            name="ck_cx_tombstone_reason",
        ),
    )
