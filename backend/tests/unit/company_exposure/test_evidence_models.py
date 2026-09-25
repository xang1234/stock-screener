from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import MetaData, create_engine, inspect
from sqlalchemy.exc import IntegrityError

import app.models  # noqa: F401
from app.database import Base
from app.models.company_exposure import (
    DocumentCaptureEvent,
    ExposurePassage,
    IssuerSecurityLinkRevision,
    PassageDerivative,
)
from app.models.economic_taxonomy_runtime_common import ImmutableRuntimePayload
from tests.fixtures.company_exposure.factory import (
    FIXED_NOW,
    make_document,
    make_issuer,
    make_revision,
    make_security,
)

MIGRATION = (
    Path(__file__).resolve().parents[3]
    / "alembic/versions/20260925_0058_exposure_identity_documents.py"
)
TASK01_TABLES = (
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


def _link(security, issuer, revision, state):
    return IssuerSecurityLinkRevision(
        security_id=security.id,
        issuer_id=issuer.id,
        revision_number=revision,
        state=state,
        acceptance_policy="administrator_reviewed",
        actor="test:admin",
        reason=f"revision {revision}",
        evidence={"reference": "filing"},
        snapshot={"ticker": security.symbol},
    )


@pytest.mark.case("I11")
@pytest.mark.exposure_layer("schema")
def test_same_security_keeps_multiple_link_revisions(db_session):
    security = make_security(db_session, "ACME")
    issuer = make_issuer(db_session)
    first = _link(security, issuer, 1, "accepted")
    db_session.add(first)
    db_session.flush()
    db_session.add(_link(security, issuer, 2, "rejected"))
    db_session.flush()

    rows = (
        db_session.query(IssuerSecurityLinkRevision)
        .filter_by(security_id=security.id)
        .order_by(IssuerSecurityLinkRevision.revision_number)
        .all()
    )
    assert [(r.revision_number, r.state) for r in rows] == [
        (1, "accepted"),
        (2, "rejected"),
    ]

    first.state = "rejected"
    with pytest.raises(ImmutableRuntimePayload):
        db_session.flush()
    db_session.rollback()


def test_duplicate_link_revision_number_is_rejected(db_session):
    security = make_security(db_session, "ACME")
    issuer = make_issuer(db_session)
    db_session.add(_link(security, issuer, 1, "accepted"))
    db_session.flush()
    db_session.add(_link(security, issuer, 1, "rejected"))
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


@pytest.mark.case("E12")
@pytest.mark.exposure_layer("schema")
def test_identical_bytes_cannot_create_a_second_revision(db_session):
    document = make_document(db_session, "sec:accession:0000000001-26-000001")
    revision = make_revision(db_session, document, b"<html>annual report</html>")
    for _ in range(2):
        db_session.add(
            DocumentCaptureEvent(
                document_id=document.id,
                revision_id=revision.id,
                outcome="unchanged",
                changed=False,
                content_hash=revision.content_hash,
                observed_metadata={},
                retrieved_at=FIXED_NOW,
            )
        )
    db_session.flush()
    assert db_session.query(DocumentCaptureEvent).count() == 2

    with pytest.raises(IntegrityError):
        make_revision(db_session, document, b"<html>annual report</html>")
    db_session.rollback()


def test_append_only_rows_reject_orm_delete(db_session):
    document = make_document(db_session, "sec:accession:0000000001-26-000002")
    revision = make_revision(db_session, document, b"body")
    db_session.delete(revision)
    with pytest.raises(ImmutableRuntimePayload):
        db_session.flush()
    db_session.rollback()


def test_derivative_cannot_claim_primary_role(db_session):
    document = make_document(db_session, "sec:accession:0000000001-26-000004")
    revision = make_revision(db_session, document, b"body")
    passage = ExposurePassage(
        document_revision_id=revision.id,
        revision_content_hash=revision.content_hash,
        preparation_policy="structure-v1",
        extractor_version="html-v1",
        locator={"section_path": ["Item 1"], "start": 0, "end": 4},
        locator_hash="a" * 64,
        original_text="body",
        text_hash="b" * 64,
        context={},
    )
    db_session.add(passage)
    db_session.flush()
    db_session.add(
        PassageDerivative(
            passage_id=passage.id,
            kind="translation",
            policy_version="translation-v3",
            model_identity="opencode-go/kimi-k2.6",
            input_hash="c" * 64,
            parent_text_hash=passage.text_hash,
            output_text="body",
            output={},
            uncertainty={},
            evidence_role="original_primary",
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def _load_migration():
    spec = importlib.util.spec_from_file_location("exposure_0058", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.exposure_layer("schema")
def test_migration_matches_models_and_downgrades(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'm.sqlite'}")
    migration = _load_migration()
    assert migration.down_revision == "20260925_0057"
    try:
        with engine.begin() as connection:
            Base.metadata.tables["stock_universe"].create(connection)
            migration.op = Operations(MigrationContext.configure(connection))
            migration.upgrade()

        subset = MetaData()
        for name in ("stock_universe", *TASK01_TABLES):
            Base.metadata.tables[name].to_metadata(subset)
        with engine.connect() as connection:
            diffs = compare_metadata(MigrationContext.configure(connection), subset)
        assert diffs == []

        with engine.begin() as connection:
            migration.op = Operations(MigrationContext.configure(connection))
            migration.downgrade()
        remaining = set(inspect(engine).get_table_names())
        assert remaining.isdisjoint(TASK01_TABLES)
    finally:
        engine.dispose()
