from __future__ import annotations

from datetime import timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

from app.database import engine
from app.models.company_exposure import (
    DocumentCaptureEvent,
    ExposureDocumentRevision,
    ExposurePassage,
)
from tests.fixtures.company_exposure.factory import (
    FIXED_NOW,
    make_document,
    make_revision,
)

pytestmark = [
    pytest.mark.skipif(
        engine.dialect.name != "postgresql",
        reason="requires PostgreSQL triggers and foreign keys",
    ),
    pytest.mark.exposure_layer("postgres"),
]


@pytest.mark.case("E12")
@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE company_exposure_document_revisions SET reporting_period = 'FY2030'",
        "DELETE FROM company_exposure_document_revisions",
        "UPDATE company_exposure_documents SET publisher = 'rewritten'",
    ],
)
def test_raw_sql_cannot_mutate_retained_evidence(db_session, statement):
    document = make_document(db_session, "sec:accession:pg-1")
    make_revision(db_session, document, b"annual report", period="FY2025")
    db_session.commit()
    with pytest.raises(DBAPIError, match="company_exposure_payload_immutable"):
        db_session.execute(text(statement))
        db_session.flush()
    db_session.rollback()


def test_capture_cannot_cite_another_documents_revision(db_session):
    first = make_document(db_session, "sec:accession:pg-2")
    second = make_document(db_session, "sec:accession:pg-3")
    revision = make_revision(db_session, first, b"first body")
    db_session.add(
        DocumentCaptureEvent(
            document_id=second.id,
            revision_id=revision.id,
            outcome="unchanged",
            changed=False,
            observed_metadata={},
            retrieved_at=FIXED_NOW,
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_passage_must_match_its_revision_content_hash(db_session):
    document = make_document(db_session, "sec:accession:pg-4")
    revision = make_revision(db_session, document, b"passage body")
    db_session.add(
        ExposurePassage(
            document_revision_id=revision.id,
            revision_content_hash="0" * 64,
            preparation_policy="structure-v1",
            extractor_version="html-v1",
            locator={"start": 0, "end": 4},
            locator_hash="a" * 64,
            original_text="body",
            text_hash="b" * 64,
            context={},
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_revision_timestamps_round_trip_timezone_aware(db_session):
    document = make_document(db_session, "sec:accession:pg-5")
    revision = make_revision(db_session, document, b"body", published_at=FIXED_NOW)
    db_session.commit()
    db_session.expire_all()
    stored = db_session.get(ExposureDocumentRevision, revision.id)
    assert stored.first_available_at.tzinfo is not None
    assert stored.published_at.astimezone(timezone.utc) == FIXED_NOW
