from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.models.company_exposure import ClaimEvidenceLink, DocumentRelationRevision
from app.services.company_exposure.provenance import origin_groups
from tests.fixtures.company_exposure.factory import (
    make_document,
    make_passage,
    make_revision,
    verified_claim,
)

REPORT = b"FY2025 annual report: the ET-9000 supports HBM testing."
QUOTE = "The ET-9000 supports HBM testing."


def _passage(db, key, content, issuer=None, provider="sec"):
    document = make_document(db, key, issuer=issuer, provider=provider)
    revision = make_revision(db, document, content)
    return document, make_passage(db, revision, QUOTE)


@pytest.fixture
def one_report_three_copies(dossier):
    db = dossier.db
    exchange, via_exchange = _passage(db, "exchange:10-K", REPORT, dossier.issuer)
    _, via_issuer_site = _passage(
        db, "issuer-site:10-K", REPORT, dossier.issuer, "issuer_ir"
    )
    translated, via_translation = _passage(
        db, "issuer-site:10-K:ja", "年次報告書".encode(), dossier.issuer, "issuer_ir"
    )
    db.add(
        DocumentRelationRevision(
            from_document_id=translated.id,
            to_document_id=exchange.id,
            relation="translation",
            revision_number=1,
            scope={},
            evidence={"reference": "translation notice"},
            actor="test",
        )
    )
    db.flush()
    return via_exchange, via_issuer_site, via_translation


@pytest.mark.case("E13")
@pytest.mark.exposure_layer("unit")
def test_mirrors_and_translation_are_one_origin(dossier, one_report_three_copies):
    ids = [p.id for p in one_report_three_copies]
    groups = origin_groups(dossier.db, ids)
    assert len(set(groups.values())) == 1

    _, independent = _passage(
        dossier.db, "trade-press:interview", b"different original"
    )
    groups = origin_groups(dossier.db, [*ids, independent.id])
    assert len(set(groups.values())) == 2


@pytest.mark.case("E13")
@pytest.mark.exposure_layer("unit")
def test_persisted_claim_links_record_the_shared_origin(
    dossier, one_report_three_copies
):
    claim = verified_claim(
        "product_application",
        passage=one_report_three_copies[0],
        supported_as_of=datetime(2026, 2, 13, tzinfo=timezone.utc),
    )
    cited = tuple(
        type(claim.evidence[0])(p.id, QUOTE, claim.evidence[0].role)
        for p in one_report_three_copies
    )
    from dataclasses import replace

    dossier.persist(dossier.attempt(replace(claim, evidence=cited)))
    origins = {
        link.attribution["origin_group"]
        for link in dossier.db.execute(select(ClaimEvidenceLink)).scalars()
    }
    assert len(origins) == 1
