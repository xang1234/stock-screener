from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Barrier

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.database import engine
from app.models.company_exposure import (
    AssessmentClaimSelection,
    AssessmentRevision,
    ExposureUseHoldRevision,
)
from app.services.company_exposure.assessments import (
    AssessmentAttemptInput,
    ExposureAssessmentService,
)
from app.services.company_exposure.claims import AssessmentScope
from app.services.company_exposure.freshness import refresh_due_holds
from tests.fixtures.company_exposure.factory import (
    FixedClock,
    make_document,
    make_issuer,
    make_passage,
    make_revision,
    make_theme,
    verified_claim,
)

pytestmark = [
    pytest.mark.skipif(
        engine.dialect.name != "postgresql", reason="requires PostgreSQL row locks"
    ),
    pytest.mark.exposure_layer("postgres"),
]

ROLE_DATE = datetime(2025, 8, 1, tzinfo=timezone.utc)


def _seed(factory):
    session = factory()
    issuer = make_issuer(session, "pg-dossier-issuer")
    theme = make_theme(session, "pg-dossier-theme")
    document = make_document(session, "sec:10-K:pg-dossier", issuer=issuer)
    revision = make_revision(session, document, b"annual report")
    passages = [
        make_passage(session, revision, f"We sell product {i} for HBM.", index=i)
        for i in range(3)
    ]
    scope = AssessmentScope(
        issuer_id=issuer.id,
        economic_theme_id=theme.id,
        theme_fingerprint="f" * 64,
        theme_label="AI Memory",
        theme_terms=("HBM",),
    )
    base = verified_claim(
        "role", passage=passages[0], product_key="product-0", supported_as_of=ROLE_DATE
    )
    service = ExposureAssessmentService(session, clock=FixedClock().now)
    result = service.assess(AssessmentAttemptInput(scope=scope, claims=(base,)))
    ref = service.persist_assessment(result)
    session.commit()
    session.close()
    return scope, passages, ref


def test_concurrent_assessments_serialize_and_keep_both_results():
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    scope, passages, base = _seed(factory)
    barrier = Barrier(2)

    def run(index):
        session = factory()
        try:
            service = ExposureAssessmentService(session, clock=FixedClock().now)
            claim = verified_claim(
                "customer_relationship",
                passage=passages[index],
                product_key=f"product-{index}",
                supported_as_of=ROLE_DATE,
            )
            # Selection is computed (outside the fence) against the same prior.
            result = service.assess(
                AssessmentAttemptInput(scope=scope, claims=(claim,))
            )
            assert result.prior_revision_id == base.id
            session.commit()
            barrier.wait(timeout=10)
            ref = service.persist_assessment(result)
            session.commit()
            return ref
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        refs = list(executor.map(run, [1, 2]))

    assert sorted(ref.revision_number for ref in refs) == [2, 3]
    assert [ref.rebuilt for ref in sorted(refs, key=lambda r: r.revision_number)] == [
        False,
        True,
    ]
    check = factory()
    latest = check.execute(
        select(AssessmentRevision)
        .order_by(AssessmentRevision.revision_number.desc())
        .limit(1)
    ).scalar_one()
    selected = check.execute(
        select(func.count())
        .select_from(AssessmentClaimSelection)
        .where(AssessmentClaimSelection.assessment_revision_id == latest.id)
    ).scalar()
    assert selected == 3  # base role plus both concurrent customer claims
    check.close()


@pytest.mark.case("R12")
def test_expiry_holds_are_recorded_without_research():
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    _seed(factory)
    session = factory()
    report = refresh_due_holds(session, datetime(2027, 1, 1, tzinfo=timezone.utc))
    session.commit()
    assert len(report.new_holds) == 1
    kinds = session.execute(select(ExposureUseHoldRevision.hold_kind)).scalars().all()
    assert kinds == ["stale"]
    session.close()
