from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy.orm import sessionmaker

from app.database import engine
from app.domain.company_exposure.contracts import SERVICE_PRINCIPAL, RegistryMatch
from app.models.company_exposure import IssuerSecurityLinkRevision
from app.services.company_exposure.issuer_identity import IssuerIdentityAdapter
from tests.fixtures.company_exposure.factory import make_security

pytestmark = [
    pytest.mark.skipif(
        engine.dialect.name != "postgresql", reason="requires PostgreSQL row locks"
    ),
    pytest.mark.exposure_layer("postgres"),
]


@pytest.mark.case("I02")
def test_concurrent_registry_acceptance_creates_one_link():
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    security = make_security(session, "RACE")
    session.commit()
    session.close()
    match = RegistryMatch(
        security_id=security.id,
        market="US",
        scheme="cik",
        value="555",
        candidate_count=1,
        ticker_confirmed=True,
        matched_ticker="RACE",
        registry_capture_revision_id=None,
        official_record_capture_revision_id=None,
    )
    barrier = Barrier(2)

    def accept(_):
        worker = factory()
        try:
            barrier.wait(timeout=10)
            ref = IssuerIdentityAdapter(worker).accept_registry_match(
                match, SERVICE_PRINCIPAL
            )
            worker.commit()
            return ref
        finally:
            worker.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        refs = list(executor.map(accept, [0, 1]))

    assert {ref.state for ref in refs} == {"accepted"}
    assert sorted(ref.created for ref in refs) == [False, True]
    check = factory()
    assert check.query(IssuerSecurityLinkRevision).count() == 1
    check.close()
