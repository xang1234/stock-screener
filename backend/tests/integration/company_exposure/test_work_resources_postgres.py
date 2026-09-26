from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import sessionmaker

from app.database import engine
from app.domain.company_exposure.contracts import ReservationState, ResourceUnit
from app.infra.db.repositories.company_exposure_work_repo import (
    ROOT_PROVIDER_ATTEMPTS,
    CompanyExposureWorkRepository,
    ReservationLedger,
)
from app.models.company_exposure import ResearchResourcePool
from tests.fixtures.company_exposure.factory import make_security, make_theme

pytestmark = [
    pytest.mark.skipif(
        engine.dialect.name != "postgresql",
        reason="requires PostgreSQL row locks and SKIP LOCKED",
    ),
    pytest.mark.exposure_layer("postgres"),
]


@pytest.fixture
def factory():
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def seeded(factory):
    session = factory()
    theme = make_theme(session)
    security = make_security(session, "ACME")
    repo = CompanyExposureWorkRepository(session)
    request, _ = repo.create_request(
        kind="verify",
        requester_principal="test:admin",
        idempotency_namespace="test:admin",
        idempotency_key="pg-1",
        economic_theme_id=theme.id,
        security_id=security.id,
    )
    pool = ReservationLedger(session).ensure_pool(
        pool_key="llm:opencode-go",
        unit=ResourceUnit.REQUESTS,
        period="2026-09-26",
        period_end=None,
        capacity=1,
    )
    repo.enqueue(
        request=request, stage="acquire", input_hash="a" * 64, policy_bundle_version="p1"
    )
    session.commit()
    session.close()
    return request, pool


@pytest.mark.case("R03")
@pytest.mark.case("R05")
def test_two_workers_contending_for_last_unit_get_one_reservation(factory, seeded):
    request, pool = seeded
    barrier = Barrier(2)

    def contend(label):
        session = factory()
        try:
            barrier.wait(timeout=10)
            outcome = ReservationLedger(session).reserve(
                pool_id=pool.id,
                amount=1,
                purpose="claim_review",
                logical_operation_key=f"op-{label}",
                root_request_id=request.id,
                root_budget_key=ROOT_PROVIDER_ATTEMPTS,
            )
            session.commit()
            return outcome
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=2) as pool_executor:
        outcomes = list(pool_executor.map(contend, ["a", "b"]))

    assert sum(o.allowed for o in outcomes) == 1
    assert {o.reason for o in outcomes if not o.allowed} == {"capacity_exhausted"}
    check = factory()
    assert check.get(ResearchResourcePool, pool.id).reserved_amount == 1
    check.close()


def test_skip_locked_allows_only_one_claimant(factory, seeded):
    barrier = Barrier(2)

    def claim(worker):
        session = factory()
        try:
            barrier.wait(timeout=10)
            item = CompanyExposureWorkRepository(session).claim_next(worker_id=worker)
            claimed = item.id if item is not None else None
            barrier.wait(timeout=10)
            session.commit()
            return claimed
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(claim, ["w1", "w2"]))
    assert sum(value is not None for value in claims) == 1


@pytest.mark.case("R04")
def test_reservation_history_cannot_be_rewritten_by_sql(factory, seeded):
    _request, pool = seeded
    session = factory()
    ledger = ReservationLedger(session)
    outcome = ledger.reserve(
        pool_id=pool.id, amount=1, purpose="claim_review", logical_operation_key="x"
    )
    ledger.transition(outcome.reservation_id, ReservationState.DISPATCHED)
    session.commit()
    with pytest.raises(DBAPIError, match="company_exposure_payload_immutable"):
        session.execute(
            text("UPDATE company_exposure_reservation_events SET state = 'released'")
        )
    session.rollback()
    session.close()
