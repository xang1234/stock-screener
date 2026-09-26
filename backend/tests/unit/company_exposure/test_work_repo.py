from __future__ import annotations

from datetime import timedelta

import pytest

from app.domain.company_exposure.contracts import ReservationState, ResourceUnit
from app.infra.db.repositories.company_exposure_work_repo import (
    ROOT_PROVIDER_ATTEMPTS,
    CompanyExposureWorkRepository,
    ReservationLedger,
    ReservationTransitionError,
    WorkLeaseError,
)
from app.models.company_exposure import (
    ResearchProviderAttempt,
    ResearchResourcePool,
    ResearchRootBudget,
)
from tests.fixtures.company_exposure.factory import (
    FixedClock,
    make_security,
    make_theme,
)


@pytest.fixture
def clock():
    return FixedClock()


@pytest.fixture
def repo(db_session, clock):
    return CompanyExposureWorkRepository(db_session, clock=clock.now)


@pytest.fixture
def ledger(db_session, clock):
    return ReservationLedger(db_session, clock=clock.now)


@pytest.fixture
def request_row(db_session, repo):
    theme = make_theme(db_session)
    security = make_security(db_session, "ACME")
    request, _ = repo.create_request(
        kind="verify",
        requester_principal="test:admin",
        idempotency_namespace="test:admin",
        idempotency_key="verify-1",
        economic_theme_id=theme.id,
        security_id=security.id,
    )
    return request


@pytest.fixture
def pool(ledger):
    return ledger.ensure_pool(
        pool_key="llm:opencode-go",
        unit=ResourceUnit.REQUESTS,
        period="2026-09-26",
        period_end=None,
        capacity=2,
    )


def _reserve(ledger, pool, request_row, key="op"):
    return ledger.reserve(
        pool_id=pool.id,
        amount=1,
        purpose="claim_review",
        logical_operation_key=key,
        root_request_id=request_row.id,
        root_budget_key=ROOT_PROVIDER_ATTEMPTS,
    )


def test_claim_heartbeat_and_complete_require_the_live_lease(repo, request_row, clock):
    item = repo.enqueue(
        request=request_row, stage="acquire", input_hash="a" * 64, policy_bundle_version="p1"
    )
    assert repo.enqueue(
        request=request_row, stage="acquire", input_hash="a" * 64, policy_bundle_version="p1"
    ).id == item.id
    claimed = repo.claim_next(worker_id="w1")
    token = claimed.lease_token
    assert repo.claim_next(worker_id="w2") is None

    with pytest.raises(WorkLeaseError):
        repo.heartbeat(item.id, lease_token=type(token)(int=0))
    repo.heartbeat(item.id, token)

    clock.advance(minutes=6)
    with pytest.raises(WorkLeaseError):
        repo.complete_step(item.id, token, status="completed")
    reclaimed = repo.claim_next(worker_id="w2")
    assert reclaimed.id == item.id and reclaimed.lease_token != token
    repo.complete_step(item.id, reclaimed.lease_token, status="completed")
    assert item.status == "completed"


def test_pause_and_resume_do_not_reset_root_budget(repo, ledger, pool, request_row, db_session):
    repo.enqueue(
        request=request_row, stage="acquire", input_hash="a" * 64, policy_bundle_version="p1"
    )
    assert _reserve(ledger, pool, request_row).allowed
    repo.pause(request_row.id, reason="paused_allowance")
    assert repo.claim_next(worker_id="w1") is None
    repo.resume(request_row.id)
    assert repo.claim_next(worker_id="w1") is not None
    budget = (
        db_session.query(ResearchRootBudget)
        .filter_by(root_request_id=request_row.id, budget_key=ROOT_PROVIDER_ATTEMPTS)
        .one()
    )
    assert budget.used_amount == 1
    assert [e.state for e in repo.events(request_row.id)] == [
        "queued",
        "paused_allowance",
        "queued",
    ]


def test_capacity_and_missing_allocation_block_reservation(ledger, pool, request_row):
    assert _reserve(ledger, pool, request_row, "a").allowed
    assert _reserve(ledger, pool, request_row, "b").allowed
    blocked = _reserve(ledger, pool, request_row, "c")
    assert (blocked.allowed, blocked.reason, blocked.available) == (
        False,
        "capacity_exhausted",
        0,
    )
    unconfigured = ledger.ensure_pool(
        pool_key="llm:opencode-go:vision",
        unit=ResourceUnit.REQUESTS,
        period="2026-09-26",
        period_end=None,
        capacity=None,
    )
    outcome = _reserve(ledger, unconfigured, request_row, "d")
    assert (outcome.allowed, outcome.reason) == (False, "allocation_not_configured")


@pytest.mark.case("R05")
@pytest.mark.exposure_layer("unit")
def test_root_budget_is_cumulative_across_attempts(ledger, request_row, db_session):
    big_pool = ledger.ensure_pool(
        pool_key="llm:opencode-go",
        unit=ResourceUnit.REQUESTS,
        period="2026-09-27",
        period_end=None,
        capacity=1000,
    )
    outcomes = [_reserve(ledger, big_pool, request_row, f"op-{i}") for i in range(25)]
    assert sum(o.allowed for o in outcomes) == 24
    assert outcomes[-1].reason == "root_budget_exhausted"


@pytest.mark.case("R04")
@pytest.mark.exposure_layer("unit")
def test_cancellation_keeps_dispatched_attempt_immutable(
    repo, ledger, pool, request_row, db_session
):
    dispatched = _reserve(ledger, pool, request_row, "dispatched")
    pending = _reserve(ledger, pool, request_row, "pending")
    ledger.transition(dispatched.reservation_id, ReservationState.DISPATCHED)
    attempt = ResearchProviderAttempt(
        logical_operation_key="dispatched",
        attempt_number=1,
        request_id=request_row.id,
        operation="claim_review",
        route="opencode-go",
        model="kimi-k2.6",
        parameters={},
        input_hash="a" * 64,
        policy_hash="b" * 64,
        reservation_id=dispatched.reservation_id,
    )
    db_session.add(attempt)
    db_session.flush()

    repo.cancel(request_row.id, reason="user_cancelled")
    db_session.expire_all()

    assert db_session.get(ResearchProviderAttempt, attempt.id) is not None
    assert ledger.state(dispatched.reservation_id) is ReservationState.UNCERTAIN
    assert ledger.state(pending.reservation_id) is ReservationState.RELEASED
    pool_row = db_session.get(ResearchResourcePool, pool.id)
    assert pool_row.reserved_amount == 1


def test_dispatched_release_requires_pre_dispatch_phase(ledger, pool, request_row):
    outcome = _reserve(ledger, pool, request_row)
    ledger.transition(outcome.reservation_id, ReservationState.DISPATCHED)
    with pytest.raises(ReservationTransitionError):
        ledger.transition(outcome.reservation_id, ReservationState.RELEASED)
    ledger.transition(
        outcome.reservation_id, ReservationState.RELEASED, dispatch_phase="pre_dispatch"
    )
    with pytest.raises(ReservationTransitionError):
        ledger.transition(outcome.reservation_id, ReservationState.DISPATCHED)


def test_reconcile_adjusts_reserved_bound_to_reported_usage(ledger, request_row, db_session):
    tokens = ledger.ensure_pool(
        pool_key="llm:opencode-go",
        unit=ResourceUnit.REPORTED_TOKENS,
        period="2026-09-26",
        period_end=None,
        capacity=10_000,
    )
    outcome = ledger.reserve(
        pool_id=tokens.id, amount=4000, purpose="claim_review", logical_operation_key="t"
    )
    ledger.transition(outcome.reservation_id, ReservationState.DISPATCHED)
    ledger.transition(
        outcome.reservation_id,
        ReservationState.RECONCILED,
        dispatch_phase="dispatched",
        actual_amount=1234,
    )
    assert db_session.get(ResearchResourcePool, tokens.id).reserved_amount == 1234


@pytest.mark.case("R04")
@pytest.mark.exposure_layer("unit")
def test_close_period_expires_uncertain_and_never_refunds(ledger, pool, request_row, db_session):
    uncertain = _reserve(ledger, pool, request_row, "u")
    ledger.transition(uncertain.reservation_id, ReservationState.DISPATCHED)
    ledger.transition(
        uncertain.reservation_id, ReservationState.UNCERTAIN, dispatch_phase="uncertain"
    )
    in_flight = _reserve(ledger, pool, request_row, "f")
    ledger.transition(in_flight.reservation_id, ReservationState.DISPATCHED)

    expired = ledger.close_period(
        pool_key="llm:opencode-go", unit=ResourceUnit.REQUESTS, period="2026-09-26"
    )

    assert set(expired) == {uncertain.reservation_id, in_flight.reservation_id}
    assert ledger.state(uncertain.reservation_id) is ReservationState.EXPIRED_UNCERTAIN
    assert db_session.get(ResearchResourcePool, pool.id).reserved_amount == 2
    closed = _reserve(ledger, pool, request_row, "late")
    assert (closed.allowed, closed.reason) == (False, "period_closed")
    next_day = ledger.ensure_pool(
        pool_key="llm:opencode-go",
        unit=ResourceUnit.REQUESTS,
        period="2026-09-27",
        period_end=None,
        capacity=2,
    )
    assert next_day.reserved_amount == 0


def test_events_are_append_only_ordered(repo, request_row, clock):
    clock.advance(seconds=timedelta(seconds=1).total_seconds())
    repo.append_event(request_row.id, "researching", {"stage": "acquire"})
    assert [e.sequence for e in repo.events(request_row.id)] == [1, 2]
