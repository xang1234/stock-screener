from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Barrier
from uuid import uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

from app.database import engine
from app.infra.db.repositories.economic_taxonomy_work_repo import (
    EconomicTaxonomyWorkRepository,
)
from app.models.economic_taxonomy_runtime import ProcessingRequest, TaxonomyAuthority
from app.services.economic_source_admission import (
    EconomicSourceAdmissionService,
    EvidenceAdmission,
)

NOW = datetime(2026, 9, 21, tzinfo=timezone.utc)

pytestmark = pytest.mark.skipif(
    engine.dialect.name != "postgresql",
    reason="requires PostgreSQL row locks and SKIP LOCKED",
)


@pytest.fixture
def factory():
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    session.add(
        TaxonomyAuthority(
            id=1,
            mode="shadow",
            processing_head_revision=11,
            authority_epoch=3,
            writes_fenced=False,
            rollback_state="ready",
        )
    )
    session.flush()
    admitted = EconomicSourceAdmissionService(session).admit_content(
        EvidenceAdmission(
            provider="x",
            canonical_item_id="concurrent-post",
            capture_route="legacy",
            original_text="Memory demand.",
            preparation_version="prep-v1",
            captured_at=datetime(2026, 9, 21, tzinfo=timezone.utc),
            available_at=datetime(2026, 9, 21, tzinfo=timezone.utc),
        )
    )
    request = EconomicTaxonomyWorkRepository(session).enqueue_request(
        source_lineage_id=admitted.source_lineage_id,
        evidence_packet_id=admitted.packet_id,
        policy_bundle_version="bundle-v1",
        available_at=datetime(2026, 9, 21, tzinfo=timezone.utc),
    )
    session.commit()
    session.close()
    return factory, request.id


def test_skip_locked_allows_only_one_claimant(factory):
    sessions, request_id = factory
    barrier = Barrier(2)

    def claim(worker_id):
        session = sessions()
        try:
            barrier.wait(timeout=5)
            row = EconomicTaxonomyWorkRepository(session).claim_next(
                worker_id=worker_id,
                now=datetime(2026, 9, 21, tzinfo=timezone.utc),
            )
            session.commit()
            return row.id if row else None
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = list(pool.map(claim, ("worker-1", "worker-2")))

    assert claimed.count(request_id) == 1
    assert claimed.count(None) == 1


def test_provider_attempt_numbers_are_concurrency_safe(factory):
    sessions, request_id = factory
    barrier = Barrier(2)

    def begin(dispatch_id):
        session = sessions()
        try:
            barrier.wait(timeout=5)
            attempt = EconomicTaxonomyWorkRepository(session).begin_provider_attempt(
                request_id,
                operation="extract",
                dispatch_id=dispatch_id,
            )
            session.commit()
            return attempt.attempt_number
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        numbers = list(pool.map(begin, ("dispatch-1", "dispatch-2")))

    assert sorted(numbers) == [1, 2]


def test_requeue_leaves_a_request_claimed_while_it_waited_for_the_row_lock(factory):
    from scripts.requeue_economic_processing_requests import requeue

    sessions, request_id = factory
    with sessions.begin() as session:
        request = session.get(ProcessingRequest, request_id)
        request.status = "terminal_failure"
        request.completion_code = "provider_outcome_uncertain"
    holder = sessions()
    request = holder.execute(
        select(ProcessingRequest)
        .where(ProcessingRequest.id == request_id)
        .with_for_update()
    ).scalar_one()

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(requeue, sessions, NOW, apply=True)
        deadline = time.monotonic() + 5
        while not _waiting_on_row_lock() and not pending.done():
            assert time.monotonic() < deadline, "requeue never reached the row lock"
            time.sleep(0.05)
        lease_token = uuid4()
        request.status = "leased"
        request.lease_token = lease_token
        request.lease_owner = "worker-1"
        request.lease_expires_at = NOW
        holder.commit()
        holder.close()
        counts = pending.result(timeout=10)

    with sessions() as session:
        stored = session.get(ProcessingRequest, request_id)
        assert (counts, stored.status, stored.lease_token) == (Counter(), "leased", lease_token)


def _waiting_on_row_lock() -> bool:
    with engine.connect() as connection:
        return bool(
            connection.scalar(
                text(
                    "SELECT count(*) FROM pg_locks "
                    "WHERE NOT granted AND locktype = 'transactionid'"
                )
            )
        )
