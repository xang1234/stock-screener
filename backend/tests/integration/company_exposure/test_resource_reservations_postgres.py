from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy.orm import sessionmaker

from app.database import engine
from app.models.company_exposure import ResearchProviderAttempt
from app.services.company_exposure.config import ExposureRuntimeConfig
from app.services.company_exposure.resources import DispatchRequest, ResearchResources
from tests.fixtures.company_exposure.factory import FixedClock

pytestmark = [
    pytest.mark.skipif(
        engine.dialect.name != "postgresql", reason="requires PostgreSQL row locks"
    ),
    pytest.mark.exposure_layer("postgres"),
]

CONFIG = ExposureRuntimeConfig(
    text_route_enabled=True,
    subscription_key_present=True,
    daily_request_limit=1,
    daily_token_limit=10_000,
)


@pytest.mark.case("R02")
@pytest.mark.case("R03")
def test_concurrent_dispatches_share_the_last_daily_unit():
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    clock = FixedClock()
    barrier = Barrier(2)

    def reserve(label):
        session = factory()
        try:
            barrier.wait(timeout=10)
            ticket = ResearchResources(session, CONFIG, clock=clock.now).reserve(
                DispatchRequest(
                    logical_operation_key=f"op-{label}",
                    operation="claim_review",
                    capability="text",
                    input_hash="a" * 64,
                    policy_hash="b" * 64,
                    max_output_tokens=1000,
                    estimated_input_tokens=100,
                )
            )
            session.commit()
            return ticket
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        tickets = list(executor.map(reserve, ["a", "b"]))

    assert sum(ticket.allowed for ticket in tickets) == 1
    assert {t.reason for t in tickets if not t.allowed} == {"capacity_exhausted"}
    check = factory()
    assert check.query(ResearchProviderAttempt).count() == 1
    check.close()
