from __future__ import annotations

import httpx
import pytest

from app.models.company_exposure import (
    ResearchArtifact,
    ResearchProviderAttempt,
    ResearchProviderResult,
)
from app.services.company_exposure.config import ExposureRuntimeConfig
from app.services.company_exposure.providers import (
    ProviderInput,
    SubscriptionArtifactRunner,
    SubscriptionProvider,
    default_client_factory,
)
from app.services.company_exposure.resources import ResearchResources
from tests.fixtures.company_exposure.factory import FakeGoTransport, FixedClock

CONFIG = ExposureRuntimeConfig(
    text_route_enabled=True,
    subscription_key_present=True,
    daily_request_limit=10,
    daily_token_limit=100_000,
)


@pytest.fixture
def clock():
    return FixedClock()


@pytest.fixture
def go_transport():
    return FakeGoTransport()


@pytest.fixture
def resources(db_session, clock):
    return ResearchResources(db_session, CONFIG, clock=clock.now)


@pytest.fixture
def subscription_runner(db_session, resources, go_transport):
    provider = SubscriptionProvider(
        api_key="test-key",
        client_factory=default_client_factory(go_transport.transport),
    )
    return SubscriptionArtifactRunner(db_session, resources, provider)


@pytest.fixture
def provider_input():
    return ProviderInput(
        operation="claim_review",
        messages=[{"role": "user", "content": "Passage: ..."}],
        input_hash="a" * 64,
        policy_hash="b" * 64,
        max_output_tokens=1000,
        logical_operation_key="claim_review:" + "a" * 64,
    )


@pytest.mark.case("R03")
@pytest.mark.exposure_layer("unit")
def test_retry_records_two_go_attempts_and_reuses_success(
    subscription_runner, provider_input, go_transport, db_session
):
    go_transport.queue_status(503)
    go_transport.queue_json({"claims": []}, usage={"total_tokens": 321})
    first = subscription_runner.run(provider_input)
    assert first.retryable is True
    success = subscription_runner.run(provider_input)
    repeated = subscription_runner.run(provider_input)
    assert repeated.artifact_id == success.artifact_id
    assert repeated.reused is True
    assert len(go_transport.requests) == 2
    assert db_session.query(ResearchProviderAttempt).count() == 2
    assert [
        a.attempt_number
        for a in db_session.query(ResearchProviderAttempt).order_by(
            ResearchProviderAttempt.attempt_number
        )
    ] == [1, 2]
    assert db_session.query(ResearchArtifact).count() == 1
    assert {request.json["model"] for request in go_transport.requests} == {"kimi-k2.6"}
    assert all(r.url.endswith("/chat/completions") for r in go_transport.requests)


@pytest.mark.case("R04")
@pytest.mark.exposure_layer("unit")
@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("refused"), httpx.ConnectTimeout("connect"), httpx.PoolTimeout("pool")],
)
def test_connect_phase_failure_is_pre_dispatch_and_released(
    subscription_runner, provider_input, go_transport, resources, error
):
    go_transport.queue_exception(error)
    result = subscription_runner.run(provider_input)
    usage = resources.read(result.ticket_id)
    assert usage.dispatch_phase == "pre_dispatch"
    assert usage.state == "released"


@pytest.mark.case("R04")
@pytest.mark.exposure_layer("unit")
@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("read"),
        httpx.WriteTimeout("write"),
        httpx.ReadError("reset"),
        httpx.RemoteProtocolError("eof"),
    ],
)
def test_post_send_failure_stays_uncertain(
    subscription_runner, provider_input, go_transport, resources, error
):
    go_transport.queue_exception(error)
    result = subscription_runner.run(provider_input)
    usage = resources.read(result.ticket_id)
    assert usage.state == "uncertain"
    assert usage.actual_dollar_cost is None


@pytest.mark.case("R04")
@pytest.mark.exposure_layer("unit")
def test_uncertain_reservation_expires_with_its_period(
    subscription_runner, provider_input, go_transport, resources, clock
):
    go_transport.queue_exception(httpx.ReadTimeout("read"))
    result = subscription_runner.run(provider_input)
    period, period_end = CONFIG.allocation_period(clock.now())
    clock.advance_to(period_end)
    reports = resources.close_ended_periods()
    assert [r.period for r in reports] == [period]
    assert resources.read(result.ticket_id).state == "expired_uncertain"
    next_period, _ = CONFIG.allocation_period(clock.now())
    assert (
        resources.available("llm:opencode-go", period=next_period).requests
        == resources.limit("llm:opencode-go").requests
    )


def test_success_without_reported_usage_keeps_tokens_unknown(
    subscription_runner, provider_input, go_transport, db_session
):
    go_transport.queue_json({"claims": []})
    subscription_runner.run(provider_input)
    result = db_session.query(ResearchProviderResult).one()
    assert (result.outcome, result.usage_known, result.reported_usage) == (
        "success",
        False,
        {},
    )


def test_missing_key_is_pre_dispatch_and_makes_no_call(
    db_session, resources, go_transport, provider_input
):
    provider = SubscriptionProvider(
        api_key="", client_factory=default_client_factory(go_transport.transport)
    )
    runner = SubscriptionArtifactRunner(db_session, resources, provider)
    result = runner.run(provider_input)
    assert result.failure_code == "subscription_credentials_missing"
    assert resources.read(result.ticket_id).state == "released"
    assert go_transport.requests == []
