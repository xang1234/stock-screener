from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from app.services.company_exposure.config import ExposureRuntimeConfig
from app.services.company_exposure.providers import (
    ProviderInput,
    SubscriptionArtifactRunner,
    SubscriptionProvider,
    default_client_factory,
)
from app.services.company_exposure.resources import DispatchRequest, ResearchResources
from tests.fixtures.company_exposure.factory import FakeGoTransport, FixedClock

ENABLED = dict(
    text_route_enabled=True,
    subscription_key_present=True,
    daily_request_limit=5,
    daily_token_limit=100_000,
)


@pytest.fixture
def clock():
    return FixedClock()


@pytest.fixture
def go_transport():
    return FakeGoTransport()


def _resources(db_session, clock, **overrides):
    return ResearchResources(
        db_session, ExposureRuntimeConfig(**{**ENABLED, **overrides}), clock=clock.now
    )


def _dispatch(key="op-1"):
    return DispatchRequest(
        logical_operation_key=key,
        operation="claim_review",
        capability="text",
        input_hash="a" * 64,
        policy_hash="b" * 64,
        max_output_tokens=1000,
        estimated_input_tokens=200,
    )


def _runner(db_session, resources, go_transport, api_key="test-key"):
    provider = SubscriptionProvider(
        api_key=api_key, client_factory=default_client_factory(go_transport.transport)
    )
    return SubscriptionArtifactRunner(db_session, resources, provider)


@pytest.fixture
def metered_spy(monkeypatch):
    from app.services.llm import llm_service

    spy = MagicMock()
    monkeypatch.setattr(llm_service.LLMService, "completion", spy)
    return spy


@pytest.mark.case("R02")
@pytest.mark.exposure_layer("unit")
def test_subscription_pause_never_falls_back_to_metered(
    db_session, clock, go_transport, metered_spy
):
    resources = _resources(db_session, clock, daily_request_limit=0)
    ticket = resources.reserve(_dispatch())
    assert ticket.state == "paused_allowance"
    assert ticket.reason == "capacity_exhausted"
    runner = _runner(db_session, resources, go_transport)
    result = runner.run(
        ProviderInput(
            operation="claim_review",
            messages=[{"role": "user", "content": "x"}],
            input_hash="a" * 64,
            policy_hash="b" * 64,
            max_output_tokens=100,
            logical_operation_key="op-1",
        )
    )
    assert result.pause_reason == "capacity_exhausted"
    assert go_transport.requests == []
    metered_spy.assert_not_called()


@pytest.mark.case("R02")
@pytest.mark.exposure_layer("unit")
@pytest.mark.parametrize(
    ("overrides", "state", "reason"),
    [
        ({"daily_request_limit": None}, "paused_allowance", "allocation_not_configured"),
        ({"text_route_enabled": False}, "unavailable_capability", "route_not_approved"),
        (
            {"subscription_key_present": False},
            "unavailable_capability",
            "subscription_credentials_missing",
        ),
    ],
)
def test_missing_configuration_blocks_before_any_dispatch(
    db_session, clock, overrides, state, reason
):
    ticket = _resources(db_session, clock, **overrides).reserve(_dispatch())
    assert (ticket.allowed, ticket.state, ticket.reason) == (False, state, reason)


def test_vision_needs_its_own_route_approval(db_session, clock):
    resources = _resources(db_session, clock)
    vision = replace(_dispatch(), capability="vision")
    assert resources.reserve(vision).reason == "route_not_approved"


def test_other_routes_are_never_accepted(db_session, clock):
    resources = _resources(db_session, clock)
    metered = replace(_dispatch(), route="groq")
    assert resources.reserve(metered).reason == "route_not_approved"
