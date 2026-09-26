from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.domain.company_exposure.contracts import ResearchMode
from app.services.company_exposure.config import ExposureRuntimeConfig, load_config


@pytest.mark.case("R01")
@pytest.mark.case("R15")
@pytest.mark.exposure_layer("unit")
def test_defaults_do_not_spend_even_with_credentials():
    config = load_config(
        SimpleNamespace(opencode_go_api_key="secret-value", tavily_api_key="tvly")
    )
    assert config.research_mode is ResearchMode.DISABLED
    assert config.paid_search_enabled is False
    assert not config.route_approved("text")
    assert not config.route_approved("vision")
    assert config.daily_request_limit is None
    assert config.subscription_key_present is True


def test_public_status_never_contains_secret_values():
    config = load_config(SimpleNamespace(opencode_go_api_key="secret-value"))
    status = config.public_status()
    assert "secret-value" not in repr(status)
    assert status["provider_remaining_allowance"] == "unknown"
    assert status["subscription_route"] == "opencode-go/kimi-k2.6"


def test_allocation_period_uses_configured_timezone():
    config = ExposureRuntimeConfig(allocation_timezone="Asia/Singapore")
    at = datetime(2026, 9, 26, 17, 30, tzinfo=timezone.utc)  # 01:30 SGT next day
    period, end = config.allocation_period(at)
    assert period == "2026-09-27"
    assert end.astimezone(timezone.utc) == datetime(2026, 9, 27, 16, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        ExposureRuntimeConfig(allocation_timezone="Mars/Olympus")
