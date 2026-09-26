from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.config.exposure_settings import ExposureSettings
from app.domain.company_exposure.contracts import ResearchMode
from app.services.company_exposure.config import ExposureRuntimeConfig, load_config


@pytest.mark.case("R01")
@pytest.mark.case("R15")
@pytest.mark.exposure_layer("unit")
def test_defaults_do_not_spend_even_with_credentials():
    config = load_config(
        ExposureSettings(_env_file=None), subscription_key="secret-value"
    )
    assert config.research_mode is ResearchMode.DISABLED
    assert config.paid_search_enabled is False
    assert not config.route_approved("text")
    assert not config.route_approved("vision")
    assert config.daily_request_limit is None
    assert config.subscription_key_present is True


def test_public_status_never_contains_secret_values():
    config = load_config(
        ExposureSettings(_env_file=None), subscription_key="secret-value"
    )
    status = config.public_status()
    assert "secret-value" not in repr(status)
    assert status["provider_remaining_allowance"] == "unknown"
    assert status["subscription_route"] == "opencode-go/kimi-k2.6"


def test_allocation_period_uses_configured_timezone():
    config = ExposureRuntimeConfig(allocation_timezone="Asia/Singapore")
    at = datetime(2026, 9, 26, 17, 30, tzinfo=timezone.utc)  # 01:30 SGT next day
    period, end = config.allocation_period(at)
    assert period == "2026-09-27"
    assert end.astimezone(timezone.utc) == datetime(
        2026, 9, 27, 16, 0, tzinfo=timezone.utc
    )
    with pytest.raises(ValueError):
        ExposureRuntimeConfig(allocation_timezone="Mars/Olympus")


def test_environment_names_and_blank_limits(monkeypatch):
    monkeypatch.setenv("EXPOSURE_RESEARCH_MODE", "shadow")
    monkeypatch.setenv("EXPOSURE_LLM_DAILY_REQUEST_LIMIT", "")
    monkeypatch.setenv("EXPOSURE_LLM_DAILY_TOKEN_LIMIT", "5000")
    settings = ExposureSettings(_env_file=None)
    assert settings.research_mode is ResearchMode.SHADOW
    assert (settings.llm_daily_request_limit, settings.llm_daily_token_limit) == (
        None,
        5000,
    )
    monkeypatch.setenv("EXPOSURE_RESEARCH_MODE", "sometimes")
    with pytest.raises(ValueError):
        ExposureSettings(_env_file=None)
