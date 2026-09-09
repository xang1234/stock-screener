from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.config.settings import Settings


def test_social_ingest_defaults_fail_closed():
    settings = Settings(_env_file=None)

    assert settings.social_signals_mode == "off"
    assert settings.social_ingest_provider == "disabled"
    assert settings.social_refresh_hours == 6
    assert settings.social_manual_refresh_cooldown_seconds == 3600
    assert settings.social_stale_after_hours == 7
    assert settings.social_initial_backfill_days == 14
    assert settings.social_initial_backfill_limit_per_source == 1000
    assert settings.social_incremental_limit_per_source == 200
    assert settings.social_xui_config_path == "/app/data/xui-reader/config.toml"
    assert settings.social_xui_profile == "automation"
    assert settings.social_official_daily_post_limit == 2000
    assert settings.social_llm_daily_budget_usd == Decimal("2.00")
    assert settings.social_llm_budget_timezone == "Asia/Singapore"
    assert settings.social_llm_batch_size == 20
    assert settings.social_llm_min_interval_seconds == 5
    assert settings.social_llm_max_calls_per_run == 20
    assert settings.social_llm_max_calls_per_day == 80
    assert settings.social_market_close_grace_minutes == 120
    assert settings.capability_flags()["social_signals"] is False


@pytest.mark.parametrize("value", ["disabled", "official", "xui"])
def test_social_provider_accepts_only_explicit_modes(value):
    assert (
        Settings(_env_file=None, social_ingest_provider=value).social_ingest_provider
        == value
    )


@pytest.mark.parametrize("value", ["off", "validation", "live"])
def test_social_signals_mode_accepts_only_explicit_modes(value):
    assert Settings(_env_file=None, social_signals_mode=value).social_signals_mode == value


@pytest.mark.parametrize("field", ["social_signals_mode", "social_ingest_provider"])
def test_social_modes_reject_unknown_values(field):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: "automatic"})


@pytest.mark.parametrize("mode", ["off", "validation", "live"])
@pytest.mark.parametrize("provider", ["disabled", "official", "xui"])
def test_social_capability_requires_live_mode_and_configured_provider(mode, provider):
    settings = Settings(
        _env_file=None,
        social_signals_mode=mode,
        social_ingest_provider=provider,
    )

    assert settings.capability_flags()["social_signals"] is (
        mode == "live" and provider != "disabled"
    )


def test_social_budget_timezone_requires_an_iana_timezone():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, social_llm_budget_timezone="Singapore-ish")


def test_social_refresh_cadence_accepts_even_day_divisors_only():
    assert Settings(_env_file=None, social_refresh_hours=3).social_refresh_hours == 3
    for value in (0, 5, 25):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, social_refresh_hours=value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("social_llm_batch_size", 0),
        ("social_llm_batch_size", 51),
        ("social_llm_min_interval_seconds", -1),
        ("social_llm_max_calls_per_run", 0),
        ("social_llm_max_calls_per_day", 0),
    ],
)
def test_social_llm_request_guards_reject_unsafe_values(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})
