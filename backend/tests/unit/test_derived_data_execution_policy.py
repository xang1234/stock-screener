from datetime import date

import pytest

from app.services.derived_data_execution_policy import (
    DerivedDataExecutionMode,
    resolve_derived_data_execution_policy,
)


TODAY = date(2026, 7, 16)
HISTORICAL = date(2026, 7, 15)


@pytest.mark.parametrize(
    ("requested", "target", "cache_only", "strict", "warmup", "partial"),
    [
        (None, HISTORICAL, False, False, False, False),
        ("auto", HISTORICAL, False, False, False, False),
        ("auto", TODAY, True, True, True, False),
        ("strict_cache_only", HISTORICAL, True, True, False, False),
        ("strict_cache_only", TODAY, True, True, False, False),
        ("refresh_guarded", HISTORICAL, True, False, False, True),
        ("refresh_guarded", TODAY, True, False, False, True),
    ],
)
def test_policy_matrix(
    requested,
    target,
    cache_only,
    strict,
    warmup,
    partial,
):
    policy = resolve_derived_data_execution_policy(
        execution_policy=requested,
        target_date=target,
        current_date=TODAY,
    )

    assert policy.cache_only is cache_only
    assert policy.strict_completeness is strict
    assert policy.requires_warmup_metadata is warmup
    assert policy.tolerates_partial_coverage is partial


def test_force_cache_only_has_legacy_precedence():
    policy = resolve_derived_data_execution_policy(
        execution_policy="refresh_guarded",
        force_cache_only=True,
        refresh_guarded_cache_only=True,
        target_date=HISTORICAL,
        current_date=TODAY,
    )

    assert policy.mode is DerivedDataExecutionMode.STRICT_CACHE_ONLY


def test_legacy_guarded_flag_maps_to_guarded_mode():
    policy = resolve_derived_data_execution_policy(
        refresh_guarded_cache_only=True,
        target_date=HISTORICAL,
        current_date=TODAY,
    )

    assert policy.mode is DerivedDataExecutionMode.REFRESH_GUARDED


def test_same_day_auto_bypass_removes_warmup_requirement():
    policy = resolve_derived_data_execution_policy(
        target_date=TODAY,
        current_date=TODAY,
        allow_same_day_warmup_bypass=True,
    )

    assert policy.mode is DerivedDataExecutionMode.AUTO
    assert policy.cache_only is True
    assert policy.strict_completeness is True
    assert policy.requires_warmup_metadata is False


def test_invalid_serialized_policy_is_rejected():
    with pytest.raises(ValueError, match="Unknown derived-data execution policy"):
        resolve_derived_data_execution_policy(
            execution_policy="provider_if_maybe",
            target_date=HISTORICAL,
            current_date=TODAY,
        )
