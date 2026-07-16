import ast
from dataclasses import fields
from datetime import date
from pathlib import Path

import pytest

from app.services.derived_data_execution_policy import (
    DerivedDataExecutionMode,
    DerivedDataExecutionPolicy,
    DerivedDataTargetKind,
    DerivedDataValidationProfile,
    resolve_derived_data_execution_policy,
)


TODAY = date(2026, 7, 16)
HISTORICAL = date(2026, 7, 15)
BACKEND_ROOT = Path(__file__).resolve().parents[2]
LEGACY_NAMES = {
    "force_cache_only",
    "refresh_guarded_cache_only",
}


@pytest.mark.parametrize(
    ("requested", "target", "profile"),
    [
        (
            None,
            HISTORICAL,
            DerivedDataValidationProfile.PROVIDER_ALLOWED,
        ),
        (
            "auto",
            HISTORICAL,
            DerivedDataValidationProfile.PROVIDER_ALLOWED,
        ),
        (
            "auto",
            TODAY,
            DerivedDataValidationProfile.STRICT_WITH_WARMUP,
        ),
        (
            "strict_cache_only",
            HISTORICAL,
            DerivedDataValidationProfile.STRICT_WITHOUT_WARMUP,
        ),
        (
            "strict_cache_only",
            TODAY,
            DerivedDataValidationProfile.STRICT_WITHOUT_WARMUP,
        ),
        (
            "refresh_guarded",
            HISTORICAL,
            DerivedDataValidationProfile.TOLERANT_CACHE_ONLY,
        ),
        (
            "refresh_guarded",
            TODAY,
            DerivedDataValidationProfile.TOLERANT_CACHE_ONLY,
        ),
    ],
)
def test_policy_resolves_one_validation_profile(requested, target, profile):
    policy = resolve_derived_data_execution_policy(
        execution_policy=requested,
        target_date=target,
        current_date=TODAY,
    )

    assert policy.validation_profile is profile


def test_policy_stores_only_request_state():
    assert [field.name for field in fields(DerivedDataExecutionPolicy)] == [
        "mode",
        "target_kind",
        "same_day_warmup_bypassed",
    ]


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
    assert policy.requires_strict_completeness is True
    assert policy.requires_warmup_metadata is False


def test_auto_same_day_gap_fill_becomes_provider_allowed_historical_policy():
    policy = resolve_derived_data_execution_policy(
        target_date=TODAY,
        current_date=TODAY,
    )

    gap_policy = policy.for_gap_fill()

    assert gap_policy.mode is DerivedDataExecutionMode.AUTO
    assert gap_policy.target_kind is DerivedDataTargetKind.HISTORICAL
    assert (
        gap_policy.validation_profile
        is DerivedDataValidationProfile.PROVIDER_ALLOWED
    )


def test_guarded_gap_fill_preserves_guarded_policy():
    policy = resolve_derived_data_execution_policy(
        execution_policy="refresh_guarded",
        target_date=TODAY,
        current_date=TODAY,
    )

    assert policy.for_gap_fill() is policy


def test_policy_owns_guarded_response_metadata():
    policy = resolve_derived_data_execution_policy(
        execution_policy="refresh_guarded",
        target_date=HISTORICAL,
        current_date=TODAY,
    )

    result = policy.annotate_response({"status": "failed"})

    assert result == {
        "status": "failed",
        "cache_only": True,
        "cache_policy": "refresh_guarded",
    }


def test_invalid_serialized_policy_is_rejected():
    with pytest.raises(ValueError, match="Unknown derived-data execution policy"):
        resolve_derived_data_execution_policy(
            execution_policy="provider_if_maybe",
            target_date=HISTORICAL,
            current_date=TODAY,
        )


def test_legacy_policy_names_do_not_branch_below_task_boundary():
    for relative_path in (
        "app/services/breadth_calculator_service.py",
        "app/services/ibd_group_rank_service.py",
        "app/services/group_rank_input_loader.py",
    ):
        source = (BACKEND_ROOT / relative_path).read_text()
        for legacy_name in LEGACY_NAMES:
            assert legacy_name not in source

    for relative_path in (
        "app/tasks/breadth_tasks.py",
        "app/tasks/group_rank_tasks.py",
    ):
        source = (BACKEND_ROOT / relative_path).read_text()
        assert "DerivedDataExecutionMode.AUTO" not in source
        assert "DerivedDataExecutionMode.REFRESH_GUARDED" not in source
        assert "'policy' in locals()" not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.If, ast.IfExp, ast.While)):
                continue
            referenced = {
                child.id
                for child in ast.walk(node.test)
                if isinstance(child, ast.Name)
            }
            assert LEGACY_NAMES.isdisjoint(referenced), (
                relative_path,
                node.lineno,
                referenced,
            )
