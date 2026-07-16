"""Execution policy shared by breadth and group derived-data calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class DerivedDataExecutionMode(str, Enum):
    AUTO = "auto"
    STRICT_CACHE_ONLY = "strict_cache_only"
    REFRESH_GUARDED = "refresh_guarded"


@dataclass(frozen=True)
class DerivedDataExecutionPolicy:
    mode: DerivedDataExecutionMode
    cache_only: bool
    strict_completeness: bool
    requires_warmup_metadata: bool
    tolerates_partial_coverage: bool

    @classmethod
    def provider_allowed(cls) -> "DerivedDataExecutionPolicy":
        return cls(
            mode=DerivedDataExecutionMode.AUTO,
            cache_only=False,
            strict_completeness=False,
            requires_warmup_metadata=False,
            tolerates_partial_coverage=False,
        )


def resolve_derived_data_execution_policy(
    *,
    target_date: date,
    current_date: date,
    execution_policy: str | DerivedDataExecutionMode | None = None,
    force_cache_only: bool = False,
    refresh_guarded_cache_only: bool = False,
    allow_same_day_warmup_bypass: bool = False,
) -> DerivedDataExecutionPolicy:
    if force_cache_only:
        mode = DerivedDataExecutionMode.STRICT_CACHE_ONLY
    elif refresh_guarded_cache_only:
        mode = DerivedDataExecutionMode.REFRESH_GUARDED
    elif execution_policy is None:
        mode = DerivedDataExecutionMode.AUTO
    else:
        try:
            mode = DerivedDataExecutionMode(execution_policy)
        except ValueError as exc:
            raise ValueError(
                f"Unknown derived-data execution policy: {execution_policy}"
            ) from exc

    if mode is DerivedDataExecutionMode.STRICT_CACHE_ONLY:
        return DerivedDataExecutionPolicy(
            mode=mode,
            cache_only=True,
            strict_completeness=True,
            requires_warmup_metadata=False,
            tolerates_partial_coverage=False,
        )

    if mode is DerivedDataExecutionMode.REFRESH_GUARDED:
        return DerivedDataExecutionPolicy(
            mode=mode,
            cache_only=True,
            strict_completeness=False,
            requires_warmup_metadata=False,
            tolerates_partial_coverage=True,
        )

    same_day = target_date == current_date
    return DerivedDataExecutionPolicy(
        mode=mode,
        cache_only=same_day,
        strict_completeness=same_day,
        requires_warmup_metadata=(
            same_day and not allow_same_day_warmup_bypass
        ),
        tolerates_partial_coverage=False,
    )
