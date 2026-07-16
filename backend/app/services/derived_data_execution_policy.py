"""Execution policy shared by breadth and group derived-data calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class DerivedDataExecutionMode(str, Enum):
    AUTO = "auto"
    STRICT_CACHE_ONLY = "strict_cache_only"
    REFRESH_GUARDED = "refresh_guarded"


class DerivedDataTargetKind(str, Enum):
    SAME_DAY = "same_day"
    HISTORICAL = "historical"


class DerivedDataValidationProfile(str, Enum):
    PROVIDER_ALLOWED = "provider_allowed"
    STRICT_WITH_WARMUP = "strict_with_warmup"
    STRICT_WITHOUT_WARMUP = "strict_without_warmup"
    TOLERANT_CACHE_ONLY = "tolerant_cache_only"


@dataclass(frozen=True)
class DerivedDataExecutionPolicy:
    mode: DerivedDataExecutionMode
    target_kind: DerivedDataTargetKind
    same_day_warmup_bypassed: bool = False

    def __post_init__(self) -> None:
        if self.same_day_warmup_bypassed and (
            self.mode is not DerivedDataExecutionMode.AUTO
            or self.target_kind is not DerivedDataTargetKind.SAME_DAY
        ):
            raise ValueError(
                "Same-day warmup can only be bypassed for automatic same-day runs"
            )

    @classmethod
    def provider_allowed(cls) -> "DerivedDataExecutionPolicy":
        return cls(
            mode=DerivedDataExecutionMode.AUTO,
            target_kind=DerivedDataTargetKind.HISTORICAL,
        )

    @property
    def validation_profile(self) -> DerivedDataValidationProfile:
        if self.mode is DerivedDataExecutionMode.STRICT_CACHE_ONLY:
            return DerivedDataValidationProfile.STRICT_WITHOUT_WARMUP
        if self.mode is DerivedDataExecutionMode.REFRESH_GUARDED:
            return DerivedDataValidationProfile.TOLERANT_CACHE_ONLY
        if self.target_kind is DerivedDataTargetKind.HISTORICAL:
            return DerivedDataValidationProfile.PROVIDER_ALLOWED
        if self.same_day_warmup_bypassed:
            return DerivedDataValidationProfile.STRICT_WITHOUT_WARMUP
        return DerivedDataValidationProfile.STRICT_WITH_WARMUP

    @property
    def allows_provider_reads(self) -> bool:
        return (
            self.validation_profile
            is DerivedDataValidationProfile.PROVIDER_ALLOWED
        )

    @property
    def cache_only(self) -> bool:
        return not self.allows_provider_reads

    @property
    def requires_strict_completeness(self) -> bool:
        return self.validation_profile in {
            DerivedDataValidationProfile.STRICT_WITH_WARMUP,
            DerivedDataValidationProfile.STRICT_WITHOUT_WARMUP,
        }

    @property
    def strict_completeness(self) -> bool:
        """Compatibility alias for callers not yet migrated to the closed model."""
        return self.requires_strict_completeness

    @property
    def requires_warmup_metadata(self) -> bool:
        return (
            self.validation_profile
            is DerivedDataValidationProfile.STRICT_WITH_WARMUP
        )

    @property
    def tolerates_partial_coverage(self) -> bool:
        return (
            self.validation_profile
            is DerivedDataValidationProfile.TOLERANT_CACHE_ONLY
        )

    @property
    def response_cache_policy(self) -> str | None:
        if self.mode is DerivedDataExecutionMode.REFRESH_GUARDED:
            return "refresh_guarded"
        return None

    def for_gap_fill(self) -> "DerivedDataExecutionPolicy":
        if self.mode is DerivedDataExecutionMode.AUTO:
            return self.provider_allowed()
        return self

    def annotate_response(
        self,
        result: dict,
        *,
        include_cache_only: bool = False,
    ) -> dict:
        if include_cache_only:
            result["cache_only"] = self.cache_only
        if self.response_cache_policy is not None:
            result["cache_only"] = True
            result["cache_policy"] = self.response_cache_policy
        return result


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

    same_day = target_date == current_date
    return DerivedDataExecutionPolicy(
        mode=mode,
        target_kind=(
            DerivedDataTargetKind.SAME_DAY
            if same_day
            else DerivedDataTargetKind.HISTORICAL
        ),
        same_day_warmup_bypassed=(
            mode is DerivedDataExecutionMode.AUTO
            and same_day
            and allow_same_day_warmup_bypass
        ),
    )
