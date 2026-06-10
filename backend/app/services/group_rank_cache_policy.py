"""Cache coverage requirement for group-rank calculations."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.price_coverage_policy import CACHE_ONLY_MIN_PRICE_COVERAGE


STRICT_GROUP_RANK_CACHE_COVERAGE = CACHE_ONLY_MIN_PRICE_COVERAGE


@dataclass(frozen=True)
class GroupRankCacheRequirement:
    enabled: bool
    min_coverage: float
    reason: str

    @classmethod
    def disabled(cls) -> "GroupRankCacheRequirement":
        return cls(enabled=False, min_coverage=0.0, reason="not_required")

    @classmethod
    def strict(cls) -> "GroupRankCacheRequirement":
        return cls.minimum(
            STRICT_GROUP_RANK_CACHE_COVERAGE,
            reason="strict_cache_only",
        )

    @classmethod
    def minimum(cls, min_coverage: float, *, reason: str) -> "GroupRankCacheRequirement":
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError("Group rank cache coverage must be between 0.0 and 1.0")
        return cls(
            enabled=True,
            min_coverage=min_coverage,
            reason=reason,
        )
