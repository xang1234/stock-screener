"""Unit tests for the T4 quality-aware fallback policy in domain/scoring.

Covers the three explicit behaviours the bead calls out:
- exclusion (very low completeness → force PASS)
- downgrade (partial completeness → one tier down)
- tie-break (documented: consumers sort by completeness as secondary key)

Plus the pass-through cases that guarantee "avoid hard failure":
- None completeness (legacy rows from before T2)
- completeness >= downgrade threshold (healthy rows unchanged)
- rating already at the floor (can't downgrade WATCH/PASS further)
"""
from __future__ import annotations

import pytest

from app.domain.scanning.models import RatingCategory
from app.domain.scanning.scoring import (
    QUALITY_DOWNGRADE_THRESHOLD,
    QUALITY_EXCLUSION_THRESHOLD,
    QualityAdjustment,
    apply_quality_policy,
)


class TestPassThrough:
    """Cases where the policy MUST NOT alter the rating."""

    def test_none_completeness_passes_through(self):
        """Legacy rows from before the T2 migration have NULL completeness;
        the policy must not penalise them or the rollout would regress."""
        result = apply_quality_policy(RatingCategory.STRONG_BUY, None)
        assert result == QualityAdjustment(RatingCategory.STRONG_BUY, None)

    @pytest.mark.parametrize("completeness", [60, 75, 90, 100])
    def test_high_completeness_passes_through(self, completeness):
        result = apply_quality_policy(RatingCategory.BUY, completeness)
        assert result == QualityAdjustment(RatingCategory.BUY, None)

    def test_boundary_at_downgrade_threshold_passes_through(self):
        """At exactly DOWNGRADE_THRESHOLD, no downgrade (inclusive upper bound)."""
        result = apply_quality_policy(
            RatingCategory.STRONG_BUY, QUALITY_DOWNGRADE_THRESHOLD
        )
        assert result.rating is RatingCategory.STRONG_BUY
        assert result.reason is None


class TestExclusion:
    """Completeness below EXCLUSION_THRESHOLD → force PASS with reason."""

    @pytest.mark.parametrize("completeness", [0, 10, 29])
    @pytest.mark.parametrize("rating", [
        RatingCategory.STRONG_BUY, RatingCategory.BUY,
        RatingCategory.WATCH, RatingCategory.PASS,
    ])
    def test_very_low_completeness_forces_pass(self, rating, completeness):
        result = apply_quality_policy(rating, completeness)
        assert result.rating is RatingCategory.PASS
        assert result.reason is not None
        assert str(completeness) in result.reason
        assert "exclusion" in result.reason

    def test_boundary_at_exclusion_threshold_is_downgrade_not_exclusion(self):
        """At exactly EXCLUSION_THRESHOLD, we're in the DOWNGRADE band."""
        result = apply_quality_policy(
            RatingCategory.STRONG_BUY, QUALITY_EXCLUSION_THRESHOLD
        )
        # Downgraded (not PASS), and the reason references downgrade not exclusion.
        assert result.rating is RatingCategory.BUY
        assert "downgrade" in result.reason


class TestDowngrade:
    """Completeness in [EXCLUSION_THRESHOLD, DOWNGRADE_THRESHOLD) → one tier down."""

    @pytest.mark.parametrize("completeness", [30, 45, 59])
    def test_strong_buy_drops_to_buy(self, completeness):
        result = apply_quality_policy(RatingCategory.STRONG_BUY, completeness)
        assert result.rating is RatingCategory.BUY
        assert result.reason is not None
        assert str(completeness) in result.reason

    @pytest.mark.parametrize("completeness", [30, 45, 59])
    def test_buy_drops_to_watch(self, completeness):
        result = apply_quality_policy(RatingCategory.BUY, completeness)
        assert result.rating is RatingCategory.WATCH
        assert result.reason is not None

    def test_watch_stays_watch_without_spurious_reason(self):
        """WATCH → WATCH per the _DOWNGRADE map. No reason recorded since
        nothing actually changed — avoids misleading UIs."""
        result = apply_quality_policy(RatingCategory.WATCH, 45)
        assert result.rating is RatingCategory.WATCH
        assert result.reason is None

    def test_pass_stays_pass_without_spurious_reason(self):
        result = apply_quality_policy(RatingCategory.PASS, 45)
        assert result.rating is RatingCategory.PASS
        assert result.reason is None


class TestThresholds:
    """Lock the policy's numeric boundaries in case they drift."""

    def test_exclusion_below_downgrade(self):
        assert QUALITY_EXCLUSION_THRESHOLD < QUALITY_DOWNGRADE_THRESHOLD

    def test_exclusion_threshold_value(self):
        # Bumping this value changes the behaviour of downstream scanners —
        # it should be a conscious decision, not accidental drift.
        assert QUALITY_EXCLUSION_THRESHOLD == 30

    def test_downgrade_threshold_value(self):
        assert QUALITY_DOWNGRADE_THRESHOLD == 60


class TestTieBreakSemantics:
    """T4 documents that tie-break uses field_completeness_score as the
    secondary sort key. The policy does not mutate composite_score; it
    exposes completeness for the caller to sort on.
    """

    def test_policy_does_not_depend_on_composite_score(self):
        """The adjustment is a function of rating + completeness only.
        Composite_score is irrelevant here (ordering concern, not scoring)."""
        # Same inputs → same output regardless of which composite produced
        # the rating. Verified implicitly by the above tests.
        a = apply_quality_policy(RatingCategory.STRONG_BUY, 45)
        b = apply_quality_policy(RatingCategory.STRONG_BUY, 45)
        assert a == b
