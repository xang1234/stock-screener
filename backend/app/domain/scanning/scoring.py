"""Pure scoring and rating policies for the scanning domain.

Extracted from scan_orchestrator.py so that core business rules
live inside the domain layer with zero infrastructure dependencies.

All functions are pure: no I/O, no side effects, fully deterministic.

Quality-aware fallback (T4)
---------------------------
The ``apply_quality_policy`` function adjusts the overall rating when the
underlying fundamentals are incomplete (per T2's
``field_completeness_score``). The policy has three explicit behaviours:

- **Exclusion**: completeness below ``QUALITY_EXCLUSION_THRESHOLD``
  forces ``RatingCategory.PASS``. Rows this sparse cannot be trusted
  to rank against fuller peers; they are scored out.
- **Downgrade**: completeness between ``QUALITY_EXCLUSION_THRESHOLD``
  and ``QUALITY_DOWNGRADE_THRESHOLD`` drops the rating one tier
  (STRONG_BUY → BUY, BUY → WATCH, WATCH → WATCH).
- **Tie-break**: rows with equal ``composite_score`` should break by
  higher ``field_completeness_score``. Consumers enforce this via an
  ``ORDER BY composite_score DESC, field_completeness_score DESC``
  secondary sort; this module documents the semantics but does not
  mutate the score itself (keeps the displayed score honest).

Unknown completeness (``None``) is a pass-through: legacy rows from
before the T2 migration are treated as "unknown quality" rather than
forced into PASS, so the rollout doesn't regress existing results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import CompositeMethod, RatingCategory, ScreenerOutputDomain

# ---------------------------------------------------------------------------
# Scoring thresholds (centralised so they can be referenced by tests)
# ---------------------------------------------------------------------------

STRONG_BUY_THRESHOLD: float = 80.0
BUY_THRESHOLD: float = 70.0
WATCH_THRESHOLD: float = 60.0

# Downgrade map: current rating → downgraded rating (one level lower)
_DOWNGRADE: dict[RatingCategory, RatingCategory] = {
    RatingCategory.STRONG_BUY: RatingCategory.BUY,
    RatingCategory.BUY: RatingCategory.WATCH,
    RatingCategory.WATCH: RatingCategory.WATCH,
    RatingCategory.PASS: RatingCategory.PASS,
}


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


def calculate_composite_score(
    screener_outputs: dict[str, ScreenerOutputDomain],
    method: CompositeMethod,
    weights: dict[str, float] | None = None,
) -> float:
    """Combine per-screener scores into a single composite score.

    Args:
        screener_outputs: Mapping of screener name → result.
        method: Aggregation strategy (weighted_average, maximum, minimum).
        weights: Optional per-screener weights (keyed by screener name).
                 If ``None`` or missing keys, equal weight is assumed.

    Returns:
        Composite score in the range 0-100.
    """
    if not screener_outputs:
        return 0.0

    if method is CompositeMethod.MAXIMUM:
        return max(o.score for o in screener_outputs.values())

    if method is CompositeMethod.MINIMUM:
        return min(o.score for o in screener_outputs.values())

    # weighted_average (default)
    if weights:
        total_weight = 0.0
        weighted_sum = 0.0
        for name, output in screener_outputs.items():
            w = weights.get(name, 1.0)
            weighted_sum += output.score * w
            total_weight += w
        return weighted_sum / total_weight if total_weight else 0.0

    # equal-weight average (original behaviour)
    scores = [o.score for o in screener_outputs.values()]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Overall rating
# ---------------------------------------------------------------------------


def calculate_overall_rating(
    composite_score: float,
    screener_outputs: dict[str, ScreenerOutputDomain],
) -> RatingCategory:
    """Derive a human-readable rating from the composite score.

    Two-step policy:
      1. Map composite score to a base rating via thresholds.
      2. If fewer than half of the screeners passed, downgrade one level.
         If *none* passed, force ``PASS`` regardless of score.

    Args:
        composite_score: Composite score (0-100).
        screener_outputs: Per-screener results (used for pass-rate check).

    Returns:
        A :class:`RatingCategory` enum member.
    """
    # --- Step 1: threshold-based base rating ---
    if composite_score >= STRONG_BUY_THRESHOLD:
        base = RatingCategory.STRONG_BUY
    elif composite_score >= BUY_THRESHOLD:
        base = RatingCategory.BUY
    elif composite_score >= WATCH_THRESHOLD:
        base = RatingCategory.WATCH
    else:
        base = RatingCategory.PASS

    # --- Step 2: pass-rate adjustment ---
    pass_count = sum(1 for o in screener_outputs.values() if o.passes)
    total_count = len(screener_outputs)

    if pass_count == 0:
        return RatingCategory.PASS

    if pass_count < total_count / 2:
        return _DOWNGRADE[base]

    return base


# ---------------------------------------------------------------------------
# Quality-aware fallback (T4)
# ---------------------------------------------------------------------------

QUALITY_EXCLUSION_THRESHOLD: int = 30
"""Below this ``field_completeness_score``, force rating to PASS.

Rationale: with fewer than ~30% of tier-weighted core fields present,
the composite score is effectively noise — downgrading to WATCH would
still let the row compete with fuller peers in the same tier.
"""

QUALITY_DOWNGRADE_THRESHOLD: int = 60
"""Below this ``field_completeness_score`` (but above exclusion), drop
one rating tier. Rows in this band have usable but partial data; the
displayed score is retained and the rating carries the quality signal.

Note: this value coincidentally equals ``WATCH_THRESHOLD`` (60), but the
semantics are orthogonal — WATCH_THRESHOLD is a 0-100 composite-score
cutoff, this is a 0-100 completeness-score cutoff. Changing one does
not imply changing the other.
"""


@dataclass(frozen=True)
class QualityAdjustment:
    """Result of applying quality-aware fallback to a rating.

    - ``rating``: the adjusted rating (may equal the input if no
      adjustment was needed).
    - ``reason``: human-readable explanation when an adjustment was
      made, or ``None`` when the rating passed through unchanged.
      Surfaced in the scan result so operators can trace *why* a row
      was downgraded.
    """
    rating: RatingCategory
    reason: Optional[str]


def apply_quality_policy(
    rating: RatingCategory,
    field_completeness_score: Optional[int],
) -> QualityAdjustment:
    """Downgrade or exclude ``rating`` based on ``field_completeness_score``.

    See module docstring for full policy semantics. Exhaustive behaviour:

    - ``None`` completeness → pass-through (unknown quality, don't penalise).
    - ``score < QUALITY_EXCLUSION_THRESHOLD`` → force ``PASS``.
    - ``QUALITY_EXCLUSION_THRESHOLD <= score < QUALITY_DOWNGRADE_THRESHOLD``
      → one tier down via ``_DOWNGRADE``.
    - ``score >= QUALITY_DOWNGRADE_THRESHOLD`` → pass-through.

    This function is a *further* adjustment applied after
    :func:`calculate_overall_rating`; the pass-rate downgrade happens
    first, then the quality-based downgrade refines it.
    """
    if field_completeness_score is None:
        return QualityAdjustment(rating=rating, reason=None)

    if field_completeness_score < QUALITY_EXCLUSION_THRESHOLD:
        if rating is RatingCategory.PASS:
            # Row was already PASS for other reasons (low composite score
            # or pass-rate downgrade). Don't attribute the PASS to the
            # quality policy — misleading. Floor-case consistent with
            # the downgrade branch below.
            return QualityAdjustment(rating=rating, reason=None)
        return QualityAdjustment(
            rating=RatingCategory.PASS,
            reason=(
                f"completeness {field_completeness_score} below "
                f"exclusion threshold {QUALITY_EXCLUSION_THRESHOLD}"
            ),
        )

    if field_completeness_score < QUALITY_DOWNGRADE_THRESHOLD:
        downgraded = _DOWNGRADE[rating]
        if downgraded is rating:
            # Already at the floor (WATCH/PASS) — no reason to record a
            # downgrade that didn't happen.
            return QualityAdjustment(rating=rating, reason=None)
        return QualityAdjustment(
            rating=downgraded,
            reason=(
                f"completeness {field_completeness_score} below "
                f"downgrade threshold {QUALITY_DOWNGRADE_THRESHOLD}"
            ),
        )

    return QualityAdjustment(rating=rating, reason=None)


__all__ = [
    "STRONG_BUY_THRESHOLD",
    "BUY_THRESHOLD",
    "WATCH_THRESHOLD",
    "QUALITY_EXCLUSION_THRESHOLD",
    "QUALITY_DOWNGRADE_THRESHOLD",
    "QualityAdjustment",
    "calculate_composite_score",
    "calculate_overall_rating",
    "apply_quality_policy",
]
