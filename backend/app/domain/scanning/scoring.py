"""Pure scoring and rating policies for the scanning domain.

Extracted from scan_orchestrator.py so that core business rules
live inside the domain layer with zero infrastructure dependencies.

All functions are pure: no I/O, no side effects, fully deterministic.
"""

from __future__ import annotations

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


__all__ = [
    "STRONG_BUY_THRESHOLD",
    "BUY_THRESHOLD",
    "WATCH_THRESHOLD",
    "calculate_composite_score",
    "calculate_overall_rating",
]
