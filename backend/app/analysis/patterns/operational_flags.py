"""Operational invalidation flags for the Setup Engine.

These flags surface operational risks (extended entries, broken support,
low liquidity, imminent earnings) that are distinct from data-quality flags.
They are informational warnings — they do NOT affect ``derived_ready`` or
``setup_ready``.

Architecture: pure functions operating on pre-computed inputs, making them
trivially testable without any database, API, or DataFrame dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.analysis.patterns.config import SetupEngineParameters
from app.analysis.patterns.report import InvalidationFlag


@dataclass(frozen=True)
class OperationalFlagInputs:
    """Pre-computed values needed to evaluate operational flags.

    All fields are optional — when ``None``, the corresponding flag is
    silently skipped (permissive by design).
    """

    distance_to_pivot_pct: float | None = None
    current_price: float | None = None
    ma_50: float | None = None
    adtv_usd: float | None = None
    next_earnings_date: date | None = None
    reference_date: date | None = None


def compute_operational_flags(
    inputs: OperationalFlagInputs,
    parameters: SetupEngineParameters,
) -> list[InvalidationFlag]:
    """Evaluate operational risk flags from pre-computed inputs.

    Returns a list of ``InvalidationFlag`` instances (may be empty).
    Each flag carries ``is_hard`` to distinguish structural breaks from
    soft caution warnings.
    """
    flags: list[InvalidationFlag] = []

    # 1. Too extended past pivot
    if (
        inputs.distance_to_pivot_pct is not None
        and inputs.distance_to_pivot_pct > parameters.too_extended_pivot_distance_pct
    ):
        flags.append(InvalidationFlag(
            code="too_extended",
            is_hard=False,
        ))

    # 2. Breaks 50-day moving average support
    if inputs.current_price is not None and inputs.ma_50 is not None:
        cushion = parameters.breaks_50d_support_cushion_pct
        threshold = inputs.ma_50 * (1.0 - cushion / 100.0)
        if inputs.current_price < threshold:
            flags.append(InvalidationFlag(
                code="breaks_50d_support",
                is_hard=True,
            ))

    # 3. Low liquidity (ADTV below minimum)
    if (
        inputs.adtv_usd is not None
        and inputs.adtv_usd < parameters.low_liquidity_adtv_min_usd
    ):
        flags.append(InvalidationFlag(
            code="low_liquidity",
            is_hard=False,
        ))

    # 4. Earnings soon (within risk window)
    if (
        inputs.next_earnings_date is not None
        and inputs.reference_date is not None
    ):
        days_until = (inputs.next_earnings_date - inputs.reference_date).days
        if 0 <= days_until <= parameters.earnings_soon_window_days:
            flags.append(InvalidationFlag(
                code="earnings_soon",
                is_hard=False,
            ))

    return flags
