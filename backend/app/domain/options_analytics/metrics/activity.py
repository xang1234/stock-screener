"""Open-interest and volume activity metrics without directional claims."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from ..models import MetricValue, NormalizedOptionContract

ACTIVITY_VOLUME_FLOOR = 100


@dataclass(frozen=True)
class ActivityMetrics:
    volume_oi_ratio: MetricValue
    near_spot_volume_concentration: MetricValue
    activity_intensity: MetricValue
    qualifying_volume: int | None


def _non_negative(value: int | None) -> int | None:
    if value is None or not math.isfinite(float(value)) or value < 0:
        return None
    return int(value)


def _complete_total(
    contracts: tuple[NormalizedOptionContract, ...],
    field: str,
) -> int | None:
    values = tuple(_non_negative(getattr(row, field)) for row in contracts)
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def calculate_activity_metrics(
    contracts: tuple[NormalizedOptionContract, ...], *, spot: float
) -> ActivityMetrics:
    total_volume = _complete_total(contracts, "volume")
    total_open_interest = _complete_total(contracts, "open_interest")
    if total_volume is None or total_open_interest is None:
        ratio = MetricValue(
            available=False,
            reason_codes=("activity_totals_incomplete",),
        )
    elif total_open_interest == 0:
        ratio = MetricValue(available=False, reason_codes=("open_interest_zero",))
    else:
        ratio = MetricValue(available=True, value=total_volume / total_open_interest)

    if total_volume is None:
        concentration = MetricValue(
            available=False,
            reason_codes=("volume_total_incomplete",),
        )
    elif total_volume == 0 or not math.isfinite(float(spot)) or spot <= 0:
        concentration = MetricValue(
            available=False,
            reason_codes=("volume_concentration_unavailable",),
        )
    else:
        near_volume = sum(
            value
            for row in contracts
            if abs(row.strike - spot) / spot <= 0.05
            and (value := _non_negative(row.volume)) is not None
        )
        concentration = MetricValue(available=True, value=near_volume / total_volume)

    if total_volume is None:
        intensity = MetricValue(
            available=False,
            reason_codes=("activity_totals_incomplete",),
        )
    elif total_volume < ACTIVITY_VOLUME_FLOOR:
        intensity = MetricValue(
            available=False,
            reason_codes=("activity_volume_floor_not_met",),
        )
    elif not ratio.available:
        intensity = MetricValue(available=False, reason_codes=ratio.reason_codes)
    else:
        intensity = MetricValue(
            available=True,
            value=float(ratio.value),
            label="Activity Intensity",
        )
    return ActivityMetrics(
        volume_oi_ratio=ratio,
        near_spot_volume_concentration=concentration,
        activity_intensity=intensity,
        qualifying_volume=total_volume,
    )


def rank_activity(values: Mapping[str, MetricValue]) -> dict[str, int | None]:
    available = sorted(
        (
            (symbol.strip().upper(), float(metric.value))
            for symbol, metric in values.items()
            if metric.available and metric.value is not None
        ),
        key=lambda row: (-row[1], row[0]),
    )
    ranks = {symbol: rank for rank, (symbol, _) in enumerate(available, 1)}
    return {
        symbol.strip().upper(): ranks.get(symbol.strip().upper()) for symbol in values
    }
