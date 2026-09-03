from __future__ import annotations

import pytest

from app.domain.options_analytics.metrics.activity import (
    calculate_activity_metrics,
    rank_activity,
)
from app.domain.options_analytics.models import MetricValue, NormalizedOptionContract, OptionSide


def _contract(strike: float, volume: int | None, oi: int | None):
    return NormalizedOptionContract(
        side=OptionSide.CALL,
        strike=strike,
        bid=1,
        ask=2,
        last_price=1.5,
        volume=volume,
        open_interest=oi,
        implied_volatility=0.2,
        last_trade_at=None,
        contract_size="REGULAR",
        multiplier=100,
    )


def test_activity_handles_zero_open_interest_without_division() -> None:
    metrics = calculate_activity_metrics((_contract(100, 120, 0),), spot=100)

    assert metrics.volume_oi_ratio.available is False
    assert metrics.volume_oi_ratio.reason_codes == ("open_interest_zero",)
    assert metrics.activity_intensity.available is False


def test_activity_uses_five_percent_concentration_and_100_contract_floor() -> None:
    metrics = calculate_activity_metrics(
        (_contract(100, 80, 100), _contract(104, 40, 100), _contract(110, 80, 200)),
        spot=100,
    )

    assert metrics.volume_oi_ratio.value == pytest.approx(0.5)
    assert metrics.near_spot_volume_concentration.value == pytest.approx(0.6)
    assert metrics.activity_intensity.value == pytest.approx(0.3)
    assert metrics.qualifying_volume == 200


def test_activity_below_100_contracts_is_unavailable() -> None:
    metrics = calculate_activity_metrics((_contract(100, 99, 100),), spot=100)

    assert metrics.activity_intensity.available is False
    assert metrics.activity_intensity.reason_codes == ("activity_volume_floor_not_met",)


def test_cross_sectional_activity_rank_is_stable_and_skips_unavailable() -> None:
    ranked = rank_activity(
        {
            "MSFT": MetricValue(available=True, value=0.5),
            "AAPL": MetricValue(available=True, value=0.5),
            "NVDA": MetricValue(available=True, value=0.8),
            "NONE": MetricValue(available=False, reason_codes=("unavailable",)),
        }
    )

    assert ranked == {"NVDA": 1, "AAPL": 2, "MSFT": 3, "NONE": None}

