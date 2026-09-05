from __future__ import annotations

import pytest

from app.domain.options_analytics.metrics.activity import (
    calculate_activity_metrics,
    rank_activity,
)
from app.domain.options_analytics.models import (
    MetricValue,
    NormalizedOptionContract,
    OptionSide,
)


def _contract(
    strike: float,
    volume: int | None,
    oi: int | None,
    *,
    side: OptionSide = OptionSide.CALL,
):
    return NormalizedOptionContract(
        side=side,
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


def test_activity_calculates_call_put_volume_ratio() -> None:
    metrics = calculate_activity_metrics(
        (
            _contract(100, 120, 200),
            _contract(100, 80, 200, side=OptionSide.PUT),
        ),
        spot=100,
    )

    assert metrics.call_put_volume_ratio.value == pytest.approx(1.5)


def test_call_put_volume_ratio_rejects_incomplete_side_volume() -> None:
    metrics = calculate_activity_metrics(
        (
            _contract(100, 120, 200),
            _contract(100, None, 200, side=OptionSide.PUT),
        ),
        spot=100,
    )

    assert metrics.call_put_volume_ratio.available is False
    assert metrics.call_put_volume_ratio.reason_codes == ("side_volume_incomplete",)


def test_call_put_volume_ratio_rejects_an_absent_side() -> None:
    metrics = calculate_activity_metrics((_contract(100, 120, 200),), spot=100)

    assert metrics.call_put_volume_ratio.available is False
    assert metrics.call_put_volume_ratio.reason_codes == ("side_volume_incomplete",)


def test_call_put_volume_ratio_rejects_zero_put_volume() -> None:
    metrics = calculate_activity_metrics(
        (
            _contract(100, 120, 200),
            _contract(100, 0, 200, side=OptionSide.PUT),
        ),
        spot=100,
    )

    assert metrics.call_put_volume_ratio.available is False
    assert metrics.call_put_volume_ratio.reason_codes == ("put_volume_zero",)


def test_call_put_volume_ratio_preserves_zero_call_volume() -> None:
    metrics = calculate_activity_metrics(
        (
            _contract(100, 0, 200),
            _contract(100, 80, 200, side=OptionSide.PUT),
        ),
        spot=100,
    )

    assert metrics.call_put_volume_ratio.available is True
    assert metrics.call_put_volume_ratio.value == 0


def test_activity_handles_zero_open_interest_without_division() -> None:
    metrics = calculate_activity_metrics((_contract(100, 120, 0),), spot=100)

    assert metrics.volume_oi_ratio.available is False
    assert metrics.volume_oi_ratio.reason_codes == ("open_interest_zero",)
    assert metrics.near_spot_open_interest_concentration.available is False
    assert metrics.near_spot_open_interest_concentration.reason_codes == (
        "open_interest_concentration_unavailable",
    )
    assert metrics.activity_intensity.available is False


def test_activity_uses_five_percent_concentration() -> None:
    metrics = calculate_activity_metrics(
        (_contract(100, 80, 100), _contract(104, 40, 100), _contract(110, 80, 200)),
        spot=100,
    )

    assert metrics.volume_oi_ratio.value == pytest.approx(0.5)
    assert metrics.near_spot_volume_concentration.value == pytest.approx(0.6)
    assert metrics.near_spot_open_interest_concentration.value == pytest.approx(0.5)
    assert metrics.activity_intensity.value == pytest.approx(0.5)
    assert metrics.highest_contract_activity_ratio.available is False


def test_activity_below_100_contracts_keeps_aggregate_intensity_available() -> None:
    metrics = calculate_activity_metrics((_contract(100, 99, 100),), spot=100)

    assert metrics.activity_intensity.available is True
    assert metrics.activity_intensity.value == pytest.approx(0.99)
    assert metrics.highest_contract_activity_ratio.available is False
    assert metrics.highest_contract_activity_ratio.reason_codes == (
        "contract_volume_floor_not_met",
    )


def test_contract_activity_floor_is_not_satisfied_by_dispersed_volume() -> None:
    metrics = calculate_activity_metrics(
        (_contract(100, 60, 100), _contract(105, 60, 100)),
        spot=100,
    )

    assert metrics.activity_intensity.value == pytest.approx(0.6)
    assert metrics.highest_contract_activity_ratio.available is False


def test_contract_activity_reports_highest_qualifying_volume_oi_ratio() -> None:
    metrics = calculate_activity_metrics(
        (_contract(100, 120, 200), _contract(105, 200, 100)),
        spot=100,
    )

    assert metrics.highest_contract_activity_ratio.available is True
    assert metrics.highest_contract_activity_ratio.value == pytest.approx(2.0)


@pytest.mark.parametrize("volumes", [(None, None), (120, None)])
def test_activity_rejects_incomplete_volume_totals(
    volumes: tuple[int | None, int | None],
) -> None:
    metrics = calculate_activity_metrics(
        tuple(
            _contract(strike, volume, 200)
            for strike, volume in zip((100, 105), volumes, strict=True)
        ),
        spot=100,
    )

    assert metrics.volume_oi_ratio.available is False
    assert metrics.volume_oi_ratio.reason_codes == ("activity_totals_incomplete",)
    assert metrics.near_spot_volume_concentration.available is False
    assert metrics.activity_intensity.available is False


def test_activity_rejects_incomplete_open_interest_total() -> None:
    metrics = calculate_activity_metrics(
        (_contract(100, 120, 200), _contract(105, 80, None)),
        spot=100,
    )

    assert metrics.volume_oi_ratio.available is False
    assert metrics.volume_oi_ratio.reason_codes == ("activity_totals_incomplete",)
    assert metrics.near_spot_volume_concentration.available is True
    assert metrics.near_spot_open_interest_concentration.available is False
    assert metrics.near_spot_open_interest_concentration.reason_codes == (
        "open_interest_total_incomplete",
    )
    assert metrics.activity_intensity.available is False


def test_open_interest_concentration_preserves_zero_near_spot_interest() -> None:
    metrics = calculate_activity_metrics(
        (
            _contract(110, 120, 200),
            _contract(115, 80, 100, side=OptionSide.PUT),
        ),
        spot=100,
    )

    assert metrics.near_spot_open_interest_concentration.available is True
    assert metrics.near_spot_open_interest_concentration.value == 0


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
