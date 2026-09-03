from __future__ import annotations

import pytest

from app.domain.options_analytics.metrics.volatility import (
    calculate_25_delta_skew,
    calculate_atm_iv,
    calculate_realized_volatility,
    calculate_volatility_risk_premium,
)
from app.domain.options_analytics.models import NormalizedOptionContract, OptionSide


def _contract(side: OptionSide, strike: float, iv: float | None, delta: float | None = None):
    return NormalizedOptionContract(
        side=side,
        strike=strike,
        bid=1,
        ask=2,
        last_price=1.5,
        volume=10,
        open_interest=20,
        implied_volatility=iv,
        last_trade_at=None,
        contract_size="REGULAR",
        multiplier=100,
        delta=delta,
    )


def test_atm_iv_requires_valid_call_and_put_at_closest_strike() -> None:
    result = calculate_atm_iv(
        (
            _contract(OptionSide.CALL, 100, 0.20),
            _contract(OptionSide.PUT, 100, 0.30),
            _contract(OptionSide.CALL, 105, 0.40),
            _contract(OptionSide.PUT, 105, 0.50),
        ),
        spot=101,
    )

    assert result.value == pytest.approx(0.25)


def test_atm_iv_is_unavailable_when_either_side_is_invalid() -> None:
    result = calculate_atm_iv(
        (_contract(OptionSide.CALL, 100, 0.2), _contract(OptionSide.PUT, 100, float("nan"))),
        spot=100,
    )

    assert result.available is False
    assert result.reason_codes == ("atm_two_sided_iv_unavailable",)


def test_25_delta_skew_requires_both_sides_inside_delta_band() -> None:
    valid = calculate_25_delta_skew(
        (
            _contract(OptionSide.CALL, 105, 0.22, delta=0.25),
            _contract(OptionSide.PUT, 95, 0.31, delta=-0.24),
        )
    )
    invalid = calculate_25_delta_skew(
        (
            _contract(OptionSide.CALL, 105, 0.22, delta=0.19),
            _contract(OptionSide.PUT, 95, 0.31, delta=-0.24),
        )
    )

    assert valid.value == pytest.approx(0.09)
    assert invalid.available is False
    assert invalid.reason_codes == ("twenty_five_delta_pair_unavailable",)


def test_realized_volatility_uses_20_unfilled_returns_and_sqrt_252() -> None:
    result = calculate_realized_volatility(tuple(float(close) for close in range(100, 121)))

    assert result.value == pytest.approx(0.007808767470741185, rel=1e-12)


def test_realized_volatility_does_not_fill_missing_closes() -> None:
    closes = tuple(float(close) for close in range(100, 120)) + (None,)

    result = calculate_realized_volatility(closes)

    assert result.available is False
    assert result.reason_codes == ("realized_volatility_history_unavailable",)


def test_vrp_is_atm_iv_minus_realized_volatility() -> None:
    result = calculate_volatility_risk_premium(atm_iv=0.30, realized_volatility=0.20)

    assert result.value == pytest.approx(0.10)

