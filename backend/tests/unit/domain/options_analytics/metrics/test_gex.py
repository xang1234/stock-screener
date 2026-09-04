from __future__ import annotations

import pytest

from app.domain.options_analytics.metrics.gex import (
    black_scholes_unit_gamma,
    estimate_contract_gex,
    estimate_gamma_flip,
    estimate_open_interest_walls,
)
from app.domain.options_analytics.models import NormalizedOptionContract, OptionSide


def _contract(
    side: OptionSide,
    strike: float,
    *,
    oi: int = 10,
    iv: float = 0.2,
    multiplier: int | None = 100,
) -> NormalizedOptionContract:
    return NormalizedOptionContract(
        side=side,
        strike=strike,
        bid=1,
        ask=2,
        last_price=1.5,
        volume=10,
        open_interest=oi,
        implied_volatility=iv,
        last_trade_at=None,
        contract_size="REGULAR" if multiplier == 100 else "NON_STANDARD",
        multiplier=multiplier,
    )


def test_black_scholes_gamma_and_dollar_gex_per_one_percent_move() -> None:
    gamma = black_scholes_unit_gamma(
        spot=100, strike=100, time_years=0.25, rate=0.04, dividend_yield=0.01, volatility=0.2
    )
    result = estimate_contract_gex(
        _contract(OptionSide.CALL, 100),
        spot=100,
        time_years=0.25,
        rate=0.04,
        dividend_yield=0.01,
    )

    assert gamma == pytest.approx(0.039484932869, rel=1e-10)
    assert result.value == pytest.approx(3948.4932869, rel=1e-10)
    assert result.label == "Estimated Call GEX"
    assert result.evidence["dealer_proxy_sign"] == "calls_positive_puts_negative"


def test_put_gex_is_negative_under_the_disclosed_dealer_proxy() -> None:
    result = estimate_contract_gex(
        _contract(OptionSide.PUT, 100),
        spot=100,
        time_years=0.25,
        rate=0.04,
        dividend_yield=0.01,
    )

    assert result.value == pytest.approx(-3948.4932869, rel=1e-10)
    assert result.label == "Estimated Put GEX"


def test_non_regular_multiplier_makes_gex_unavailable() -> None:
    result = estimate_contract_gex(
        _contract(OptionSide.CALL, 100, multiplier=None),
        spot=100,
        time_years=0.25,
        rate=0.04,
        dividend_yield=0.01,
    )

    assert result.available is False
    assert result.reason_codes == ("contract_multiplier_unavailable",)


def test_gamma_flip_interpolates_a_real_sign_crossing() -> None:
    result = estimate_gamma_flip(((90.0, -20.0), (100.0, 20.0), (110.0, 30.0)))

    assert result.value == pytest.approx(95.0)
    assert result.label == "Estimated Gamma Flip"


def test_gamma_flip_is_unavailable_when_profile_never_crosses_zero() -> None:
    result = estimate_gamma_flip(((90.0, 10.0), (100.0, 20.0)))

    assert result.available is False
    assert result.reason_codes == ("gamma_crossing_unavailable",)


def test_open_interest_walls_are_estimates_with_stable_lower_strike_ties() -> None:
    call_wall, put_wall = estimate_open_interest_walls(
        (
            _contract(OptionSide.CALL, 100, oi=50),
            _contract(OptionSide.CALL, 110, oi=50),
            _contract(OptionSide.PUT, 90, oi=60),
            _contract(OptionSide.PUT, 80, oi=60),
        )
    )

    assert call_wall.value == 100
    assert call_wall.label == "Estimated Call Wall"
    assert put_wall.value == 80
    assert put_wall.label == "Estimated Put Wall"
