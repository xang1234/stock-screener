from __future__ import annotations

import pytest
from app.domain.options_analytics.metrics.gex import (
    black_scholes_unit_gamma,
    estimate_contract_gex,
    estimate_gamma_flip,
    estimate_gex_walls,
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


def test_gamma_flip_reprices_the_chain_across_hypothetical_spot_levels() -> None:
    contracts = (
        _contract(OptionSide.CALL, 90, oi=1_000),
        _contract(OptionSide.PUT, 110, oi=1_000),
    )
    result = estimate_gamma_flip(
        contracts,
        pinned_spot=100,
        time_years=0.25,
        rate=0.04,
        dividend_yield=0.01,
    )

    assert 90 < result.value < 110
    assert result.label == "Estimated Gamma Flip"
    assert result.evidence["method"] == "chain_repricing_linear_interpolation"
    assert result.evidence["grid_step"] == 1.0


def test_gamma_flip_is_unavailable_when_profile_never_crosses_zero() -> None:
    result = estimate_gamma_flip(
        (_contract(OptionSide.CALL, 90, oi=1_000),),
        pinned_spot=100,
        time_years=0.25,
        rate=0.04,
        dividend_yield=0.01,
    )

    assert result.available is False
    assert result.reason_codes == ("gamma_crossing_unavailable",)


def test_gamma_flip_is_unavailable_for_a_flat_zero_profile() -> None:
    result = estimate_gamma_flip(
        (
            _contract(OptionSide.CALL, 90, oi=100),
            _contract(OptionSide.PUT, 90, oi=100),
            _contract(OptionSide.CALL, 100, oi=100),
            _contract(OptionSide.PUT, 100, oi=100),
            _contract(OptionSide.CALL, 110, oi=100),
            _contract(OptionSide.PUT, 110, oi=100),
        ),
        pinned_spot=100,
        time_years=0.25,
        rate=0.04,
        dividend_yield=0.01,
    )

    assert result.available is False
    assert result.reason_codes == ("gamma_crossing_unavailable",)


def test_walls_use_absolute_estimated_side_gex_not_raw_open_interest() -> None:
    call_wall, put_wall = estimate_gex_walls(
        (
            _contract(OptionSide.CALL, 100, oi=100),
            _contract(OptionSide.CALL, 130, oi=500),
            _contract(OptionSide.PUT, 100, oi=100),
            _contract(OptionSide.PUT, 70, oi=500),
        ),
        spot=100,
        time_years=0.25,
        rate=0.04,
        dividend_yield=0.01,
    )

    assert call_wall.value == 100
    assert call_wall.label == "Estimated Call Wall"
    assert call_wall.evidence["method"] == "maximum_absolute_estimated_side_gex"
    assert put_wall.value == 100
    assert put_wall.label == "Estimated Put Wall"
