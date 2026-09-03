from __future__ import annotations

from app.domain.options_analytics.metrics.max_pain import calculate_max_pain
from app.domain.options_analytics.models import NormalizedOptionContract, OptionSide


def _contract(side: OptionSide, strike: float, oi: int | None) -> NormalizedOptionContract:
    return NormalizedOptionContract(
        side=side,
        strike=strike,
        bid=1,
        ask=2,
        last_price=1.5,
        volume=10,
        open_interest=oi,
        implied_volatility=0.25,
        last_trade_at=None,
        contract_size="REGULAR",
        multiplier=100,
    )


def test_max_pain_uses_every_usable_strike_and_resolves_ties_lower() -> None:
    contracts = (
        _contract(OptionSide.CALL, 90, 1),
        _contract(OptionSide.CALL, 100, 0),
        _contract(OptionSide.PUT, 110, 1),
    )

    result = calculate_max_pain(contracts)

    assert result.available is True
    assert result.value == 90


def test_max_pain_is_unavailable_without_non_negative_open_interest() -> None:
    result = calculate_max_pain(
        (_contract(OptionSide.CALL, 100, None), _contract(OptionSide.PUT, 100, -1))
    )

    assert result.available is False
    assert result.reason_codes == ("open_interest_unavailable",)

