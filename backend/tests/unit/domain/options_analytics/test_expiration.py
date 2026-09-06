from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.domain.options_analytics.expiration import (
    retain_contracts_for_persistence,
    select_monthly_expiration,
)
from app.domain.options_analytics.models import NormalizedOptionContract, OptionSide


class _SessionCalendar:
    def __init__(self, holidays: set[date] | None = None) -> None:
        self.holidays = holidays or set()

    def is_session(self, value: date) -> bool:
        return value.weekday() < 5 and value not in self.holidays


def _contract(strike: float) -> NormalizedOptionContract:
    return NormalizedOptionContract(
        side=OptionSide.CALL,
        strike=strike,
        bid=1.0,
        ask=1.2,
        last_price=1.1,
        volume=100,
        open_interest=200,
        implied_volatility=0.3,
        last_trade_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        contract_size="REGULAR",
        multiplier=100,
    )


def test_selects_nearest_standard_monthly_between_14_and_45_calendar_dte() -> None:
    selected = select_monthly_expiration(
        as_of_date=date(2026, 3, 1),
        listed_expirations=[date(2026, 3, 13), date(2026, 4, 17), date(2026, 3, 20)],
        calendar=_SessionCalendar(),
    )

    assert selected == date(2026, 3, 20)


def test_monthly_expiration_moves_to_thursday_for_friday_holiday() -> None:
    selected = select_monthly_expiration(
        as_of_date=date(2025, 3, 20),
        listed_expirations=[date(2025, 4, 17), date(2025, 4, 25)],
        calendar=_SessionCalendar({date(2025, 4, 18)}),
    )

    assert selected == date(2025, 4, 17)


@pytest.mark.parametrize(
    ("listed_expiration", "holidays"),
    [
        (date(2025, 4, 18), {date(2025, 4, 18)}),
        (date(2026, 4, 16), set()),
    ],
)
def test_accepts_provider_representation_of_thursday_or_friday_in_monthly_week(
    listed_expiration: date,
    holidays: set[date],
) -> None:
    selected = select_monthly_expiration(
        as_of_date=date(listed_expiration.year, 3, 20),
        listed_expirations=[listed_expiration],
        calendar=_SessionCalendar(holidays),
    )

    assert selected == listed_expiration


def test_weeklies_and_out_of_window_expirations_are_rejected() -> None:
    selected = select_monthly_expiration(
        as_of_date=date(2026, 3, 1),
        listed_expirations=[date(2026, 3, 6), date(2026, 3, 13), date(2026, 4, 17)],
        calendar=_SessionCalendar(),
    )

    assert selected is None


def test_persistence_keeps_anchor_plus_at_most_30_strikes_each_side() -> None:
    contracts = tuple(_contract(float(strike)) for strike in range(60, 141))

    retained = retain_contracts_for_persistence(contracts, spot_price=100.2)

    assert [contract.strike for contract in retained] == [
        float(strike) for strike in range(70, 131)
    ]
