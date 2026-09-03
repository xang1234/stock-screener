"""Expiration choice and persistence-window policies."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

from .models import NormalizedOptionContract

MIN_EXPIRATION_DTE = 14
MAX_EXPIRATION_DTE = 45
STRIKES_PER_SIDE = 30


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    first_friday = 1 + (4 - first.weekday()) % 7
    return date(year, month, first_friday + 14)


def _standard_monthly_expiration(year: int, month: int, calendar: object) -> date:
    expiration = _third_friday(year, month)
    while not calendar.is_session(expiration):
        expiration -= timedelta(days=1)
    return expiration


def select_monthly_expiration(
    *,
    as_of_date: date,
    listed_expirations: Iterable[date],
    calendar: object,
) -> date | None:
    eligible: list[date] = []
    for expiration in set(listed_expirations):
        dte = (expiration - as_of_date).days
        if not MIN_EXPIRATION_DTE <= dte <= MAX_EXPIRATION_DTE:
            continue
        expected = _standard_monthly_expiration(
            expiration.year, expiration.month, calendar
        )
        if expiration == expected:
            eligible.append(expiration)
    return min(eligible) if eligible else None


def retain_contracts_for_persistence(
    contracts: Iterable[NormalizedOptionContract],
    *,
    spot_price: float,
) -> tuple[NormalizedOptionContract, ...]:
    rows = tuple(contracts)
    strikes = sorted({row.strike for row in rows})
    if not strikes:
        return ()
    anchor = min(strikes, key=lambda strike: (abs(strike - spot_price), strike))
    lower = [strike for strike in strikes if strike < anchor][-STRIKES_PER_SIDE:]
    higher = [strike for strike in strikes if strike > anchor][:STRIKES_PER_SIDE]
    retained = set((*lower, anchor, *higher))
    return tuple(
        sorted(
            (row for row in rows if row.strike in retained),
            key=lambda row: (row.strike, row.side.value),
        )
    )
