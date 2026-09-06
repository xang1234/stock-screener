from __future__ import annotations

from datetime import date

from app.services.market_session_lag import MarketSessionWindow


class Calendar:
    def is_trading_day(self, market: str, value: date) -> bool:
        return market == "US" and value.weekday() < 5

    def trading_days(self, market: str, start: date, end: date) -> list[date]:
        assert market == "US"
        return [
            start.fromordinal(ordinal)
            for ordinal in range(start.toordinal(), end.toordinal() + 1)
            if start.fromordinal(ordinal).weekday() < 5
        ]


def test_market_session_window_returns_the_requested_sessions_ending_on_date() -> None:
    window = MarketSessionWindow(Calendar(), market="us")

    assert window.is_session(date(2026, 9, 4))
    assert window.sessions_ending_on(date(2026, 9, 7), 3) == (
        date(2026, 9, 3),
        date(2026, 9, 4),
        date(2026, 9, 7),
    )
