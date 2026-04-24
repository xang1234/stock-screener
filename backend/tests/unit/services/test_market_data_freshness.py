"""Unit tests for the symbol-scoped freshness check service.

Uses an in-memory session stub; no real DB required.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch


@dataclass
class _Row:
    symbol: str
    market: str
    last_date: date | None


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *args, **kwargs):
        return self

    def outerjoin(self, *args, **kwargs):
        return self

    def filter(self, *args, **kwargs):
        return self

    def group_by(self, *args, **kwargs):
        return self

    def all(self):
        return self._rows

    def close(self):
        pass


def _patch_session(rows):
    return patch(
        "app.services.market_data_freshness.SessionLocal",
        return_value=_FakeSession(rows),
    )


def _patch_calendar(last_completed_by_market):
    def _last_completed(market, now=None):
        return last_completed_by_market[market]

    fake_calendar = SimpleNamespace(last_completed_trading_day=_last_completed)
    return patch(
        "app.services.market_data_freshness.get_market_calendar_service",
        return_value=fake_calendar,
    )


def test_fresh_universe_returns_none():
    from app.services.market_data_freshness import check_symbol_freshness

    rows = [
        _Row(symbol="AAPL", market="US", last_date=date(2026, 4, 23)),
        _Row(symbol="MSFT", market="US", last_date=date(2026, 4, 23)),
    ]
    with _patch_session(rows), _patch_calendar({"US": date(2026, 4, 23)}):
        assert check_symbol_freshness(["AAPL", "MSFT"]) is None


def test_stale_market_returns_detail():
    from app.services.market_data_freshness import check_symbol_freshness

    rows = [
        _Row(symbol="AAPL", market="US", last_date=date(2026, 4, 22)),
    ]
    with _patch_session(rows), _patch_calendar({"US": date(2026, 4, 23)}):
        detail = check_symbol_freshness(["AAPL"])

    assert detail is not None
    assert detail["code"] == "market_data_stale"
    assert detail["stale_markets"][0]["market"] == "US"
    assert detail["stale_markets"][0]["oldest_last_cached_date"] == "2026-04-22"


def test_unresolved_symbols_flagged_even_if_covered_symbols_are_fresh():
    """Round 4 Codex P2: symbols requested by the scan but missing from
    stock_universe must be treated as stale — the freshness query silently
    drops them, so we need to compare requested-vs-resolved explicitly.
    """
    from app.services.market_data_freshness import check_symbol_freshness

    rows = [
        _Row(symbol="AAPL", market="US", last_date=date(2026, 4, 23)),
    ]
    with _patch_session(rows), _patch_calendar({"US": date(2026, 4, 23)}):
        detail = check_symbol_freshness(["AAPL", "UNKNOWN_FAKE", "MISSING"])

    assert detail is not None
    assert detail["code"] == "market_data_stale"
    assert detail["unresolved_symbols"] == ["MISSING", "UNKNOWN_FAKE"]
    assert "unknown symbols" in detail["message"]


def test_uncovered_symbols_in_known_market_flag_stale():
    """Symbol exists in stock_universe but has no stock_prices rows at all."""
    from app.services.market_data_freshness import check_symbol_freshness

    rows = [
        _Row(symbol="AAPL", market="US", last_date=date(2026, 4, 23)),
        _Row(symbol="NEWIPO", market="US", last_date=None),  # no cached prices
    ]
    with _patch_session(rows), _patch_calendar({"US": date(2026, 4, 23)}):
        detail = check_symbol_freshness(["AAPL", "NEWIPO"])

    assert detail is not None
    us = detail["stale_markets"][0]
    assert us["market"] == "US"
    assert us["uncovered_symbols"] == 1
    assert us["covered_symbols"] == 1


def test_empty_symbol_list_returns_none():
    from app.services.market_data_freshness import check_symbol_freshness

    assert check_symbol_freshness([]) is None
    assert check_symbol_freshness(["", None]) is None


def test_calendar_failure_fails_closed():
    """Round 6 CodeRabbit + Codex P1 (same finding): if
    last_completed_trading_day raises (unsupported market code, calendar
    backend outage, etc.) the gate must fail-closed — treat the market as
    stale rather than silently letting the scan through.
    """
    from app.services.market_data_freshness import check_symbol_freshness

    rows = [
        _Row(symbol="AAPL", market="US", last_date=date(2026, 4, 23)),
    ]

    def _raising_calendar(market, now=None):
        raise RuntimeError("calendar backend is down")

    with _patch_session(rows), patch(
        "app.services.market_data_freshness.get_market_calendar_service",
        return_value=SimpleNamespace(last_completed_trading_day=_raising_calendar),
    ):
        detail = check_symbol_freshness(["AAPL"])

    assert detail is not None
    assert detail["code"] == "market_data_stale"
    assert detail["stale_markets"][0]["market"] == "US"
    assert detail["stale_markets"][0]["reason"] == "calendar_unavailable"
    assert detail["stale_markets"][0]["expected_date"] is None
    assert "calendar unavailable" in detail["message"]
