"""Unit tests for the FX service (T3).

Covers the three-tier lookup chain (memo / Redis / DB / fetcher), the
USD identity short-circuit, and the FXQuote → metadata serialisation
that fundamentals storage consumes.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from app.services.fx_service import (
    FXQuote,
    FXService,
    MARKET_CURRENCY_MAP,
    SUPPORTED_CURRENCIES,
    USD,
    currency_for_market,
)


# --- currency_for_market ---------------------------------------------------

class TestCurrencyForMarket:
    def test_known_markets(self):
        assert currency_for_market("US") == "USD"
        assert currency_for_market("HK") == "HKD"
        assert currency_for_market("JP") == "JPY"
        assert currency_for_market("TW") == "TWD"

    def test_case_insensitive_and_trimmed(self):
        assert currency_for_market("  hk  ") == "HKD"
        assert currency_for_market("jp") == "JPY"

    def test_unknown_or_none_defaults_to_usd(self):
        assert currency_for_market(None) == "USD"
        assert currency_for_market("") == "USD"
        assert currency_for_market("XX") == "USD"

    def test_supported_set_matches_map(self):
        assert SUPPORTED_CURRENCIES == frozenset(MARKET_CURRENCY_MAP.values())

    def test_agrees_with_security_master_defaults(self):
        """Drift guard: fx_service.MARKET_CURRENCY_MAP mirrors
        security_master_service._MARKET_DEFAULTS. If a new market is added
        to one, CI must fail until the other catches up.
        """
        from app.services import security_master_service as sm
        for market, currency in MARKET_CURRENCY_MAP.items():
            sm_currency, _tz = sm._MARKET_DEFAULTS[market]
            assert sm_currency == currency, (
                f"Market {market!r}: fx_service says {currency!r}, "
                f"security_master says {sm_currency!r}."
            )
        assert set(MARKET_CURRENCY_MAP.keys()) == set(sm._MARKET_DEFAULTS.keys())


# --- FXQuote ---------------------------------------------------------------

class TestFXQuoteSerialisation:
    def test_metadata_roundtrip_keys(self):
        quote = FXQuote(
            from_currency="HKD", to_currency="USD",
            rate=0.128, as_of_date=date(2026, 4, 12), source="yfinance",
        )
        meta = quote.to_metadata()
        assert meta == {
            "from_currency": "HKD",
            "to_currency": "USD",
            "rate": 0.128,
            "as_of_date": "2026-04-12",
            "source": "yfinance",
        }


# --- FXService -------------------------------------------------------------

def _make_service(rate_fetcher=None, db_rows=None):
    """Return an FXService wired with mocks and no Redis/DB."""
    fake_db = MagicMock()
    # _read_database uses .order_by().first(); stub accordingly. None = no
    # persisted rate, forces the fetcher path.
    fake_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = db_rows
    return FXService(
        rate_fetcher=rate_fetcher or (lambda c: None),
        session_factory=lambda: fake_db,
        redis_client=None,
    )


class TestUSDIdentity:
    def test_usd_returns_identity_quote_without_fetcher(self):
        fetcher = MagicMock()
        svc = _make_service(rate_fetcher=fetcher)
        quote = svc.get_usd_rate("USD")
        assert quote is not None
        assert quote.rate == 1.0
        assert quote.source == FXService.SOURCE_IDENTITY
        fetcher.assert_not_called()


class TestLookupChain:
    def test_fetcher_called_when_cache_and_db_empty(self):
        calls = []

        def fetcher(currency):
            calls.append(currency)
            return 0.128

        svc = _make_service(rate_fetcher=fetcher)
        quote = svc.get_usd_rate("HKD")
        assert quote is not None
        assert quote.rate == 0.128
        assert quote.source == FXService.SOURCE_YFINANCE
        assert calls == ["HKD"]

    def test_memo_short_circuits_repeat_call(self):
        calls = []

        def fetcher(currency):
            calls.append(currency)
            return 0.128

        svc = _make_service(rate_fetcher=fetcher)
        svc.get_usd_rate("HKD")
        svc.get_usd_rate("HKD")
        svc.get_usd_rate("HKD")
        assert calls == ["HKD"]  # fetcher hit exactly once

    def test_missing_rate_returns_none(self):
        svc = _make_service(rate_fetcher=lambda c: None)
        assert svc.get_usd_rate("HKD") is None


class TestConvertToUSD:
    def test_usd_amount_unchanged(self):
        svc = _make_service()
        usd, quote = svc.convert_to_usd(1_000_000, "USD")
        assert usd == 1_000_000
        assert quote is not None
        assert quote.rate == 1.0

    def test_non_usd_amount_scaled(self):
        svc = _make_service(rate_fetcher=lambda c: 0.13)
        usd, quote = svc.convert_to_usd(1_000_000, "HKD")
        assert usd == pytest.approx(130_000)
        assert quote.from_currency == "HKD"

    def test_none_amount_returns_none(self):
        svc = _make_service(rate_fetcher=lambda c: 0.13)
        usd, quote = svc.convert_to_usd(None, "HKD")
        assert usd is None
        assert quote is None

    def test_missing_rate_returns_none_pair(self):
        svc = _make_service(rate_fetcher=lambda c: None)
        usd, quote = svc.convert_to_usd(1_000, "HKD")
        assert usd is None and quote is None

    def test_none_currency_returns_none_pair(self):
        svc = _make_service()
        usd, quote = svc.convert_to_usd(1_000, None)
        assert usd is None and quote is None


class TestInputNormalisation:
    def test_case_insensitive(self):
        svc = _make_service(rate_fetcher=lambda c: 0.128)
        quote = svc.get_usd_rate("hkd")
        assert quote is not None
        assert quote.from_currency == "HKD"

    def test_whitespace_handled(self):
        svc = _make_service(rate_fetcher=lambda c: 0.128)
        quote = svc.get_usd_rate("  HKD  ")
        assert quote is not None
        assert quote.from_currency == "HKD"

    def test_empty_string_returns_none(self):
        svc = _make_service(rate_fetcher=lambda c: 0.128)
        assert svc.get_usd_rate("") is None


class TestPrefetch:
    def test_prefetch_warms_memo_and_skips_unavailable(self):
        rates = {"HKD": 0.128, "JPY": 0.0065, "TWD": None}
        svc = _make_service(rate_fetcher=lambda c: rates.get(c))
        quotes = svc.prefetch(["HKD", "JPY", "TWD", "USD"])
        assert set(quotes.keys()) == {"HKD", "JPY", "USD"}
        # Subsequent lookup must not re-fetch.
        calls = []
        svc._rate_fetcher = lambda c: (calls.append(c), 999)[1]
        svc.get_usd_rate("HKD")
        assert calls == []
