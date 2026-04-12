"""Integration tests proving HybridFundamentalsService Phase 3 honours the
routing policy: non-US symbols must not hit finviz.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.hybrid_fundamentals_service import HybridFundamentalsService


def _make_service() -> HybridFundamentalsService:
    finviz = MagicMock()
    finviz._rate_limiter = MagicMock()
    finviz.get_finviz_only_fields.return_value = {"short_float": 0.05}
    price_cache = MagicMock()
    price_cache.get_many.return_value = {}
    price_cache.get_historical_data.return_value = None
    svc = HybridFundamentalsService(
        include_finviz=True,
        finviz_service=finviz,
        price_cache=price_cache,
    )
    # Replace bulk fetcher so we don't hit yfinance in tests.
    svc.bulk_fetcher = MagicMock()
    svc.bulk_fetcher.fetch_batch_fundamentals.return_value = {}
    return svc


class TestPhase3PolicyFiltering:
    def test_non_us_symbols_are_excluded_from_finviz_phase(self):
        svc = _make_service()
        symbols = ["AAPL", "0700.HK", "7203.T", "2330.TW"]
        market_by_symbol = {
            "AAPL": "US",
            "0700.HK": "HK",
            "7203.T": "JP",
            "2330.TW": "TW",
        }

        svc.fetch_fundamentals_batch(
            symbols,
            include_technicals=False,
            include_finviz=True,
            market_by_symbol=market_by_symbol,
        )

        # Only AAPL (US) should have finviz called.
        called_symbols = [
            call.args[0] for call in svc.finviz_service.get_finviz_only_fields.call_args_list
        ]
        assert called_symbols == ["AAPL"]

    def test_missing_market_defaults_to_us(self):
        svc = _make_service()
        symbols = ["AAPL", "MSFT"]
        # No market_by_symbol provided — legacy behaviour: all treated as US.
        svc.fetch_fundamentals_batch(
            symbols, include_technicals=False, include_finviz=True
        )
        called_symbols = [
            call.args[0] for call in svc.finviz_service.get_finviz_only_fields.call_args_list
        ]
        assert sorted(called_symbols) == ["AAPL", "MSFT"]

    def test_all_non_us_batch_skips_finviz_entirely(self):
        svc = _make_service()
        symbols = ["0700.HK", "7203.T"]
        market_by_symbol = {"0700.HK": "HK", "7203.T": "JP"}

        svc.fetch_fundamentals_batch(
            symbols,
            include_technicals=False,
            include_finviz=True,
            market_by_symbol=market_by_symbol,
        )

        svc.finviz_service.get_finviz_only_fields.assert_not_called()
