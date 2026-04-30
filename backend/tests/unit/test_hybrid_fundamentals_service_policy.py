"""Integration tests proving HybridFundamentalsService Phase 3 honours the
routing policy: non-US symbols must not hit finviz.
"""
from __future__ import annotations

from unittest.mock import MagicMock

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


def _make_cn_data_source() -> MagicMock:
    data_source = MagicMock()
    data_source.get_combined_data.side_effect = lambda symbol, market=None: {
        "fundamentals": {
            "symbol": symbol,
            "market": market,
            "market_cap": 123.0,
            "data_source": "akshare+baostock",
        },
        "growth": {"eps_growth_qq": 12.5},
        "data_source": "akshare+baostock",
    }
    return data_source


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

    def test_market_map_is_forwarded_to_bulk_growth_extractor(self):
        svc = _make_service()
        symbols = ["0700.HK", "7203.T", "AAPL"]
        market_by_symbol = {"0700.HK": "HK", "7203.T": "JP", "AAPL": "US"}

        svc.fetch_fundamentals_batch(
            symbols,
            include_technicals=False,
            include_finviz=False,
            market_by_symbol=market_by_symbol,
        )

        assert svc.bulk_fetcher.fetch_batch_fundamentals.call_count == 1
        kwargs = svc.bulk_fetcher.fetch_batch_fundamentals.call_args.kwargs
        assert kwargs.get("market_by_symbol") == market_by_symbol

    def test_cn_symbols_use_cn_provider_path_and_skip_yfinance_batch(self):
        svc = _make_service()
        svc._data_source_service = _make_cn_data_source()
        symbols = ["920118.BJ", "000001.SZ"]
        market_by_symbol = {"920118.BJ": "CN", "000001.SZ": "CN"}

        result = svc.fetch_fundamentals_batch(
            symbols,
            include_technicals=False,
            include_finviz=False,
            market_by_symbol=market_by_symbol,
        )

        svc.bulk_fetcher.fetch_batch_fundamentals.assert_not_called()
        assert svc._data_source_service.get_combined_data.call_count == 2
        assert result["920118.BJ"]["data_source"] == "akshare+baostock"
        assert result["920118.BJ"]["eps_growth_qq"] == 12.5

    def test_mixed_cn_batch_only_sends_non_cn_symbols_to_yfinance(self):
        svc = _make_service()
        svc._data_source_service = _make_cn_data_source()
        svc.bulk_fetcher.fetch_batch_fundamentals.return_value = {
            "AAPL": {"market_cap": 456.0}
        }
        symbols = ["920118.BJ", "AAPL"]
        market_by_symbol = {"920118.BJ": "CN", "AAPL": "US"}

        result = svc.fetch_fundamentals_batch(
            symbols,
            include_technicals=False,
            include_finviz=False,
            market_by_symbol=market_by_symbol,
        )

        svc.bulk_fetcher.fetch_batch_fundamentals.assert_called_once()
        assert svc.bulk_fetcher.fetch_batch_fundamentals.call_args.args[0] == ["AAPL"]
        assert result["920118.BJ"]["market"] == "CN"
        assert result["AAPL"]["market_cap"] == 456.0

    def test_cn_suffix_without_market_map_skips_yfinance_and_finviz(self):
        svc = _make_service()
        svc._data_source_service = _make_cn_data_source()

        result = svc.fetch_fundamentals_batch(
            ["920118.BJ"],
            include_technicals=False,
            include_finviz=True,
            market_by_symbol=None,
        )

        svc.bulk_fetcher.fetch_batch_fundamentals.assert_not_called()
        svc.finviz_service.get_finviz_only_fields.assert_not_called()
        assert result["920118.BJ"]["market"] == "CN"

    def test_progress_callback_reaches_completion_when_finviz_phase_is_empty(self):
        svc = _make_service()
        symbols = ["0700.HK", "7203.T"]
        market_by_symbol = {"0700.HK": "HK", "7203.T": "JP"}
        progress_calls = []

        svc.fetch_fundamentals_batch(
            symbols,
            include_technicals=False,
            include_finviz=True,
            progress_callback=lambda current, total: progress_calls.append((current, total)),
            market_by_symbol=market_by_symbol,
        )

        assert progress_calls
        assert progress_calls[-1] == (len(symbols), len(symbols))

    def test_parallel_hybrid_forwards_market_map_to_parallel_growth_extractor(self):
        svc = _make_service()
        svc.bulk_fetcher.fetch_fundamentals_parallel.return_value = {}
        symbols = ["0700.HK", "AAPL"]
        market_by_symbol = {"0700.HK": "HK", "AAPL": "US"}

        svc.fetch_fundamentals_with_parallel_finviz(
            symbols,
            include_technicals=False,
            finviz_workers=1,
            market_by_symbol=market_by_symbol,
        )

        assert svc.bulk_fetcher.fetch_fundamentals_parallel.call_count == 1
        kwargs = svc.bulk_fetcher.fetch_fundamentals_parallel.call_args.kwargs
        assert kwargs.get("market_by_symbol") == market_by_symbol

    def test_parallel_cn_suffix_without_market_map_skips_yfinance_and_finviz(self):
        svc = _make_service()
        svc._data_source_service = _make_cn_data_source()
        svc.bulk_fetcher.fetch_fundamentals_parallel.return_value = {}

        result = svc.fetch_fundamentals_with_parallel_finviz(
            ["920118.BJ"],
            include_technicals=False,
            finviz_workers=1,
            market_by_symbol=None,
        )

        svc.bulk_fetcher.fetch_fundamentals_parallel.assert_not_called()
        svc.finviz_service.get_finviz_only_fields_batch.assert_not_called()
        assert result["920118.BJ"]["market"] == "CN"
