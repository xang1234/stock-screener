"""Contract tests for StockDataProvider / DataPreparationLayer.

Verifies retry with backoff, partial failure semantics, rate-limit
handling, missing-data detection, bulk operations, and adapter delegation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.domain.common.errors import DataFetchError
from app.infra.providers.stock_data import DataPrepStockDataProvider
from app.scanners.base_screener import DataRequirements, StockData
from app.scanners.data_preparation import DataPreparationLayer
from app.services.benchmark_cache_service import BenchmarkDataBundle
from app.services.rate_limiter import RateLimitTimeoutError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(days: int = 252, price: float = 100.0) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame."""
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
    return pd.DataFrame(
        {
            "Open": price * 0.995,
            "High": price * 1.01,
            "Low": price * 0.99,
            "Close": price,
            "Volume": 1_000_000,
        },
        index=dates,
    )


REQUIREMENTS = DataRequirements(
    price_period="2y",
    needs_fundamentals=True,
    needs_benchmark=True,
)

FUNDAMENTALS = {"market_cap": 50_000_000_000, "sector": "Technology"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_price_cache():
    with patch(
        "app.services.price_cache_service.PriceCacheService"
    ) as cls:
        inst = MagicMock()
        cls.get_instance.return_value = inst
        inst.get_historical_data.return_value = None  # default: cache miss
        inst.get_many.return_value = {}
        yield inst


@pytest.fixture
def mock_yfinance():
    svc = MagicMock()
    svc.get_historical_data.return_value = _make_price_df()
    return svc


@pytest.fixture
def mock_benchmark_cache():
    bc = MagicMock()
    bc.get_benchmark_data.return_value = _make_price_df(days=252, price=450.0)
    return bc


@pytest.fixture
def mock_fundamentals_cache():
    fc = MagicMock()
    fc.get_fundamentals.return_value = FUNDAMENTALS
    fc.get_many.return_value = {}
    return fc


@pytest.fixture
def mock_sleep():
    with patch("app.scanners.data_preparation.time.sleep") as sl:
        yield sl


@pytest.fixture
def data_layer(mock_price_cache, mock_yfinance, mock_benchmark_cache, mock_fundamentals_cache, mock_sleep):
    """Real DataPreparationLayer with all external I/O mocked."""
    layer = DataPreparationLayer.__new__(DataPreparationLayer)
    layer._max_retries = 2
    layer._retry_base_delay = 0.01
    layer._yfinance_service = mock_yfinance
    layer._rate_limiter = MagicMock()
    layer.price_cache = mock_price_cache
    layer.benchmark_cache = mock_benchmark_cache
    layer.fundamentals_cache = mock_fundamentals_cache
    return layer


# ===================================================================
# Class 1: Partial failure (allow_partial=True, the default)
# ===================================================================

class TestPartialFailureAllowPartialTrue:
    """When allow_partial=True, errors are stored but no exception raised."""

    def test_price_failure_stores_error_in_fetch_errors(
        self, data_layer, mock_price_cache, mock_yfinance,
    ):
        mock_price_cache.get_historical_data.return_value = None
        mock_yfinance.get_historical_data.side_effect = ConnectionError("refused")

        result = data_layer.prepare_data("AAPL", REQUIREMENTS)

        assert isinstance(result, StockData)
        assert "price_data" in result.fetch_errors
        assert "refused" in result.fetch_errors["price_data"]

    def test_benchmark_failure_does_not_crash(
        self, data_layer, mock_benchmark_cache,
    ):
        mock_benchmark_cache.get_benchmark_data.side_effect = TimeoutError("SPY timeout")

        result = data_layer.prepare_data("AAPL", REQUIREMENTS)

        assert isinstance(result, StockData)
        assert "benchmark_data" in result.fetch_errors
        assert result.benchmark_data.empty

    def test_fundamentals_failure_returns_none_with_error(
        self, data_layer, mock_fundamentals_cache,
    ):
        mock_fundamentals_cache.get_fundamentals.side_effect = RuntimeError("DB down")

        result = data_layer.prepare_data("AAPL", REQUIREMENTS)

        assert result.fundamentals is None
        assert "fundamentals" in result.fetch_errors

    def test_prepare_data_uses_canonical_symbol_for_price_and_fundamentals(
        self, data_layer, mock_price_cache, mock_fundamentals_cache
    ):
        data_layer._resolve_identity = MagicMock(return_value=MagicMock(
            normalized_symbol="700",
            canonical_symbol="0700.HK",
            market="HK",
            exchange="XHKG",
            currency="HKD",
            timezone="Asia/Hong_Kong",
            local_code="0700",
        ))
        mock_price_cache.get_historical_data.return_value = _make_price_df()

        result = data_layer.prepare_data("700", REQUIREMENTS)

        assert result.symbol == "0700.HK"
        mock_price_cache.get_historical_data.assert_called_once_with("0700.HK", period="2y")
        mock_fundamentals_cache.get_fundamentals.assert_called_once_with("0700.HK", force_refresh=False)

    def test_all_components_fail_returns_stock_data_with_all_errors(
        self, data_layer, mock_price_cache, mock_yfinance,
        mock_benchmark_cache, mock_fundamentals_cache,
    ):
        mock_price_cache.get_historical_data.return_value = None
        mock_yfinance.get_historical_data.side_effect = ConnectionError("no net")
        mock_benchmark_cache.get_benchmark_data.side_effect = TimeoutError("spy down")
        mock_fundamentals_cache.get_fundamentals.side_effect = RuntimeError("db")

        result = data_layer.prepare_data("AAPL", REQUIREMENTS)

        assert isinstance(result, StockData)
        assert len(result.fetch_errors) == 3
        assert result.price_data.empty
        assert result.benchmark_data.empty
        assert result.fundamentals is None


# ===================================================================
# Class 2: Partial failure (allow_partial=False — strict mode)
# ===================================================================

class TestPartialFailureAllowPartialFalse:
    """When allow_partial=False, any fetch error raises DataFetchError."""

    def test_price_failure_raises_data_fetch_error(
        self, data_layer, mock_price_cache, mock_yfinance,
    ):
        mock_price_cache.get_historical_data.return_value = None
        mock_yfinance.get_historical_data.side_effect = ConnectionError("refused")

        with pytest.raises(DataFetchError) as exc_info:
            data_layer.prepare_data("AAPL", REQUIREMENTS, allow_partial=False)

        err = exc_info.value
        assert err.symbol == "AAPL"
        assert "price_data" in err.errors
        # partial_data should carry whatever succeeded
        assert isinstance(err.partial_data, StockData)

    def test_fundamentals_failure_raises_data_fetch_error(
        self, data_layer, mock_fundamentals_cache,
    ):
        mock_fundamentals_cache.get_fundamentals.side_effect = RuntimeError("oops")

        with pytest.raises(DataFetchError) as exc_info:
            data_layer.prepare_data("AAPL", REQUIREMENTS, allow_partial=False)

        assert "fundamentals" in exc_info.value.errors
        # Price data should still be present in partial_data
        assert not exc_info.value.partial_data.price_data.empty

    def test_all_succeed_returns_normally(self, data_layer):
        result = data_layer.prepare_data("AAPL", REQUIREMENTS, allow_partial=False)

        assert isinstance(result, StockData)
        assert result.fetch_errors == {}
        assert not result.price_data.empty


# ===================================================================
# Class 3: Retry with backoff
# ===================================================================

class TestRetryWithBackoff:
    """Transient errors are retried with exponential backoff."""

    def test_retry_succeeds_on_second_attempt(
        self, data_layer, mock_price_cache, mock_yfinance, mock_sleep,
    ):
        mock_price_cache.get_historical_data.return_value = None
        mock_yfinance.get_historical_data.side_effect = [
            ConnectionError("transient"),
            _make_price_df(),
        ]

        result = data_layer.prepare_data("AAPL", REQUIREMENTS)

        assert "price_data" not in result.fetch_errors
        assert mock_yfinance.get_historical_data.call_count == 2
        assert mock_sleep.call_count == 1

    def test_retry_exhausted_stores_error(
        self, data_layer, mock_price_cache, mock_yfinance, mock_sleep,
    ):
        mock_price_cache.get_historical_data.return_value = None
        mock_yfinance.get_historical_data.side_effect = ConnectionError("down")

        result = data_layer.prepare_data("AAPL", REQUIREMENTS)

        assert "price_data" in result.fetch_errors
        # 1 initial + 2 retries = 3 attempts
        assert mock_yfinance.get_historical_data.call_count == 3

    def test_retry_exhausted_raises_with_strict_mode(
        self, data_layer, mock_price_cache, mock_yfinance, mock_sleep,
    ):
        mock_price_cache.get_historical_data.return_value = None
        mock_yfinance.get_historical_data.side_effect = ConnectionError("down")

        with pytest.raises(DataFetchError):
            data_layer.prepare_data("AAPL", REQUIREMENTS, allow_partial=False)

    def test_no_retry_on_non_transient_error(
        self, data_layer, mock_price_cache, mock_yfinance, mock_sleep,
    ):
        mock_price_cache.get_historical_data.return_value = None
        mock_yfinance.get_historical_data.side_effect = ValueError("bad symbol")

        result = data_layer.prepare_data("AAPL", REQUIREMENTS)

        assert "price_data" in result.fetch_errors
        # Non-transient: no retry, just 1 call
        assert mock_yfinance.get_historical_data.call_count == 1
        assert mock_sleep.call_count == 0

    def test_backoff_delay_is_exponential(
        self, data_layer, mock_price_cache, mock_yfinance, mock_sleep,
    ):
        mock_price_cache.get_historical_data.return_value = None
        mock_yfinance.get_historical_data.side_effect = ConnectionError("down")

        data_layer.prepare_data("AAPL", REQUIREMENTS)

        # With base=0.01, delays should be ~0.01 and ~0.02 (plus jitter)
        assert mock_sleep.call_count == 2
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        # Second delay should be roughly 2x the first (within jitter tolerance)
        assert delays[1] > delays[0]

    def test_retry_on_benchmark_and_fundamentals(
        self, data_layer, mock_benchmark_cache, mock_fundamentals_cache, mock_sleep,
    ):
        mock_benchmark_cache.get_benchmark_data.side_effect = [
            TimeoutError("timeout"),
            _make_price_df(days=252, price=450.0),
        ]
        mock_fundamentals_cache.get_fundamentals.side_effect = [
            ConnectionError("conn"),
            FUNDAMENTALS,
        ]

        result = data_layer.prepare_data("AAPL", REQUIREMENTS)

        assert "benchmark_data" not in result.fetch_errors
        assert result.fundamentals == FUNDAMENTALS
        assert mock_benchmark_cache.get_benchmark_data.call_count == 2
        assert mock_fundamentals_cache.get_fundamentals.call_count == 2

    def test_non_us_symbol_routes_benchmark_by_market(self, data_layer, mock_benchmark_cache):
        result = data_layer.prepare_data("700.HK", REQUIREMENTS)

        assert "benchmark_data" not in result.fetch_errors
        mock_benchmark_cache.get_benchmark_data.assert_called_once_with(
            market="HK",
            period="2y",
        )

    def test_prepare_data_propagates_resolved_benchmark_metadata(
        self, data_layer, mock_benchmark_cache
    ):
        benchmark_df = _make_price_df(days=252, price=280.0)
        mock_benchmark_cache.get_benchmark_bundle.return_value = BenchmarkDataBundle(
            market="HK",
            period="2y",
            benchmark_symbol="2800.HK",
            benchmark_role="fallback",
            benchmark_kind="etf",
            candidate_symbols=("^HSI", "2800.HK"),
            data=benchmark_df,
        )

        result = data_layer.prepare_data("700.HK", REQUIREMENTS)

        assert result.benchmark_data is benchmark_df
        assert result.benchmark_symbol == "2800.HK"
        assert result.benchmark_role == "fallback"
        assert result.benchmark_kind == "etf"
        assert result.benchmark_candidates == ("^HSI", "2800.HK")


# ===================================================================
# Class 4: Rate limit and timeout handling
# ===================================================================

class TestRateLimitAndTimeout:
    """Rate limit and timeout errors are treated as transient."""

    def test_rate_limit_timeout_retried_then_stored(
        self, data_layer, mock_price_cache, mock_yfinance, mock_sleep,
    ):
        mock_price_cache.get_historical_data.return_value = None
        mock_yfinance.get_historical_data.side_effect = RateLimitTimeoutError(
            "rate limit exceeded"
        )

        result = data_layer.prepare_data("AAPL", REQUIREMENTS)

        assert "price_data" in result.fetch_errors
        # Should have retried (transient error)
        assert mock_yfinance.get_historical_data.call_count == 3

    def test_cache_hit_skips_api_call(
        self, data_layer, mock_price_cache, mock_yfinance,
    ):
        mock_price_cache.get_historical_data.return_value = _make_price_df()

        result = data_layer.prepare_data("AAPL", REQUIREMENTS)

        assert "price_data" not in result.fetch_errors
        mock_yfinance.get_historical_data.assert_not_called()


# ===================================================================
# Class 5: Missing data (structured errors)
# ===================================================================

class TestMissingDataStructured:
    """API returning None or empty DataFrame results in structured errors."""

    def test_api_returns_none_marks_missing(
        self, data_layer, mock_price_cache, mock_yfinance,
    ):
        mock_price_cache.get_historical_data.return_value = None
        mock_yfinance.get_historical_data.return_value = None

        result = data_layer.prepare_data("AAPL", REQUIREMENTS)

        assert result.fetch_errors["price_data"] == "No price data returned"

    def test_empty_dataframe_marks_missing(
        self, data_layer, mock_price_cache, mock_yfinance,
    ):
        mock_price_cache.get_historical_data.return_value = None
        mock_yfinance.get_historical_data.return_value = pd.DataFrame()

        result = data_layer.prepare_data("AAPL", REQUIREMENTS)

        assert result.fetch_errors["price_data"] == "No price data returned"


# ===================================================================
# Class 6: Bulk data preparation
# ===================================================================

class TestBulkDataPreparation:
    """Bulk prepare_data_bulk correctly handles multi-symbol scenarios."""

    def test_bulk_partial_cache_hits(
        self, data_layer, mock_price_cache, mock_yfinance,
        mock_fundamentals_cache,
    ):
        cached = {
            "AAPL": _make_price_df(price=180.0),
            "MSFT": _make_price_df(price=420.0),
        }
        mock_price_cache.get_many.return_value = cached
        # GOOG misses cache → falls back to yfinance
        mock_yfinance.get_historical_data.return_value = _make_price_df(price=170.0)

        results = data_layer.prepare_data_bulk(
            ["AAPL", "MSFT", "GOOG"], REQUIREMENTS,
        )

        assert len(results) == 3
        assert all(isinstance(v, StockData) for v in results.values())
        # Only GOOG should have triggered the yfinance API
        mock_yfinance.get_historical_data.assert_called_once()

    def test_bulk_allow_partial_false_processes_all_then_raises(
        self, data_layer, mock_price_cache, mock_yfinance,
        mock_fundamentals_cache,
    ):
        mock_price_cache.get_many.return_value = {}

        # Use a callable side_effect so retries don't consume other
        # symbols' results from a flat list.
        def _yf_side_effect(symbol, **kwargs):
            if symbol == "MSFT":
                raise ValueError("MSFT bad data")  # non-transient → no retry
            return _make_price_df(price=180.0)

        mock_yfinance.get_historical_data.side_effect = _yf_side_effect

        with pytest.raises(DataFetchError) as exc_info:
            data_layer.prepare_data_bulk(
                ["AAPL", "MSFT", "GOOG"], REQUIREMENTS, allow_partial=False,
            )

        err = exc_info.value
        assert "MSFT/price_data" in err.errors
        # partial_data should contain all symbols (including MSFT with errors)
        assert isinstance(err.partial_data, dict)
        assert "AAPL" in err.partial_data
        assert "GOOG" in err.partial_data

    def test_bulk_empty_symbols_returns_empty(self, data_layer):
        result = data_layer.prepare_data_bulk([], REQUIREMENTS)
        assert result == {}

    def test_bulk_benchmark_fetched_once(
        self, data_layer, mock_price_cache, mock_yfinance,
        mock_benchmark_cache,
    ):
        mock_price_cache.get_many.return_value = {
            "AAPL": _make_price_df(),
            "MSFT": _make_price_df(),
        }

        results = data_layer.prepare_data_bulk(["AAPL", "MSFT"], REQUIREMENTS)

        # Benchmark fetched once, shared across both symbols
        mock_benchmark_cache.get_benchmark_data.assert_called_once_with(
            market="US",
            period="2y",
        )
        # Both symbols should have the same benchmark_data reference
        assert results["AAPL"].benchmark_data is results["MSFT"].benchmark_data

    def test_bulk_fetches_benchmark_once_per_market_for_mixed_symbols(
        self, data_layer, mock_price_cache, mock_benchmark_cache
    ):
        hk_df = _make_price_df(price=280.0)
        us_df = _make_price_df(price=450.0)
        mock_benchmark_cache.get_benchmark_data.side_effect = lambda *, market, period: (
            hk_df if market == "HK" else us_df
        )
        mock_price_cache.get_many.return_value = {
            "AAPL": _make_price_df(),
            "0700.HK": _make_price_df(),
        }

        results = data_layer.prepare_data_bulk(["AAPL", "0700.HK"], REQUIREMENTS)

        assert mock_benchmark_cache.get_benchmark_data.call_count == 2
        assert results["AAPL"].benchmark_data is us_df
        assert results["0700.HK"].benchmark_data is hk_df

    def test_bulk_propagates_resolved_benchmark_metadata_per_market(
        self, data_layer, mock_price_cache, mock_benchmark_cache
    ):
        hk_df = _make_price_df(price=280.0)
        us_df = _make_price_df(price=450.0)
        mock_benchmark_cache.get_benchmark_bundle.side_effect = lambda *, market, period: (
            BenchmarkDataBundle(
                market=market,
                period=period,
                benchmark_symbol="^HSI" if market == "HK" else "SPY",
                benchmark_role="primary",
                benchmark_kind="index" if market == "HK" else "etf",
                candidate_symbols=("^HSI", "2800.HK") if market == "HK" else ("SPY", "IVV"),
                data=hk_df if market == "HK" else us_df,
            )
        )
        mock_price_cache.get_many.return_value = {
            "AAPL": _make_price_df(price=180.0),
            "MSFT": _make_price_df(price=190.0),
            "0700.HK": _make_price_df(price=380.0),
            "0005.HK": _make_price_df(price=60.0),
        }

        results = data_layer.prepare_data_bulk(["AAPL", "MSFT", "0700.HK", "0005.HK"], REQUIREMENTS)

        assert results["AAPL"].benchmark_symbol == "SPY"
        assert results["AAPL"].benchmark_candidates == ("SPY", "IVV")
        assert results["0700.HK"].benchmark_symbol == "^HSI"
        assert results["0700.HK"].benchmark_candidates == ("^HSI", "2800.HK")

    def test_bulk_computes_rs_universe_performances_within_each_market(
        self, data_layer, mock_price_cache, mock_benchmark_cache
    ):
        us_benchmark = _make_price_df(price=450.0)
        hk_benchmark = _make_price_df(price=280.0)
        mock_benchmark_cache.get_benchmark_bundle.side_effect = lambda *, market, period: (
            BenchmarkDataBundle(
                market=market,
                period=period,
                benchmark_symbol="SPY" if market == "US" else "^HSI",
                benchmark_role="primary",
                benchmark_kind="etf" if market == "US" else "index",
                candidate_symbols=("SPY", "IVV") if market == "US" else ("^HSI", "2800.HK"),
                data=us_benchmark if market == "US" else hk_benchmark,
            )
        )
        mock_price_cache.get_many.return_value = {
            "AAPL": _make_price_df(price=180.0),
            "MSFT": _make_price_df(price=190.0),
            "0700.HK": _make_price_df(price=380.0),
            "0005.HK": _make_price_df(price=60.0),
        }

        results = data_layer.prepare_data_bulk(["AAPL", "MSFT", "0700.HK", "0005.HK"], REQUIREMENTS)

        assert results["AAPL"].rs_universe_performances is not None
        assert results["MSFT"].rs_universe_performances is not None
        assert results["0700.HK"].rs_universe_performances is not None
        assert results["0005.HK"].rs_universe_performances is not None

        us_weighted = results["AAPL"].rs_universe_performances["weighted"]
        hk_weighted = results["0700.HK"].rs_universe_performances["weighted"]
        assert len(us_weighted) == 2
        assert len(hk_weighted) == 2
        assert us_weighted is results["MSFT"].rs_universe_performances["weighted"]
        assert hk_weighted is results["0005.HK"].rs_universe_performances["weighted"]
        assert results["AAPL"].rs_universe_performances is not results["0700.HK"].rs_universe_performances

    def test_bulk_results_preserve_requested_input_key_for_unsuffixed_symbol(
        self, data_layer, mock_price_cache
    ):
        data_layer._resolve_identity = MagicMock(side_effect=[
            MagicMock(
                normalized_symbol="700",
                canonical_symbol="0700.HK",
                market="HK",
                exchange="XHKG",
                currency="HKD",
                timezone="Asia/Hong_Kong",
                local_code="0700",
            )
        ])
        mock_price_cache.get_many.return_value = {"0700.HK": _make_price_df(price=380.0)}

        results = data_layer.prepare_data_bulk(["700"], REQUIREMENTS)

        assert "700" in results
        assert "0700.HK" not in results
        assert results["700"].symbol == "0700.HK"

    def test_bulk_keeps_entries_for_aliases_that_share_same_canonical_symbol(
        self, data_layer, mock_price_cache
    ):
        data_layer._resolve_identity = MagicMock(side_effect=[
            MagicMock(
                normalized_symbol="700",
                canonical_symbol="0700.HK",
                market="HK",
                exchange="XHKG",
                currency="HKD",
                timezone="Asia/Hong_Kong",
                local_code="0700",
            ),
            MagicMock(
                normalized_symbol="0700.HK",
                canonical_symbol="0700.HK",
                market="HK",
                exchange="XHKG",
                currency="HKD",
                timezone="Asia/Hong_Kong",
                local_code="0700",
            ),
        ])
        mock_price_cache.get_many.return_value = {"0700.HK": _make_price_df(price=380.0)}

        results = data_layer.prepare_data_bulk(["700", "0700.HK"], REQUIREMENTS)

        assert set(results.keys()) == {"700", "0700.HK"}
        assert results["700"].symbol == "0700.HK"
        assert results["0700.HK"].symbol == "0700.HK"

    def test_bulk_records_benchmark_error_when_market_benchmark_missing(
        self, data_layer, mock_price_cache, mock_benchmark_cache
    ):
        mock_benchmark_cache.get_benchmark_data.return_value = None
        mock_price_cache.get_many.return_value = {"AAPL": _make_price_df()}

        results = data_layer.prepare_data_bulk(["AAPL"], REQUIREMENTS)

        assert results["AAPL"].benchmark_data.empty
        assert "benchmark_data" in results["AAPL"].fetch_errors

    def test_bulk_strict_mode_raises_when_market_benchmark_missing(
        self, data_layer, mock_price_cache, mock_benchmark_cache
    ):
        mock_benchmark_cache.get_benchmark_data.return_value = None
        mock_price_cache.get_many.return_value = {"AAPL": _make_price_df()}

        with pytest.raises(DataFetchError) as exc_info:
            data_layer.prepare_data_bulk(["AAPL"], REQUIREMENTS, allow_partial=False)

        assert "AAPL/benchmark_data" in exc_info.value.errors

    def test_bulk_batch_only_prices_never_falls_back_to_per_symbol_fetch(
        self, data_layer, mock_price_cache, mock_yfinance,
        mock_fundamentals_cache,
    ):
        mock_price_cache.get_many.return_value = {"AAPL": _make_price_df(price=180.0)}

        results = data_layer.prepare_data_bulk(
            ["AAPL", "MSFT"],
            REQUIREMENTS,
            batch_only_prices=True,
        )

        assert results["AAPL"].fetch_errors == {}
        assert results["MSFT"].fetch_errors["price_data"] == "No price data returned from batch-only price path"
        assert results["MSFT"].price_data.empty
        mock_yfinance.get_historical_data.assert_not_called()


# ===================================================================
# Class 7: Adapter delegation
# ===================================================================

class TestAdapterDelegation:
    """DataPrepStockDataProvider correctly delegates to DataPreparationLayer."""

    def test_adapter_delegates_prepare_data(self):
        with patch.object(DataPreparationLayer, "prepare_data") as mock_pd:
            mock_pd.return_value = MagicMock(spec=StockData)
            adapter = DataPrepStockDataProvider(
                cache_bundle=MagicMock(
                    price=MagicMock(),
                    benchmark=MagicMock(),
                    fundamentals=MagicMock(),
                )
            )
            adapter.prepare_data("AAPL", REQUIREMENTS)

            mock_pd.assert_called_once_with("AAPL", REQUIREMENTS, allow_partial=True)

    def test_adapter_delegates_prepare_data_bulk(self):
        with patch.object(DataPreparationLayer, "prepare_data_bulk") as mock_pdb:
            mock_pdb.return_value = {}
            adapter = DataPrepStockDataProvider(
                cache_bundle=MagicMock(
                    price=MagicMock(),
                    benchmark=MagicMock(),
                    fundamentals=MagicMock(),
                )
            )
            adapter.prepare_data_bulk(["AAPL", "MSFT"], REQUIREMENTS)

            mock_pdb.assert_called_once_with(
                ["AAPL", "MSFT"],
                REQUIREMENTS,
                allow_partial=True,
                batch_only_prices=False,
                batch_only_fundamentals=False,
            )

    def test_adapter_delegates_prepare_data_bulk_batch_only_flags(self):
        with patch.object(DataPreparationLayer, "prepare_data_bulk") as mock_pdb:
            mock_pdb.return_value = {}
            adapter = DataPrepStockDataProvider(
                cache_bundle=MagicMock(
                    price=MagicMock(),
                    benchmark=MagicMock(),
                    fundamentals=MagicMock(),
                )
            )
            adapter.prepare_data_bulk(
                ["AAPL", "MSFT"],
                REQUIREMENTS,
                batch_only_prices=True,
                batch_only_fundamentals=True,
            )

            mock_pdb.assert_called_once_with(
                ["AAPL", "MSFT"],
                REQUIREMENTS,
                allow_partial=True,
                batch_only_prices=True,
                batch_only_fundamentals=True,
            )

    def test_adapter_forwards_allow_partial(self):
        with patch.object(DataPreparationLayer, "prepare_data") as mock_pd:
            mock_pd.return_value = MagicMock(spec=StockData)
            adapter = DataPrepStockDataProvider(
                cache_bundle=MagicMock(
                    price=MagicMock(),
                    benchmark=MagicMock(),
                    fundamentals=MagicMock(),
                )
            )
            adapter.prepare_data("AAPL", REQUIREMENTS, allow_partial=False)

            mock_pd.assert_called_once_with("AAPL", REQUIREMENTS, allow_partial=False)
