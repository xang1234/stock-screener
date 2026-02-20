"""Tests for SetupEngineScanner — the runtime entrypoint for the Setup Engine pipeline.

Covers:
- Registration in screener_registry
- Data requirements declaration
- Insufficient data paths (short bars, policy rejection)
- Full pipeline scan with synthetic data
- Error handling
- Rating calculation thresholds
- Weekly OHLCV propagation to detectors
- Scanner-level timing observability (SE-E9)
- Per-detector elapsed_ms in DetectorExecutionTrace (SE-E9)
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.analysis.patterns.aggregator import DetectorExecutionTrace
from app.analysis.patterns.models import validate_setup_engine_payload
from app.scanners.base_screener import DataRequirements, ScreenerResult, StockData
from app.scanners.screener_registry import screener_registry
from app.scanners.setup_engine_screener import SetupEngineScanner, _count_current_week_sessions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_price_data(num_days: int, start_price: float = 50.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a mild uptrend."""
    dates = pd.bdate_range(end=pd.Timestamp("2025-12-19"), periods=num_days)
    np.random.seed(42)
    close = start_price + np.cumsum(np.random.randn(num_days) * 0.5 + 0.05)
    close = np.maximum(close, 1.0)  # keep prices positive
    return pd.DataFrame(
        {
            "Open": close * (1 - np.random.uniform(0, 0.02, num_days)),
            "High": close * (1 + np.random.uniform(0, 0.03, num_days)),
            "Low": close * (1 - np.random.uniform(0, 0.03, num_days)),
            "Close": close,
            "Volume": np.random.randint(100_000, 5_000_000, size=num_days).astype(float),
        },
        index=dates,
    )


def _make_stock_data(
    symbol: str = "TEST",
    num_days: int = 350,
    with_benchmark: bool = True,
) -> StockData:
    """Build a StockData object with synthetic price and benchmark data."""
    price_data = _make_price_data(num_days)
    benchmark_data = _make_price_data(num_days, start_price=400.0) if with_benchmark else pd.DataFrame()
    return StockData(
        symbol=symbol,
        price_data=price_data,
        benchmark_data=benchmark_data,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegistration:
    """SetupEngineScanner must be discoverable via the global registry."""

    def test_is_registered(self):
        assert screener_registry.is_registered("setup_engine")

    def test_get_returns_instance(self):
        instance = screener_registry.get("setup_engine")
        assert instance is not None
        assert isinstance(instance, SetupEngineScanner)

    def test_screener_name(self):
        scanner = SetupEngineScanner()
        assert scanner.screener_name == "setup_engine"


class TestDataRequirements:
    """Data requirements should request 2y price data and benchmark."""

    def test_requirements(self):
        scanner = SetupEngineScanner()
        reqs = scanner.get_data_requirements()
        assert isinstance(reqs, DataRequirements)
        assert reqs.price_period == "2y"
        assert reqs.needs_benchmark is True
        assert reqs.needs_fundamentals is False
        assert reqs.needs_quarterly_growth is False
        assert reqs.needs_earnings_history is False


class TestInsufficientData:
    """Early-return paths for stocks with too little history."""

    def test_short_bars_returns_insufficient(self):
        """<100 daily bars should short-circuit before the policy check."""
        data = _make_stock_data(num_days=50)
        scanner = SetupEngineScanner()
        result = scanner.scan_stock("SHORT", data)

        assert isinstance(result, ScreenerResult)
        assert result.score == 0.0
        assert result.passes is False
        assert result.rating == "Insufficient Data"
        assert result.screener_name == "setup_engine"
        # No payload — just a reason string
        assert "reason" in result.details

    def test_policy_insufficient_returns_payload(self):
        """200 bars pass the quick guard (>=100) but fail the 252-bar policy.

        This path should build a real payload documenting the insufficiency.
        """
        data = _make_stock_data(num_days=200)
        scanner = SetupEngineScanner()
        result = scanner.scan_stock("POLICY_FAIL", data)

        assert result.score == 0.0
        assert result.passes is False
        assert result.rating == "Insufficient Data"
        # Payload exists with null scores
        assert "setup_engine" in result.details
        payload = result.details["setup_engine"]
        assert payload["setup_score"] is None
        assert payload["quality_score"] is None
        assert payload["readiness_score"] is None
        # Explain should document the failure
        assert "explain" in payload
        assert len(payload["explain"]["failed_checks"]) > 0


class TestScanWithSyntheticData:
    """Full pipeline scan with enough synthetic data to run all stages."""

    def test_valid_screener_result(self):
        data = _make_stock_data(num_days=350)
        scanner = SetupEngineScanner()
        result = scanner.scan_stock("SYNTH", data)

        assert isinstance(result, ScreenerResult)
        assert 0 <= result.score <= 100
        assert result.screener_name == "setup_engine"
        assert result.rating in ("Strong Buy", "Buy", "Watch", "Pass")

    def test_payload_exists_and_validates(self):
        data = _make_stock_data(num_days=350)
        scanner = SetupEngineScanner()
        result = scanner.scan_stock("SYNTH", data)

        assert "setup_engine" in result.details
        payload = result.details["setup_engine"]
        errors = validate_setup_engine_payload(payload)
        assert errors == [], f"Payload validation errors: {errors}"

    def test_passes_matches_setup_ready(self):
        data = _make_stock_data(num_days=350)
        scanner = SetupEngineScanner()
        result = scanner.scan_stock("SYNTH", data)

        assert result.passes == result.details["setup_engine"]["setup_ready"]

    def test_candidates_is_list(self):
        """Verifies weekly OHLCV was passed — detectors that need weekly data
        return insufficient_data without it, producing no candidates list.
        """
        data = _make_stock_data(num_days=350)
        scanner = SetupEngineScanner()
        result = scanner.scan_stock("SYNTH", data)

        payload = result.details["setup_engine"]
        assert isinstance(payload["candidates"], list)


class TestErrorHandling:
    """scan_stock must catch exceptions and return Error result."""

    def test_exception_returns_error_result(self):
        scanner = SetupEngineScanner()
        data = _make_stock_data(num_days=350)

        with patch.object(
            scanner._aggregator,
            "aggregate",
            side_effect=RuntimeError("detector exploded"),
        ):
            result = scanner.scan_stock("BOOM", data)

        assert result.score == 0.0
        assert result.passes is False
        assert result.rating == "Error"
        assert "error" in result.details


class TestCalculateRating:
    """Rating thresholds: score=0 should be 'Pass', never 'Insufficient Data'."""

    def test_strong_buy(self):
        scanner = SetupEngineScanner()
        details = {"setup_engine": {"setup_ready": True}}
        assert scanner.calculate_rating(85.0, details) == "Strong Buy"

    def test_buy(self):
        scanner = SetupEngineScanner()
        details = {"setup_engine": {"setup_ready": True}}
        assert scanner.calculate_rating(70.0, details) == "Buy"

    def test_watch(self):
        scanner = SetupEngineScanner()
        details = {"setup_engine": {"setup_ready": False}}
        assert scanner.calculate_rating(55.0, details) == "Watch"

    def test_pass_low_score(self):
        scanner = SetupEngineScanner()
        details = {"setup_engine": {"setup_ready": False}}
        assert scanner.calculate_rating(30.0, details) == "Pass"

    def test_zero_score_is_pass_not_insufficient(self):
        """Score=0 via calculate_rating must return 'Pass', not 'Insufficient Data'."""
        scanner = SetupEngineScanner()
        details = {"setup_engine": {"setup_ready": False}}
        assert scanner.calculate_rating(0.0, details) == "Pass"

    def test_high_score_not_ready_is_watch(self):
        """Score >= 80 but setup_ready=False should be Watch, not Strong Buy."""
        scanner = SetupEngineScanner()
        details = {"setup_engine": {"setup_ready": False}}
        assert scanner.calculate_rating(85.0, details) == "Watch"


class TestCountCurrentWeekSessions:
    """Unit tests for the _count_current_week_sessions helper."""

    def test_empty_dataframe(self):
        assert _count_current_week_sessions(pd.DataFrame()) == 0

    def test_single_day(self):
        dates = pd.DatetimeIndex([pd.Timestamp("2025-12-19")])
        df = pd.DataFrame({"Close": [100.0]}, index=dates)
        assert _count_current_week_sessions(df) == 1

    def test_full_week(self):
        dates = pd.bdate_range("2025-12-15", "2025-12-19")  # Mon-Fri
        df = pd.DataFrame({"Close": range(len(dates))}, index=dates)
        assert _count_current_week_sessions(df) == 5


class TestTimingObservability:
    """SE-E9: Scanner-level timing and per-detector elapsed_ms."""

    def test_scanner_emits_timing_log(self, caplog):
        """INFO-level timing line should appear for a valid scan."""
        data = _make_stock_data(num_days=350)
        scanner = SetupEngineScanner()

        with caplog.at_level(logging.INFO, logger="app.scanners.setup_engine_screener"):
            scanner.scan_stock("TIMING", data)

        timing_lines = [r for r in caplog.records if "SE timing TIMING" in r.message]
        assert len(timing_lines) == 1
        msg = timing_lines[0].message
        assert "prep=" in msg
        assert "detectors=" in msg
        assert "readiness=" in msg
        assert "total=" in msg
        assert "score=" in msg

    def test_detector_traces_have_elapsed_ms(self):
        """Every DetectorExecutionTrace should have a non-negative elapsed_ms."""
        from app.analysis.patterns.aggregator import SetupEngineAggregator
        from app.analysis.patterns.config import DEFAULT_SETUP_ENGINE_PARAMETERS
        from app.analysis.patterns.detectors import PatternDetectorInput

        price_data = _make_price_data(350)
        from app.analysis.patterns.technicals import resample_ohlcv
        weekly_data = resample_ohlcv(price_data, exclude_incomplete_last_period=True)

        detector_input = PatternDetectorInput(
            symbol="TRACE",
            timeframe="daily",
            daily_bars=len(price_data),
            weekly_bars=len(weekly_data),
            features={
                "daily_ohlcv": price_data,
                "weekly_ohlcv": weekly_data,
            },
        )

        aggregator = SetupEngineAggregator()
        output = aggregator.aggregate(detector_input, parameters=DEFAULT_SETUP_ENGINE_PARAMETERS)

        assert len(output.detector_traces) > 0
        for trace in output.detector_traces:
            assert isinstance(trace, DetectorExecutionTrace)
            assert trace.elapsed_ms >= 0.0

    def test_per_detector_debug_logs(self, caplog):
        """DEBUG-level per-detector lines should appear for each detector."""
        data = _make_stock_data(num_days=350)
        scanner = SetupEngineScanner()

        with caplog.at_level(logging.DEBUG, logger="app.scanners.setup_engine_screener"):
            scanner.scan_stock("DETLOG", data)

        debug_lines = [r for r in caplog.records if "SE detector" in r.message and "DETLOG" in r.message]
        assert len(debug_lines) > 0
        for line in debug_lines:
            assert "elapsed=" in line.message
            assert "outcome=" in line.message
