"""Setup Engine runtime budget regression tests.

Enforces wall-clock timing budgets at three granularity levels:
  1. Individual detectors (parametrized over all 7)
  2. Aggregator pipeline (all detectors + calibration + selection)
  3. End-to-end scanner (prep + detectors + readiness + payload)

Plus structural guards on detector registry size and uniqueness.

All tests use deterministic synthetic data (np.random.seed(42)), require
no external services, and follow the warmup/measure pattern established
in test_query_performance.py.

Note: The _verify_seed_integrity autouse fixture in conftest.py will create
the perf_engine (in-memory SQLite + 500-row seeding) for this module even
though no SE test uses it. The overhead is negligible (~50-100ms).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from app.analysis.patterns.aggregator import SetupEngineAggregator
from app.analysis.patterns.config import DEFAULT_SETUP_ENGINE_PARAMETERS
from app.analysis.patterns.detectors import (
    PatternDetectorInput,
    default_pattern_detectors,
)
from app.analysis.patterns.readiness import compute_breakout_readiness_features
from app.analysis.patterns.technicals import resample_ohlcv
from app.scanners.base_screener import StockData
from app.scanners.setup_engine_screener import SetupEngineScanner

# ---------------------------------------------------------------------------
# Budget constants
# ---------------------------------------------------------------------------

# Per-detector budget (ms). Most detectors remain in low double digits, while
# three_weeks_tight has O(n^2) behavior and can spike on shared CI runners.
# 300ms keeps regressions visible while avoiding flaky false negatives.
DETECTOR_BUDGET_MS = 300

# Full aggregator pipeline: sum of all detectors (~130ms typical) + calibration
# + selection overhead. 500ms gives ~3x CI headroom.
AGGREGATOR_BUDGET_MS = 500

# Readiness computation: vectorized ATR/Bollinger/RS on 350 bars.
READINESS_BUDGET_MS = 100

# End-to-end scanner: prep + detectors + readiness + payload assembly.
SCANNER_BUDGET_MS = 1000

# Insufficient-data short-circuit path.
EARLY_RETURN_BUDGET_MS = 10

# Detector registry ceiling: 7 currently; room for 5 before review.
MAX_DETECTOR_COUNT = 12

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_price_data(num_days: int, start_price: float = 50.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a mild uptrend."""
    dates = pd.bdate_range(end=pd.Timestamp("2025-12-19"), periods=num_days)
    np.random.seed(42)
    close = start_price + np.cumsum(np.random.randn(num_days) * 0.5 + 0.05)
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {
            "Open": close * (1 - np.random.uniform(0, 0.02, num_days)),
            "High": close * (1 + np.random.uniform(0, 0.03, num_days)),
            "Low": close * (1 - np.random.uniform(0, 0.03, num_days)),
            "Close": close,
            "Volume": np.random.randint(100_000, 5_000_000, size=num_days).astype(
                float
            ),
        },
        index=dates,
    )


def _make_stock_data(
    symbol: str = "PERF",
    num_days: int = 350,
    with_benchmark: bool = True,
) -> StockData:
    """Build a StockData object with synthetic price and benchmark data."""
    price_data = _make_price_data(num_days)
    benchmark_data = (
        _make_price_data(num_days, start_price=400.0)
        if with_benchmark
        else pd.DataFrame()
    )
    return StockData(
        symbol=symbol,
        price_data=price_data,
        benchmark_data=benchmark_data,
    )


# ---------------------------------------------------------------------------
# Module-scoped fixtures — synthetic data shared across tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_price_data() -> pd.DataFrame:
    """350-bar daily OHLCV with np.random.seed(42)."""
    return _make_price_data(350)


@pytest.fixture(scope="module")
def synthetic_weekly_data(synthetic_price_data: pd.DataFrame) -> pd.DataFrame:
    """Weekly resampled OHLCV from the daily fixture."""
    return resample_ohlcv(synthetic_price_data, exclude_incomplete_last_period=True)


@pytest.fixture(scope="module")
def synthetic_detector_input(
    synthetic_price_data: pd.DataFrame,
    synthetic_weekly_data: pd.DataFrame,
) -> PatternDetectorInput:
    """PatternDetectorInput with 350 daily bars and weekly OHLCV."""
    return PatternDetectorInput(
        symbol="PERFTEST",
        timeframe="daily",
        daily_bars=len(synthetic_price_data),
        weekly_bars=len(synthetic_weekly_data),
        features={
            "daily_ohlcv": synthetic_price_data,
            "weekly_ohlcv": synthetic_weekly_data,
        },
    )


@pytest.fixture(scope="module")
def synthetic_stock_data() -> StockData:
    """350-bar StockData for scanner-level tests."""
    return _make_stock_data(num_days=350)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Per-Detector Budgets (parametrized)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestDetectorBudgets:
    """Each detector must complete within DETECTOR_BUDGET_MS on synthetic data."""

    @pytest.mark.parametrize(
        "detector",
        default_pattern_detectors(),
        ids=lambda d: d.name,
    )
    def test_each_detector_under_budget(
        self,
        detector,
        synthetic_detector_input: PatternDetectorInput,
    ):
        # Warmup: prime Python bytecode paths and pandas internals
        detector.detect_safe(synthetic_detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS)

        t0 = time.perf_counter()
        detector.detect_safe(synthetic_detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < DETECTOR_BUDGET_MS, (
            f"Detector '{detector.name}' took {elapsed_ms:.1f}ms "
            f"(budget: {DETECTOR_BUDGET_MS}ms)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Aggregator Budget
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestAggregatorBudget:
    """Full aggregation pipeline must complete within AGGREGATOR_BUDGET_MS."""

    def test_aggregate_under_budget(
        self, synthetic_detector_input: PatternDetectorInput
    ):
        aggregator = SetupEngineAggregator()

        # Warmup
        aggregator.aggregate(
            synthetic_detector_input, parameters=DEFAULT_SETUP_ENGINE_PARAMETERS
        )

        t0 = time.perf_counter()
        aggregator.aggregate(
            synthetic_detector_input, parameters=DEFAULT_SETUP_ENGINE_PARAMETERS
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < AGGREGATOR_BUDGET_MS, (
            f"Aggregator took {elapsed_ms:.1f}ms (budget: {AGGREGATOR_BUDGET_MS}ms)"
        )

    def test_trace_timing_sums_are_consistent(
        self, synthetic_detector_input: PatternDetectorInput
    ):
        """Sum of detector traces must be positive and less than total elapsed.

        Lower bound: ensures instrumentation is actually recording time.
        Upper bound: confirms calibration/selection work happens outside
        detector traces (i.e., there's non-detector overhead in aggregate()).
        """
        aggregator = SetupEngineAggregator()

        t0 = time.perf_counter()
        output = aggregator.aggregate(
            synthetic_detector_input, parameters=DEFAULT_SETUP_ENGINE_PARAMETERS
        )
        total_elapsed_ms = (time.perf_counter() - t0) * 1000

        trace_sum_ms = sum(t.elapsed_ms for t in output.detector_traces)

        assert trace_sum_ms > 0, (
            "Detector trace elapsed_ms sum is zero — instrumentation broken?"
        )
        assert trace_sum_ms < total_elapsed_ms, (
            f"Detector trace sum ({trace_sum_ms:.1f}ms) >= total aggregator "
            f"elapsed ({total_elapsed_ms:.1f}ms) — calibration/selection "
            f"overhead should make total strictly larger"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Readiness Budget
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestReadinessBudget:
    """Readiness computation must complete within READINESS_BUDGET_MS."""

    def test_readiness_under_budget(self, synthetic_price_data: pd.DataFrame):
        last_close = float(synthetic_price_data["Close"].iloc[-1])
        pivot_price = last_close * 1.03  # realistic: 3% above current
        benchmark_close = synthetic_price_data["Close"] * 8.0  # proxy SPY scale

        # Warmup
        compute_breakout_readiness_features(
            synthetic_price_data,
            pivot_price=pivot_price,
            benchmark_close=benchmark_close,
        )

        t0 = time.perf_counter()
        compute_breakout_readiness_features(
            synthetic_price_data,
            pivot_price=pivot_price,
            benchmark_close=benchmark_close,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < READINESS_BUDGET_MS, (
            f"Readiness (with benchmark) took {elapsed_ms:.1f}ms "
            f"(budget: {READINESS_BUDGET_MS}ms)"
        )

    def test_readiness_without_benchmark_under_budget(
        self, synthetic_price_data: pd.DataFrame
    ):
        last_close = float(synthetic_price_data["Close"].iloc[-1])
        pivot_price = last_close * 1.03

        # Warmup
        compute_breakout_readiness_features(
            synthetic_price_data,
            pivot_price=pivot_price,
            benchmark_close=None,
        )

        t0 = time.perf_counter()
        compute_breakout_readiness_features(
            synthetic_price_data,
            pivot_price=pivot_price,
            benchmark_close=None,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < READINESS_BUDGET_MS, (
            f"Readiness (no benchmark) took {elapsed_ms:.1f}ms "
            f"(budget: {READINESS_BUDGET_MS}ms)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Scanner Budget (end-to-end)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestScannerBudget:
    """End-to-end scan_stock must complete within SCANNER_BUDGET_MS."""

    def test_scan_stock_under_budget(self, synthetic_stock_data: StockData):
        scanner = SetupEngineScanner()

        # Warmup
        scanner.scan_stock("WARMUP", synthetic_stock_data)

        t0 = time.perf_counter()
        scanner.scan_stock("PERF", synthetic_stock_data)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < SCANNER_BUDGET_MS, (
            f"scan_stock took {elapsed_ms:.1f}ms (budget: {SCANNER_BUDGET_MS}ms)"
        )

    def test_scan_stock_insufficient_data_is_fast(self):
        """50-bar StockData should trigger early return well under budget."""
        short_data = _make_stock_data(num_days=50)
        scanner = SetupEngineScanner()

        t0 = time.perf_counter()
        result = scanner.scan_stock("SHORT", short_data)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert result.rating == "Insufficient Data"
        assert elapsed_ms < EARLY_RETURN_BUDGET_MS, (
            f"Insufficient-data early return took {elapsed_ms:.1f}ms "
            f"(budget: {EARLY_RETURN_BUDGET_MS}ms)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Detector Count Guard
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestDetectorCountGuard:
    """Structural guards on the detector registry."""

    def test_detector_count_within_ceiling(self):
        """Adding detectors inflates runtime linearly. If you've added a new
        detector that pushes past the ceiling, update MAX_DETECTOR_COUNT and
        verify the aggregator budget still holds.
        """
        detectors = default_pattern_detectors()
        assert len(detectors) <= MAX_DETECTOR_COUNT, (
            f"Detector count ({len(detectors)}) exceeds ceiling "
            f"({MAX_DETECTOR_COUNT}). Each detector adds ~5-15ms to every "
            f"scan_stock call. Update MAX_DETECTOR_COUNT after verifying "
            f"aggregator budget still holds."
        )

    def test_detector_names_are_unique(self):
        """Duplicate detector names mean doubled runtime for the same work."""
        detectors = default_pattern_detectors()
        names = [d.name for d in detectors]
        assert len(names) == len(set(names)), (
            f"Duplicate detector names found: "
            f"{[n for n in names if names.count(n) > 1]}"
        )
