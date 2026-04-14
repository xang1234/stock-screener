"""Deterministic yfinance simulator for load/soak benchmarks.

The real yfinance API has variable latency, occasional 429s, and is unreliable
for CI. This simulator gives reproducible numbers (seeded RNG) while preserving
the pipeline-level dynamics (lock contention, batch sizing, backoff) that the
load test is actually measuring.

A single ``--live`` opt-in flag in the test runner switches to the real
yfinance — used for occasional out-of-band evidence-pack runs.
"""
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class SimulatorProfile:
    """Per-market latency + 429 distribution profile.

    Numbers are calibrated from production observations; the absolute values
    matter less than relative relationships across markets, since the load
    test gates on regression vs a committed baseline.
    """
    base_latency_s: float          # mean seconds per yf.download batch call
    latency_jitter_s: float        # uniform jitter ± this many seconds
    rate_limit_probability: float  # 0..1 chance of 429 per batch
    failure_probability: float     # 0..1 chance of non-429 transient failure


# Calibrated profiles per market. Non-US markets get higher 429 / latency
# baselines reflecting historical observations.
DEFAULT_PROFILES: Dict[str, SimulatorProfile] = {
    "US": SimulatorProfile(base_latency_s=0.15, latency_jitter_s=0.05,
                            rate_limit_probability=0.02, failure_probability=0.01),
    "HK": SimulatorProfile(base_latency_s=0.30, latency_jitter_s=0.10,
                            rate_limit_probability=0.05, failure_probability=0.02),
    "JP": SimulatorProfile(base_latency_s=0.30, latency_jitter_s=0.10,
                            rate_limit_probability=0.04, failure_probability=0.02),
    "TW": SimulatorProfile(base_latency_s=0.40, latency_jitter_s=0.15,
                            rate_limit_probability=0.06, failure_probability=0.03),
}


class YFinanceSimulator:
    """Drop-in replacement for ``yfinance.download`` used by ``BulkDataFetcher``.

    Deterministic when constructed with a fixed seed. Returns synthetic OHLCV
    DataFrames matching yfinance's MultiIndex shape so downstream code that
    parses the result works unmodified.
    """

    def __init__(
        self,
        profile: SimulatorProfile,
        seed: int = 42,
        sleep_fn=time.sleep,
    ):
        self.profile = profile
        self._rng = random.Random(seed)
        self._sleep_fn = sleep_fn
        self._call_count = 0
        self._429_count = 0
        self._latencies_s: List[float] = []

    def download(self, tickers, **kwargs) -> pd.DataFrame:
        """Mimic yfinance.download: simulate latency, possibly raise 429.

        Returns a MultiIndex DataFrame with synthetic OHLCV per ticker so the
        BulkDataFetcher's per-symbol parsing path runs end-to-end.
        """
        self._call_count += 1
        profile = self.profile  # snapshot once; set_profile() may swap it concurrently

        # Simulate per-batch latency
        latency = max(
            0.001,
            profile.base_latency_s
            + self._rng.uniform(-profile.latency_jitter_s, profile.latency_jitter_s),
        )
        self._sleep_fn(latency)
        self._latencies_s.append(latency)

        # Inject rate-limit error
        if self._rng.random() < profile.rate_limit_probability:
            self._429_count += 1
            raise Exception("YFRateLimitError: Too Many Requests (429)")

        # Inject non-429 transient failure
        if self._rng.random() < profile.failure_probability:
            raise Exception("Transient network error")

        # Build synthetic OHLCV DataFrame matching yfinance MultiIndex format
        if isinstance(tickers, str):
            symbol_list = tickers.split()
        else:
            symbol_list = list(tickers)

        raw = _build_synthetic_ohlcv(symbol_list)
        # yfinance returns flat columns (Open, High, …) for single-symbol calls,
        # MultiIndex for multi-symbol. bulk_data_fetcher.py:362 branches on this.
        if len(symbol_list) == 1:
            return raw[symbol_list[0]].copy()
        return raw

    @property
    def stats(self) -> dict:
        """Cumulative simulator stats — useful for sanity-checking the harness."""
        sorted_lat = sorted(self._latencies_s)
        n = len(sorted_lat)
        return {
            "calls": self._call_count,
            "rate_limited": self._429_count,
            "p50_latency_s": sorted_lat[n // 2] if n else 0.0,
            "p95_latency_s": sorted_lat[int(n * 0.95)] if n else 0.0,
            "p99_latency_s": sorted_lat[min(int(n * 0.99), n - 1)] if n else 0.0,
        }


_SYNTHETIC_DAYS = 60
_PRICE_FIELDS = ("Open", "High", "Low", "Close", "Adj Close")
# Date index is stable across calls — the harness uses utcnow().date() once at
# import time. Test runs across day boundaries are fine because the values
# are synthetic; only shape matters to downstream parsing.
_CACHED_DATES = pd.date_range(end=datetime.utcnow().date(), periods=_SYNTHETIC_DAYS, freq="B")


def _stable_symbol_hash(symbol: str) -> int:
    """Deterministic hash that doesn't depend on PYTHONHASHSEED.

    Python's built-in ``hash()`` for strings is randomized per-process, which
    would make synthetic prices vary across runs and break baseline diffs that
    eventually inspect data values.
    """
    return int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)


def _build_synthetic_ohlcv(symbols: List[str], days: int = _SYNTHETIC_DAYS) -> pd.DataFrame:
    """Build a yfinance-shaped MultiIndex DataFrame for the given symbols.

    Uses a module-level cached date index when ``days`` matches the default,
    avoiding ~60 microseconds of pd.date_range churn per batch.
    """
    dates = _CACHED_DATES if days == _SYNTHETIC_DAYS else pd.date_range(
        end=datetime.utcnow().date(), periods=days, freq="B"
    )
    columns = pd.MultiIndex.from_product(
        [symbols, [*_PRICE_FIELDS, "Volume"]],
        names=["Ticker", "Field"],
    )
    data = {}
    for sym in symbols:
        base = 100.0 + (_stable_symbol_hash(sym) % 100)
        for field in _PRICE_FIELDS:
            data[(sym, field)] = [base + i * 0.1 for i in range(days)]
        data[(sym, "Volume")] = [1_000_000 + i * 10 for i in range(days)]
    return pd.DataFrame(data, index=dates, columns=columns)


def build_market_simulators(seed: int = 42) -> Dict[str, YFinanceSimulator]:
    """Return one simulator per market, all driven by the same seed offset."""
    return {
        market: YFinanceSimulator(profile, seed=seed + i)
        for i, (market, profile) in enumerate(DEFAULT_PROFILES.items())
    }


class MultiMarketSimulator:
    """Simulator that routes calls to per-market sub-simulators based on the
    symbol prefix (``LOAD_<MARKET>_NNNN`` — see conftest).

    Why a single object: the load harness runs markets in parallel via
    ThreadPoolExecutor, and ``patch.object(bdf_module, "yf", ...)`` would
    race across threads if each thread patched a different simulator.
    Patching ONCE with this multi-market router avoids the race; each
    sub-simulator's stats remain isolated.

    No internal lock: each market's sub-simulator is exercised by exactly
    one harness thread (one ``_run_one_market`` worker per market). Sub-sim
    state (call_count, _429_count, _latencies_s) is therefore single-threaded
    by construction, despite the parent simulator being shared.
    """

    SYMBOL_PREFIX = "LOAD_"

    def __init__(self, sub_simulators: Dict[str, YFinanceSimulator]):
        self._sims = sub_simulators

    def set_profile(self, market: str, profile: SimulatorProfile) -> None:
        """Replace the per-market profile at runtime.

        Used by fault-injection helpers (bead 9.4) to swap a victim market's
        latency/429/failure distribution mid-test without reaching into the
        sub-simulator's private state.
        """
        if market not in self._sims:
            raise KeyError(
                f"Unknown market {market!r}; known: {sorted(self._sims)}"
            )
        self._sims[market].profile = profile

    def _market_for_symbol(self, symbol: str) -> Optional[str]:
        if not symbol.startswith(self.SYMBOL_PREFIX):
            return None
        rest = symbol[len(self.SYMBOL_PREFIX):]
        return rest.split("_", 1)[0] if "_" in rest else None

    def download(self, tickers, **kwargs) -> pd.DataFrame:
        if isinstance(tickers, str):
            symbols = tickers.split()
        else:
            symbols = list(tickers)
        if not symbols:
            return _build_synthetic_ohlcv([])

        markets = {self._market_for_symbol(sym) for sym in symbols}
        if None in markets:
            raise ValueError(
                f"All load-harness symbols must use {self.SYMBOL_PREFIX}<MARKET>_<ID>: {symbols!r}"
            )
        if len(markets) != 1:
            raise ValueError(f"Mixed-market batch is not supported: {symbols!r}")

        market = next(iter(markets))
        if market not in self._sims:
            raise KeyError(f"Unknown market {market!r}; known: {sorted(self._sims)}")

        return self._sims[market].download(tickers, **kwargs)

    @property
    def stats(self) -> Dict[str, dict]:
        """Per-market stats dict. Use this instead of per-sim ``.stats``
        when running through the multi-market router."""
        return {market: sim.stats for market, sim in self._sims.items()}


def build_multi_market_simulator(
    seed: int = 42, sleep_fn=time.sleep
) -> MultiMarketSimulator:
    """Build a single simulator covering all default markets.

    Asserts that ``DEFAULT_PROFILES`` covers every supported market — if a
    new market is added to ``SUPPORTED_MARKETS`` without a calibrated
    profile, the harness fails loudly here rather than silently producing
    empty DataFrames for the new market.
    """
    from app.tasks.market_queues import SUPPORTED_MARKETS
    missing = set(SUPPORTED_MARKETS) - set(DEFAULT_PROFILES)
    if missing:
        raise RuntimeError(
            f"YFinance simulator is missing profile(s) for {sorted(missing)}; "
            f"add to DEFAULT_PROFILES with calibrated latency + 429 numbers."
        )
    sub_sims = {
        market: YFinanceSimulator(profile, seed=seed + i, sleep_fn=sleep_fn)
        for i, (market, profile) in enumerate(DEFAULT_PROFILES.items())
    }
    return MultiMarketSimulator(sub_sims)
