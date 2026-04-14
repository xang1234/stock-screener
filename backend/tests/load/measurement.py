"""Measurement collection, JSON snapshot, and regression-gating for load tests.

Records per-market wall-clock + 429 count + tail-latency + worker memory/CPU
into a stable JSON shape. Compares against a committed baseline and reports
regressions so a CI gate can fail the build on threshold breaches.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# CI-gating regression thresholds.
#
# We gate on two **deterministic** metrics:
#   - yfinance_calls: purely a function of seed + pipeline logic. ANY change
#     means the fetcher's batching/retry behavior shifted (which is what
#     9.3 actually wants to detect). Threshold is 0% — exact match required.
#   - rate_limit_429s: deterministic from the seeded simulator. +50% allows
#     for one extra injection on the boundary.
#
# wall_clock_s and tail-latency are reported in the snapshot for human
# review (perf trend analysis) but NOT gated because real-world CI host
# noise produces 20-30% wall-clock variance between identical runs. The
# original +20% wall-clock threshold from the bead notes turned out to be
# tighter than measurement noise.
YFINANCE_CALLS_REGRESSION_PCT = 0.0   # exact match
RATE_LIMIT_REGRESSION_PCT = 50.0


@dataclass
class MarketMetrics:
    """Per-market measurements collected during a load run."""
    market: str
    wall_clock_s: float
    symbols_processed: int
    yfinance_calls: int
    rate_limit_429s: int
    transient_failures: int
    p50_batch_latency_s: float
    p95_batch_latency_s: float
    p99_batch_latency_s: float


@dataclass
class WorkerResourceMetrics:
    """Process-level memory/CPU sampling. Reported, not gated."""
    peak_rss_mb: float
    avg_cpu_percent: float
    samples: int


@dataclass
class LoadRunSnapshot:
    """Top-level artifact written to disk and compared against baseline."""
    schema_version: int = 1
    run_id: str = ""
    git_sha: str = ""
    timestamp_utc: str = ""
    scenario: str = ""
    seed: int = 42
    universe_size_per_market: Dict[str, int] = field(default_factory=dict)
    markets: List[MarketMetrics] = field(default_factory=list)
    worker_resources: Optional[WorkerResourceMetrics] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "LoadRunSnapshot":
        markets = [MarketMetrics(**m) for m in data.get("markets", [])]
        wr = data.get("worker_resources")
        worker_resources = WorkerResourceMetrics(**wr) if wr else None
        return cls(
            schema_version=data.get("schema_version", 1),
            run_id=data.get("run_id", ""),
            git_sha=data.get("git_sha", ""),
            timestamp_utc=data.get("timestamp_utc", ""),
            scenario=data.get("scenario", ""),
            seed=data.get("seed", 42),
            universe_size_per_market=data.get("universe_size_per_market", {}),
            markets=markets,
            worker_resources=worker_resources,
        )


@dataclass
class RegressionFinding:
    """A single per-market regression detected by the comparison."""
    market: str
    metric: str
    baseline: float
    current: float
    delta_pct: float
    threshold_pct: float

    def __str__(self) -> str:
        return (
            f"[{self.market}] {self.metric}: {self.baseline:.2f} → {self.current:.2f} "
            f"({self.delta_pct:+.1f}%, threshold ±{self.threshold_pct:.0f}%)"
        )


@dataclass
class RegressionReport:
    """Aggregated regression diff between current run and baseline."""
    regressions: List[RegressionFinding] = field(default_factory=list)
    improvements: List[RegressionFinding] = field(default_factory=list)
    new_markets: List[str] = field(default_factory=list)
    removed_markets: List[str] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)

    def format_summary(self) -> str:
        lines = []
        if self.regressions:
            lines.append(f"Regressions ({len(self.regressions)}):")
            for r in self.regressions:
                lines.append(f"  ✗ {r}")
        if self.improvements:
            lines.append(f"Improvements ({len(self.improvements)}):")
            for r in self.improvements:
                lines.append(f"  ✓ {r}")
        if self.new_markets:
            lines.append(f"New markets in current run: {sorted(self.new_markets)}")
        if self.removed_markets:
            lines.append(f"Markets missing from current run: {sorted(self.removed_markets)}")
        if not lines:
            lines.append("No changes vs baseline.")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Snapshot I/O
# ----------------------------------------------------------------------------

def write_snapshot(snapshot: LoadRunSnapshot, path: Path) -> None:
    """Persist a snapshot as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(snapshot.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")


def read_snapshot(path: Path) -> Optional[LoadRunSnapshot]:
    """Read a baseline snapshot. Returns None when the file is missing —
    the harness uses this to decide whether to bootstrap a new baseline."""
    if not path.exists():
        return None
    with path.open() as f:
        return LoadRunSnapshot.from_dict(json.load(f))


# ----------------------------------------------------------------------------
# Regression comparison
# ----------------------------------------------------------------------------

def compare_to_baseline(
    current: LoadRunSnapshot,
    baseline: LoadRunSnapshot,
    yfinance_calls_threshold_pct: float = YFINANCE_CALLS_REGRESSION_PCT,
    rate_limit_threshold_pct: float = RATE_LIMIT_REGRESSION_PCT,
) -> RegressionReport:
    """Compute a per-market regression diff against the baseline.

    Gates on the two deterministic metrics (yfinance_calls and rate_limit_429s).
    Wall-clock + tail-latency are recorded in the snapshot for human-review
    trend analysis but not gated, because real-world CI host noise produces
    wall-clock variance that swamps any threshold tighter than ~50%.
    """
    report = RegressionReport()

    baseline_by_market = {m.market: m for m in baseline.markets}
    current_by_market = {m.market: m for m in current.markets}

    for market in sorted(set(baseline_by_market) | set(current_by_market)):
        if market not in current_by_market:
            report.removed_markets.append(market)
            continue
        if market not in baseline_by_market:
            report.new_markets.append(market)
            continue

        base = baseline_by_market[market]
        cur = current_by_market[market]

        for metric, threshold in [
            ("yfinance_calls", yfinance_calls_threshold_pct),
            ("rate_limit_429s", rate_limit_threshold_pct),
        ]:
            base_val = float(getattr(base, metric))
            cur_val = float(getattr(cur, metric))
            delta_pct = _pct_change(base_val, cur_val)
            finding = RegressionFinding(
                market=market,
                metric=metric,
                baseline=base_val,
                current=cur_val,
                delta_pct=delta_pct,
                threshold_pct=threshold,
            )
            if delta_pct > threshold:
                report.regressions.append(finding)
            elif delta_pct < -threshold:
                report.improvements.append(finding)

    return report


def _pct_change(baseline: float, current: float) -> float:
    """Percent change from baseline to current. 0 baseline with non-zero
    current returns +inf (treated as regression); 0 to 0 returns 0."""
    if baseline == 0:
        return float("inf") if current > 0 else 0.0
    return ((current - baseline) / baseline) * 100.0


# ----------------------------------------------------------------------------
# Resource sampling
# ----------------------------------------------------------------------------

class ResourceSampler:
    """Lightweight psutil-based memory/CPU sampler.

    Polls in the calling thread at a fixed interval; user threads wrap
    workload between ``start()`` and ``stop()`` calls. Skipped entirely
    when psutil is unavailable so the harness still runs without the
    optional dependency.
    """

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self._rss_samples_mb: List[float] = []
        self._cpu_samples_pct: List[float] = []
        self._proc = None
        self._enabled = False

    def __enter__(self):
        try:
            import psutil
            self._proc = psutil.Process(os.getpid())
            # Prime CPU percent measurement
            self._proc.cpu_percent(interval=None)
            self._enabled = True
            self._start_time = time.monotonic()
        except ImportError:
            self._enabled = False
        return self

    def sample(self) -> None:
        if not self._enabled or self._proc is None:
            return
        try:
            mem = self._proc.memory_info().rss / (1024 * 1024)
            cpu = self._proc.cpu_percent(interval=None)
            self._rss_samples_mb.append(mem)
            self._cpu_samples_pct.append(cpu)
        except Exception:
            pass

    def __exit__(self, *args):
        # No cleanup needed; sample() is opt-in by callers.
        pass

    def metrics(self) -> Optional[WorkerResourceMetrics]:
        if not self._rss_samples_mb:
            return None
        return WorkerResourceMetrics(
            peak_rss_mb=max(self._rss_samples_mb),
            avg_cpu_percent=statistics.mean(self._cpu_samples_pct) if self._cpu_samples_pct else 0.0,
            samples=len(self._rss_samples_mb),
        )
