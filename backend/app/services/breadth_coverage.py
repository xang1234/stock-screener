"""Authoritative cache-coverage accounting for breadth calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


CACHE_MISS_SYMBOL_SAMPLE_LIMIT = 20


@dataclass(frozen=True)
class BreadthCoverageReport:
    candidate_stocks: int
    symbols_with_cached_history: int
    cache_miss_stocks: int
    cache_miss_symbols_sample: tuple[str, ...]
    cache_coverage_ratio: float
    total_stocks_scanned: int
    skipped_stocks: int
    insufficient_data_stocks: int
    error_stocks: int
    insufficient_history_observations: int

    def to_daily_dict(self) -> dict[str, Any]:
        return {
            "candidate_stocks": self.candidate_stocks,
            "total_stocks_scanned": self.total_stocks_scanned,
            "symbols_with_cached_history": self.symbols_with_cached_history,
            "skipped_stocks": self.skipped_stocks,
            "cache_miss_stocks": self.cache_miss_stocks,
            "insufficient_data_stocks": self.insufficient_data_stocks,
            "error_stocks": self.error_stocks,
            "cache_coverage_ratio": self.cache_coverage_ratio,
            "cache_miss_symbols_sample": list(self.cache_miss_symbols_sample),
        }

    def to_backfill_dict(self) -> dict[str, Any]:
        return {
            "target_symbols": self.candidate_stocks,
            "symbols_with_cached_history": self.symbols_with_cached_history,
            "cache_miss_stocks": self.cache_miss_stocks,
            "cache_miss_symbols_sample": list(self.cache_miss_symbols_sample),
            "cache_coverage_ratio": self.cache_coverage_ratio,
            "insufficient_history_observations": (
                self.insufficient_history_observations
            ),
        }


@dataclass
class BreadthCoverageAccumulator:
    _candidate_symbols: set[str] = field(default_factory=set)
    _cached_symbols: set[str] = field(default_factory=set)
    _cache_miss_symbols: set[str] = field(default_factory=set)
    _scanned: int = 0
    _skipped: int = 0
    _insufficient: int = 0
    _errors: int = 0
    _insufficient_observations: int = 0

    def record_price_batch(
        self,
        candidate_symbols: Iterable[str],
        cache_miss_symbols: Iterable[str],
    ) -> None:
        candidates = set(candidate_symbols)
        misses = set(cache_miss_symbols)
        self._candidate_symbols.update(candidates)
        self._cache_miss_symbols.update(misses)
        self._cached_symbols.update(candidates - misses)

    def record_scanned(self) -> None:
        self._scanned += 1

    def record_cache_miss(self) -> None:
        self._skipped += 1

    def record_insufficient(self) -> None:
        self._insufficient += 1
        self._insufficient_observations += 1
        self._skipped += 1

    def record_error(self) -> None:
        self._errors += 1
        self._skipped += 1

    def report(self) -> BreadthCoverageReport:
        candidate_count = len(self._candidate_symbols)
        cached_count = len(self._cached_symbols)
        return BreadthCoverageReport(
            candidate_stocks=candidate_count,
            symbols_with_cached_history=cached_count,
            cache_miss_stocks=len(self._cache_miss_symbols),
            cache_miss_symbols_sample=tuple(
                sorted(self._cache_miss_symbols)[
                    :CACHE_MISS_SYMBOL_SAMPLE_LIMIT
                ]
            ),
            cache_coverage_ratio=(
                cached_count / candidate_count if candidate_count else 0.0
            ),
            total_stocks_scanned=self._scanned,
            skipped_stocks=self._skipped,
            insufficient_data_stocks=self._insufficient,
            error_stocks=self._errors,
            insufficient_history_observations=self._insufficient_observations,
        )


@dataclass(frozen=True)
class BreadthCalculationResult:
    indicators: Mapping[str, Any]
    coverage: BreadthCoverageReport

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            **dict(self.indicators),
            **self.coverage.to_daily_dict(),
        }
