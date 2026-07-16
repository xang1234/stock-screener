"""Authoritative cache-coverage accounting for breadth calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


CACHE_MISS_SYMBOL_SAMPLE_LIMIT = 20


@dataclass(frozen=True)
class BreadthPriceCoverage:
    candidate_stocks: int
    symbols_with_cached_history: int
    cache_miss_stocks: int
    cache_miss_symbols_sample: tuple[str, ...]
    cache_coverage_ratio: float


@dataclass
class BreadthPriceCoverageAccumulator:
    _candidate_symbols: set[str] = field(default_factory=set)
    _cached_symbols: set[str] = field(default_factory=set)
    _cache_miss_symbols: set[str] = field(default_factory=set)

    def record_batch(
        self,
        candidate_symbols: Iterable[str],
        cache_miss_symbols: Iterable[str],
    ) -> None:
        candidates = set(candidate_symbols)
        misses = set(cache_miss_symbols)
        self._candidate_symbols.update(candidates)
        self._cache_miss_symbols.update(misses)
        self._cached_symbols.update(candidates - misses)

    def report(self) -> BreadthPriceCoverage:
        candidate_count = len(self._candidate_symbols)
        cached_count = len(self._cached_symbols)
        return BreadthPriceCoverage(
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
        )


@dataclass(frozen=True)
class BreadthOutcomeReport:
    scanned: int = 0
    cache_misses: int = 0
    insufficient: int = 0
    errors: int = 0

    @property
    def skipped(self) -> int:
        return self.cache_misses + self.insufficient + self.errors

    def __add__(self, other: "BreadthOutcomeReport") -> "BreadthOutcomeReport":
        return BreadthOutcomeReport(
            scanned=self.scanned + other.scanned,
            cache_misses=self.cache_misses + other.cache_misses,
            insufficient=self.insufficient + other.insufficient,
            errors=self.errors + other.errors,
        )


@dataclass
class BreadthOutcomeCounter:
    _scanned: int = 0
    _cache_misses: int = 0
    _insufficient: int = 0
    _errors: int = 0

    def record_scanned(self) -> None:
        self._scanned += 1

    def record_cache_miss(self) -> None:
        self._cache_misses += 1

    def record_insufficient(self) -> None:
        self._insufficient += 1

    def record_error(self) -> None:
        self._errors += 1

    def report(self) -> BreadthOutcomeReport:
        return BreadthOutcomeReport(
            scanned=self._scanned,
            cache_misses=self._cache_misses,
            insufficient=self._insufficient,
            errors=self._errors,
        )


@dataclass(frozen=True)
class BreadthCoverageReport:
    price_coverage: BreadthPriceCoverage
    outcomes: BreadthOutcomeReport

    @classmethod
    def from_parts(
        cls,
        price_coverage: BreadthPriceCoverage,
        outcomes: BreadthOutcomeReport,
    ) -> "BreadthCoverageReport":
        return cls(
            price_coverage=price_coverage,
            outcomes=outcomes,
        )

    @property
    def candidate_stocks(self) -> int:
        return self.price_coverage.candidate_stocks

    @property
    def symbols_with_cached_history(self) -> int:
        return self.price_coverage.symbols_with_cached_history

    @property
    def cache_miss_stocks(self) -> int:
        return self.price_coverage.cache_miss_stocks

    @property
    def cache_miss_symbols_sample(self) -> tuple[str, ...]:
        return self.price_coverage.cache_miss_symbols_sample

    @property
    def cache_coverage_ratio(self) -> float:
        return self.price_coverage.cache_coverage_ratio

    @property
    def total_stocks_scanned(self) -> int:
        return self.outcomes.scanned

    @property
    def skipped_stocks(self) -> int:
        return self.outcomes.skipped

    @property
    def insufficient_data_stocks(self) -> int:
        return self.outcomes.insufficient

    @property
    def error_stocks(self) -> int:
        return self.outcomes.errors

    @property
    def insufficient_history_observations(self) -> int:
        return self.outcomes.insufficient

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

@dataclass(frozen=True)
class BreadthCalculationResult:
    indicators: Mapping[str, Any]
    coverage: BreadthCoverageReport

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            **dict(self.indicators),
            **self.coverage.to_daily_dict(),
        }
