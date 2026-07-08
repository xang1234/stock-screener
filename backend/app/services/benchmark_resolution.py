"""Benchmark candidate resolution and diagnostics.

This module owns benchmark selection policy. The cache service supplies Redis,
database, and yfinance primitives; the resolver decides which candidate is
usable and returns explicit diagnostics for callers that need failure context.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Protocol

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkCandidateStatus:
    """Structured diagnostic for one benchmark candidate lookup attempt."""

    symbol: str
    role: str
    source: "BenchmarkCandidateSource"
    outcome: "BenchmarkCandidateOutcome"
    latest_date: date | None

    def as_diagnostic(self) -> dict[str, str | None]:
        return {
            "symbol": self.symbol,
            "role": self.role,
            "source": self.source.value,
            "status": self.outcome.value,
            "latest_date": self.latest_date.isoformat() if self.latest_date else None,
        }


class BenchmarkCandidateSource(str, Enum):
    """Where a benchmark candidate came from."""

    REDIS = "redis"
    DATABASE = "database"
    FETCH = "fetch"


class BenchmarkCandidateOutcome(str, Enum):
    """Outcome of inspecting a benchmark candidate."""

    SELECTED = "selected"
    EMPTY = "empty"
    STALE_CACHE = "stale_cache"
    STALE_REQUIRED_DATE = "stale_required_date"


@dataclass(frozen=True)
class BenchmarkDataBundle:
    """Resolved benchmark payload with selection metadata and OHLCV data."""

    market: str
    period: str
    benchmark_symbol: str
    benchmark_role: str
    benchmark_kind: str | None
    candidate_symbols: tuple[str, ...]
    data: pd.DataFrame
    candidate_statuses: tuple[BenchmarkCandidateStatus, ...] = ()


@dataclass(frozen=True)
class BenchmarkResolution:
    """Explicit result object for benchmark resolution attempts."""

    bundle: BenchmarkDataBundle | None
    candidate_statuses: tuple[BenchmarkCandidateStatus, ...] = ()
    error: str | None = None

    def candidate_diagnostics(self) -> list[dict[str, str | None]]:
        return [status.as_diagnostic() for status in self.candidate_statuses]

    def error_payload(self, *, market: str, as_of_date: date) -> dict:
        payload = {
            "error": self.error or "no_benchmark_data",
            "market": market,
            "date": as_of_date.isoformat(),
        }
        diagnostics = self.candidate_diagnostics()
        if diagnostics:
            payload["benchmark_candidates"] = diagnostics
        return payload


class BenchmarkFallbackPolicy(str, Enum):
    """Controls whether benchmark fallback symbols are eligible for a lookup."""

    ALLOW = "allow"
    PRIMARY_ONLY = "primary_only"


class BenchmarkResolutionAdapter(Protocol):
    def load_benchmark_from_redis(
        self,
        benchmark_symbol: str,
        period: str,
        market: str,
    ) -> pd.DataFrame | None: ...

    def load_benchmark_from_database(
        self,
        benchmark_symbol: str,
        period: str,
        market: str = "US",
    ) -> pd.DataFrame | None: ...

    def benchmark_data_is_fresh(
        self,
        data: pd.DataFrame,
        market: str = "US",
        max_age_hours: int = 24,
    ) -> bool: ...

    def store_benchmark_in_redis(
        self,
        benchmark_symbol: str,
        period: str,
        data: pd.DataFrame,
        market: str = "US",
    ) -> None: ...

    def fetch_and_cache_benchmark(
        self,
        benchmark_symbol: str,
        market: str,
        period: str,
    ) -> pd.DataFrame | None: ...


@dataclass(frozen=True)
class _BenchmarkResolutionContext:
    normalized_market: str
    period: str
    registry_entry: Any
    candidate_symbols: tuple[str, ...]
    required_as_of_date: date | None


@dataclass(frozen=True)
class _CandidateProbe:
    bundle: BenchmarkDataBundle | None = None
    status: BenchmarkCandidateStatus | None = None
    stop_candidate: bool = False


class BenchmarkResolver:
    """Resolve benchmark candidates using cache/fetch primitives from an adapter."""

    def __init__(self, *, adapter: BenchmarkResolutionAdapter, registry: Any):
        self._adapter = adapter
        self._registry = registry

    def resolve(
        self,
        *,
        market: str = "US",
        period: str = "2y",
        force_refresh: bool = False,
        fallback_policy: BenchmarkFallbackPolicy = BenchmarkFallbackPolicy.ALLOW,
        required_as_of_date: date | None = None,
    ) -> BenchmarkResolution:
        normalized_market = self._registry.normalize_market(market)
        context = _BenchmarkResolutionContext(
            normalized_market=normalized_market,
            period=period,
            registry_entry=self._registry.get_entry(normalized_market),
            candidate_symbols=self._candidate_symbols_for_policy(
                normalized_market,
                fallback_policy,
            ),
            required_as_of_date=required_as_of_date,
        )
        statuses: list[BenchmarkCandidateStatus] = []

        if not force_refresh:
            bundle = self._resolve_cached_candidates(context=context, statuses=statuses)
            if bundle is not None:
                return self._resolution_for_bundle(bundle)

        bundle = self._resolve_fetched_candidates(
            context=context,
            statuses=statuses,
            force_refresh=force_refresh,
        )
        if bundle is not None:
            return self._resolution_for_bundle(bundle)

        return self._missing_benchmark_resolution(context, statuses)

    @staticmethod
    def _resolution_for_bundle(bundle: BenchmarkDataBundle) -> BenchmarkResolution:
        return BenchmarkResolution(
            bundle=bundle,
            candidate_statuses=bundle.candidate_statuses,
        )

    def _missing_benchmark_resolution(
        self,
        context: _BenchmarkResolutionContext,
        statuses: list[BenchmarkCandidateStatus],
    ) -> BenchmarkResolution:
        logger.warning(
            "No benchmark candidate produced data for market=%s period=%s",
            context.normalized_market,
            context.period,
        )
        status_tuple = tuple(statuses)
        error = (
            "benchmark_not_current"
            if any(
                status.outcome == BenchmarkCandidateOutcome.STALE_REQUIRED_DATE
                for status in status_tuple
            )
            else "no_benchmark_data"
        )
        return BenchmarkResolution(
            bundle=None,
            candidate_statuses=status_tuple,
            error=error,
        )

    def _resolve_cached_candidates(
        self,
        *,
        context: _BenchmarkResolutionContext,
        statuses: list[BenchmarkCandidateStatus],
    ) -> BenchmarkDataBundle | None:
        for idx, benchmark_symbol in enumerate(context.candidate_symbols):
            role = self._candidate_role(idx)
            for source in (BenchmarkCandidateSource.REDIS, BenchmarkCandidateSource.DATABASE):
                probe = self._probe_cached_candidate(
                    context=context,
                    benchmark_symbol=benchmark_symbol,
                    role=role,
                    source=source,
                    prior_statuses=statuses,
                )
                if probe.status is not None:
                    statuses.append(probe.status)
                if probe.bundle is not None:
                    return probe.bundle
                if probe.stop_candidate:
                    break
        return None

    def _probe_cached_candidate(
        self,
        *,
        context: _BenchmarkResolutionContext,
        benchmark_symbol: str,
        role: str,
        source: BenchmarkCandidateSource,
        prior_statuses: list[BenchmarkCandidateStatus],
    ) -> _CandidateProbe:
        cached_data = self._get_cached_candidate(
            source=source,
            benchmark_symbol=benchmark_symbol,
            context=context,
        )
        if cached_data is None:
            return _CandidateProbe()
        if not self._adapter.benchmark_data_is_fresh(cached_data, market=context.normalized_market):
            logger.info(
                "Cache STALE for %s benchmark %s (%s, %s) (%s)",
                context.normalized_market,
                benchmark_symbol,
                context.period,
                role,
                source.value.capitalize(),
            )
            return _CandidateProbe(
                status=self._candidate_status(
                    benchmark_symbol=benchmark_symbol,
                    role=role,
                    source=source,
                    outcome=BenchmarkCandidateOutcome.STALE_CACHE,
                    data=cached_data,
                    context=context,
                )
            )

        probe = self._probe_usable_candidate(
            benchmark_symbol=benchmark_symbol,
            role=role,
            source=source,
            data=cached_data,
            context=context,
            prior_statuses=prior_statuses,
        )
        if probe.bundle is None:
            return probe
        logger.info(
            "Cache HIT for %s benchmark %s (%s, %s) (%s)",
            context.normalized_market,
            benchmark_symbol,
            context.period,
            role,
            source.value.capitalize(),
        )
        if source == BenchmarkCandidateSource.DATABASE:
            self._adapter.store_benchmark_in_redis(
                benchmark_symbol=benchmark_symbol,
                period=context.period,
                data=cached_data,
                market=context.normalized_market,
            )
        return probe

    def _resolve_fetched_candidates(
        self,
        *,
        context: _BenchmarkResolutionContext,
        statuses: list[BenchmarkCandidateStatus],
        force_refresh: bool,
    ) -> BenchmarkDataBundle | None:
        for idx, benchmark_symbol in enumerate(context.candidate_symbols):
            role = self._candidate_role(idx)
            logger.info(
                self._fetch_log_message(force_refresh),
                context.normalized_market,
                benchmark_symbol,
                context.period,
                role,
            )
            fetched = self._adapter.fetch_and_cache_benchmark(
                benchmark_symbol=benchmark_symbol,
                market=context.normalized_market,
                period=context.period,
            )
            probe = self._probe_usable_candidate(
                benchmark_symbol=benchmark_symbol,
                role=role,
                source=BenchmarkCandidateSource.FETCH,
                data=fetched,
                context=context,
                prior_statuses=statuses,
            )
            if probe.status is not None:
                statuses.append(probe.status)
            if probe.bundle is not None:
                return probe.bundle
        return None

    def _probe_usable_candidate(
        self,
        *,
        benchmark_symbol: str,
        role: str,
        source: BenchmarkCandidateSource,
        data: pd.DataFrame | None,
        context: _BenchmarkResolutionContext,
        prior_statuses: list[BenchmarkCandidateStatus],
    ) -> _CandidateProbe:
        if data is None or data.empty:
            return _CandidateProbe(
                status=self._candidate_status(
                    benchmark_symbol=benchmark_symbol,
                    role=role,
                    source=source,
                    outcome=BenchmarkCandidateOutcome.EMPTY,
                    data=data,
                    context=context,
                )
            )
        if not self._data_meets_required_date(
            data,
            required_as_of_date=context.required_as_of_date,
        ):
            logger.info(
                "%s %s benchmark %s (%s, %s) but latest eligible date is not %s",
                "Fetched" if source == BenchmarkCandidateSource.FETCH else "Cache HIT for",
                context.normalized_market,
                benchmark_symbol,
                context.period,
                role,
                context.required_as_of_date,
            )
            return _CandidateProbe(
                status=self._candidate_status(
                    benchmark_symbol=benchmark_symbol,
                    role=role,
                    source=source,
                    outcome=BenchmarkCandidateOutcome.STALE_REQUIRED_DATE,
                    data=data,
                    context=context,
                ),
                stop_candidate=(source != BenchmarkCandidateSource.FETCH),
            )

        selected_status = self._candidate_status(
            benchmark_symbol=benchmark_symbol,
            role=role,
            source=source,
            outcome=BenchmarkCandidateOutcome.SELECTED,
            data=data,
            context=context,
        )
        return _CandidateProbe(
            bundle=BenchmarkDataBundle(
                market=context.normalized_market,
                period=context.period,
                benchmark_symbol=benchmark_symbol,
                benchmark_role=role,
                benchmark_kind=(
                    context.registry_entry.primary_kind
                    if role == "primary"
                    else context.registry_entry.fallback_kind
                ),
                candidate_symbols=context.candidate_symbols,
                data=data,
                candidate_statuses=tuple((*prior_statuses, selected_status)),
            ),
            status=selected_status,
        )

    def _candidate_symbols_for_policy(
        self,
        market: str,
        fallback_policy: BenchmarkFallbackPolicy,
    ) -> tuple[str, ...]:
        all_candidates = self._registry.get_candidate_symbols(market)
        normalized_policy = BenchmarkFallbackPolicy(fallback_policy)
        if normalized_policy == BenchmarkFallbackPolicy.ALLOW:
            return tuple(all_candidates)
        return tuple(all_candidates[:1])

    @staticmethod
    def _candidate_role(index: int) -> str:
        return "primary" if index == 0 else "fallback"

    def _candidate_status(
        self,
        *,
        benchmark_symbol: str,
        role: str,
        source: BenchmarkCandidateSource,
        outcome: BenchmarkCandidateOutcome,
        data: pd.DataFrame | None,
        context: _BenchmarkResolutionContext,
    ) -> BenchmarkCandidateStatus:
        return BenchmarkCandidateStatus(
            symbol=benchmark_symbol,
            role=role,
            source=source,
            outcome=outcome,
            latest_date=self._latest_data_date(
                data,
                as_of_date=context.required_as_of_date,
            ),
        )

    def _get_cached_candidate(
        self,
        *,
        source: BenchmarkCandidateSource,
        benchmark_symbol: str,
        context: _BenchmarkResolutionContext,
    ) -> pd.DataFrame | None:
        if source == BenchmarkCandidateSource.REDIS:
            return self._adapter.load_benchmark_from_redis(
                benchmark_symbol=benchmark_symbol,
                period=context.period,
                market=context.normalized_market,
            )
        return self._adapter.load_benchmark_from_database(
            benchmark_symbol=benchmark_symbol,
            period=context.period,
            market=context.normalized_market,
        )

    @staticmethod
    def _fetch_log_message(force_refresh: bool) -> str:
        if force_refresh:
            return "Force refresh requested for %s benchmark %s (%s, %s)"
        return "Cache MISS for %s benchmark %s (%s, %s) - fetching from yfinance"

    def _data_meets_required_date(
        self,
        data: pd.DataFrame | None,
        *,
        required_as_of_date: date | None,
    ) -> bool:
        if required_as_of_date is None:
            return True
        return self._latest_data_date(data, as_of_date=required_as_of_date) == required_as_of_date

    @staticmethod
    def _latest_data_date(
        data: pd.DataFrame | None,
        *,
        as_of_date: date | None = None,
    ) -> date | None:
        if data is None or data.empty:
            return None
        eligible_dates = [
            row_date
            for row_date in (pd.Timestamp(value).date() for value in data.index)
            if as_of_date is None or row_date <= as_of_date
        ]
        return max(eligible_dates) if eligible_dates else None
