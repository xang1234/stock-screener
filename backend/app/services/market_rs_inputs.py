"""Load one bounded, point-in-time input set for canonical Market RS."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import math

from sqlalchemy.orm import Session

from app.domain.relative_strength import HORIZON_SESSIONS
from app.models.stock import StockPrice
from app.services.benchmark_registry_service import (
    BenchmarkRegistryService,
    benchmark_registry,
)
from app.services.market_calendar_service import MarketCalendarService
from app.services.point_in_time_universe_service import (
    PointInTimeUniverseService,
    PointInTimeUniverseUnavailable,
)


EMPTY_UNIVERSE_HASH = hashlib.sha256(b"").hexdigest()
MINIMUM_CURRENT_PRICE_COVERAGE = 0.90


@dataclass(frozen=True)
class MarketRsInputs:
    market: str
    as_of_date: date
    benchmark_symbol: str
    benchmark_as_of_date: date
    universe_hash: str
    expected_symbols: tuple[str, ...]
    excess_returns_by_symbol: dict[str, dict[str, float]]
    exclusions: dict[str, str]
    current_price_coverage: float


class MarketRsInputUnavailable(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        diagnostics: dict[str, object],
        benchmark_symbol: str = "SPY",
        universe_hash: str = EMPTY_UNIVERSE_HASH,
        expected_symbol_count: int = 0,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.diagnostics = diagnostics
        self.benchmark_symbol = benchmark_symbol
        self.universe_hash = universe_hash
        self.expected_symbol_count = expected_symbol_count


class MarketRsInputLoader:
    def __init__(
        self,
        *,
        point_in_time_universe: PointInTimeUniverseService | None = None,
        market_calendar: MarketCalendarService | None = None,
        benchmark_registry: BenchmarkRegistryService = benchmark_registry,
    ) -> None:
        self._point_in_time_universe = (
            point_in_time_universe or PointInTimeUniverseService()
        )
        self._market_calendar = market_calendar or MarketCalendarService()
        self._benchmark_registry = benchmark_registry

    @staticmethod
    def _valid_price(value: float | None) -> bool:
        return value is not None and math.isfinite(float(value)) and float(value) > 0

    def load(
        self,
        db: Session,
        *,
        market: str,
        as_of_date: date,
    ) -> MarketRsInputs:
        normalized = self._benchmark_registry.normalize_market(market)
        benchmark_candidates = tuple(
            self._benchmark_registry.get_candidate_symbols(normalized)
        )
        primary_benchmark = self._benchmark_registry.get_primary_symbol(normalized)
        try:
            universe = self._point_in_time_universe.resolve(
                db,
                market=normalized,
                as_of_date=as_of_date,
            )
        except PointInTimeUniverseUnavailable as exc:
            raise MarketRsInputUnavailable(
                str(exc),
                reason_code="point_in_time_universe_unavailable",
                diagnostics={"error": str(exc)},
                benchmark_symbol=primary_benchmark,
            ) from exc
        context = {
            "benchmark_symbol": primary_benchmark,
            "universe_hash": universe.universe_hash,
            "expected_symbol_count": len(universe.symbols),
        }

        try:
            anchors = self._market_calendar.session_anchors(
                normalized,
                as_of_date,
                offsets=tuple(HORIZON_SESSIONS.values()),
            )
        except ValueError as exc:
            raise MarketRsInputUnavailable(
                str(exc),
                reason_code="session_anchors_unavailable",
                diagnostics={"error": str(exc)},
                **context,
            ) from exc

        anchor_dates = set(anchors.values())
        query_symbols = tuple(
            dict.fromkeys((*universe.symbols, *benchmark_candidates))
        )
        rows = (
            db.query(
                StockPrice.symbol,
                StockPrice.date,
                StockPrice.adj_close,
            )
            .filter(
                StockPrice.symbol.in_(query_symbols),
                StockPrice.date.in_(anchor_dates),
            )
            .all()
        )
        prices: dict[tuple[str, date], float] = {}
        for row in rows:
            if self._valid_price(row.adj_close):
                prices[(row.symbol, row.date)] = float(row.adj_close)

        benchmark_symbol = next(
            (
                candidate
                for candidate in benchmark_candidates
                if all((candidate, anchor) in prices for anchor in anchor_dates)
            ),
            None,
        )
        if benchmark_symbol is None:
            missing_by_candidate = {
                candidate: sorted(
                    anchor.isoformat()
                    for anchor in anchor_dates
                    if (candidate, anchor) not in prices
                )
                for candidate in benchmark_candidates
            }
            raise MarketRsInputUnavailable(
                f"No {normalized} benchmark has every exact RS session anchor",
                reason_code="benchmark_adjusted_anchor_missing",
                diagnostics={"missing_anchor_dates": missing_by_candidate},
                **context,
            )

        context["benchmark_symbol"] = benchmark_symbol
        current_date = anchors[0]
        current_available = sum(
            (symbol, current_date) in prices for symbol in universe.symbols
        )
        current_price_coverage = (
            current_available / len(universe.symbols) if universe.symbols else 0.0
        )
        if current_price_coverage < MINIMUM_CURRENT_PRICE_COVERAGE:
            raise MarketRsInputUnavailable(
                f"{normalized} current price coverage is "
                f"{current_price_coverage:.1%}; 90.0% required",
                reason_code="current_adjusted_price_coverage_below_threshold",
                diagnostics={
                    "current_price_coverage": current_price_coverage,
                    "current_prices_available": current_available,
                    "expected_symbol_count": len(universe.symbols),
                },
                **context,
            )

        benchmark_current = prices[(benchmark_symbol, current_date)]
        benchmark_returns = {
            horizon: benchmark_current / prices[(benchmark_symbol, anchors[offset])]
            - 1.0
            for horizon, offset in HORIZON_SESSIONS.items()
        }
        excess_returns_by_symbol: dict[str, dict[str, float]] = {}
        exclusions: dict[str, str] = {}
        required_offsets = (0, *HORIZON_SESSIONS.values())
        for symbol in universe.symbols:
            missing_offset = next(
                (
                    offset
                    for offset in required_offsets
                    if (symbol, anchors[offset]) not in prices
                ),
                None,
            )
            if missing_offset is not None:
                label = "current" if missing_offset == 0 else str(missing_offset)
                exclusions[symbol] = f"missing_adjusted_{label}_session_anchor"
                continue

            current_stock = prices[(symbol, current_date)]
            excess_returns_by_symbol[symbol] = {
                horizon: (
                    current_stock / prices[(symbol, anchors[offset])] - 1.0
                )
                - benchmark_returns[horizon]
                for horizon, offset in HORIZON_SESSIONS.items()
            }

        return MarketRsInputs(
            market=normalized,
            as_of_date=as_of_date,
            benchmark_symbol=benchmark_symbol,
            benchmark_as_of_date=current_date,
            universe_hash=universe.universe_hash,
            expected_symbols=universe.symbols,
            excess_returns_by_symbol=excess_returns_by_symbol,
            exclusions=exclusions,
            current_price_coverage=current_price_coverage,
        )
