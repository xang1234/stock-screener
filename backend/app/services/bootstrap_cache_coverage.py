"""Coverage gate for cache-only bootstrap feature snapshots."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Sequence

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.provider_snapshot import (
    ProviderSnapshotPointer,
    ProviderSnapshotRow,
    ProviderSnapshotRun,
)
from app.models.stock import StockFundamental, StockPrice
from app.services.price_coverage_policy import (
    PriceCoveragePolicy,
    normalize_market_code,
    price_coverage_policy_for_market,
)
from app.services.provider_snapshot_service import WEEKLY_REFERENCE_SNAPSHOT_KEYS

MISSING_SYMBOL_PREVIEW_LIMIT = 20


@dataclass(frozen=True)
class BootstrapPriceCoverageReport(Mapping[str, Any]):
    policy: PriceCoveragePolicy
    price_coverage_date: date | str
    price_total_symbols: int
    price_covered_symbols: int
    price_missing_symbols: tuple[str, ...]

    @property
    def market(self) -> str:
        return self.policy.market

    @property
    def threshold(self) -> float:
        return self.policy.price_min_coverage

    @property
    def price_missing_symbol_count(self) -> int:
        return len(self.price_missing_symbols)

    @property
    def price_coverage_ratio(self) -> float:
        return _ratio(self.price_covered_symbols, self.price_total_symbols)

    @property
    def eligible(self) -> bool:
        return self.price_coverage_ratio >= self.threshold

    @property
    def mode(self) -> str:
        return "price_ready" if self.eligible else "waiting_for_prices"

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "threshold": self.threshold,
            "eligible": self.eligible,
            "mode": self.mode,
            "price_coverage_date": _date_string(self.price_coverage_date),
            "price_total_symbols": self.price_total_symbols,
            "price_covered_symbols": self.price_covered_symbols,
            "price_missing_symbols": self.price_missing_symbol_count,
            "price_coverage_ratio": self.price_coverage_ratio,
            "price_missing_symbols_preview": list(
                self.price_missing_symbols[:MISSING_SYMBOL_PREVIEW_LIMIT]
            ),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class BootstrapCacheCoverageReport(Mapping[str, Any]):
    price_report: BootstrapPriceCoverageReport
    fundamentals_coverage_date: str | None
    fundamentals_total_symbols: int
    fundamentals_covered_symbols: int
    fundamentals_missing_symbols: tuple[str, ...]

    @property
    def market(self) -> str:
        return self.price_report.market

    @property
    def threshold(self) -> float:
        return self.price_report.threshold

    @property
    def price_threshold(self) -> float:
        return self.price_report.threshold

    @property
    def fundamentals_threshold(self) -> float:
        return self.price_report.policy.fundamentals_min_coverage

    @property
    def fundamentals_missing_symbol_count(self) -> int:
        return len(self.fundamentals_missing_symbols)

    @property
    def fundamentals_coverage_ratio(self) -> float:
        return _ratio(self.fundamentals_covered_symbols, self.fundamentals_total_symbols)

    @property
    def eligible(self) -> bool:
        return (
            self.price_report.eligible
            and self.fundamentals_coverage_ratio >= self.fundamentals_threshold
        )

    @property
    def mode(self) -> str:
        return "cache_only" if self.eligible else "fallback_existing"

    def to_dict(self) -> dict[str, Any]:
        price_payload = self.price_report.to_dict()
        return {
            "market": self.market,
            "threshold": self.threshold,
            "price_threshold": self.price_report.threshold,
            "fundamentals_threshold": self.fundamentals_threshold,
            "eligible": self.eligible,
            "mode": self.mode,
            "price_coverage_date": price_payload["price_coverage_date"],
            "fundamentals_coverage_date": self.fundamentals_coverage_date,
            "price_total_symbols": price_payload["price_total_symbols"],
            "price_covered_symbols": price_payload["price_covered_symbols"],
            "price_missing_symbols": price_payload["price_missing_symbols"],
            "price_coverage_ratio": price_payload["price_coverage_ratio"],
            "price_missing_symbols_preview": price_payload[
                "price_missing_symbols_preview"
            ],
            "fundamentals_total_symbols": self.fundamentals_total_symbols,
            "fundamentals_covered_symbols": self.fundamentals_covered_symbols,
            "fundamentals_missing_symbols": self.fundamentals_missing_symbol_count,
            "fundamentals_coverage_ratio": self.fundamentals_coverage_ratio,
            "fundamentals_missing_symbols_preview": list(
                self.fundamentals_missing_symbols[:MISSING_SYMBOL_PREVIEW_LIMIT]
            ),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def _normalize_symbols(symbols: Sequence[str]) -> list[str]:
    return sorted({str(symbol).upper() for symbol in symbols if symbol})


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _report_meets_policy(
    payload: Mapping[str, Any],
    policy: PriceCoveragePolicy,
) -> bool:
    price_ratio = _optional_float(payload.get("price_coverage_ratio"))
    fundamentals_ratio = _optional_float(payload.get("fundamentals_coverage_ratio"))
    if price_ratio is None or fundamentals_ratio is None:
        return False
    return (
        price_ratio >= policy.price_min_coverage
        and fundamentals_ratio >= policy.fundamentals_min_coverage
    )


def normalize_bootstrap_gate_report(
    *,
    market: str | None,
    report: Mapping[str, Any] | None,
    unsupported_symbols: Sequence[str],
) -> dict[str, Any]:
    payload = dict(report or {})
    policy = price_coverage_policy_for_market(market)
    eligible = _report_meets_policy(payload, policy)
    payload.update(
        {
            "eligible": eligible,
            "threshold": policy.price_min_coverage,
            "price_threshold": policy.price_min_coverage,
            "fundamentals_threshold": policy.fundamentals_min_coverage,
            "mode": "cache_only" if eligible else "waiting_for_cache_coverage",
            "unsupported_skipped_count": len(unsupported_symbols) if eligible else 0,
            "unsupported_symbols_preview": (
                list(unsupported_symbols[:MISSING_SYMBOL_PREVIEW_LIMIT])
                if eligible
                else []
            ),
        }
    )
    return payload


def _ratio(covered: int, total: int) -> float:
    return covered / total if total > 0 else 0.0


def _date_string(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)


def evaluate_bootstrap_price_cache_coverage(
    db: Session,
    *,
    market: str,
    symbols: Sequence[str],
    as_of_date: date,
) -> BootstrapPriceCoverageReport:
    """Return bootstrap price coverage without requiring later fundamentals stages."""
    normalized_market = normalize_market_code(market)
    normalized_symbols = _normalize_symbols(symbols)
    total = len(normalized_symbols)

    latest_price_rows = (
        db.query(StockPrice.symbol, func.max(StockPrice.date))
        .filter(StockPrice.symbol.in_(normalized_symbols))
        .group_by(StockPrice.symbol)
        .all()
        if normalized_symbols
        else []
    )
    latest_price_by_symbol = {
        str(symbol).upper(): latest_date for symbol, latest_date in latest_price_rows
    }
    price_missing = tuple(
        symbol
        for symbol in normalized_symbols
        if latest_price_by_symbol.get(symbol) is None
        or latest_price_by_symbol[symbol] < as_of_date
    )
    policy = price_coverage_policy_for_market(normalized_market)
    return BootstrapPriceCoverageReport(
        policy=policy,
        price_coverage_date=as_of_date,
        price_total_symbols=total,
        price_covered_symbols=total - len(price_missing),
        price_missing_symbols=price_missing,
    )


def evaluate_bootstrap_cache_coverage(
    db: Session,
    *,
    market: str,
    symbols: Sequence[str],
    as_of_date: date,
) -> BootstrapCacheCoverageReport:
    """Return a JSON-ready coverage report for bootstrap cache-only eligibility."""
    normalized_market = normalize_market_code(market)
    normalized_symbols = _normalize_symbols(symbols)
    total = len(normalized_symbols)
    price_report = evaluate_bootstrap_price_cache_coverage(
        db,
        market=normalized_market,
        symbols=normalized_symbols,
        as_of_date=as_of_date,
    )

    snapshot_key = WEEKLY_REFERENCE_SNAPSHOT_KEYS.get(normalized_market)
    snapshot_run = None
    snapshot_symbols: set[str] = set()
    if snapshot_key is not None:
        pointer = (
            db.query(ProviderSnapshotPointer)
            .filter(ProviderSnapshotPointer.snapshot_key == snapshot_key)
            .first()
        )
        if pointer is not None:
            snapshot_run = db.get(ProviderSnapshotRun, pointer.run_id)
            if snapshot_run is not None and normalized_symbols:
                rows = (
                    db.query(ProviderSnapshotRow.symbol)
                    .filter(
                        ProviderSnapshotRow.run_id == snapshot_run.id,
                        ProviderSnapshotRow.symbol.in_(normalized_symbols),
                    )
                    .all()
                )
                snapshot_symbols = {str(row[0]).upper() for row in rows}

    fundamentals_rows = (
        db.query(StockFundamental.symbol, StockFundamental.updated_at)
        .filter(StockFundamental.symbol.in_(normalized_symbols))
        .all()
        if normalized_symbols
        else []
    )
    fundamentals_symbols = {str(symbol).upper() for symbol, _ in fundamentals_rows}
    fundamentals_dates = [updated_at for _, updated_at in fundamentals_rows if updated_at]
    covered_fundamentals = snapshot_symbols | fundamentals_symbols
    fundamentals_missing = tuple(
        symbol for symbol in normalized_symbols if symbol not in covered_fundamentals
    )
    fundamentals_covered = total - len(fundamentals_missing)

    fundamentals_coverage_date = None
    if snapshot_run is not None:
        fundamentals_coverage_date = _date_string(
            snapshot_run.published_at or snapshot_run.created_at
        )
    elif fundamentals_dates:
        fundamentals_coverage_date = _date_string(max(fundamentals_dates))

    return BootstrapCacheCoverageReport(
        price_report=price_report,
        fundamentals_coverage_date=fundamentals_coverage_date,
        fundamentals_total_symbols=total,
        fundamentals_covered_symbols=fundamentals_covered,
        fundamentals_missing_symbols=fundamentals_missing,
    )
