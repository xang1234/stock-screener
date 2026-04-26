"""Coverage gate for cache-only bootstrap feature snapshots."""

from __future__ import annotations

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
from app.services.provider_snapshot_service import WEEKLY_REFERENCE_SNAPSHOT_KEYS

BOOTSTRAP_CACHE_ONLY_MIN_COVERAGE = 0.95
MISSING_SYMBOL_PREVIEW_LIMIT = 20


def _normalize_symbols(symbols: Sequence[str]) -> list[str]:
    return sorted({str(symbol).upper() for symbol in symbols if symbol})


def _ratio(covered: int, total: int) -> float:
    return covered / total if total > 0 else 1.0


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


def evaluate_bootstrap_cache_coverage(
    db: Session,
    *,
    market: str,
    symbols: Sequence[str],
    as_of_date: date,
) -> dict[str, Any]:
    """Return a JSON-ready coverage report for bootstrap cache-only eligibility."""
    normalized_market = str(market or "US").strip().upper() or "US"
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
    price_missing = [
        symbol
        for symbol in normalized_symbols
        if latest_price_by_symbol.get(symbol) is None
        or latest_price_by_symbol[symbol] < as_of_date
    ]
    price_covered = total - len(price_missing)
    price_ratio = _ratio(price_covered, total)

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
    fundamentals_missing = [
        symbol for symbol in normalized_symbols if symbol not in covered_fundamentals
    ]
    fundamentals_covered = total - len(fundamentals_missing)
    fundamentals_ratio = _ratio(fundamentals_covered, total)

    fundamentals_coverage_date = None
    if snapshot_run is not None:
        fundamentals_coverage_date = _date_string(
            snapshot_run.published_at or snapshot_run.created_at
        )
    elif fundamentals_dates:
        fundamentals_coverage_date = _date_string(max(fundamentals_dates))

    threshold = BOOTSTRAP_CACHE_ONLY_MIN_COVERAGE
    eligible = price_ratio >= threshold and fundamentals_ratio >= threshold
    return {
        "market": normalized_market,
        "threshold": threshold,
        "eligible": eligible,
        "mode": "cache_only" if eligible else "fallback_existing",
        "price_coverage_date": as_of_date.isoformat(),
        "fundamentals_coverage_date": fundamentals_coverage_date,
        "price_total_symbols": total,
        "price_covered_symbols": price_covered,
        "price_missing_symbols": len(price_missing),
        "price_coverage_ratio": price_ratio,
        "price_missing_symbols_preview": price_missing[:MISSING_SYMBOL_PREVIEW_LIMIT],
        "fundamentals_total_symbols": total,
        "fundamentals_covered_symbols": fundamentals_covered,
        "fundamentals_missing_symbols": len(fundamentals_missing),
        "fundamentals_coverage_ratio": fundamentals_ratio,
        "fundamentals_missing_symbols_preview": fundamentals_missing[
            :MISSING_SYMBOL_PREVIEW_LIMIT
        ],
    }
