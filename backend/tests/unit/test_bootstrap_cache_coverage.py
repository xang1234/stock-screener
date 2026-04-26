"""Coverage gate tests for bootstrap cache-only snapshots."""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.provider_snapshot import (
    ProviderSnapshotPointer,
    ProviderSnapshotRow,
    ProviderSnapshotRun,
)
from app.models.stock import StockFundamental, StockPrice
from app.services.bootstrap_cache_coverage import evaluate_bootstrap_cache_coverage


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _price(symbol: str, day: date) -> StockPrice:
    return StockPrice(
        symbol=symbol,
        date=day,
        open=10,
        high=11,
        low=9,
        close=10,
        volume=1000,
    )


def _fundamental(symbol: str, updated_at: datetime) -> StockFundamental:
    return StockFundamental(symbol=symbol, updated_at=updated_at)


def _published_snapshot(
    *,
    snapshot_key: str,
    symbols: list[str],
    published_at: datetime,
) -> tuple[ProviderSnapshotRun, list[ProviderSnapshotRow], ProviderSnapshotPointer]:
    run = ProviderSnapshotRun(
        snapshot_key=snapshot_key,
        run_mode="publish",
        status="published",
        source_revision=f"rev-{snapshot_key}",
        symbols_total=len(symbols),
        symbols_published=len(symbols),
        created_at=published_at,
        published_at=published_at,
    )
    rows = [
        ProviderSnapshotRow(
            run_id=1,
            symbol=symbol,
            row_hash=f"hash-{symbol}",
            normalized_payload_json='{"market": "US"}',
        )
        for symbol in symbols
    ]
    pointer = ProviderSnapshotPointer(
        snapshot_key=snapshot_key,
        run_id=1,
        updated_at=published_at,
    )
    return run, rows, pointer


def test_bootstrap_cache_coverage_is_eligible_when_price_and_fundamentals_meet_threshold():
    db = _session()
    symbols = [f"SYM{i}" for i in range(20)]
    as_of = date(2026, 4, 24)
    published_at = datetime(2026, 4, 20, tzinfo=timezone.utc)
    run, rows, pointer = _published_snapshot(
        snapshot_key="fundamentals_v1_us",
        symbols=symbols[:19],
        published_at=published_at,
    )
    db.add(run)
    db.flush()
    for row in rows:
        row.run_id = run.id
        db.add(row)
    pointer.run_id = run.id
    db.add(pointer)
    db.add_all([_price(symbol, as_of) for symbol in symbols[:19]])
    db.commit()

    report = evaluate_bootstrap_cache_coverage(
        db,
        market="US",
        symbols=symbols,
        as_of_date=as_of,
        threshold=0.95,
    )

    assert report["eligible"] is True
    assert report["mode"] == "cache_only"
    assert report["price_coverage_date"] == "2026-04-24"
    assert report["fundamentals_coverage_date"] == "2026-04-20"
    assert report["price_coverage_ratio"] == 0.95
    assert report["fundamentals_coverage_ratio"] == 0.95
    assert report["price_missing_symbols_preview"] == ["SYM19"]
    assert report["fundamentals_missing_symbols_preview"] == ["SYM19"]


def test_bootstrap_cache_coverage_falls_back_when_either_side_is_below_threshold():
    db = _session()
    symbols = [f"SYM{i}" for i in range(20)]
    as_of = date(2026, 4, 24)
    db.add_all([_price(symbol, as_of) for symbol in symbols[:18]])
    db.add_all(
        [
            _fundamental(symbol, datetime(2026, 4, 21, tzinfo=timezone.utc))
            for symbol in symbols
        ]
    )
    db.commit()

    report = evaluate_bootstrap_cache_coverage(
        db,
        market="US",
        symbols=symbols,
        as_of_date=as_of,
        threshold=0.95,
    )

    assert report["eligible"] is False
    assert report["mode"] == "fallback_existing"
    assert report["price_coverage_ratio"] == 0.9
    assert report["fundamentals_coverage_ratio"] == 1.0
    assert report["price_missing_symbols_preview"] == ["SYM18", "SYM19"]


def test_bootstrap_cache_coverage_uses_fundamentals_updated_at_when_no_snapshot_exists():
    db = _session()
    symbols = ["AAPL", "MSFT"]
    as_of = date(2026, 4, 24)
    updated_at = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)
    db.add_all([_price(symbol, as_of) for symbol in symbols])
    db.add_all([_fundamental(symbol, updated_at) for symbol in symbols])
    db.commit()

    report = evaluate_bootstrap_cache_coverage(
        db,
        market="US",
        symbols=symbols,
        as_of_date=as_of,
        threshold=0.95,
    )

    assert report["eligible"] is True
    assert report["fundamentals_coverage_date"] == "2026-04-22"
    assert report["fundamentals_covered_symbols"] == 2
