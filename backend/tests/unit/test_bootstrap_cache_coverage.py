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
from app.services.bootstrap_cache_coverage import (
    BootstrapCacheCoverageReport,
    BootstrapPriceCoverageReport,
    evaluate_bootstrap_cache_coverage,
    evaluate_bootstrap_price_cache_coverage,
    normalize_bootstrap_gate_report,
)
import app.services.bootstrap_cache_coverage as bootstrap_cache_coverage_module
from app.domain.markets.catalog import get_market_catalog
from app.services.price_coverage_policy import price_coverage_policy_for_market


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
    )

    assert report["eligible"] is True
    assert report["fundamentals_coverage_date"] == "2026-04-22"
    assert report["fundamentals_covered_symbols"] == 2


def test_bootstrap_price_cache_coverage_ignores_fundamentals_before_later_bootstrap_stage():
    db = _session()
    symbols = [f"SYM{i}" for i in range(20)]
    as_of = date(2026, 4, 24)
    db.add_all([_price(symbol, as_of) for symbol in symbols[:19]])
    db.commit()

    report = evaluate_bootstrap_price_cache_coverage(
        db,
        market="US",
        symbols=symbols,
        as_of_date=as_of,
    )

    assert isinstance(report, BootstrapPriceCoverageReport)
    assert report["eligible"] is True
    assert report.to_dict()["mode"] == "price_ready"
    assert report["price_coverage_ratio"] == 0.95
    assert report["price_covered_symbols"] == 19
    assert report["price_missing_symbols"] == 1
    assert report["price_missing_symbols_preview"] == ["SYM19"]


def test_bootstrap_price_cache_coverage_uses_market_specific_threshold():
    db = _session()
    symbols = [f"SYM{i}" for i in range(20)]
    as_of = date(2026, 4, 24)
    db.add_all([_price(symbol, as_of) for symbol in symbols[:11]])
    db.commit()

    report = evaluate_bootstrap_price_cache_coverage(
        db,
        market="TW",
        symbols=symbols,
        as_of_date=as_of,
    )

    assert report["threshold"] == 0.50
    assert report["price_coverage_ratio"] == 0.55
    assert report["eligible"] is True


def test_bootstrap_cache_coverage_keeps_fundamentals_threshold_strict_for_partial_price_markets():
    db = _session()
    symbols = [f"SYM{i}" for i in range(20)]
    as_of = date(2026, 4, 24)
    db.add_all([_price(symbol, as_of) for symbol in symbols[:11]])
    db.add_all(
        [
            _fundamental(symbol, datetime(2026, 4, 21, tzinfo=timezone.utc))
            for symbol in symbols[:18]
        ]
    )
    db.commit()

    report = evaluate_bootstrap_cache_coverage(
        db,
        market="TW",
        symbols=symbols,
        as_of_date=as_of,
    )

    assert report["price_threshold"] == 0.50
    assert report["fundamentals_threshold"] == 0.95
    assert report["price_coverage_ratio"] == 0.55
    assert report["fundamentals_coverage_ratio"] == 0.90
    assert report["eligible"] is False


def test_bootstrap_cache_report_uses_price_threshold_as_compatibility_alias_without_duplicate_state():
    assert "threshold" not in BootstrapCacheCoverageReport.__dataclass_fields__

    db = _session()
    symbols = [f"SYM{i}" for i in range(20)]
    as_of = date(2026, 4, 24)
    db.add_all([_price(symbol, as_of) for symbol in symbols[:11]])
    db.add_all(
        [
            _fundamental(symbol, datetime(2026, 4, 21, tzinfo=timezone.utc))
            for symbol in symbols
        ]
    )
    db.commit()

    report = evaluate_bootstrap_cache_coverage(
        db,
        market="TW",
        symbols=symbols,
        as_of_date=as_of,
    )

    assert report["threshold"] == report["price_threshold"] == 0.50
    assert report["fundamentals_threshold"] == 0.95


def test_bootstrap_coverage_uses_shared_price_policy_without_alias_exports():
    for alias in (
        "BOOTSTRAP_CACHE_ONLY_MIN_COVERAGE",
        "BOOTSTRAP_CACHE_ONLY_MIN_FUNDAMENTALS_COVERAGE",
        "BOOTSTRAP_PRICE_MIN_COVERAGE_BY_MARKET",
        "BootstrapCoveragePolicy",
        "bootstrap_coverage_policy_for_market",
    ):
        assert not hasattr(bootstrap_cache_coverage_module, alias), alias


def test_normalize_bootstrap_gate_report_owns_threshold_and_unsupported_metadata():
    report = normalize_bootstrap_gate_report(
        market="TW",
        report={
            "eligible": True,
            "price_coverage_ratio": 0.55,
            "fundamentals_coverage_ratio": 1.0,
        },
        unsupported_symbols=["1234.BAD", "5678.BAD"],
    )

    assert report["threshold"] == report["price_threshold"] == 0.50
    assert report["fundamentals_threshold"] == 0.95
    assert report["eligible"] is True
    assert report["mode"] == "cache_only"
    assert report["unsupported_skipped_count"] == 2
    assert report["unsupported_symbols_preview"] == ["1234.BAD", "5678.BAD"]


def test_normalize_bootstrap_gate_report_derives_eligibility_from_ratios_not_claim():
    report = normalize_bootstrap_gate_report(
        market="TW",
        report={
            "eligible": True,
            "threshold": 0.10,
            "price_threshold": 0.10,
            "fundamentals_threshold": 0.10,
            "price_coverage_ratio": 0.49,
            "fundamentals_coverage_ratio": 1.0,
        },
        unsupported_symbols=["1234.BAD"],
    )

    assert report["threshold"] == report["price_threshold"] == 0.50
    assert report["fundamentals_threshold"] == 0.95
    assert report["eligible"] is False
    assert report["mode"] == "waiting_for_cache_coverage"
    assert report["unsupported_skipped_count"] == 0
    assert report["unsupported_symbols_preview"] == []


def test_normalize_bootstrap_gate_report_defaults_closed_when_ratios_are_missing():
    report = normalize_bootstrap_gate_report(
        market="TW",
        report={"eligible": True},
        unsupported_symbols=["1234.BAD"],
    )

    assert report["eligible"] is False
    assert report["mode"] == "waiting_for_cache_coverage"
    assert report["unsupported_skipped_count"] == 0
    assert report["unsupported_symbols_preview"] == []


def test_normalize_bootstrap_gate_report_hides_unsupported_symbols_until_gate_passes():
    report = normalize_bootstrap_gate_report(
        market="TW",
        report={"eligible": False},
        unsupported_symbols=["1234.BAD"],
    )

    assert report["mode"] == "waiting_for_cache_coverage"
    assert report["unsupported_skipped_count"] == 0
    assert report["unsupported_symbols_preview"] == []


def test_bootstrap_price_thresholds_cover_every_supported_market():
    expected_price_thresholds = {
        "AU": 0.90,
        "CA": 0.75,
        "CN": 0.90,
        "DE": 0.90,
        "HK": 0.80,
        "IN": 0.50,
        "JP": 0.90,
        "KR": 0.95,
        "MY": 0.85,
        "SG": 0.60,
        "TW": 0.50,
        "US": 0.95,
    }

    assert set(expected_price_thresholds) == set(
        get_market_catalog().supported_market_codes()
    )

    for market, expected_threshold in expected_price_thresholds.items():
        policy = price_coverage_policy_for_market(market)

        assert policy.market == market
        assert policy.price_min_coverage == expected_threshold, market
        assert policy.fundamentals_min_coverage == 0.95, market


def test_bootstrap_price_cache_coverage_rejects_empty_candidate_set():
    db = _session()

    report = evaluate_bootstrap_price_cache_coverage(
        db,
        market="US",
        symbols=[],
        as_of_date=date(2026, 4, 24),
    )

    assert report["eligible"] is False
    assert report["mode"] == "waiting_for_prices"
    assert report["price_total_symbols"] == 0
    assert report["price_coverage_ratio"] == 0.0
