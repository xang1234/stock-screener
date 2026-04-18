"""CLI for building the read-only static-site data bundle."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import func

from app.database import SessionLocal
from app.infra.db.models.feature_store import FeatureRunPointer
from app.models.industry import IBDGroupRank
from app.models.market_breadth import MarketBreadth
from app.models.stock import StockPrice
from app.models.stock_universe import StockUniverse
from app.scripts._runtime import prepare_runtime, repo_root
from app.services.breadth_calculator_service import BreadthCalculatorService
from app.services.bulk_data_fetcher import BulkDataFetcher
from app.services.ibd_industry_service import IBDIndustryService
from app.services.static_site_export_service import StaticSiteExportService
from app.tasks.data_fetch_lock import disable_serialized_data_fetch_lock
from app.utils.market_hours import get_last_market_close, is_trading_day
from app.utils.symbol_support import split_supported_price_symbols
from app.wiring.bootstrap import (
    get_group_rank_service,
    get_price_cache,
    get_provider_snapshot_service,
)


STATIC_DAILY_PRICE_REFRESH_PERIOD = "7d"
STATIC_DAILY_PRICE_REFRESH_BATCH_SIZE = 250
STATIC_GROUP_HISTORY_LOOKBACK_DAYS = 100
STATIC_BREADTH_HISTORY_MIN_TRADING_DAYS = 20
STATIC_BREADTH_HISTORY_LOOKBACK_DAYS = 90
STATIC_BUILD_MODE_PRICE_DELTA = "price_delta"
STATIC_BUILD_MODE_FULL = "full"
STATIC_EXPORT_MARKETS = ("US", "HK", "JP", "TW")
STATIC_DEFAULT_MARKET = "US"


def _default_output_dir() -> Path:
    return repo_root() / "frontend" / "public" / "static-data"


def _tracked_ibd_csv_path() -> Path:
    return repo_root() / "data" / "IBD_industry_group.csv"


def _resolve_latest_completed_us_trading_date() -> date:
    """Return the latest completed NYSE session date for static exports."""
    return get_last_market_close().date()


def _iter_chunks(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[index:index + chunk_size] for index in range(0, len(items), chunk_size)]


def _market_pointer_key(market: str) -> str:
    return f"latest_published_market:{market.upper()}"


def _upsert_feature_run_pointer(*, pointer_key: str, run_id: int) -> None:
    with SessionLocal() as db:
        if not hasattr(db, "query"):
            return
        pointer = (
            db.query(FeatureRunPointer)
            .filter(FeatureRunPointer.key == pointer_key)
            .first()
        )
        if pointer is None:
            db.add(FeatureRunPointer(key=pointer_key, run_id=run_id))
        else:
            pointer.run_id = run_id
        db.commit()


def _refresh_static_daily_prices(*, as_of_date: date) -> dict[str, Any]:
    """Refresh recent price bars in batches without Redis or warmup metadata."""
    price_cache = get_price_cache()
    fetcher = BulkDataFetcher()

    with SessionLocal() as db:
        active_symbols = [
            symbol
            for symbol, in db.query(StockUniverse.symbol)
            .filter(StockUniverse.is_active.is_(True))
            .order_by(StockUniverse.market_cap.desc().nullslast(), StockUniverse.symbol.asc())
            .all()
        ]
        supported_symbols, skipped_symbols = split_supported_price_symbols(active_symbols)
        latest_rows = (
            db.query(StockPrice.symbol, func.max(StockPrice.date))
            .filter(StockPrice.symbol.in_(supported_symbols))
            .group_by(StockPrice.symbol)
            .all()
        )

    latest_by_symbol = {symbol: latest_date for symbol, latest_date in latest_rows}
    db_fresh_symbols = [
        symbol for symbol in supported_symbols if latest_by_symbol.get(symbol) is not None and latest_by_symbol[symbol] >= as_of_date
    ]
    stale_symbols = [
        symbol for symbol in supported_symbols if latest_by_symbol.get(symbol) is not None and latest_by_symbol[symbol] < as_of_date
    ]
    no_history_symbols = [
        symbol for symbol in supported_symbols if symbol not in latest_by_symbol
    ]

    if not stale_symbols:
        if no_history_symbols:
            print(
                "[static-daily prices] No reusable cached price history found; "
                "relying on the batch-only feature snapshot path for full history fetch.",
                flush=True,
            )
        else:
            print(
                f"[static-daily prices] Database already has fresh price rows for "
                f"{len(db_fresh_symbols):,} supported symbols as of {as_of_date}.",
                flush=True,
            )
        return {
            "status": "skipped",
            "as_of_date": as_of_date.isoformat(),
            "total_active_symbols": len(active_symbols),
            "supported_symbols": len(supported_symbols),
            "db_fresh_symbols": len(db_fresh_symbols),
            "stale_symbols": len(stale_symbols),
            "no_history_symbols": len(no_history_symbols),
            "skipped_unsupported_symbols": len(skipped_symbols),
            "yahoo_fetched_symbols": 0,
            "yahoo_failed_symbols": 0,
        }

    refreshed = 0
    failed = 0
    total = len(stale_symbols)
    total_batches = (total + STATIC_DAILY_PRICE_REFRESH_BATCH_SIZE - 1) // STATIC_DAILY_PRICE_REFRESH_BATCH_SIZE

    print(
        f"[static-daily prices] Refreshing {total:,} stale symbols in {total_batches} batches "
        f"for {as_of_date} (DB fresh: {len(db_fresh_symbols):,}, unsupported skipped: {len(skipped_symbols):,}).",
        flush=True,
    )

    for batch_index, batch_symbols in enumerate(
        _iter_chunks(stale_symbols, STATIC_DAILY_PRICE_REFRESH_BATCH_SIZE),
        start=1,
    ):
        processed_before = refreshed + failed
        print(
            f"[static-daily prices] Batch {batch_index}/{total_batches}: "
            f"{processed_before:,}/{total:,} processed, fetching {len(batch_symbols):,} symbols from Yahoo.",
            flush=True,
        )
        batch_results = fetcher.fetch_prices_in_batches(
            batch_symbols,
            period=STATIC_DAILY_PRICE_REFRESH_PERIOD,
        )
        batch_to_store: dict[str, Any] = {}
        for symbol, payload in batch_results.items():
            price_data = payload.get("price_data")
            if not payload.get("has_error") and price_data is not None and not price_data.empty:
                batch_to_store[symbol] = price_data
                refreshed += 1
            else:
                failed += 1
        if batch_to_store:
            price_cache.store_batch_in_cache(batch_to_store, also_store_db=True)
        print(
            f"[static-daily prices] Batch {batch_index}/{total_batches} complete: "
            f"{refreshed + failed:,}/{total:,} processed, {refreshed:,} refreshed, {failed:,} failed.",
            flush=True,
        )

    return {
        "status": "completed",
        "as_of_date": as_of_date.isoformat(),
        "total_active_symbols": len(active_symbols),
        "supported_symbols": len(supported_symbols),
        "db_fresh_symbols": len(db_fresh_symbols),
        "stale_symbols": len(stale_symbols),
        "no_history_symbols": len(no_history_symbols),
        "skipped_unsupported_symbols": len(skipped_symbols),
        "yahoo_fetched_symbols": refreshed,
        "yahoo_failed_symbols": failed,
    }


def _generate_trading_dates(start_date: date, end_date: date) -> list[date]:
    trading_dates: list[date] = []
    current = start_date
    while current <= end_date:
        if is_trading_day(current):
            trading_dates.append(current)
        current += timedelta(days=1)
    return trading_dates


def _ensure_group_rank_history(*, as_of_date: date) -> dict[str, Any]:
    """Backfill recent group-rank history so 1W/1M/3M deltas can be rendered."""
    start_date = as_of_date - timedelta(days=STATIC_GROUP_HISTORY_LOOKBACK_DAYS)
    desired_dates = _generate_trading_dates(start_date, as_of_date)

    with SessionLocal() as db:
        existing_dates = {
            record_date
            for record_date, in db.query(IBDGroupRank.date)
            .filter(IBDGroupRank.date >= start_date, IBDGroupRank.date <= as_of_date)
            .distinct()
            .all()
        }
        missing_dates = [calc_date for calc_date in desired_dates if calc_date not in existing_dates]
        if not missing_dates:
            print(
                f"[static-groups] Existing rankings already cover {len(desired_dates):,} trading dates "
                f"through {as_of_date}.",
                flush=True,
            )
            return {
                "status": "skipped",
                "as_of_date": as_of_date.isoformat(),
                "lookback_start_date": start_date.isoformat(),
                "missing_dates": 0,
                "processed": 0,
                "errors": 0,
            }

        print(
            f"[static-groups] Backfilling {len(missing_dates):,} missing trading dates "
            f"from {missing_dates[0]} to {missing_dates[-1]} before publishing {as_of_date}.",
            flush=True,
        )
        stats = get_group_rank_service().fill_gaps_optimized(db, missing_dates)
        stats.update(
            {
                "status": "completed",
                "as_of_date": as_of_date.isoformat(),
                "lookback_start_date": start_date.isoformat(),
                "missing_dates": len(missing_dates),
            }
        )
        return stats


def _ensure_breadth_history(
    *,
    as_of_date: date,
    min_trading_days: int = STATIC_BREADTH_HISTORY_MIN_TRADING_DAYS,
) -> dict[str, Any]:
    """Backfill recent breadth history so static snapshots include multi-day context."""
    start_date = as_of_date - timedelta(days=STATIC_BREADTH_HISTORY_LOOKBACK_DAYS)
    desired_dates = _generate_trading_dates(start_date, as_of_date)
    target_dates = desired_dates[-min_trading_days:] if min_trading_days > 0 else desired_dates
    if not target_dates:
        return {
            "status": "skipped",
            "as_of_date": as_of_date.isoformat(),
            "lookback_start_date": start_date.isoformat(),
            "target_trading_days": 0,
            "recomputed_dates": 0,
        }

    with SessionLocal() as db:
        existing_dates = {
            record_date
            for record_date, in db.query(MarketBreadth.date)
            .filter(MarketBreadth.date >= target_dates[0], MarketBreadth.date <= as_of_date)
            .all()
        }
        missing_dates = [calc_date for calc_date in target_dates if calc_date not in existing_dates]
        recompute_dates = sorted(set(missing_dates + [as_of_date]))

        if len(recompute_dates) == 1 and as_of_date in existing_dates:
            print(
                f"[static-breadth] Existing breadth history already covers the last "
                f"{len(target_dates)} trading days through {as_of_date}.",
                flush=True,
            )
            return {
                "status": "skipped",
                "as_of_date": as_of_date.isoformat(),
                "lookback_start_date": start_date.isoformat(),
                "target_trading_days": len(target_dates),
                "missing_dates": 0,
                "recomputed_dates": 0,
            }

        print(
            f"[static-breadth] Recomputing {len(recompute_dates)} dates "
            f"({len(missing_dates)} missing) to ensure {len(target_dates)} trading-day history "
            f"through {as_of_date}.",
            flush=True,
        )
        stats = BreadthCalculatorService(db, get_price_cache()).backfill_range(
            start_date=recompute_dates[0],
            end_date=recompute_dates[-1],
            trading_dates=recompute_dates,
            cache_only=True,
        )
        stats.update(
            {
                "status": "completed",
                "as_of_date": as_of_date.isoformat(),
                "lookback_start_date": start_date.isoformat(),
                "target_trading_days": len(target_dates),
                "missing_dates": len(missing_dates),
                "recomputed_dates": len(recompute_dates),
            }
        )
        return stats


def _run_daily_refresh(
    *,
    skip_universe_refresh: bool = False,
    skip_fundamentals_refresh: bool = False,
    build_mode: Literal["price_delta", "full"] = STATIC_BUILD_MODE_PRICE_DELTA,
    hydrate_published_snapshot: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    from app.interfaces.tasks.feature_store_tasks import (
        build_daily_snapshot,
    )
    from app.tasks.fundamentals_tasks import refresh_all_fundamentals
    from app.tasks.universe_tasks import refresh_stock_universe

    warnings: list[str] = []
    as_of_date = _resolve_latest_completed_us_trading_date()

    def _snapshot_ready(snapshot: dict[str, Any]) -> bool:
        status = snapshot.get("status")
        if status == "published":
            return True
        if status == "skipped" and snapshot.get("reason") == "already_published":
            return True
        if status is None and (snapshot.get("run_id") is not None or snapshot.get("existing_run_id") is not None):
            return True
        return False

    with disable_serialized_data_fetch_lock():
        results: dict[str, Any] = {}
        if not skip_universe_refresh:
            results["universe_refresh"] = refresh_stock_universe.run()

        if not skip_fundamentals_refresh:
            results["fundamentals_refresh"] = refresh_all_fundamentals.run()

        if build_mode == STATIC_BUILD_MODE_FULL and hydrate_published_snapshot:
            provider_snapshot_service = get_provider_snapshot_service()
            with SessionLocal() as db:
                results["fundamentals_hydrate"] = provider_snapshot_service.hydrate_all_published_snapshots(
                    db,
                    allow_yahoo_hydration=False,
                )

        with SessionLocal() as db:
            results["ibd_seed_refresh"] = {
                "csv_path": str(_tracked_ibd_csv_path()),
                "loaded": IBDIndustryService.load_from_csv(db, csv_path=_tracked_ibd_csv_path()),
            }

        results["price_refresh"] = _refresh_static_daily_prices(as_of_date=as_of_date)
        feature_snapshots: dict[str, Any] = {}
        for market in STATIC_EXPORT_MARKETS:
            market_result = build_daily_snapshot.run(
                as_of_date_str=as_of_date.isoformat(),
                static_daily_mode=True,
                universe_name=f"market:{market.lower()}",
                market=market,
                publish_pointer_key=_market_pointer_key(market),
                ignore_runtime_market_gate=True,
            )
            feature_snapshots[market] = market_result

        results["feature_snapshots"] = feature_snapshots
        for market, snapshot in feature_snapshots.items():
            if market == STATIC_DEFAULT_MARKET:
                continue
            if _snapshot_ready(snapshot):
                continue
            status = snapshot.get("status")
            reason = snapshot.get("reason")
            message = f"Static export market {market} snapshot returned status {status!r}"
            if reason:
                message += f" ({reason})."
            else:
                message += "."
            warnings.append(message)

        default_snapshot = feature_snapshots.get(STATIC_DEFAULT_MARKET, {})
        default_snapshot_status = default_snapshot.get("status")
        default_snapshot_ready = _snapshot_ready(default_snapshot)
        default_run_id = (
            default_snapshot.get("run_id")
            or default_snapshot.get("existing_run_id")
        )
        if default_snapshot_ready and default_run_id is not None:
            _upsert_feature_run_pointer(
                pointer_key="latest_published",
                run_id=default_run_id,
            )
            results["default_market_pointer"] = {
                "market": STATIC_DEFAULT_MARKET,
                "pointer_key": "latest_published",
                "run_id": default_run_id,
            }
        elif default_run_id is not None:
            warnings.append(
                f"{STATIC_DEFAULT_MARKET} feature snapshot returned status "
                f"{default_snapshot_status!r}; 'latest_published' was not updated."
            )
        else:
            warnings.append(
                f"No {STATIC_DEFAULT_MARKET} feature snapshot produced a run id; "
                "'latest_published' was not updated."
            )

    return results, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(_default_output_dir()),
        help="Directory to receive the generated static JSON bundle.",
    )
    parser.add_argument(
        "--refresh-daily",
        action="store_true",
        help="Run the synchronous daily refresh/build steps before exporting.",
    )
    parser.add_argument(
        "--build-mode",
        choices=(STATIC_BUILD_MODE_PRICE_DELTA, STATIC_BUILD_MODE_FULL),
        default=STATIC_BUILD_MODE_PRICE_DELTA,
        help="Refresh mode to use before static export. price_delta is the optimized default.",
    )
    parser.add_argument(
        "--skip-universe-refresh",
        action="store_true",
        help="Do not refresh the live stock universe before exporting.",
    )
    parser.add_argument(
        "--skip-fundamentals-refresh",
        action="store_true",
        help="Do not run the live weekly fundamentals refresh before exporting.",
    )
    parser.add_argument(
        "--hydrate-published-snapshot",
        action="store_true",
        help="Hydrate stock_fundamentals from the currently imported published snapshot.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete the output directory before exporting.",
    )
    args = parser.parse_args()

    prepare_runtime()

    refresh_warnings: list[str] = []
    if args.refresh_daily:
        refresh_results, refresh_warnings = _run_daily_refresh(
            skip_universe_refresh=args.skip_universe_refresh,
            skip_fundamentals_refresh=args.skip_fundamentals_refresh,
            build_mode=args.build_mode,
            hydrate_published_snapshot=args.hydrate_published_snapshot,
        )
        print("Daily refresh complete:")
        for name, result in refresh_results.items():
            print(f"  - {name}: {result}")
        for warning in refresh_warnings:
            print(f"  - warning: {warning}")

    service = StaticSiteExportService(SessionLocal)
    result = service.export(Path(args.output_dir), clean=not args.no_clean)

    print("Static site export complete:")
    print(f"  - output_dir: {result.output_dir}")
    print(f"  - generated_at: {result.generated_at}")
    print(f"  - as_of_date: {result.as_of_date}")
    for warning in (*refresh_warnings, *result.warnings):
        print(f"  - warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
