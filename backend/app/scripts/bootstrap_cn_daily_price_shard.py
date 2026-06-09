"""Refresh and export one resumable CN daily price shard."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from app.database import SessionLocal
from app.models.stock_universe import StockUniverse
from app.scripts._runtime import prepare_runtime, repo_root
from app.services.bulk_data_fetcher import BulkDataFetcher
from app.wiring.bootstrap import get_daily_price_bundle_service


DEFAULT_BATCH_SIZE = 100
CheckpointPublisher = Callable[[Path, Path], None]


def _default_output_dir() -> Path:
    return repo_root() / ".tmp" / "cn-daily-price-shards"


def _iter_chunks(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[index:index + chunk_size] for index in range(0, len(items), chunk_size)]


def select_shard_symbols(
    symbols: list[str] | tuple[str, ...],
    *,
    shard_index: int,
    shard_count: int,
) -> list[str]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 1 or shard_index > shard_count:
        raise ValueError("shard_index must be between 1 and shard_count")
    ordered = sorted(
        {
            str(symbol or "").strip().upper()
            for symbol in symbols
            if str(symbol or "").strip()
        }
    )
    return [
        symbol for offset, symbol in enumerate(ordered)
        if offset % shard_count == shard_index - 1
    ]


def _load_active_cn_symbols(db: Any) -> list[str]:
    rows = (
        db.query(StockUniverse.symbol)
        .filter(
            StockUniverse.active_filter(),
            StockUniverse.market == "CN",
        )
        .order_by(StockUniverse.symbol.asc())
        .all()
    )
    return [symbol for symbol, in rows]


def _shard_asset_name(*, as_of_date: date, shard_index: int, shard_count: int) -> str:
    compact_date = as_of_date.isoformat().replace("-", "")
    return f"daily-price-cn-{compact_date}-shard-{shard_index}-of-{shard_count}.json.gz"


def _shard_manifest_name(*, as_of_date: date, shard_index: int, shard_count: int) -> str:
    return _shard_asset_name(
        as_of_date=as_of_date,
        shard_index=shard_index,
        shard_count=shard_count,
    ).removesuffix(".json.gz") + ".manifest.json"


def _export_shard_checkpoint(
    *,
    db: Any,
    service: Any,
    as_of_date: date,
    shard_index: int,
    shard_count: int,
    output_dir: Path,
    shard_symbols: list[str],
) -> tuple[dict[str, Any], Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = _shard_asset_name(
        as_of_date=as_of_date,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    manifest_name = _shard_manifest_name(
        as_of_date=as_of_date,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    bundle_path = output_dir / bundle_name
    manifest_path = output_dir / manifest_name
    export_stats = service.export_daily_price_bundle(
        db,
        market="CN",
        output_path=bundle_path,
        bundle_asset_name=bundle_name,
        latest_manifest_path=manifest_path,
        as_of_date=as_of_date,
        symbols=shard_symbols,
    )
    return export_stats, bundle_path, manifest_path


def _build_github_release_checkpoint_publisher(release_tag: str) -> CheckpointPublisher:
    def _publish(bundle_path: Path, manifest_path: Path) -> None:
        print(
            f"[cn-price-shard] publishing durable checkpoint {bundle_path.name} "
            f"and {manifest_path.name} to {release_tag}",
            flush=True,
        )
        for path in (bundle_path, manifest_path):
            subprocess.run(
                ["gh", "release", "upload", release_tag, str(path), "--clobber"],
                check=True,
            )

    return _publish


def _validate_cn_as_of_date(service: Any, as_of_date: date) -> None:
    latest_completed = service.market_calendar.last_completed_trading_day("CN")
    if as_of_date > latest_completed:
        raise ValueError(
            f"CN as-of date {as_of_date.isoformat()} is after the latest completed "
            f"trading day {latest_completed.isoformat()}"
        )
    if not service.market_calendar.is_trading_day("CN", as_of_date):
        raise ValueError(f"CN as-of date {as_of_date.isoformat()} is not a trading day")


def bootstrap_cn_daily_price_shard(
    *,
    db: Any,
    service: Any,
    fetcher: BulkDataFetcher,
    shard_index: int,
    shard_count: int,
    as_of_date: date,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    checkpoint_publisher: CheckpointPublisher | None = None,
) -> dict[str, Any]:
    active_symbols = _load_active_cn_symbols(db)
    shard_symbols = select_shard_symbols(
        active_symbols,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    if not shard_symbols:
        raise RuntimeError(f"CN shard {shard_index}/{shard_count} resolved to zero symbols")

    missing_symbols = service.symbols_missing_as_of(
        db,
        symbols=shard_symbols,
        as_of_date=as_of_date,
    )
    refreshed_symbols = 0
    failed_symbols = 0
    export_stats: dict[str, Any] | None = None

    total_batches = max(1, (len(missing_symbols) + batch_size - 1) // batch_size)
    print(
        f"[cn-price-shard] shard={shard_index}/{shard_count} "
        f"symbols={len(shard_symbols):,} missing={len(missing_symbols):,} "
        f"as_of={as_of_date.isoformat()}",
        flush=True,
    )

    for batch_number, batch_symbols in enumerate(
        _iter_chunks(missing_symbols, batch_size),
        start=1,
    ):
        print(
            f"[cn-price-shard] batch {batch_number}/{total_batches}: "
            f"fetching {len(batch_symbols):,} symbols",
            flush=True,
        )
        batch_results = fetcher.fetch_prices_in_batches(
            batch_symbols,
            period=service.DAILY_PRICE_BAR_PERIOD,
            market="CN",
        )
        batch_to_store = {}
        for symbol, payload in batch_results.items():
            price_data = payload.get("price_data")
            if not payload.get("has_error") and price_data is not None and not price_data.empty:
                batch_to_store[symbol] = price_data
                refreshed_symbols += 1
            else:
                failed_symbols += 1
        if batch_to_store:
            service.price_cache.store_batch_in_cache(batch_to_store, also_store_db=True)
        export_stats, bundle_path, manifest_path = _export_shard_checkpoint(
            db=db,
            service=service,
            as_of_date=as_of_date,
            shard_index=shard_index,
            shard_count=shard_count,
            output_dir=output_dir,
            shard_symbols=shard_symbols,
        )
        if checkpoint_publisher is not None:
            checkpoint_publisher(bundle_path, manifest_path)
        print(
            f"[cn-price-shard] batch {batch_number}/{total_batches} complete: "
            f"refreshed={refreshed_symbols:,} failed={failed_symbols:,}",
            flush=True,
        )

    if export_stats is None:
        export_stats, bundle_path, manifest_path = _export_shard_checkpoint(
            db=db,
            service=service,
            as_of_date=as_of_date,
            shard_index=shard_index,
            shard_count=shard_count,
            output_dir=output_dir,
            shard_symbols=shard_symbols,
        )
        if checkpoint_publisher is not None:
            checkpoint_publisher(bundle_path, manifest_path)

    stats = {
        "market": "CN",
        "as_of_date": as_of_date.isoformat(),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "active_symbols": len(active_symbols),
        "shard_symbols": len(shard_symbols),
        "missing_symbols": len(missing_symbols),
        "refreshed_symbols": refreshed_symbols,
        "failed_symbols": failed_symbols,
        **export_stats,
    }
    print("CN daily price shard complete:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-index",
        type=int,
        required=True,
        help="1-based shard index to refresh.",
    )
    parser.add_argument("--shard-count", type=int, required=True, help="Total number of CN shards.")
    parser.add_argument(
        "--output-dir",
        default=str(_default_output_dir()),
        help="Directory to receive the shard bundle and manifest.",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Bundle as-of date (YYYY-MM-DD). Defaults to CN's latest completed trading day.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of missing symbols to fetch before checkpointing to the database.",
    )
    parser.add_argument(
        "--checkpoint-release-tag",
        default=None,
        help="Upload the shard bundle and manifest to this GitHub release after each batch.",
    )
    args = parser.parse_args()

    prepare_runtime()
    service = get_daily_price_bundle_service()
    as_of_date = (
        date.fromisoformat(args.as_of_date)
        if args.as_of_date
        else service.market_calendar.last_completed_trading_day("CN")
    )
    _validate_cn_as_of_date(service, as_of_date)
    checkpoint_publisher = (
        _build_github_release_checkpoint_publisher(args.checkpoint_release_tag)
        if args.checkpoint_release_tag
        else None
    )

    with SessionLocal() as db:
        bootstrap_cn_daily_price_shard(
            db=db,
            service=service,
            fetcher=BulkDataFetcher(),
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            as_of_date=as_of_date,
            output_dir=Path(args.output_dir),
            batch_size=max(1, args.batch_size),
            checkpoint_publisher=checkpoint_publisher,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
