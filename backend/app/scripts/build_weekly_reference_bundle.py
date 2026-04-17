"""Build market-scoped weekly fundamentals reference bundles for static-site workflows."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
from typing import Any

from app.database import SessionLocal
from app.models.stock_universe import StockUniverse
from app.scripts._runtime import prepare_runtime, repo_root
from app.services.official_market_universe_source_service import (
    OfficialMarketUniverseSourceService,
)
from app.services.provider_snapshot_service import ProviderSnapshotService
from app.wiring.bootstrap import (
    get_fundamentals_cache,
    get_hybrid_fundamentals_service,
    get_provider_snapshot_service,
    get_stock_universe_service,
)


def _default_output_dir() -> Path:
    return repo_root() / ".tmp" / "weekly-reference"


def _default_bundle_name(market: str, published_run) -> str:
    as_of = (published_run.published_at or published_run.created_at).date().isoformat().replace("-", "")
    revision = (published_run.source_revision or "snapshot").replace(":", "-").replace("/", "-")
    return f"weekly-reference-{market.lower()}-{as_of}-{revision}.json.gz"


def _default_latest_manifest_name(market: str) -> str:
    return ProviderSnapshotService.weekly_reference_latest_manifest_name_for_market(market)


def _print_progress(event: dict[str, object]) -> None:
    stage = event.get("stage")
    if stage == "snapshot_fetch_complete":
        print(
            "[snapshot] "
            f"{event['completed_fetches']}/{event['total_fetches']} "
            f"({event['percent_complete']}%) "
            f"{event['exchange']} {event['category']} rows={event['rows']}",
            flush=True,
        )
        return

    if stage == "hydrate_start":
        print(
            "[hydrate] "
            f"starting {event['total_symbols']} symbols in {event['total_chunks']} chunks "
            f"(chunk_size={event['chunk_size']})",
            flush=True,
        )
        return

    if stage == "hydrate_chunk_complete":
        print(
            "[hydrate] "
            f"chunk {event['chunk_index']}/{event['total_chunks']} "
            f"processed {event['processed_symbols']}/{event['total_symbols']} "
            f"({event['percent_complete']}%) "
            f"live_price={event['live_price_symbols']} "
            f"cached_only={event['cached_only_symbols']} "
            f"yahoo_hydrated={event['yahoo_hydrated']} "
            f"missing_prices={event['missing_prices']} "
            f"missing_yahoo={event['missing_yahoo']} "
            f"skipped_yahoo_price={event['skipped_yahoo_price_symbols']} "
            f"skipped_yahoo_fields={event['skipped_yahoo_field_symbols']}",
            flush=True,
        )


def _print_snapshot_publish_summary(snapshot_stats: dict[str, Any]) -> None:
    thresholds = snapshot_stats.get("coverage_thresholds") or {}
    coverage = snapshot_stats.get("coverage") or {}
    if not thresholds or not coverage:
        return
    print(
        "[publish] "
        f"market={thresholds.get('market')} "
        f"coverage={thresholds.get('active_coverage', 0.0):.2%} "
        f"(min={thresholds.get('min_active_coverage', 0.0):.2%}) "
        f"missing_ratio={thresholds.get('missing_ratio', 0.0):.2%} "
        f"(max={thresholds.get('max_missing_ratio', 0.0):.2%}) "
        f"snapshot_rows={coverage.get('snapshot_symbols', 0)} "
        f"active_symbols={coverage.get('active_symbols', 0)}",
        flush=True,
    )


def _write_step_summary(market: str, summary: dict[str, Any]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    snapshot_stats = summary.get("snapshot_publish") or {}
    thresholds = snapshot_stats.get("coverage_thresholds") or {}
    coverage = snapshot_stats.get("coverage") or {}
    fundamentals_stats = summary.get("fundamentals_refresh") or {}
    export_stats = summary.get("export") or {}
    provider_error_counts = fundamentals_stats.get("provider_error_counts") or {}

    lines = [
        f"## Weekly Reference Bundle: {market}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Coverage gate market | {thresholds.get('market', market)} |",
        f"| Active coverage | {thresholds.get('active_coverage', 0.0):.2%} |",
        f"| Minimum coverage | {thresholds.get('min_active_coverage', 0.0):.2%} |",
        f"| Missing ratio | {thresholds.get('missing_ratio', 0.0):.2%} |",
        f"| Maximum missing ratio | {thresholds.get('max_missing_ratio', 0.0):.2%} |",
        f"| Snapshot rows | {coverage.get('snapshot_symbols', 0)} |",
        f"| Active symbols | {coverage.get('active_symbols', 0)} |",
        f"| Persisted symbols | {fundamentals_stats.get('persisted_symbols', 'n/a')} |",
        f"| Failed persistence symbols | {fundamentals_stats.get('failed_persistence_symbols', 0)} |",
        f"| Failed fetch/store symbols | {fundamentals_stats.get('failed', 0)} |",
        f"| Bundle rows exported | {export_stats.get('rows', 0)} |",
    ]
    if provider_error_counts:
        lines.extend(
            [
                "",
                "| Provider error bucket | Count |",
                "| --- | --- |",
            ]
        )
        for key, value in sorted(provider_error_counts.items()):
            lines.append(f"| `{key}` | {value} |")
    lines.extend(["", ""])

    with Path(summary_path).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _raise_publish_blocked(
    *,
    market: str,
    summary: dict[str, Any],
    snapshot_stats: dict[str, Any],
) -> None:
    _write_step_summary(market, summary)
    raise RuntimeError(
        "Weekly fundamentals snapshot did not publish: "
        f"{snapshot_stats.get('warnings') or 'coverage gate blocked publish'}"
    )


def _ingest_official_market_snapshot(db, stock_universe_service, snapshot) -> dict[str, Any]:
    if snapshot.market == "HK":
        return stock_universe_service.ingest_hk_snapshot_rows(
            db,
            rows=snapshot.rows,
            source_name=snapshot.source_name,
            snapshot_id=snapshot.snapshot_id,
            snapshot_as_of=snapshot.snapshot_as_of,
            source_metadata=snapshot.source_metadata,
        )
    if snapshot.market == "JP":
        return stock_universe_service.ingest_jp_snapshot_rows(
            db,
            rows=snapshot.rows,
            source_name=snapshot.source_name,
            snapshot_id=snapshot.snapshot_id,
            snapshot_as_of=snapshot.snapshot_as_of,
            source_metadata=snapshot.source_metadata,
        )
    if snapshot.market == "TW":
        return stock_universe_service.ingest_tw_snapshot_rows(
            db,
            rows=snapshot.rows,
            source_name=snapshot.source_name,
            snapshot_id=snapshot.snapshot_id,
            snapshot_as_of=snapshot.snapshot_as_of,
            source_metadata=snapshot.source_metadata,
        )
    raise ValueError(f"Unsupported official weekly reference market {snapshot.market!r}")


def _build_us_bundle(
    db,
    *,
    provider_snapshot_service,
    stock_universe_service,
    market: str,
    output_dir: Path,
    bundle_name: str | None,
    latest_manifest_name: str,
) -> dict[str, Any]:
    snapshot_key = ProviderSnapshotService.snapshot_key_for_market(market)

    print("Starting stock universe refresh from Finviz...", flush=True)
    universe_stats = stock_universe_service.populate_universe(db)
    print(f"Universe refresh complete: {universe_stats}", flush=True)

    print("Starting published fundamentals snapshot build from Finviz...", flush=True)
    snapshot_stats = provider_snapshot_service.create_snapshot_run(
        db,
        run_mode="publish",
        snapshot_key=snapshot_key,
        market=market,
        publish=True,
        progress_callback=_print_progress,
        show_finviz_progress=True,
    )
    summary = {
        "output_dir": output_dir,
        "universe_refresh": universe_stats,
        "snapshot_publish": snapshot_stats,
    }
    if not snapshot_stats.get("published"):
        _raise_publish_blocked(
            market=market,
            summary=summary,
            snapshot_stats=snapshot_stats,
        )
    _print_snapshot_publish_summary(snapshot_stats)

    published_run = provider_snapshot_service.get_published_run(db, snapshot_key=snapshot_key)
    if published_run is None:
        raise RuntimeError("Published weekly fundamentals snapshot was not found after publish")

    resolved_bundle_name = bundle_name or _default_bundle_name(market, published_run)
    bundle_path = output_dir / resolved_bundle_name
    latest_manifest_path = output_dir / latest_manifest_name
    export_stats = provider_snapshot_service.export_weekly_reference_bundle(
        db,
        output_path=bundle_path,
        bundle_asset_name=resolved_bundle_name,
        latest_manifest_path=latest_manifest_path,
        snapshot_key=snapshot_key,
        market=market,
    )

    summary.update(
        {
            "bundle": bundle_path,
            "latest_manifest": latest_manifest_path,
            "export": export_stats,
        }
    )
    return summary


def _build_asia_bundle(
    db,
    *,
    provider_snapshot_service,
    stock_universe_service,
    market: str,
    output_dir: Path,
    bundle_name: str | None,
    latest_manifest_name: str,
) -> dict[str, Any]:
    snapshot_key = ProviderSnapshotService.snapshot_key_for_market(market)
    official_source_service = OfficialMarketUniverseSourceService()
    fundamentals_cache = get_fundamentals_cache()
    hybrid_service = get_hybrid_fundamentals_service()

    print(f"Starting official universe refresh for {market}...", flush=True)
    official_snapshot = official_source_service.fetch_market_snapshot(market)
    universe_stats = _ingest_official_market_snapshot(db, stock_universe_service, official_snapshot)
    print(f"Universe refresh complete: {universe_stats}", flush=True)

    active_rows = (
        db.query(StockUniverse)
        .filter(
            StockUniverse.active_filter(),
            StockUniverse.market == market,
        )
        .order_by(StockUniverse.symbol.asc())
        .all()
    )
    if not active_rows:
        raise RuntimeError(f"No active {market} universe rows found after official-source ingest")

    symbols = [row.symbol for row in active_rows]
    market_by_symbol = {row.symbol: market for row in active_rows}

    print(f"Starting hybrid fundamentals refresh for {market}...", flush=True)
    all_data = hybrid_service.fetch_fundamentals_batch(
        symbols,
        include_technicals=True,
        include_finviz=False,
        market_by_symbol=market_by_symbol,
    )
    fundamentals_stats = hybrid_service.store_all_caches(
        all_data,
        fundamentals_cache,
        session_factory=SessionLocal,
        include_quarterly=True,
        market_by_symbol=market_by_symbol,
    )
    print(f"Fundamentals refresh complete: {fundamentals_stats}", flush=True)

    cached_fundamentals = fundamentals_cache.get_many(symbols)
    snapshot_rows = []
    for row in active_rows:
        payload = dict(cached_fundamentals.get(row.symbol) or {})
        if not payload:
            continue
        payload.setdefault("company_name", row.name)
        payload.setdefault("sector", row.sector)
        payload.setdefault("industry", row.industry)
        payload.setdefault("market_cap", row.market_cap)
        snapshot_rows.append(
            provider_snapshot_service.build_market_snapshot_row(
                market=market,
                symbol=row.symbol,
                exchange=row.exchange,
                normalized_payload=payload,
                raw_payload=None,
            )
        )

    coverage_stats = {
        "active_symbols": len(symbols),
        "snapshot_symbols": len(snapshot_rows),
        "covered_active_symbols": len(snapshot_rows),
        "missing_active_symbols": max(len(symbols) - len(snapshot_rows), 0),
    }
    warnings: list[str] = []
    if fundamentals_stats.get("failed"):
        warnings.append(
            f"{fundamentals_stats['failed']} symbols failed during {market} hybrid fundamentals refresh"
        )
    if fundamentals_stats.get("failed_persistence_symbols"):
        warnings.append(
            f"{fundamentals_stats['failed_persistence_symbols']} symbols failed to persist during "
            f"{market} hybrid fundamentals refresh"
        )

    source_revision = f"{snapshot_key}:{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    snapshot_stats = provider_snapshot_service.publish_market_snapshot_run(
        db,
        snapshot_key=snapshot_key,
        market=market,
        source_revision=source_revision,
        rows=snapshot_rows,
        coverage_stats=coverage_stats,
        warnings=warnings,
    )
    summary = {
        "output_dir": output_dir,
        "universe_refresh": universe_stats,
        "fundamentals_refresh": fundamentals_stats,
        "snapshot_publish": snapshot_stats,
    }
    if not snapshot_stats.get("published"):
        _raise_publish_blocked(
            market=market,
            summary=summary,
            snapshot_stats=snapshot_stats,
        )
    _print_snapshot_publish_summary(snapshot_stats)

    published_run = provider_snapshot_service.get_published_run(db, snapshot_key=snapshot_key)
    if published_run is None:
        raise RuntimeError(f"Published weekly fundamentals snapshot for {market} was not found")

    resolved_bundle_name = bundle_name or _default_bundle_name(market, published_run)
    bundle_path = output_dir / resolved_bundle_name
    latest_manifest_path = output_dir / latest_manifest_name
    export_stats = provider_snapshot_service.export_weekly_reference_bundle(
        db,
        output_path=bundle_path,
        bundle_asset_name=resolved_bundle_name,
        latest_manifest_path=latest_manifest_path,
        snapshot_key=snapshot_key,
        market=market,
    )

    summary.update(
        {
            "bundle": bundle_path,
            "latest_manifest": latest_manifest_path,
            "export": export_stats,
        }
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market",
        required=True,
        choices=list(ProviderSnapshotService.SNAPSHOT_KEY_FUNDAMENTALS_BY_MARKET),
        help="Market code to build the weekly reference bundle for.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_default_output_dir()),
        help="Directory to receive the generated bundle and latest manifest.",
    )
    parser.add_argument(
        "--bundle-name",
        default=None,
        help="Bundle asset filename. Defaults to weekly-reference-<market>-<YYYYMMDD>-<revision>.json.gz",
    )
    parser.add_argument(
        "--latest-manifest-name",
        default=None,
        help="Filename for the latest-pointer manifest JSON. Defaults to the market-scoped name.",
    )
    args = parser.parse_args()

    prepare_runtime()
    provider_snapshot_service = get_provider_snapshot_service()
    stock_universe_service = get_stock_universe_service()

    market = args.market.upper()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_manifest_name = args.latest_manifest_name or _default_latest_manifest_name(market)

    with SessionLocal() as db:
        if market == "US":
            summary = _build_us_bundle(
                db,
                provider_snapshot_service=provider_snapshot_service,
                stock_universe_service=stock_universe_service,
                market=market,
                output_dir=output_dir,
                bundle_name=args.bundle_name,
                latest_manifest_name=latest_manifest_name,
            )
        else:
            summary = _build_asia_bundle(
                db,
                provider_snapshot_service=provider_snapshot_service,
                stock_universe_service=stock_universe_service,
                market=market,
                output_dir=output_dir,
                bundle_name=args.bundle_name,
                latest_manifest_name=latest_manifest_name,
            )

    _write_step_summary(market, summary)
    print(f"Weekly reference bundle complete for {market}:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
