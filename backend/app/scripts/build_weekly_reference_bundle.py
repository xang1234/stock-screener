"""Build market-scoped weekly fundamentals reference bundles for static-site workflows."""

from __future__ import annotations

import argparse
import json
import math
import time
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

_WEEKLY_NON_US_YFINANCE_DELAY_PER_TICKER = 0.2
_DEFAULT_FETCH_CHUNK_SIZE = 250


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


def _configure_weekly_hybrid_service(market: str, hybrid_service: Any) -> None:
    if market == "US" or not hasattr(hybrid_service, "yfinance_delay_per_ticker"):
        return
    hybrid_service.yfinance_delay_per_ticker = _WEEKLY_NON_US_YFINANCE_DELAY_PER_TICKER
    print(
        "Using weekly non-US yfinance per-ticker delay "
        f"{_WEEKLY_NON_US_YFINANCE_DELAY_PER_TICKER:.2f}s for {market}",
        flush=True,
    )


def _print_fundamentals_progress(market: str, completed: float, total: int) -> None:
    total_count = max(int(total), 1)
    completed_count = min(int(completed), total_count)
    percent = (completed_count / total_count) * 100
    print(
        f"[fundamentals] {market} {completed_count}/{total_count} ({percent:.1f}%)",
        flush=True,
    )


_FUNDAMENTALS_STATS_NUMERIC_KEYS = (
    "fundamentals_stored",
    "quarterly_stored",
    "ownership_updated",
    "failed",
    "persisted_symbols",
    "failed_persistence_symbols",
)


def _empty_fundamentals_stats() -> dict[str, Any]:
    stats: dict[str, Any] = {key: 0 for key in _FUNDAMENTALS_STATS_NUMERIC_KEYS}
    stats["provider_error_counts"] = {}
    return stats


def _merge_fundamentals_stats(
    accumulator: dict[str, Any],
    chunk_stats: dict[str, Any] | None,
) -> None:
    if not chunk_stats:
        return
    for key in _FUNDAMENTALS_STATS_NUMERIC_KEYS:
        accumulator[key] = int(accumulator.get(key, 0) or 0) + int(
            chunk_stats.get(key, 0) or 0
        )
    chunk_errors = chunk_stats.get("provider_error_counts") or {}
    if chunk_errors:
        bucket = accumulator.setdefault("provider_error_counts", {})
        for key, value in chunk_errors.items():
            bucket[key] = int(bucket.get(key, 0) or 0) + int(value or 0)


def _as_nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(parsed, 0)


def _published_run_is_incomplete_partial_seed(
    provider_snapshot_service: Any, db, snapshot_key: str
) -> bool:
    published_run = provider_snapshot_service.get_published_run(db, snapshot_key=snapshot_key)
    if published_run is None or not published_run.coverage_stats_json:
        return False
    try:
        coverage = json.loads(published_run.coverage_stats_json)
    except (TypeError, ValueError):
        return False
    if not coverage.get("partial_run"):
        return False

    missing_active_symbols = _as_nonnegative_int(coverage.get("missing_active_symbols"))
    if missing_active_symbols is not None:
        return missing_active_symbols > 0

    active_symbols = _as_nonnegative_int(coverage.get("active_symbols"))
    snapshot_symbols = _as_nonnegative_int(coverage.get("snapshot_symbols"))
    if active_symbols is not None and snapshot_symbols is not None:
        return snapshot_symbols < active_symbols

    covered_active_symbols = _as_nonnegative_int(coverage.get("covered_active_symbols"))
    if active_symbols is not None and covered_active_symbols is not None:
        return covered_active_symbols < active_symbols

    return False


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
    if snapshot.market == "IN":
        return stock_universe_service.ingest_in_snapshot_rows(
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
    if snapshot.market == "KR":
        return stock_universe_service.ingest_kr_snapshot_rows(
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
    if snapshot.market == "CN":
        return stock_universe_service.ingest_cn_snapshot_rows(
            db,
            rows=snapshot.rows,
            source_name=snapshot.source_name,
            snapshot_id=snapshot.snapshot_id,
            snapshot_as_of=snapshot.snapshot_as_of,
            source_metadata=snapshot.source_metadata,
        )
    if snapshot.market == "CA":
        return stock_universe_service.ingest_ca_snapshot_rows(
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

    # Yahoo hydration writes market_cap / growth metrics / eps_rating / ipo_date
    # into stock_fundamentals so `export_weekly_reference_bundle` can merge them
    # into the Finviz snapshot. Without this, the US bundle carries only what
    # Finviz's screener returned, which regularly omits market_cap for delisted
    # tickers and partial responses. The Asia bundle path gets this implicitly
    # via `hybrid_service.fetch_fundamentals_batch`.
    print("Starting Yahoo hydration for US published snapshot...", flush=True)
    hydrate_stats = provider_snapshot_service.hydrate_published_snapshot(
        db,
        snapshot_key=snapshot_key,
        progress_callback=_print_progress,
    )
    summary["fundamentals_hydrate"] = hydrate_stats
    print(f"Hydration complete: {hydrate_stats}", flush=True)

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


def _run_chunked_fundamentals_refresh(
    *,
    hybrid_service: Any,
    fundamentals_cache: Any,
    market: str,
    symbols: list[str],
    market_by_symbol: dict[str, str],
    chunk_size: int,
    max_runtime_seconds: float,
) -> tuple[dict[str, Any], list[str], bool]:
    """Fetch + persist fundamentals in chunks, honouring an optional wall-clock budget.

    Returns ``(stats, attempted_symbols, deadline_hit)``. When the budget is
    exhausted, the loop exits cleanly between chunks so anything already
    persisted survives in the cache and DB. Skipped symbols inherit whatever
    data was previously hydrated from the prior weekly bundle.
    """
    stats = _empty_fundamentals_stats()
    attempted_symbols: list[str] = []
    deadline_hit = False
    if not symbols:
        return stats, attempted_symbols, deadline_hit

    chunk_size = max(1, int(chunk_size))
    total = len(symbols)
    chunks_total = math.ceil(total / chunk_size)
    deadline = (
        time.monotonic() + float(max_runtime_seconds)
        if max_runtime_seconds and max_runtime_seconds > 0
        else None
    )

    for chunk_index in range(chunks_total):
        if deadline is not None and time.monotonic() >= deadline:
            deadline_hit = True
            print(
                f"[fundamentals] {market} deadline reached before chunk "
                f"{chunk_index + 1}/{chunks_total}; stopping with "
                f"{len(attempted_symbols)}/{total} symbols attempted",
                flush=True,
            )
            break

        chunk = symbols[chunk_index * chunk_size : (chunk_index + 1) * chunk_size]
        if not chunk:
            break
        chunk_market_by_symbol = {s: market_by_symbol[s] for s in chunk if s in market_by_symbol}
        attempted_so_far = len(attempted_symbols)

        def _chunk_progress_cb(completed: float, _chunk_total: int) -> None:
            _print_fundamentals_progress(market, attempted_so_far + int(completed), total)

        chunk_data = hybrid_service.fetch_fundamentals_batch(
            chunk,
            include_technicals=True,
            include_finviz=False,
            progress_callback=_chunk_progress_cb,
            market_by_symbol=chunk_market_by_symbol,
        )
        chunk_stats = hybrid_service.store_all_caches(
            chunk_data,
            fundamentals_cache,
            session_factory=SessionLocal,
            include_quarterly=True,
            market_by_symbol=chunk_market_by_symbol,
        )
        _merge_fundamentals_stats(stats, chunk_stats)
        attempted_symbols.extend(chunk)
        print(
            f"[fundamentals] {market} chunk {chunk_index + 1}/{chunks_total} "
            f"persisted={(chunk_stats or {}).get('persisted_symbols', 0)} "
            f"failed={(chunk_stats or {}).get('failed', 0)}",
            flush=True,
        )

    return stats, attempted_symbols, deadline_hit


def _build_asia_bundle(
    db,
    *,
    provider_snapshot_service,
    stock_universe_service,
    market: str,
    output_dir: Path,
    bundle_name: str | None,
    latest_manifest_name: str,
    max_runtime_seconds: float = 0.0,
    fetch_chunk_size: int = _DEFAULT_FETCH_CHUNK_SIZE,
    allow_partial_publish: bool = False,
    resume_partial_seed: bool = False,
) -> dict[str, Any]:
    snapshot_key = ProviderSnapshotService.snapshot_key_for_market(market)
    official_source_service = OfficialMarketUniverseSourceService()
    fundamentals_cache = get_fundamentals_cache()
    hybrid_service = get_hybrid_fundamentals_service()
    _configure_weekly_hybrid_service(market, hybrid_service)

    print(f"Starting official universe refresh for {market}...", flush=True)
    stale_universe = False
    universe_error: str | None = None
    try:
        official_snapshot = official_source_service.fetch_market_snapshot(market)
        universe_stats = _ingest_official_market_snapshot(db, stock_universe_service, official_snapshot)
        print(f"Universe refresh complete: {universe_stats}", flush=True)
    except Exception as exc:
        if not allow_partial_publish:
            raise
        # _ingest_official_market_snapshot may have raised mid-transaction
        # (after bulk_save_objects but before commit), leaving the session in
        # a doomed state. Roll back before the seeded-rows query so we don't
        # mask the original error with a PendingRollbackError. The rollback
        # also reverts any partial bulk-insert, so the seeded-count below
        # reflects only rows that pre-existed this run.
        try:
            db.rollback()
        except Exception:  # pragma: no cover - defensive; rollback failures fall through
            pass
        # Assumption: the ``Seed prior weekly reference bundle`` step in
        # weekly-reference-data.yml is the only writer of CN rows in the
        # CI Postgres before this script runs, so any active rows present
        # here came from ``import_weekly_reference_bundle``. The rollback
        # above guarantees no partially-ingested rows survive. Long-running
        # deployments without a fresh Postgres should not rely on this
        # rescue path without first verifying that prior rows reflect the
        # intended baseline.
        seeded_count = (
            db.query(StockUniverse)
            .filter(
                StockUniverse.active_filter(),
                StockUniverse.market == market,
            )
            .count()
        )
        if seeded_count == 0:
            raise
        stale_universe = True
        universe_error = str(exc)
        universe_stats = {
            "stale_universe": True,
            "error": universe_error,
            "fallback_rows": seeded_count,
        }
        print(
            f"[universe] {market} official fetch failed ({universe_error}); "
            f"falling back to {seeded_count} seeded prior-week rows",
            flush=True,
        )

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
    seeded_symbols: list[str] = []
    if resume_partial_seed and _published_run_is_incomplete_partial_seed(
        provider_snapshot_service, db, snapshot_key
    ):
        seeded_payloads = fundamentals_cache.get_many(symbols)
        seeded_symbols = [symbol for symbol in symbols if seeded_payloads.get(symbol)]
        if seeded_symbols:
            print(
                f"[fundamentals] {market} resuming partial seed: "
                f"skipping {len(seeded_symbols)} cached symbols and fetching "
                f"{len(symbols) - len(seeded_symbols)} remaining symbols",
                flush=True,
            )
    seeded_symbol_set = set(seeded_symbols)
    fetch_symbols = [symbol for symbol in symbols if symbol not in seeded_symbol_set]

    print(f"Starting hybrid fundamentals refresh for {market}...", flush=True)
    fundamentals_stats, attempted_symbols, deadline_hit = _run_chunked_fundamentals_refresh(
        hybrid_service=hybrid_service,
        fundamentals_cache=fundamentals_cache,
        market=market,
        symbols=fetch_symbols,
        market_by_symbol=market_by_symbol,
        chunk_size=fetch_chunk_size,
        max_runtime_seconds=max_runtime_seconds,
    )
    attempted_symbol_set = set(attempted_symbols)
    skipped_symbols = [s for s in fetch_symbols if s not in attempted_symbol_set]
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
        "attempted_symbols": len(attempted_symbols),
        "seeded_symbols": len(seeded_symbols),
        "fetch_symbols": len(fetch_symbols),
        "skipped_due_to_deadline": len(skipped_symbols) if deadline_hit else 0,
        "partial_run": deadline_hit or stale_universe,
        "stale_universe": stale_universe,
    }
    warnings: list[str] = []
    if stale_universe:
        warnings.append(
            f"Official {market} universe fetch failed ({universe_error}); "
            f"reused {len(symbols)} seeded prior-week rows."
        )
    if fundamentals_stats.get("failed"):
        warnings.append(
            f"{fundamentals_stats['failed']} symbols failed during {market} hybrid fundamentals refresh"
        )
    if fundamentals_stats.get("failed_persistence_symbols"):
        warnings.append(
            f"{fundamentals_stats['failed_persistence_symbols']} symbols failed to persist during "
            f"{market} hybrid fundamentals refresh"
        )
    if deadline_hit:
        warnings.append(
            f"Weekly fetch deadline reached after {len(attempted_symbols)}/{len(symbols)} "
            f"{market} symbols ({len(seeded_symbols)} seeded); "
            f"{len(skipped_symbols)} symbols inherit prior-bundle data."
        )
        if not allow_partial_publish:
            warnings.append(
                "Partial publish disabled; blocking publish because the weekly fetch deadline was reached."
            )

    source_revision = f"{snapshot_key}:{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    publish = not (deadline_hit and not allow_partial_publish)
    snapshot_stats = provider_snapshot_service.publish_market_snapshot_run(
        db,
        snapshot_key=snapshot_key,
        market=market,
        source_revision=source_revision,
        rows=snapshot_rows,
        coverage_stats=coverage_stats,
        warnings=warnings,
        publish=publish,
        force_publish=bool(allow_partial_publish and (deadline_hit or stale_universe)),
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
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=0.0,
        help=(
            "Soft wall-clock budget for the fundamentals refresh phase. "
            "When > 0, the Asia path fetches in chunks and exits cleanly "
            "between chunks once the budget is exhausted. The deadline is "
            "checked before each chunk; leave at least one chunk's runtime "
            "of headroom under the GitHub Actions 6-hour job cap."
        ),
    )
    parser.add_argument(
        "--fetch-chunk-size",
        type=int,
        default=_DEFAULT_FETCH_CHUNK_SIZE,
        help=(
            "Number of symbols processed per chunk when --max-runtime-minutes "
            f"is set (default {_DEFAULT_FETCH_CHUNK_SIZE})."
        ),
    )
    parser.add_argument(
        "--allow-partial-publish",
        action="store_true",
        help=(
            "Allow partial publish in two scenarios: (a) --max-runtime-minutes "
            "triggers an early exit between fundamentals chunks, or (b) the "
            "official market-universe fetch fails and prior-week seeded "
            "universe rows are available. The snapshot is force-published "
            "even if the coverage gate would otherwise block. Skipped symbols "
            "inherit the prior weekly bundle's data; the manifest records "
            "partial_run=True with a deadline or stale_universe warning."
        ),
    )
    parser.add_argument(
        "--resume-partial-seed",
        action="store_true",
        help=(
            "When the seeded prior bundle is marked partial_run=True, skip symbols "
            "already present in the hydrated fundamentals cache and fetch only the "
            "remaining symbols. Intended for initial CN bootstrap continuation."
        ),
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
                max_runtime_seconds=max(0.0, float(args.max_runtime_minutes)) * 60.0,
                fetch_chunk_size=max(1, int(args.fetch_chunk_size)),
                allow_partial_publish=bool(args.allow_partial_publish),
                resume_partial_seed=bool(args.resume_partial_seed),
            )

    _write_step_summary(market, summary)
    print(f"Weekly reference bundle complete for {market}:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
