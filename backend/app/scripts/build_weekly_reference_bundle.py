"""Build market-scoped weekly fundamentals reference bundles for static-site workflows."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date, datetime
import os
from pathlib import Path
from typing import Any

import requests

from app.config import settings
from app.database import SessionLocal
from app.domain.providers.price_symbol_support import is_bse_scrip_code_yahoo_symbol
from app.services.asx_official_universe_source import is_au_securitisation_listing
from app.models.provider_snapshot import ProviderSnapshotRow
from app.models.stock_universe import StockUniverse
from app.scripts._runtime import prepare_runtime, repo_root
from app.services.official_market_universe_source_service import (
    OfficialMarketUniverseSourceService,
)
from app.services.official_universe_dispatch import ingest_official_market_snapshot
from app.services.finviz_screener_slices import FinvizReadError
from app.services.provider_snapshot_service import (
    PRIOR_SEED_ROW_SOURCE,
    SEEDED_CACHE_ROW_SOURCE,
    ProviderSnapshotService,
)
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


def _seed_source_note(coverage: dict[str, Any]) -> str | None:
    """How a snapshot rebuilt from the prior seed describes its data source."""
    if not coverage.get("seed_source_revision"):
        return None
    return (
        f"prior seed {coverage['seed_source_revision']} "
        f"(data as of {coverage.get('seed_as_of_date')}); Finviz snapshot failed"
    )


def _print_snapshot_publish_summary(snapshot_stats: dict[str, Any]) -> None:
    thresholds = snapshot_stats.get("coverage_thresholds") or {}
    coverage = snapshot_stats.get("coverage") or {}
    seed_note = _seed_source_note(coverage)
    if seed_note:
        # A GitHub annotation, so a green run built from old data stands out.
        market = thresholds.get("market") or snapshot_stats.get("market") or "US"
        print(f"::warning title=Weekly reference {market} reused prior seed::{seed_note}", flush=True)
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


def _rotate_to_resume_cursor(
    provider_snapshot_service: Any, db, snapshot_key: str, symbols: list[str]
) -> list[str]:
    """Start at the published run's ``fundamentals_resume_from`` and wrap around.

    A market whose full fetch outlasts the runtime budget (CN) otherwise
    restarts at the first symbol every week and never refreshes the tail.
    ``symbols`` is sorted, so a cursor that has since delisted still resumes
    at its successor.
    """
    published_run = provider_snapshot_service.get_published_run(db, snapshot_key=snapshot_key)
    try:
        cursor = json.loads(getattr(published_run, "coverage_stats_json", None) or "{}").get(
            "fundamentals_resume_from"
        )
    except (AttributeError, TypeError, ValueError):
        cursor = None
    if not isinstance(cursor, str) or not cursor:
        return symbols
    start = next((i for i, symbol in enumerate(symbols) if symbol >= cursor), 0)
    if start:
        print(f"[fundamentals] resuming at {symbols[start]} (prior run stopped there)", flush=True)
    return symbols[start:] + symbols[:start]


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
    seed_note = _seed_source_note(coverage)
    if seed_note:
        lines.append(f"| Data source | {seed_note} |")
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


def _snapshot_row_payload(row: ProviderSnapshotRow) -> dict[str, Any]:
    return {
        "symbol": row.symbol,
        "exchange": row.exchange,
        "row_hash": row.row_hash,
        "normalized_payload": json.loads(row.normalized_payload_json),
        "raw_payload": json.loads(row.raw_payload_json) if row.raw_payload_json else None,
    }


def _run_rows(db, run_id: int) -> list[ProviderSnapshotRow]:
    return db.query(ProviderSnapshotRow).filter(ProviderSnapshotRow.run_id == run_id).all()


def _prior_seed(
    db,
    *,
    provider_snapshot_service,
    snapshot_key: str,
    as_of_keys: tuple[str, ...] = ("seed_as_of_date",),
) -> tuple[Any, dict[str, Any], str | None]:
    """The imported prior bundle's run and provenance, or why it cannot stand in for this week.

    Its data must be within ``github_weekly_reference_max_age_days``, the age past
    which consumers already treat a weekly bundle as stale. ``as_of_keys`` name
    the coverage fields (first present wins) holding the date of the data, so a
    seed is dated by its data, e.g. its official listing date, not by when it
    was published.
    """
    seed_run = provider_snapshot_service.get_published_run(db, snapshot_key=snapshot_key)
    if seed_run is None:
        return None, {}, "No prior weekly reference seed is loaded to reuse."
    try:
        seed_coverage = json.loads(seed_run.coverage_stats_json or "{}")
    except (TypeError, ValueError):
        seed_coverage = {}
    if not isinstance(seed_coverage, dict):
        seed_coverage = {}
    data_as_of = next((seed_coverage[key] for key in as_of_keys if seed_coverage.get(key)), None)
    if (
        "universe_as_of_date" in as_of_keys
        and seed_coverage.get("stale_universe")
        and not data_as_of
    ):
        # Bundles from before #521 reused a universe without dating it.
        return None, {}, (
            f"Prior weekly reference seed {seed_run.source_revision} reused an undated universe."
        )
    # A seed that itself reused an older seed is as old as that seed's data.
    try:
        seed_as_of = date.fromisoformat(
            data_as_of
            or (seed_run.published_at or seed_run.created_at).date().isoformat()
        ).isoformat()
    except (AttributeError, TypeError, ValueError):
        return None, {}, f"Prior weekly reference seed {seed_run.source_revision} has no usable data date."
    # Compared in UTC, the session timezone in CI and Docker.
    age_days = (datetime.utcnow().date() - date.fromisoformat(seed_as_of)).days
    max_age_days = max(int(settings.github_weekly_reference_max_age_days or 0), 0)
    if max_age_days and age_days > max_age_days:
        return None, {}, (
            f"Prior weekly reference seed {seed_run.source_revision} is {age_days} day(s) old "
            f"(as of {seed_as_of}); max age {max_age_days} day(s)."
        )
    return seed_run, {
        "finviz_snapshot_failed": True,
        "seed_source_revision": seed_run.source_revision,
        "seed_as_of_date": seed_as_of,
    }, None


def _publish_us_seeded_cache_fallback(
    db,
    *,
    provider_snapshot_service,
    market: str,
    snapshot_key: str,
    blocked_snapshot_stats: dict[str, Any],
) -> dict[str, Any]:
    blocked_run_id = blocked_snapshot_stats.get("run_id")
    if not blocked_run_id and not blocked_snapshot_stats.get("finviz_snapshot_failed"):
        return blocked_snapshot_stats

    active_rows = (
        db.query(StockUniverse)
        .filter(
            StockUniverse.active_filter(),
            StockUniverse.market == market,
        )
        .order_by(StockUniverse.symbol.asc())
        .all()
    )
    active_symbols = {row.symbol for row in active_rows}
    current_rows = _run_rows(db, blocked_run_id) if blocked_run_id else []
    rows_by_symbol = {
        row.symbol: _snapshot_row_payload(row)
        for row in current_rows
        if row.symbol in active_symbols
    }
    missing_rows = [row for row in active_rows if row.symbol not in rows_by_symbol]
    if not missing_rows:
        return blocked_snapshot_stats

    seed_run = None
    seed_provenance: dict[str, Any] = {}
    backfill_provenance: dict[str, Any] = {}
    # ponytail: majority cutoff. Finviz failed (an HTTP error, or pages that stop
    # carrying screener rows) when it covers under half the active universe, so
    # most rows would come from the prior seed and it must be recent enough.
    # A normal partial week (e.g. 95% from Finviz) stays a dated-today backfill.
    if len(rows_by_symbol) * 2 < len(active_symbols):
        warnings = list(blocked_snapshot_stats.get("warnings") or [])
        if not blocked_snapshot_stats.get("finviz_snapshot_failed"):
            warnings.append(
                f"Finviz snapshot returned rows for only {len(rows_by_symbol)} of "
                f"{len(active_symbols)} active US symbols"
            )
        seed_run, seed_provenance, seed_problem = _prior_seed(
            db,
            provider_snapshot_service=provider_snapshot_service,
            snapshot_key=snapshot_key,
        )
        if seed_problem:
            return {**blocked_snapshot_stats, "warnings": [*warnings, seed_problem]}
        blocked_snapshot_stats = {**blocked_snapshot_stats, "warnings": warnings}
    else:
        # Finviz covered most symbols; the rest come from the cache the prior
        # seed hydrated, so that seed must be recent too. The bundle keeps its
        # own date (mostly fresh); the seed is recorded under backfill_* keys.
        _, prior, seed_problem = _prior_seed(
            db,
            provider_snapshot_service=provider_snapshot_service,
            snapshot_key=snapshot_key,
        )
        if seed_problem:
            return {
                **blocked_snapshot_stats,
                "warnings": [*(blocked_snapshot_stats.get("warnings") or []), seed_problem],
            }
        backfill_provenance = {
            "backfill_seed_source_revision": prior["seed_source_revision"],
            "backfill_seed_as_of_date": prior["seed_as_of_date"],
        }

    backfilled_symbols: list[str] = []
    seed_backfilled = 0
    if seed_run is not None:
        # Fill from the validated seed's own rows, so the recorded seed revision
        # and data date describe exactly what is republished.
        seed_rows = {row.symbol: row for row in _run_rows(db, seed_run.id)}
        for universe_row in missing_rows:
            seed_row = seed_rows.get(universe_row.symbol)
            if seed_row is None:
                continue
            rows_by_symbol[universe_row.symbol] = {
                **_snapshot_row_payload(seed_row),
                "raw_payload": {
                    "source": PRIOR_SEED_ROW_SOURCE,
                    "seed_source_revision": seed_run.source_revision,
                },
            }
            backfilled_symbols.append(universe_row.symbol)
        seed_backfilled = len(backfilled_symbols)
        # Symbols the seed lacks (e.g. new listings) can still come from the
        # cache below, labelled as cache rows.
        missing_rows = [row for row in missing_rows if row.symbol not in rows_by_symbol]

    cache_failure = None
    try:
        seeded_payloads = (
            get_fundamentals_cache().get_many([row.symbol for row in missing_rows])
            if missing_rows
            else {}
        )
    except Exception as exc:
        print(
            f"[publish] US seeded cache fallback unavailable: {exc}",
            flush=True,
        )
        if seed_run is None:
            return blocked_snapshot_stats
        # The validated seed rows still stand; the coverage gate decides.
        cache_failure = f"Seeded cache backfill failed: {type(exc).__name__}: {exc}"
        seeded_payloads = {}

    for universe_row in missing_rows:
        payload = dict(seeded_payloads.get(universe_row.symbol) or {})
        if not payload:
            continue
        payload.setdefault("company_name", universe_row.name)
        payload.setdefault("sector", universe_row.sector)
        payload.setdefault("industry", universe_row.industry)
        payload.setdefault("market_cap", universe_row.market_cap)
        payload.setdefault("symbol", universe_row.symbol)
        payload.setdefault("market", market)
        payload.setdefault("exchange", universe_row.exchange)
        payload.setdefault("currency", universe_row.currency)
        payload.setdefault("timezone", universe_row.timezone)
        payload.setdefault("local_code", universe_row.local_code)
        fallback_row = provider_snapshot_service.build_market_snapshot_row(
            market=market,
            symbol=universe_row.symbol,
            exchange=universe_row.exchange,
            normalized_payload=payload,
            raw_payload={"source": SEEDED_CACHE_ROW_SOURCE},
        )
        rows_by_symbol[universe_row.symbol] = fallback_row
        backfilled_symbols.append(universe_row.symbol)

    if not backfilled_symbols:
        return blocked_snapshot_stats

    missing_active = sorted(
        symbol for symbol in active_symbols if symbol not in rows_by_symbol
    )
    coverage_stats = {
        "active_symbols": len(active_symbols),
        "snapshot_symbols": len(rows_by_symbol),
        "covered_active_symbols": len(active_symbols.intersection(rows_by_symbol)),
        "missing_active_symbols": len(missing_active),
        "backfilled_active_symbols": len(backfilled_symbols),
        **seed_provenance,
        **backfill_provenance,
    }
    warnings = list(blocked_snapshot_stats.get("warnings") or [])
    if cache_failure:
        warnings.append(cache_failure)
    if seed_provenance:
        cache_backfilled = len(backfilled_symbols) - seed_backfilled
        warnings.append(
            f"Republished {seed_backfilled} US active symbols from the prior weekly "
            f"reference seed {seed_provenance['seed_source_revision']} "
            f"(data as of {seed_provenance['seed_as_of_date']}) because the Finviz snapshot failed"
            + (
                f"; backfilled {cache_backfilled} the seed lacked from the seeded cache"
                if cache_backfilled
                else ""
            )
        )
    else:
        warnings.append(
            "Backfilled "
            f"{len(backfilled_symbols)} US active symbols from seeded weekly reference cache "
            "because the current Finviz snapshot omitted them: "
            f"{', '.join(sorted(backfilled_symbols)[:25])}"
            + ("..." if len(backfilled_symbols) > 25 else "")
        )
    source_revision = (
        f"{snapshot_key}:{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-seeded-fallback"
    )

    return provider_snapshot_service.publish_market_snapshot_run(
        db,
        snapshot_key=snapshot_key,
        market=market,
        source_revision=source_revision,
        rows=[rows_by_symbol[symbol] for symbol in sorted(rows_by_symbol)],
        coverage_stats=coverage_stats,
        warnings=warnings,
        publish=True,
    )


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
    # The universe imported from the prior bundle has no reconciliation run in
    # this fresh database, so without a baseline the refresh could never
    # deactivate symbols Finviz has dropped; they would stay active every week
    # and erode snapshot coverage. Removals still pass the Finviz safety gates.
    baseline = stock_universe_service.seed_reconciliation_baseline_from_active_rows(
        db,
        market=market,
        source_name="finviz",
        snapshot_id=f"weekly-reference-seed:{datetime.utcnow():%Y%m%d%H%M%S}",
    )
    if baseline:
        print(
            f"Seeded Finviz reconciliation baseline: {baseline['baseline_rows']} active rows "
            f"({baseline['snapshot_id']})",
            flush=True,
        )
    universe_stats = stock_universe_service.populate_universe(db)
    print(f"Universe refresh complete: {universe_stats}", flush=True)

    print("Starting published fundamentals snapshot build from Finviz...", flush=True)
    try:
        snapshot_stats = provider_snapshot_service.create_snapshot_run(
            db,
            run_mode="publish",
            snapshot_key=snapshot_key,
            market=market,
            publish=True,
            progress_callback=_print_progress,
        )
    except (requests.RequestException, FinvizReadError) as exc:
        # create_snapshot_run rolled back its run; the seed fallback below decides.
        print(f"[publish] Finviz snapshot fetch failed: {exc}", flush=True)
        snapshot_stats = {
            "run_id": None,
            "finviz_snapshot_failed": True,
            "published": False,
            "warnings": [f"Finviz snapshot fetch failed: {type(exc).__name__}: {exc}"],
        }
    summary = {
        "output_dir": output_dir,
        "universe_refresh": universe_stats,
        "snapshot_publish": snapshot_stats,
    }
    if not snapshot_stats.get("published"):
        snapshot_stats = _publish_us_seeded_cache_fallback(
            db,
            provider_snapshot_service=provider_snapshot_service,
            market=market,
            snapshot_key=snapshot_key,
            blocked_snapshot_stats=snapshot_stats,
        )
        summary["snapshot_publish"] = snapshot_stats
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
    started_at: float | None = None,
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
    # Counted from script start (``started_at``), so setup and the universe
    # refresh spend the same budget the job timeout does.
    deadline = (
        (time.monotonic() if started_at is None else started_at) + float(max_runtime_seconds)
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
            # Inside the chunk too: one slow chunk overran CN's budget until
            # the runner cancelled the job before any partial publish (#522).
            deadline=deadline,
        )
        # The batch returns only the symbols it started before the deadline.
        chunk_attempted = [symbol for symbol in chunk if symbol in chunk_data]
        chunk_stats = hybrid_service.store_all_caches(
            chunk_data,
            fundamentals_cache,
            session_factory=SessionLocal,
            include_quarterly=True,
            market_by_symbol=chunk_market_by_symbol,
        )
        _merge_fundamentals_stats(stats, chunk_stats)
        attempted_symbols.extend(chunk_attempted)
        print(
            f"[fundamentals] {market} chunk {chunk_index + 1}/{chunks_total} "
            f"persisted={(chunk_stats or {}).get('persisted_symbols', 0)} "
            f"failed={(chunk_stats or {}).get('failed', 0)}",
            flush=True,
        )
        if len(chunk_attempted) < len(chunk):
            deadline_hit = True
            print(
                f"[fundamentals] {market} deadline reached inside chunk "
                f"{chunk_index + 1}/{chunks_total}; stopping with "
                f"{len(attempted_symbols)}/{total} symbols attempted",
                flush=True,
            )
            break

    return stats, attempted_symbols, deadline_hit


def _without_excluded_listings(market: str, rows: list[Any]) -> list[Any]:
    """Drop listings the official source now excludes but prior bundles seeded.

    Ingestion does not deactivate rows missing from a snapshot, so without
    this they would reach the bundle: IN BSE scrip codes (#480, IN is
    NSE-only) and AU securitisation trusts (#481, debt Yahoo never prices).
    """
    if market == "IN":
        excluded = lambda row: is_bse_scrip_code_yahoo_symbol(row.symbol)  # noqa: E731
    elif market == "AU":
        excluded = lambda row: is_au_securitisation_listing(row.name)  # noqa: E731
    else:
        return rows
    kept = [row for row in rows if not excluded(row)]
    if len(kept) != len(rows):
        print(f"[universe] {market} dropping {len(rows) - len(kept)} excluded listings", flush=True)
    return kept


def _seed_official_reconciliation_baseline(db, stock_universe_service, snapshot) -> None:
    """Let this refresh deactivate listings the official source has dropped.

    Same gap as the US build: the universe imported from the prior bundle has
    no reconciliation run, so nothing missing from the source is ever retired.
    The workflow enables destructive apply for this job only, and removals
    still pass the Asia safety gates. Fallback seed CSVs (``*_manual_csv``)
    are partial lists, so they never become the comparison; nor does an IN
    snapshot built from NSE alone while BSE is unreachable, or any snapshot
    with a failed fetch part (e.g. a TMX letter bucket) or board counts
    outside the validated KRX/CN baseline (e.g. a BaoStock CN fallback that
    omits BJSE).
    """
    if snapshot.source_name.endswith("_manual_csv"):
        return
    metadata = snapshot.source_metadata or {}
    if metadata.get("bse_unavailable"):
        return
    if any((metadata.get("fetch_errors") or {}).values()):
        return
    if any(value for key, value in metadata.items() if key.endswith("_baseline_breaches")):
        return
    market = snapshot.market.upper()
    baseline = stock_universe_service.seed_reconciliation_baseline_from_active_rows(
        db,
        market=market,
        source_name=snapshot.source_name,
        row_source=f"{market.lower()}_ingest",
        snapshot_id=f"weekly-reference-seed:{datetime.utcnow():%Y%m%d%H%M%S}",
    )
    if baseline:
        print(
            f"Seeded {snapshot.source_name} reconciliation baseline: "
            f"{baseline['baseline_rows']} active rows ({baseline['snapshot_id']})",
            flush=True,
        )


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
    allow_stale_universe: bool = False,
) -> dict[str, Any]:
    snapshot_key = ProviderSnapshotService.snapshot_key_for_market(market)
    official_source_service = OfficialMarketUniverseSourceService()
    fundamentals_cache = get_fundamentals_cache()
    hybrid_service = get_hybrid_fundamentals_service()
    _configure_weekly_hybrid_service(market, hybrid_service)

    print(f"Starting official universe refresh for {market}...", flush=True)
    stale_universe = False
    universe_error: str | None = None
    universe_seed: dict[str, Any] = {}
    universe_as_of: str | None = None
    try:
        official_snapshot = official_source_service.fetch_market_snapshot(market)
        universe_as_of = getattr(official_snapshot, "snapshot_as_of", None)
        _seed_official_reconciliation_baseline(db, stock_universe_service, official_snapshot)
        universe_stats = ingest_official_market_snapshot(
            db, stock_universe_service, official_snapshot
        )
        print(f"Universe refresh complete: {universe_stats}", flush=True)
    except Exception as exc:
        # --allow-stale-universe covers source outages only; a parser or
        # ingest defect must fail the job. --allow-partial-publish keeps its
        # broader historical scope.
        source_outage = isinstance(exc, requests.RequestException)
        if not (allow_partial_publish or (allow_stale_universe and source_outage)):
            raise
        # The shared official ingestion dispatch may have raised mid-transaction
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
            raise RuntimeError(
                f"Official {market} universe fetch failed ({exc}) and no prior-week "
                "universe is loaded to fall back on."
            ) from exc
        # Reused listings must be recent enough to pass as this week's, and
        # the bundle says how old they are.
        _, seed_provenance, seed_problem = _prior_seed(
            db,
            provider_snapshot_service=provider_snapshot_service,
            snapshot_key=snapshot_key,
            # universe_seed_as_of_date: bundles from this change's first revision.
            as_of_keys=("universe_as_of_date", "universe_seed_as_of_date"),
        )
        if seed_problem:
            raise RuntimeError(
                f"Official {market} universe fetch failed ({exc}); {seed_problem}"
            ) from exc
        stale_universe = True
        universe_error = str(exc)
        universe_as_of = seed_provenance["seed_as_of_date"]
        universe_seed = {
            "universe_error": universe_error,
            "universe_seed_source_revision": seed_provenance["seed_source_revision"],
            "universe_seed_as_of_date": seed_provenance["seed_as_of_date"],
        }
        universe_stats = {
            "stale_universe": True,
            "error": universe_error,
            "fallback_rows": seeded_count,
            **universe_seed,
        }
        print(
            f"[universe] {market} official fetch failed ({universe_error}); "
            f"falling back to {seeded_count} seeded rows from "
            f"{universe_seed['universe_seed_source_revision']} "
            f"(as of {universe_seed['universe_seed_as_of_date']})",
            flush=True,
        )
        # Surfaces as a run annotation so a reused universe is never silent.
        print(
            f"::warning title=Stale {market} universe::Official {market} universe fetch "
            f"failed; reused the universe as of {universe_seed['universe_seed_as_of_date']}.",
            flush=True,
        )

    db_active_rows = (
        db.query(StockUniverse)
        .filter(
            StockUniverse.active_filter(),
            StockUniverse.market == market,
        )
        .order_by(StockUniverse.symbol.asc())
        .all()
    )
    active_rows = _without_excluded_listings(market, db_active_rows)
    # The export re-queries active rows, so it must be told what was dropped.
    excluded_symbols = {row.symbol for row in db_active_rows} - {row.symbol for row in active_rows}
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
    fetch_symbols = _rotate_to_resume_cursor(
        provider_snapshot_service,
        db,
        snapshot_key,
        [symbol for symbol in symbols if symbol not in seeded_symbol_set],
    )

    print(f"Starting hybrid fundamentals refresh for {market}...", flush=True)
    fundamentals_stats, attempted_symbols, deadline_hit = _run_chunked_fundamentals_refresh(
        hybrid_service=hybrid_service,
        fundamentals_cache=fundamentals_cache,
        market=market,
        symbols=fetch_symbols,
        market_by_symbol=market_by_symbol,
        chunk_size=fetch_chunk_size,
        max_runtime_seconds=max_runtime_seconds,
        started_at=_SCRIPT_STARTED_AT,
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
        # Where next week's fetch starts, so the deadline rotates through the market.
        "fundamentals_resume_from": skipped_symbols[0] if deadline_hit and skipped_symbols else None,
        "partial_run": deadline_hit or stale_universe,
        "stale_universe": stale_universe,
        **universe_seed,
    }
    if universe_as_of:
        # The listings' own date, so the next week's fallback ages them correctly.
        coverage_stats["universe_as_of_date"] = universe_as_of
    warnings: list[str] = []
    if stale_universe:
        warnings.append(
            f"Official {market} universe fetch failed ({universe_error}); "
            f"reused {len(symbols)} seeded rows as of {universe_seed['universe_seed_as_of_date']}."
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
        excluded_symbols=excluded_symbols,
    )

    summary.update(
        {
            "bundle": bundle_path,
            "latest_manifest": latest_manifest_path,
            "export": export_stats,
        }
    )
    return summary


# Set by main(): the fundamentals deadline counts from here (#522).
_SCRIPT_STARTED_AT: float | None = None


def main() -> int:
    global _SCRIPT_STARTED_AT
    _SCRIPT_STARTED_AT = time.monotonic()
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
        "--allow-stale-universe",
        action="store_true",
        help=(
            "If the official market-universe fetch fails, reuse the imported "
            "prior bundle's universe when it is within "
            "github_weekly_reference_max_age_days. Unlike --allow-partial-publish "
            "this neither tolerates a fundamentals deadline nor bypasses the "
            "coverage gate; the manifest records stale_universe with the seed's "
            "revision, data date and the source error."
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
                allow_stale_universe=bool(args.allow_stale_universe),
            )

    _write_step_summary(market, summary)
    print(f"Weekly reference bundle complete for {market}:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
