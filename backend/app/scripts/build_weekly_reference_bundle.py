"""Build the weekly fundamentals reference bundle for static-site workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.database import SessionLocal
from app.scripts._runtime import prepare_runtime, repo_root
from app.services.provider_snapshot_service import ProviderSnapshotService
from app.wiring.bootstrap import get_provider_snapshot_service, get_stock_universe_service


def _default_output_dir() -> Path:
    return repo_root() / ".tmp" / "weekly-reference"


def _default_bundle_name(published_run) -> str:
    as_of = (published_run.published_at or published_run.created_at).date().isoformat().replace("-", "")
    revision = (published_run.source_revision or "snapshot").replace(":", "-").replace("/", "-")
    return f"weekly-reference-{as_of}-{revision}.json.gz"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(_default_output_dir()),
        help="Directory to receive the generated bundle and latest manifest.",
    )
    parser.add_argument(
        "--bundle-name",
        default=None,
        help="Bundle asset filename. Defaults to weekly-reference-<YYYYMMDD>.json.gz",
    )
    parser.add_argument(
        "--latest-manifest-name",
        default=ProviderSnapshotService.WEEKLY_REFERENCE_LATEST_MANIFEST_NAME,
        help="Filename for the latest-pointer manifest JSON.",
    )
    args = parser.parse_args()

    prepare_runtime()
    provider_snapshot_service = get_provider_snapshot_service()
    stock_universe_service = get_stock_universe_service()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as db:
        print("Starting stock universe refresh from Finviz...", flush=True)
        universe_stats = stock_universe_service.populate_universe(db)
        print(f"Universe refresh complete: {universe_stats}", flush=True)

        print("Starting published fundamentals snapshot build from Finviz...", flush=True)
        snapshot_stats = provider_snapshot_service.create_snapshot_run(
            db,
            run_mode="publish",
            publish=True,
            progress_callback=_print_progress,
            show_finviz_progress=True,
        )
        if not snapshot_stats.get("published"):
            raise RuntimeError(
                "Weekly fundamentals snapshot did not publish: "
                f"{snapshot_stats.get('warnings') or 'coverage gate blocked publish'}"
            )

        print("Starting snapshot hydration (batched price enrichment + Yahoo-only fallback)...", flush=True)
        hydrate_stats = provider_snapshot_service.hydrate_published_snapshot(
            db,
            allow_yahoo_hydration=True,
            progress_callback=_print_progress,
        )
        published_run = provider_snapshot_service.get_published_run(db)
        if published_run is None:
            raise RuntimeError("Published weekly fundamentals snapshot was not found after hydration")

        bundle_name = args.bundle_name or _default_bundle_name(published_run)
        bundle_path = output_dir / bundle_name
        latest_manifest_path = output_dir / args.latest_manifest_name
        export_stats = provider_snapshot_service.export_weekly_reference_bundle(
            db,
            output_path=bundle_path,
            bundle_asset_name=bundle_name,
            latest_manifest_path=latest_manifest_path,
        )

    print("Weekly reference bundle complete:")
    print(f"  - output_dir: {output_dir}")
    print(f"  - bundle: {bundle_path}")
    print(f"  - latest_manifest: {latest_manifest_path}")
    print(f"  - universe_refresh: {universe_stats}")
    print(f"  - snapshot_publish: {snapshot_stats}")
    print(f"  - snapshot_hydrate: {hydrate_stats}")
    print(f"  - export: {export_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
