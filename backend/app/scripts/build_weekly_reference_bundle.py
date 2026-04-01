"""Build the weekly fundamentals reference bundle for static-site workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.database import SessionLocal
from app.scripts._runtime import prepare_runtime, repo_root
from app.services.provider_snapshot_service import provider_snapshot_service
from app.services.stock_universe_service import stock_universe_service


def _default_output_dir() -> Path:
    return repo_root() / ".tmp" / "weekly-reference"


def _default_bundle_name(published_run) -> str:
    as_of = (published_run.published_at or published_run.created_at).date().isoformat().replace("-", "")
    revision = (published_run.source_revision or "snapshot").replace(":", "-").replace("/", "-")
    return f"weekly-reference-{as_of}-{revision}.json.gz"


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
        default=provider_snapshot_service.WEEKLY_REFERENCE_LATEST_MANIFEST_NAME,
        help="Filename for the latest-pointer manifest JSON.",
    )
    args = parser.parse_args()

    prepare_runtime()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as db:
        universe_stats = stock_universe_service.populate_universe(db)
        snapshot_stats = provider_snapshot_service.create_snapshot_run(
            db,
            run_mode="publish",
            publish=True,
        )
        if not snapshot_stats.get("published"):
            raise RuntimeError(
                "Weekly fundamentals snapshot did not publish: "
                f"{snapshot_stats.get('warnings') or 'coverage gate blocked publish'}"
            )

        hydrate_stats = provider_snapshot_service.hydrate_published_snapshot(
            db,
            allow_yahoo_hydration=True,
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
