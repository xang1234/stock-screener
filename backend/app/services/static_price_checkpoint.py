"""Cross-run checkpoints of the static price stage (#502).

When the price stage reaches its deadline, the market's committed rows are
written as a daily-price bundle under checkpoint names, with a manifest naming
its kind and fingerprint. A re-run of the same session imports it in
checkpoint mode (rows replaced, completed daily-import state untouched); the
refresh's coverage check then finds the finished symbols fresh and fetches
only the rest. A checkpoint never becomes the latest daily bundle or a site
artifact: only a later complete export does.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.domain.providers.data_plan import DATASET_PRICES, provider_data_plan_registry
from app.models.provider_snapshot import ProviderSnapshotPointer, ProviderSnapshotRun
from app.services.daily_price_bundle_contract import (
    DAILY_PRICE_BAR_PERIOD,
    DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
    DAILY_PRICE_MANIFEST_SCHEMA_VERSION,
    DAILY_PRICE_RELEASE_TAG,
    REQUIRED_DAILY_PRICE_MANIFEST_KEYS,
    expected_bundle_metadata_from_manifest,
    normalize_daily_price_market,
)
from app.services.daily_price_bundle_service import DailyPriceBundleService
from app.services.github_release_sync_service import GitHubReleaseSyncService
from app.services.provider_snapshot_service import ProviderSnapshotService
from app.services.static_daily_price_refresh_service import (
    STATIC_ADJUSTMENT_DRIFT_TOLERANCE,
    STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
    STATIC_DAILY_PRICE_REFRESH_PERIOD,
)

CHECKPOINT_KIND = "price_checkpoint"
CHECKPOINT_FORMAT = "static-price-checkpoint-v1"


def checkpoint_asset_names(market: str) -> tuple[str, str]:
    """(bundle, manifest) names; disjoint from ``daily-price-*`` discovery."""
    lower = normalize_daily_price_market(market).lower()
    return f"price-checkpoint-{lower}.json.gz", f"price-checkpoint-{lower}.json"


def _weekly_reference_revision(db: Session, market: str) -> str | None:
    snapshot_key = ProviderSnapshotService.snapshot_key_for_market(market)
    run = (
        db.query(ProviderSnapshotRun.source_revision)
        .join(ProviderSnapshotPointer, ProviderSnapshotPointer.run_id == ProviderSnapshotRun.id)
        .filter(ProviderSnapshotPointer.snapshot_key == snapshot_key)
        .first()
    )
    return run.source_revision if run is not None else None


def checkpoint_fingerprint(db: Session, *, market: str, as_of_date: date) -> dict[str, Any]:
    """Every input a reused checkpoint row depends on; any change means refetch."""
    normalized = normalize_daily_price_market(market)
    baseline = DailyPriceBundleService().get_import_state(db, normalized) or {}
    fingerprint = {
        "format": CHECKPOINT_FORMAT,
        "market": normalized,
        "as_of_date": as_of_date.isoformat(),
        "bundle_schema_version": DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
        "bar_period": DAILY_PRICE_BAR_PERIOD,
        "baseline_daily_price_revision": baseline.get("source_revision"),
        "weekly_reference_revision": _weekly_reference_revision(db, normalized),
        "price_provider_plan": provider_data_plan_registry.plan_for(
            normalized, DATASET_PRICES
        ).provenance_metadata(),
        "adjustment_drift_tolerance": STATIC_ADJUSTMENT_DRIFT_TOLERANCE,
        "refresh_periods": [
            STATIC_DAILY_PRICE_REFRESH_PERIOD,
            STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
        ],
    }
    # Compare in the manifest's JSON form (tuples become lists).
    return json.loads(json.dumps(fingerprint))


def write_price_checkpoint(
    db: Session, *, market: str, as_of_date: date, output_dir: Path
) -> dict[str, Any]:
    """Write the checkpoint bundle and, last, its manifest into ``output_dir``."""
    bundle_name, manifest_name = checkpoint_asset_names(market)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / manifest_name
    DailyPriceBundleService().export_daily_price_bundle(
        db,
        market=market,
        output_path=output_dir / bundle_name,
        bundle_asset_name=bundle_name,
        latest_manifest_path=manifest_path,
        as_of_date=as_of_date,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["kind"] = CHECKPOINT_KIND
    manifest["fingerprint"] = checkpoint_fingerprint(
        db, market=market, as_of_date=as_of_date
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def resume_price_checkpoint(
    db: Session,
    *,
    market: str,
    as_of_date: date,
    sync_service: GitHubReleaseSyncService | None = None,
) -> dict[str, Any]:
    """Import the market's compatible checkpoint.

    Returns ``status``: ``imported``, ``missing``, ``incompatible`` (kind or
    fingerprint differs) or ``invalid`` (download, checksum or row validation
    failed; the import is rolled back). Only ``imported`` changes the database.
    """
    bundle_name, manifest_name = checkpoint_asset_names(market)
    expected = checkpoint_fingerprint(db, market=market, as_of_date=as_of_date)

    def incompatibility(manifest: dict[str, Any]) -> tuple[bool, str | None]:
        if manifest.get("kind") != CHECKPOINT_KIND:
            return True, f"{manifest_name} is not a price checkpoint manifest"
        found = manifest.get("fingerprint")
        found = found if isinstance(found, dict) else {}
        changed = sorted(
            key for key in expected.keys() | found.keys() if found.get(key) != expected.get(key)
        )
        if changed:
            return True, "checkpoint fingerprint differs: " + ", ".join(changed)
        return False, None

    sync = sync_service or GitHubReleaseSyncService(api_base=settings.github_data_api_base)
    download_dir = Path(tempfile.mkdtemp(prefix=f"{bundle_name}-"))
    try:
        result = sync.fetch_latest_bundle(
            repository_full_name=settings.github_data_repository,
            release_tag=settings.github_daily_price_release_tag or DAILY_PRICE_RELEASE_TAG,
            manifest_asset_name=manifest_name,
            expected_manifest_schema=DAILY_PRICE_MANIFEST_SCHEMA_VERSION,
            required_manifest_keys=REQUIRED_DAILY_PRICE_MANIFEST_KEYS,
            stale_validator=incompatibility,
            github_token=settings.github_data_token,
            request_timeout_seconds=settings.github_data_timeout_seconds,
            output_dir=download_dir,
        )
        status = result.get("status")
        reason = result.get("stale_reason") or result.get("reason") or result.get("error")
        if status == "missing_manifest":
            return {"status": "missing", "reason": reason}
        if status == "stale":
            return {"status": "incompatible", "reason": reason}
        manifest = result.get("manifest") or {}
        if status != "success" or manifest.get("bundle_asset_name") != bundle_name:
            return {"status": "invalid", "reason": reason or status}
        try:
            stats = DailyPriceBundleService().import_daily_price_bundle(
                db,
                input_path=Path(str(result["bundle_path"])),
                expected_metadata=expected_bundle_metadata_from_manifest(manifest),
                checkpoint=True,
            )
        except ValueError as exc:  # row/metadata validation; already rolled back
            return {"status": "invalid", "reason": str(exc)}
        return {"status": "imported", **stats}
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)
