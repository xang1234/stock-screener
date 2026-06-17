"""Bundle/manifest format + import for the IBD classification release artifact.

Mirrors the weekly-reference / daily-price convention: a gzipped JSON ``bundle``
plus a small ``latest`` manifest carrying the bundle filename and its sha256.
The artifact is published to the ``ibd-classification-data`` GitHub release and
consumed by the daily static-site build.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy.orm import Session

from ..config import settings
from ..models.industry import IBDIndustryGroup
from .github_release_sync_service import GitHubReleaseSyncService
from .ibd_industry_service import IBDIndustryService

# fetch_latest_bundle statuses that are not a hard failure: there was simply no
# bundle to import (live_only deployment, or already up to date).
NON_FATAL_SYNC_STATUSES = frozenset({"live_only", "up_to_date"})

# String schema identifiers, matching the weekly-reference convention. The
# GitHub release-sync service compares the manifest's schema_version against the
# expected value with ``!=``, so producer and consumer must use the same type;
# strings avoid the int-vs-str mismatch that would silently reject every import.
IBD_CLASSIFICATION_BUNDLE_SCHEMA_VERSION = "ibd-classification-bundle-v1"
IBD_CLASSIFICATION_MANIFEST_SCHEMA_VERSION = "ibd-classification-manifest-v1"

RELEASE_TAG = "ibd-classification-data"


def bundle_asset_name(market: str, as_of_compact: str, revision: str) -> str:
    rev = (revision or "snapshot").replace(":", "-").replace("/", "-")
    return f"ibd-classification-{market.lower()}-{as_of_compact}-{rev}.json.gz"


def latest_manifest_name(market: str) -> str:
    return f"ibd-classification-latest-{market.lower()}.json"


def write_bundle(output_path: Path, payload: dict) -> str:
    """Write the gzipped JSON bundle; return its sha256 hex digest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    # filename="" + mtime=0 keep the gzip header path- and time-independent, so
    # the same payload yields the same bytes (and sha256) regardless of output path.
    with output_path.open("wb") as fh:
        with gzip.GzipFile(fileobj=fh, filename="", mode="wb", mtime=0) as gz:
            gz.write(data)
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def read_bundle(input_path: Path) -> dict:
    with gzip.open(input_path, "rb") as gz:
        return json.loads(gz.read().decode("utf-8"))


def build_payload(
    *,
    market: str,
    as_of_date: str,
    source_revision: str,
    generated_at: str | None,
    model_id: str | None,
    assignments: Iterable[Any],
    summary: dict,
) -> dict:
    rows = [
        {
            "symbol": a.symbol,
            "market": a.market,
            "industry_group": a.industry_group,
            "source": a.source,
            "confidence": a.confidence,
            "method": a.method,
            "model_id": a.model_id,
        }
        for a in assignments
    ]
    return {
        "schema_version": IBD_CLASSIFICATION_BUNDLE_SCHEMA_VERSION,
        "market": market.upper(),
        "generated_at": generated_at,
        "as_of_date": as_of_date,
        "source_revision": source_revision,
        "generator": "ibd_classification",
        "model_id": model_id,
        "summary": summary,
        "classifications": rows,
    }


def build_manifest(*, payload: dict, bundle_name: str, sha256: str) -> dict:
    return {
        "schema_version": IBD_CLASSIFICATION_MANIFEST_SCHEMA_VERSION,
        "market": payload["market"],
        "generated_at": payload.get("generated_at"),
        "as_of_date": payload.get("as_of_date"),
        "source_revision": payload.get("source_revision"),
        "bundle_asset_name": bundle_name,
        "sha256": sha256,
        "summary": payload.get("summary"),
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def import_classifications(db: Session, payload: dict) -> dict:
    """Upsert a bundle's classifications into ``ibd_industry_groups``.

    Authoritative rows (``source in {csv, manual}``) are never overwritten — the
    curated CSV and human overrides win. Non-authoritative rows are updated in
    place; missing symbols are inserted.
    """
    if payload.get("schema_version") != IBD_CLASSIFICATION_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported IBD classification bundle schema: {payload.get('schema_version')}"
        )

    rows = payload.get("classifications", [])
    symbols = [r["symbol"] for r in rows]

    existing: dict[str, IBDIndustryGroup] = {}
    for start in range(0, len(symbols), 500):
        chunk = symbols[start:start + 500]
        for row in db.query(IBDIndustryGroup).filter(IBDIndustryGroup.symbol.in_(chunk)).all():
            existing[row.symbol] = row

    inserted = updated = skipped_authoritative = 0
    for r in rows:
        symbol = r["symbol"]
        current = existing.get(symbol)
        if current is not None and current.source in IBDIndustryService.AUTHORITATIVE_SOURCES:
            skipped_authoritative += 1
            continue
        if current is None:
            db.add(IBDIndustryGroup(
                symbol=symbol,
                industry_group=r["industry_group"],
                market=r.get("market") or payload.get("market") or "US",
                source=r.get("source") or "embedding",
                confidence=r.get("confidence"),
                method=r.get("method"),
                model_version=r.get("model_id"),
            ))
            inserted += 1
        else:
            current.industry_group = r["industry_group"]
            current.market = r.get("market") or current.market
            current.source = r.get("source") or current.source
            current.confidence = r.get("confidence")
            current.method = r.get("method")
            current.model_version = r.get("model_id")
            updated += 1

    db.commit()
    return {
        "market": payload.get("market"),
        "rows": len(rows),
        "inserted": inserted,
        "updated": updated,
        "skipped_authoritative": skipped_authoritative,
    }


def sync_ibd_classification_from_github(
    db: Session,
    *,
    market: str,
    allow_stale: bool = True,
    output_dir: str | None = None,
    github_sync_service: GitHubReleaseSyncService | None = None,
) -> dict:
    """Fetch the latest per-market IBD bundle from the release and import it.

    Mirrors ``DailyPriceBundleService.sync_from_github``: the caller owns ``db``.
    Returns ``{"market", "status", "imported"|None, "reason"|None}``. ``live_only``
    and ``up_to_date`` are non-fatal — there is simply no bundle to import (see
    ``NON_FATAL_SYNC_STATUSES``). Authoritative csv/manual rows are preserved by
    ``import_classifications``.
    """
    normalized_market = market.strip().upper()
    sync_service = github_sync_service or GitHubReleaseSyncService(
        api_base=settings.github_data_api_base
    )
    result = sync_service.fetch_latest_bundle(
        repository_full_name=settings.github_data_repository,
        release_tag=RELEASE_TAG,
        manifest_asset_name=latest_manifest_name(normalized_market),
        source_mode=settings.market_data_source_mode,
        expected_manifest_schema=IBD_CLASSIFICATION_MANIFEST_SCHEMA_VERSION,
        required_manifest_keys=("bundle_asset_name", "sha256"),
        allow_stale=allow_stale,
        github_token=settings.github_data_token,
        request_timeout_seconds=settings.github_data_timeout_seconds,
        output_dir=output_dir or str(Path(".tmp") / "ibd-classification"),
    )

    status = result.get("status")
    if status != "success":
        reason = result.get("reason") or result.get("stale_reason") or result.get("error")
        return {"market": normalized_market, "status": status, "imported": None, "reason": reason}

    # The downloaded bundle is transient: read it, import, then delete it so
    # long-lived live workers don't accumulate one gzip per market per run.
    # (DailyPriceBundleService uses a temp dir + rmtree; here output_dir is
    # shared/default, so we unlink just this run's file.)
    bundle_path = Path(result["bundle_path"])
    try:
        payload = read_bundle(bundle_path)
        # Guard the download/trust boundary: the manifest asset is market-specific,
        # so a market mismatch means a mispublished bundle. Mirror
        # DailyPriceBundleService.sync_from_github, which rejects the same case.
        payload_market = str(payload.get("market") or "").strip().upper()
        if payload_market and payload_market != normalized_market:
            return {
                "market": normalized_market,
                "status": "market_mismatch",
                "imported": None,
                "reason": f"bundle market {payload_market!r} does not match requested {normalized_market!r}",
            }

        stats = import_classifications(db, payload)
        return {"market": normalized_market, "status": status, "imported": stats, "reason": None}
    finally:
        bundle_path.unlink(missing_ok=True)
