"""GitHub release policy for weekly-reference manifests and bundles."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings

from .github_release_sync_service import (
    GitHubReleaseSyncService,
    retry_github_operation,
)


WEEKLY_REFERENCE_MANIFEST_SCHEMA_VERSION = "weekly-reference-manifest-v1"
WEEKLY_REFERENCE_RELEASE_TAG = "weekly-reference-data"
WEEKLY_REFERENCE_LEGACY_MANIFEST_NAME = "weekly-reference-latest.json"
_COMMON_REQUIRED_KEYS = (
    "as_of_date",
    "source_revision",
    "bundle_asset_name",
    "sha256",
)


@dataclass(frozen=True)
class WeeklyReferenceManifestCandidate:
    asset_name: str
    required_keys: tuple[str, ...]


def weekly_reference_manifest_name(market: str) -> str:
    return f"weekly-reference-latest-{str(market or '').strip().lower()}.json"


def weekly_reference_manifest_candidates(
    market: str,
) -> tuple[WeeklyReferenceManifestCandidate, ...]:
    normalized_market = str(market or "").strip().upper()
    candidates = [
        WeeklyReferenceManifestCandidate(
            asset_name=weekly_reference_manifest_name(normalized_market),
            required_keys=("market", *_COMMON_REQUIRED_KEYS),
        )
    ]
    if normalized_market == "US":
        candidates.append(
            WeeklyReferenceManifestCandidate(
                asset_name=WEEKLY_REFERENCE_LEGACY_MANIFEST_NAME,
                required_keys=_COMMON_REQUIRED_KEYS,
            )
        )
    return tuple(candidates)


def fetch_weekly_reference_bundle(
    *,
    sync_service: GitHubReleaseSyncService,
    market: str,
    current_revision: str | None,
    output_dir: Path,
    stale_validator: Callable[[dict[str, Any]], Any],
    allow_stale: bool,
) -> dict[str, Any]:
    """Try the market manifest, then the compatible US legacy manifest."""
    result: dict[str, Any] = {"status": "missing_manifest"}
    for candidate in weekly_reference_manifest_candidates(market):
        result = sync_service.fetch_latest_bundle(
            repository_full_name=settings.github_data_repository,
            release_tag=(
                settings.github_weekly_reference_release_tag
                or WEEKLY_REFERENCE_RELEASE_TAG
            ),
            manifest_asset_name=candidate.asset_name,
            source_mode=settings.market_data_source_mode,
            current_revision=current_revision,
            expected_manifest_schema=WEEKLY_REFERENCE_MANIFEST_SCHEMA_VERSION,
            required_manifest_keys=candidate.required_keys,
            stale_validator=stale_validator,
            allow_stale=allow_stale,
            github_token=settings.github_data_token,
            request_timeout_seconds=settings.github_data_timeout_seconds,
            output_dir=output_dir,
        )
        if result.get("status") != "missing_manifest":
            return result
    return result


def retry_weekly_reference_sync(
    operation: Callable[[], dict[str, Any]],
    *,
    attempts: int = 3,
    retry_delay_seconds: float = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Retry transient weekly-reference release failures only."""
    return retry_github_operation(
        operation,
        should_retry=lambda result: result.get("status") == "network_error",
        attempts=attempts,
        retry_delay_seconds=retry_delay_seconds,
        sleep=sleep,
    )
