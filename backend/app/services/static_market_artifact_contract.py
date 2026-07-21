"""Shared contract for static-site per-market artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATIC_SITE_SCHEMA_VERSION = "static-site-v3"
STATIC_MARKET_METADATA_FILENAME = "manifest.market.json"


class StaticMarketArtifactContractError(RuntimeError):
    """A static market artifact manifest violates the supported contract."""


def read_static_market_manifest(
    manifest_path: Path,
    *,
    expected_schema_version: str = STATIC_SITE_SCHEMA_VERSION,
    expected_market: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StaticMarketArtifactContractError(
            f"{manifest_path}: manifest.market.json must be an object."
        )
    schema_version = payload.get("schema_version")
    if schema_version != expected_schema_version:
        raise StaticMarketArtifactContractError(
            f"{manifest_path}: schema_version {schema_version!r}; "
            f"expected {expected_schema_version!r}."
        )
    market = str(payload.get("market") or "").strip().upper()
    if not market:
        raise StaticMarketArtifactContractError(f"{manifest_path}: market is required.")
    if expected_market is not None:
        normalized_expected = str(expected_market).strip().upper()
        if market != normalized_expected:
            raise StaticMarketArtifactContractError(
                f"{manifest_path}: market {market!r}; expected {normalized_expected!r}."
            )
    return payload


def market_from_static_market_artifact_name(artifact_name: str) -> str:
    prefix = "static-market-"
    if not artifact_name.startswith(prefix):
        return ""
    market = artifact_name[len(prefix):].upper()
    if "-" in market:
        return ""
    return market
