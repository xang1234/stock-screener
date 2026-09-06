"""Top-level static-site manifest and per-market metadata assembly."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from app.services.static_market_artifact_contract import (
    STATIC_MARKET_METADATA_FILENAME,
    STATIC_SITE_SCHEMA_VERSION,
)


def coerce_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def static_market_metadata_path(market: str) -> Path:
    return Path("markets") / market.lower() / STATIC_MARKET_METADATA_FILENAME


def write_static_market_metadata(
    *,
    output_dir: Path,
    generated_at: str,
    market: str,
    entry: dict[str, Any],
    warnings: list[str],
    json_writer: Callable[[Path, dict[str, Any]], None],
) -> None:
    json_writer(
        output_dir / static_market_metadata_path(market),
        {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "market": market,
            "entry": entry,
            "warnings": list(warnings),
        },
    )


def build_static_site_manifest(
    *,
    market_entries: dict[str, dict[str, Any]],
    generated_at: str,
    warnings: list[str],
    supported_markets: tuple[str, ...],
    default_market: str,
) -> dict[str, Any]:
    if not market_entries:
        raise RuntimeError(
            "No market artifacts are available to build a static-site manifest"
        )
    ordered_markets = [
        market for market in supported_markets if market in market_entries
    ]
    ordered_entries = {market: market_entries[market] for market in ordered_markets}
    selected_default = (
        default_market
        if default_market in ordered_entries
        else next(iter(ordered_entries))
    )
    default_entry = ordered_entries[selected_default]
    return {
        "schema_version": STATIC_SITE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "as_of_date": default_entry["as_of_date"],
        "default_market": selected_default,
        "supported_markets": ordered_markets,
        "features": dict(default_entry["features"]),
        "pages": dict(default_entry["pages"]),
        "assets": dict(default_entry["assets"]),
        "markets": ordered_entries,
        "warnings": list(warnings),
    }


__all__ = [
    "build_static_site_manifest",
    "coerce_datetime",
    "static_market_metadata_path",
    "write_static_market_metadata",
]
