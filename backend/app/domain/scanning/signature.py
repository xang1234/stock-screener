"""Helpers for exact scan-profile and universe signature matching."""

from __future__ import annotations

import hashlib
import json
from typing import Any


SCAN_SIGNATURE_VERSION = 1


def build_scan_signature_payload(
    *,
    universe_type: str,
    screeners: list[str] | None,
    composite_method: str | None,
    criteria: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the canonical payload used to hash exact scan inputs."""
    normalized_universe_type = getattr(universe_type, "value", universe_type)
    normalized_screeners = sorted({
        str(screener).strip()
        for screener in (screeners or [])
        if str(screener).strip()
    })
    return {
        "signature_version": SCAN_SIGNATURE_VERSION,
        "universe_type": str(normalized_universe_type).strip().lower(),
        "screeners": normalized_screeners,
        "composite_method": (composite_method or "weighted_average").strip().lower(),
        "criteria": _normalize_json(criteria or {}),
    }


def hash_scan_signature(payload: dict[str, Any]) -> str:
    """Hash a canonical scan payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_universe_symbols(symbols: list[str] | tuple[str, ...]) -> str:
    """Hash a resolved universe membership independent of input order."""
    normalized = sorted({
        str(symbol).strip().upper()
        for symbol in symbols
        if str(symbol).strip()
    })
    canonical = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_json(value: Any) -> Any:
    """Recursively normalize dict keys and JSON-like containers."""
    if isinstance(value, dict):
        return {
            str(key): _normalize_json(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    return value
