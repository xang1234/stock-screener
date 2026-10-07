"""Check that a static market artifact contains every file it advertises."""

from __future__ import annotations

import json
from pathlib import Path

# Root-level files a market entry may advertise but that ship as their own
# artifacts (static-options-US) and are validated by their own selectors.
ROOT_LEVEL_ADVERTISED_PATHS = frozenset({"options/manifest.json"})


class StaticAdvertisedPathError(ValueError):
    """An advertised page, asset or chart payload is missing or unsafe."""


def _resolve(*, market: str, market_dir: Path, advertised: object) -> Path:
    text = str(advertised or "").strip()
    if not text:
        raise StaticAdvertisedPathError("advertises an empty path")
    relative = Path(text)
    if relative.parts[:1] == ("markets",):
        if relative.parts[1:2] != (market.lower(),):
            raise StaticAdvertisedPathError(
                f"advertised path belongs to another market: {text!r}"
            )
        relative = Path(*relative.parts[2:])
    # Paths without the markets/<m>/ prefix are relative to the market root
    # (older artifacts advertise them that way).
    root = market_dir.resolve()
    try:
        resolved = (root / relative).resolve()
    except (OSError, ValueError) as exc:
        raise StaticAdvertisedPathError(
            f"advertised path is invalid: {text!r} ({exc})"
        ) from exc
    if not resolved.is_relative_to(root):
        raise StaticAdvertisedPathError(
            f"advertised path escapes its artifact: {text!r}"
        )
    if not resolved.is_file():
        raise StaticAdvertisedPathError(f"advertised file is absent: {text!r}")
    return resolved


def validate_advertised_paths(*, market: str, entry: dict, market_dir: Path) -> None:
    """Every advertised page/asset must exist inside the artifact and parse.

    Chart payloads listed by the chart index are checked for presence only:
    there can be hundreds and the browser parses each one lazily.
    """
    pages = entry.get("pages") if isinstance(entry.get("pages"), dict) else {}
    assets = entry.get("assets") if isinstance(entry.get("assets"), dict) else {}
    for descriptor in (*pages.values(), *assets.values()):
        if not isinstance(descriptor, dict) or "path" not in descriptor:
            continue
        if str(descriptor["path"]).strip() in ROOT_LEVEL_ADVERTISED_PATHS:
            continue
        path = _resolve(market=market, market_dir=market_dir, advertised=descriptor["path"])
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StaticAdvertisedPathError(
                f"advertised file {descriptor['path']!r} does not parse: {exc}"
            ) from exc
        if descriptor is not assets.get("charts"):
            continue
        symbols = payload.get("symbols") if isinstance(payload, dict) else None
        if not isinstance(symbols, list) or not all(
            isinstance(symbol, dict) for symbol in symbols
        ):
            raise StaticAdvertisedPathError(
                f"chart index {descriptor['path']!r} must list symbol objects"
            )
        for symbol in symbols:
            _resolve(market=market, market_dir=market_dir, advertised=symbol.get("path"))
