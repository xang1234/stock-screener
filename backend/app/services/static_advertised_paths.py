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


def _load_json(*, market: str, market_dir: Path, advertised: object) -> object:
    path = _resolve(market=market, market_dir=market_dir, advertised=advertised)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StaticAdvertisedPathError(
            f"advertised file {advertised!r} does not parse: {exc}"
        ) from exc


def validate_advertised_paths(*, market: str, entry: dict, market_dir: Path) -> None:
    """Every advertised page and asset, and every file they list, must parse.

    The scan manifest lists its chunks and the chart index its payloads; the
    browser fetches each, so a missing or truncated one is as broken as a
    missing page. A few hundred small files per market are cheap to parse.
    """
    for section in ("pages", "assets"):
        if section in entry and not isinstance(entry[section], dict):
            raise StaticAdvertisedPathError(f"{section} must be an object")
    pages = entry.get("pages") or {}
    assets = entry.get("assets") or {}
    listed_files = {
        id(pages.get("scan")): ("scan manifest", "chunks"),
        id(assets.get("charts")): ("chart index", "symbols"),
    }
    for section, descriptors in (("pages", pages), ("assets", assets)):
        for name, descriptor in descriptors.items():
            # A page the browser loads needs a path; an asset may advertise
            # another key (breadth_contributors uses index_path, checked by
            # its own validator).
            if not isinstance(descriptor, dict) or (
                section == "pages" and "path" not in descriptor
            ):
                raise StaticAdvertisedPathError(
                    f"{section}.{name} descriptor must be an object with a path"
                )
    for descriptor in (*pages.values(), *assets.values()):
        if "path" not in descriptor:
            continue
        if str(descriptor["path"]).strip() in ROOT_LEVEL_ADVERTISED_PATHS:
            continue
        payload = _load_json(
            market=market, market_dir=market_dir, advertised=descriptor["path"]
        )
        if id(descriptor) not in listed_files:
            continue
        label, key = listed_files[id(descriptor)]
        refs = payload.get(key, []) if isinstance(payload, dict) else None
        if not isinstance(refs, list) or not all(isinstance(ref, dict) for ref in refs):
            raise StaticAdvertisedPathError(
                f"{label} {descriptor['path']!r} must list {key} as objects"
            )
        for ref in refs:
            _load_json(market=market, market_dir=market_dir, advertised=ref.get("path"))
