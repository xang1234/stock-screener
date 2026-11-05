"""Check that a static market artifact contains every file it advertises."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

from app.services.static_market_artifact_contract import (
    STATIC_SCOREBOARD_HORIZONS,
    STATIC_SCOREBOARD_SCHEMA_VERSION,
)

# Root-level files a market entry may advertise but that ship as their own
# artifacts (static-options-US) and are validated by their own selectors.
ROOT_LEVEL_ADVERTISED_PATHS = frozenset({"options/manifest.json"})

# Assets that advertise a key other than ``path`` and are validated by their
# own validator (breadth_contributors: index_path).
ASSETS_WITHOUT_PATH = frozenset({"breadth_contributors"})


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
    except (OSError, ValueError, RecursionError) as exc:
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
    # A broken scoreboard must not cost the market, so it has its own validator
    # and the combiner drops the entry rather than rejecting the artifact.
    assets = {
        name: descriptor
        for name, descriptor in (entry.get("assets") or {}).items()
        if name != "scoreboard"
    }
    listed_files = {
        id(pages.get("scan")): ("scan manifest", "chunks"),
        id(assets.get("charts")): ("chart index", "symbols"),
    }
    for section, descriptors in (("pages", pages), ("assets", assets)):
        for name, descriptor in descriptors.items():
            # Every descriptor the browser loads needs a path, except assets
            # that advertise another key and have their own validator.
            if not isinstance(descriptor, dict) or (
                "path" not in descriptor and name not in ASSETS_WITHOUT_PATH
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


Check = Callable[[object], bool]


def _is_text(value: object) -> bool:
    return isinstance(value, str)


def _is_flag(value: object) -> bool:
    return isinstance(value, bool)


def _is_count(value: object) -> bool:
    return type(value) is int and value >= 0


def _is_finite_number(value: object) -> bool:
    # Comparing, not math.isfinite: a huge JSON integer would overflow there, and only
    # StaticAdvertisedPathError keeps a bad scoreboard from costing the market.
    return type(value) in (int, float) and -sys.float_info.max <= value <= sys.float_info.max


def _or_null(check: Check) -> Check:
    return lambda value: value is None or check(value)


def _record(fields: dict[str, Check]) -> Check:
    return lambda value: isinstance(value, dict) and all(
        name in value and check(value[name]) for name, check in fields.items()
    )


def _list_of(check: Check) -> Check:
    return lambda value: isinstance(value, list) and all(check(item) for item in value)


_is_horizon = _record(
    {
        "horizon_sessions": _is_count,
        "sample_size": _is_count,
        "signal_dates": _is_count,
        "effective_dates": _is_finite_number,
        "pending_windows": _is_count,
        "oldest_open_sessions": _or_null(_is_count),
        "thin": _is_flag,
        "positive_rate": _or_null(_is_finite_number),
        "beat_benchmark_rate": _or_null(_is_finite_number),
        "median_return_pct": _or_null(_is_finite_number),
        "median_excess_return_pct": _or_null(_is_finite_number),
        "excess_sample_size": _is_count,
        "skipped_missing_history": _is_count,
        "suppressed_overlapping": _is_count,
    }
)


def _holds_each_horizon_once(value: object) -> bool:
    return _list_of(_is_horizon)(value) and sorted(
        block["horizon_sessions"] for block in value
    ) == sorted(STATIC_SCOREBOARD_HORIZONS)


# The fields the static page and the MCP tool screen_track_record read without a guard.
_is_scoreboard = _record(
    {
        "schema_version": lambda value: value == STATIC_SCOREBOARD_SCHEMA_VERSION,
        "market": _is_text,
        "as_of_date": _is_text,
        "benchmark": _or_null(_is_text),
        "recording_since": _or_null(_is_text),
        "scored_since": _or_null(_is_text),
        "ledger_through": _or_null(_is_text),
        "missing_sessions": _is_count,
        "cohorts": _list_of(
            _record(
                {
                    "preset": _is_text,
                    "name": _is_text,
                    "retired": _is_flag,
                    "definition": _is_text,
                    "origin": lambda value: value in ("live", "artifact", "artifact_pre_stage_fix"),
                    "first_date": _is_text,
                    "last_date": _is_text,
                    "events": _is_count,
                    "horizons": _holds_each_horizon_once,
                }
            )
        ),
    }
)


def validate_scoreboard_asset(*, market: str, market_dir: Path, descriptor: object) -> None:
    """Raise unless ``descriptor`` names a scoreboard file for ``market`` its readers can use."""
    if not isinstance(descriptor, dict) or "path" not in descriptor:
        raise StaticAdvertisedPathError("descriptor must be an object with a path")
    payload = _load_json(market=market, market_dir=market_dir, advertised=descriptor["path"])
    if not _is_scoreboard(payload) or payload["market"] != market:
        raise StaticAdvertisedPathError(
            f"advertised file {descriptor['path']!r} is not a "
            f"{STATIC_SCOREBOARD_SCHEMA_VERSION} file for {market}"
        )
