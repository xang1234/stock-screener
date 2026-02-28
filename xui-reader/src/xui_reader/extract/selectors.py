"""Selector-pack defaults and override resolution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

SelectorPack = dict[str, tuple[str, ...]]

DEFAULT_SELECTOR_PACK: dict[str, tuple[str, ...]] = {
    "tweet.article": ('article[data-testid="tweet"]', "article"),
    "tweet.status_link": ('a[href*="/status/"]',),
    "tweet.text": ('div[data-testid="tweetText"]',),
    "tweet.author_handle": ('a[role="link"][href^="/"] span',),
    "tweet.time": ("time",),
    "tweet.reply_badge": ('div[data-testid="reply"]',),
    "tweet.repost_badge": ('div[data-testid="socialContext"]',),
    "tweet.pinned_badge": ('svg[aria-label*="Pinned"]',),
    "tweet.quote_container": ('article[data-testid="tweet"] article',),
}


@dataclass(frozen=True)
class SelectorPackResolution:
    selectors: SelectorPack
    warnings: tuple[str, ...] = ()
    loaded_override: bool = False


def default_selector_pack() -> SelectorPack:
    """Return a mutable copy of built-in selector defaults."""
    return {key: tuple(value) for key, value in DEFAULT_SELECTOR_PACK.items()}


def resolve_selector_pack(
    override_path: str | Path | None = None,
    *,
    override_data: Mapping[str, Any] | None = None,
) -> SelectorPackResolution:
    """Resolve effective selector pack with override-wins policy and safe fallback."""
    selectors = default_selector_pack()
    warnings: list[str] = []
    loaded_override = False

    if override_path is not None:
        payload = _load_override_payload(Path(override_path).expanduser(), warnings)
        if payload is not None:
            _merge_override(selectors, payload, warnings)
            loaded_override = True

    if override_data is not None:
        _merge_override(selectors, override_data, warnings)
        loaded_override = True

    return SelectorPackResolution(
        selectors=selectors,
        warnings=tuple(warnings),
        loaded_override=loaded_override,
    )


def _load_override_payload(path: Path, warnings: list[str]) -> Mapping[str, Any] | None:
    if not path.exists():
        warnings.append(f"Selector override file '{path}' was not found; using defaults.")
        return None
    if path.is_dir():
        warnings.append(f"Selector override path '{path}' is a directory; using defaults.")
        return None

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        warnings.append(f"Could not read selector override file '{path}': {exc}. Using defaults.")
        return None

    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            warnings.append(
                f"Selector override file '{path}' contains invalid JSON: {exc}. Using defaults."
            )
            return None
    elif suffix == ".toml":
        data = _load_toml_override(raw, path, warnings)
        if data is None:
            return None
    else:
        warnings.append(
            f"Unsupported selector override extension '{suffix or '<none>'}' for '{path}'. "
            "Supported extensions: .json, .toml. Using defaults."
        )
        return None

    if not isinstance(data, Mapping):
        warnings.append(
            f"Selector override file '{path}' must contain a mapping/table at top level; using defaults."
        )
        return None
    return data


def _load_toml_override(raw: str, path: Path, warnings: list[str]) -> Mapping[str, Any] | None:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            warnings.append(
                "TOML parsing unavailable for selector override (requires Python 3.11+ or tomli). "
                f"Could not parse '{path}'. Using defaults."
            )
            return None

    try:
        parsed = tomllib.loads(raw)
    except Exception as exc:
        warnings.append(f"Selector override file '{path}' contains invalid TOML: {exc}. Using defaults.")
        return None

    if not isinstance(parsed, Mapping):
        warnings.append(
            f"Selector override file '{path}' must parse to a table; using defaults."
        )
        return None
    return parsed


def _merge_override(
    selectors: SelectorPack,
    override_payload: Mapping[str, Any],
    warnings: list[str],
) -> None:
    payload: Mapping[str, Any]
    raw_selectors = override_payload.get("selectors")
    if raw_selectors is not None:
        if not isinstance(raw_selectors, Mapping):
            warnings.append(
                "Selector override key 'selectors' must be a mapping/table; ignoring override payload."
            )
            return
        payload = raw_selectors
    else:
        payload = override_payload

    flattened = _flatten_mapping(payload)
    allowed = set(DEFAULT_SELECTOR_PACK)
    for key in sorted(flattened):
        value = flattened[key]
        if key not in allowed:
            warnings.append(
                f"Unknown selector override key '{key}'. Allowed keys: {', '.join(sorted(allowed))}."
            )
            continue
        normalized = _normalize_selector_value(value)
        if normalized is None:
            warnings.append(
                f"Selector override for '{key}' must be a non-empty string or list of strings; ignoring."
            )
            continue
        selectors[key] = normalized


def _normalize_selector_value(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, str):
        trimmed = value.strip()
        return (trimmed,) if trimmed else None

    if isinstance(value, list | tuple):
        normalized: list[str] = []
        for entry in value:
            if not isinstance(entry, str):
                return None
            trimmed = entry.strip()
            if not trimmed:
                return None
            normalized.append(trimmed)
        return tuple(normalized) if normalized else None

    return None


def _flatten_mapping(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for raw_key, value in payload.items():
        key = str(raw_key)
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten_mapping(value, dotted))
            continue
        flattened[dotted] = value
    return flattened
