"""Fail-closed guard that keeps live Social data out of static artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.infra.serialization import json_safe


LIVE_ONLY_STATIC_KEY_FRAGMENTS = (
    "social",
    "x_post",
    "tweet",
    "source_metrics",
    "social_signal",
)


class StaticSocialIsolationError(ValueError):
    """Raised when live-only Social data reaches a static publication."""


def find_live_only_static_key(value: Any, *, location: str) -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            if any(
                fragment in normalized_key
                for fragment in LIVE_ONLY_STATIC_KEY_FRAGMENTS
            ):
                return f"{key} at {location}"
            violation = find_live_only_static_key(
                nested,
                location=f"{location}.{key}",
            )
            if violation is not None:
                return violation
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            violation = find_live_only_static_key(
                nested,
                location=f"{location}[{index}]",
            )
            if violation is not None:
                return violation
    return None


def assert_live_only_static_isolation(output_dir: Path) -> None:
    """Validate generated/copied artifacts before they can be published."""
    root = Path(output_dir)
    for path in root.rglob("*"):
        relative_path = path.relative_to(root).as_posix()
        normalized_path = relative_path.lower().replace("-", "_")
        if any(
            fragment in normalized_path
            for fragment in LIVE_ONLY_STATIC_KEY_FRAGMENTS
        ):
            raise StaticSocialIsolationError(
                f"Live-only static path {relative_path} is forbidden"
            )
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StaticSocialIsolationError(
                f"Cannot validate static JSON isolation for {relative_path}: {exc}"
            ) from exc
        violation = find_live_only_static_key(payload, location=relative_path)
        if violation is not None:
            raise StaticSocialIsolationError(
                f"Live-only static key {violation} is forbidden in {relative_path}"
            )


def write_isolated_json(path: Path, payload: dict[str, Any]) -> None:
    safe_payload = json_safe(payload)
    violation = find_live_only_static_key(safe_payload, location=path.name)
    if violation is not None:
        raise StaticSocialIsolationError(
            f"Live-only static key {violation} is forbidden in {path.name}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe_payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
