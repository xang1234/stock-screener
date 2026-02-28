"""Profile lifecycle helpers and safety guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil

from .config import load_runtime_config, resolve_config_path
from .errors import ProfileError

PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PROFILE_SUBDIRS = ("session", "artifacts", "logs")


@dataclass(frozen=True)
class ProfileInfo:
    name: str
    active: bool


def profiles_root(config_path: str | Path | None = None) -> Path:
    """Return the root directory that stores all profile state."""
    return resolve_config_path(config_path).parent / "profiles"


def list_profiles(config_path: str | Path | None = None) -> tuple[list[ProfileInfo], str]:
    """List known profile directories and include active marker from config."""
    config = load_runtime_config(config_path)
    active_name = config.app.default_profile
    root = profiles_root(config_path)
    try:
        names = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    except OSError as exc:
        raise ProfileError(
            f"Could not list profiles under '{root}': {exc}. Check directory permissions."
        ) from exc
    infos = [ProfileInfo(name=name, active=(name == active_name)) for name in names]
    return infos, active_name


def create_profile(
    name: str,
    config_path: str | Path | None = None,
    *,
    switch: bool = False,
    force: bool = False,
) -> Path:
    """Create a profile directory with a deterministic bootstrap layout."""
    normalized_name = _validate_profile_name(name)
    load_runtime_config(config_path)
    profile_dir = _profile_path(normalized_name, config_path)
    root = profiles_root(config_path)
    try:
        root.mkdir(parents=True, exist_ok=True)

        if profile_dir.exists() and not profile_dir.is_dir():
            raise ProfileError(
                f"Profile path '{profile_dir}' exists but is not a directory. "
                "Choose a different profile name or remove the conflicting file."
            )
        if profile_dir.exists() and not force:
            raise ProfileError(
                f"Profile '{normalized_name}' already exists. Re-run with --force to re-bootstrap."
            )

        profile_dir.mkdir(parents=True, exist_ok=True)
        for child in PROFILE_SUBDIRS:
            (profile_dir / child).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProfileError(
            f"Could not create or initialize profile '{normalized_name}' at '{profile_dir}': {exc}. "
            "Check directory permissions."
        ) from exc

    if switch:
        _set_default_profile(normalized_name, config_path)

    return profile_dir


def switch_profile(
    name: str,
    config_path: str | Path | None = None,
    *,
    create_missing: bool = False,
) -> Path:
    """Switch active profile, optionally creating missing target."""
    normalized_name = _validate_profile_name(name)
    config = load_runtime_config(config_path)
    target = _profile_path(normalized_name, config_path)

    try:
        if not target.exists():
            if not create_missing:
                raise ProfileError(
                    f"Profile '{normalized_name}' does not exist. "
                    f"Run `xui profiles create {normalized_name}` or retry with --create."
                )
            create_profile(normalized_name, config_path, switch=False, force=False)
        elif not target.is_dir():
            raise ProfileError(f"Profile path '{target}' is not a directory.")
    except OSError as exc:
        raise ProfileError(
            f"Could not access profile '{normalized_name}' at '{target}': {exc}. "
            "Check directory permissions."
        ) from exc

    if config.app.default_profile != normalized_name:
        _set_default_profile(normalized_name, config_path)
    return target


def delete_profile(name: str, config_path: str | Path | None = None) -> Path:
    """Delete a non-active profile directory."""
    normalized_name = _validate_profile_name(name)
    config = load_runtime_config(config_path)
    if config.app.default_profile == normalized_name:
        raise ProfileError(
            f"Cannot delete active profile '{normalized_name}'. "
            "Switch to another profile first with `xui profiles switch <name>`."
        )

    target = _profile_path(normalized_name, config_path)
    try:
        if not target.exists():
            raise ProfileError(
                f"Profile '{normalized_name}' does not exist. Run `xui profiles list` to inspect available profiles."
            )
        if not target.is_dir():
            raise ProfileError(f"Profile path '{target}' is not a directory.")

        shutil.rmtree(target)
    except OSError as exc:
        raise ProfileError(
            f"Could not delete profile '{normalized_name}' at '{target}': {exc}. "
            "Check directory permissions."
        ) from exc
    return target


def _validate_profile_name(name: str) -> str:
    normalized = name.strip()
    if not PROFILE_NAME_RE.match(normalized):
        raise ProfileError(
            "Invalid profile name. Use 1-64 chars matching [A-Za-z0-9._-] and start with an alphanumeric character."
        )
    return normalized


def _profile_path(name: str, config_path: str | Path | None = None) -> Path:
    return profiles_root(config_path) / name


def _set_default_profile(name: str, config_path: str | Path | None = None) -> None:
    path = resolve_config_path(config_path)
    if not path.exists():
        raise ProfileError(
            f"Config file not found at '{path}'. Run `xui config init --path \"{path}\"` first."
        )
    if path.is_dir():
        raise ProfileError(
            f"Config path '{path}' is a directory; pass a file path ending in 'config.toml'."
        )

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProfileError(f"Could not read config file '{path}': {exc}.") from exc

    updated = _replace_default_profile_in_app_table(text, name)

    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        raise ProfileError(f"Could not write config file '{path}': {exc}.") from exc

    # Re-parse for immediate feedback if a malformed edit slipped through.
    load_runtime_config(path)


def _replace_default_profile_in_app_table(config_text: str, profile_name: str) -> str:
    lines = config_text.splitlines()
    had_trailing_newline = config_text.endswith("\n")
    app_start = None
    app_end = len(lines)

    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not (stripped.startswith("[") and stripped.endswith("]")):
            continue
        if stripped == "[app]":
            app_start = index
            continue
        if app_start is not None and index > app_start:
            app_end = index
            break

    default_line = f'default_profile = "{_toml_escape(profile_name)}"'

    if app_start is None:
        new_lines = ["[app]", default_line, ""]
        new_lines.extend(lines)
        output = "\n".join(new_lines)
        return output if had_trailing_newline else f"{output}\n"

    default_index = None
    for index in range(app_start + 1, app_end):
        if "=" not in lines[index]:
            continue
        key = lines[index].split("=", 1)[0].strip()
        if key == "default_profile":
            default_index = index
            break

    if default_index is None:
        lines.insert(app_start + 1, default_line)
    else:
        lines[default_index] = default_line

    output = "\n".join(lines)
    if had_trailing_newline:
        output = f"{output}\n"
    return output


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
