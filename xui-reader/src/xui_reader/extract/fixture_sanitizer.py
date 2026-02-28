"""Sanitize HTML fixtures so snapshot tests stay representative without leaking sensitive data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import re
from typing import Sequence

from xui_reader.diagnostics.artifacts import redact_text

_EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d().-]*[()\s][\d().\-\s]*\d)(?!\d)")
_STATUS_LINK_RE = re.compile(
    r"(?P<prefix>(?:https?://(?:www\.)?(?:x|twitter)\.com)?/)"
    r"(?P<handle>[A-Za-z0-9_]{1,15})/status/(?P<tweet_id>\d+)"
)
_PROFILE_HREF_RE = re.compile(
    r'(?P<prefix>href=["\'](?:https?://(?:www\.)?(?:x|twitter)\.com/|/))'
    r"(?P<handle>[A-Za-z0-9_]{1,15})(?P<suffix>[\"'])"
)
_MENTION_RE = re.compile(r"(?<![\w/])@(?P<handle>[A-Za-z0-9_]{1,15})\b")
_NUMERIC_ID_ATTR_RE = re.compile(
    r'(?P<prefix>(?:data-(?:user|item|conversation|tweet)-id|rest_id|user_id|tweet_id)\s*=\s*["\'])'
    r"(?P<value>\d+)(?P<suffix>[\"'])",
    re.IGNORECASE,
)
_RESERVED_HANDLES = {
    "about",
    "compose",
    "explore",
    "hashtag",
    "help",
    "home",
    "i",
    "intent",
    "login",
    "messages",
    "notifications",
    "privacy",
    "search",
    "settings",
    "share",
    "signup",
    "tos",
}


@dataclass(frozen=True)
class FixtureSanitizationResult:
    sanitized_html: str
    handle_replacements: int
    status_id_replacements: int
    numeric_id_replacements: int
    email_replacements: int
    phone_replacements: int
    output_path: Path | None = None


def sanitize_fixture_html(raw_html: str) -> FixtureSanitizationResult:
    """Sanitize sensitive values while preserving selector-relevant DOM structure."""
    text = redact_text(str(raw_html))
    handle_aliases: dict[str, str] = {}
    status_id_aliases: dict[str, str] = {}
    numeric_id_aliases: dict[str, str] = {}
    handle_replacements = 0
    status_id_replacements = 0
    numeric_id_replacements = 0

    text, email_replacements = _EMAIL_RE.subn("sanitized@example.test", text)

    def _alias_handle(raw_handle: str) -> str:
        key = raw_handle.lower()
        if key in _RESERVED_HANDLES:
            return raw_handle
        alias = handle_aliases.get(key)
        if alias is None:
            alias = f"user{len(handle_aliases) + 1:02d}"
            handle_aliases[key] = alias
        return alias

    def _alias_status_id(raw_id: str) -> str:
        alias = status_id_aliases.get(raw_id)
        if alias is None:
            width = max(6, len(raw_id))
            alias = f"{len(status_id_aliases) + 1:0{width}d}"
            status_id_aliases[raw_id] = alias
        return alias

    def _alias_numeric_id(raw_id: str) -> str:
        alias = numeric_id_aliases.get(raw_id)
        if alias is None:
            width = max(6, len(raw_id))
            alias = f"{len(numeric_id_aliases) + 1:0{width}d}"
            numeric_id_aliases[raw_id] = alias
        return alias

    def _replace_status_link(match: re.Match[str]) -> str:
        nonlocal handle_replacements, status_id_replacements
        alias_handle = _alias_handle(match.group("handle"))
        alias_status_id = _alias_status_id(match.group("tweet_id"))
        if alias_handle != match.group("handle"):
            handle_replacements += 1
        status_id_replacements += 1
        return f"{match.group('prefix')}{alias_handle}/status/{alias_status_id}"

    text = _STATUS_LINK_RE.sub(_replace_status_link, text)

    def _replace_profile_href(match: re.Match[str]) -> str:
        nonlocal handle_replacements
        alias_handle = _alias_handle(match.group("handle"))
        if alias_handle == match.group("handle"):
            return match.group(0)
        handle_replacements += 1
        return f"{match.group('prefix')}{alias_handle}{match.group('suffix')}"

    text = _PROFILE_HREF_RE.sub(_replace_profile_href, text)

    def _replace_mention(match: re.Match[str]) -> str:
        nonlocal handle_replacements
        alias_handle = _alias_handle(match.group("handle"))
        if alias_handle == match.group("handle"):
            return match.group(0)
        handle_replacements += 1
        return f"@{alias_handle}"

    text = _MENTION_RE.sub(_replace_mention, text)

    def _replace_numeric_id_attr(match: re.Match[str]) -> str:
        nonlocal numeric_id_replacements
        numeric_id_replacements += 1
        alias = _alias_numeric_id(match.group("value"))
        return f"{match.group('prefix')}{alias}{match.group('suffix')}"

    text = _NUMERIC_ID_ATTR_RE.sub(_replace_numeric_id_attr, text)
    text, phone_replacements = _PHONE_RE.subn("<redacted-phone>", text)

    return FixtureSanitizationResult(
        sanitized_html=text,
        handle_replacements=handle_replacements,
        status_id_replacements=status_id_replacements,
        numeric_id_replacements=numeric_id_replacements,
        email_replacements=email_replacements,
        phone_replacements=phone_replacements,
    )


def sanitize_fixture_file(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> FixtureSanitizationResult:
    """Read, sanitize, and write an HTML fixture."""
    source = Path(input_path)
    target = Path(output_path) if output_path is not None else _default_output_path(source)
    result = sanitize_fixture_html(source.read_text(encoding="utf-8"))
    target.write_text(result.sanitized_html, encoding="utf-8")
    return FixtureSanitizationResult(
        sanitized_html=result.sanitized_html,
        handle_replacements=result.handle_replacements,
        status_id_replacements=result.status_id_replacements,
        numeric_id_replacements=result.numeric_id_replacements,
        email_replacements=result.email_replacements,
        phone_replacements=result.phone_replacements,
        output_path=target,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by fixture-refresh workflows."""
    parser = argparse.ArgumentParser(description="Sanitize XUI Reader HTML fixtures.")
    parser.add_argument("input", help="Path to raw fixture HTML.")
    parser.add_argument("--output", help="Optional output path for sanitized fixture.")
    args = parser.parse_args(argv)
    result = sanitize_fixture_file(args.input, output_path=args.output)
    assert result.output_path is not None
    print(
        f"Wrote {result.output_path} "
        f"(handles={result.handle_replacements}, status_ids={result.status_id_replacements}, "
        f"numeric_ids={result.numeric_id_replacements}, emails={result.email_replacements}, "
        f"phones={result.phone_replacements})"
    )
    return 0


def _default_output_path(source: Path) -> Path:
    suffix = source.suffix or ".html"
    return source.with_name(f"{source.stem}.sanitized{suffix}")


if __name__ == "__main__":
    raise SystemExit(main())
