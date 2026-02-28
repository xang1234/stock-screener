"""Debug artifact helpers with default redaction and raw-html opt-in gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

from xui_reader.errors import DiagnosticsError

RAW_HTML_OPT_IN_ENV = "XUI_DEBUG_RAW_HTML"
DEFAULT_HTML_SNIPPET_CHARS = 4000
REDACTED = "<redacted>"

_SENSITIVE_KEY_MARKERS = (
    "storage_state",
    "cookie",
    "token",
    "authorization",
    "password",
    "secret",
    "session",
    "csrf",
)
_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*)(bearer\s+[a-z0-9._~+/-]+)"),
    re.compile(r"(?i)(set-cookie\s*[:=]\s*)([^;\n]+)"),
    re.compile(r"(?i)\b(sessionid|auth_token|ct0)\s*=\s*[^;\"'\s<>]+"),
    re.compile(r"(?i)\"(sessionid|auth_token|ct0|password|token)\"\s*:\s*\"[^\"]+\""),
)


@dataclass(frozen=True)
class HtmlArtifact:
    content: str
    truncated: bool
    redacted: bool
    raw_html_opt_in: bool


def resolve_raw_html_opt_in(raw_html_opt_in: bool | None = None) -> bool:
    """Resolve raw-html capture opt-in from explicit flag or environment."""
    if raw_html_opt_in is not None:
        return bool(raw_html_opt_in)
    raw_env = os.getenv(RAW_HTML_OPT_IN_ENV, "").strip().lower()
    return raw_env in {"1", "true", "yes", "on"}


def redact_text(value: str) -> str:
    """Redact credential-bearing markers from free-form text."""
    redacted = value
    for pattern in _SENSITIVE_VALUE_PATTERNS:
        redacted = pattern.sub(_replace_with_redacted, redacted)
    return redacted


def redact_value(value: Any) -> Any:
    """Recursively redact mapping/list/scalar values for safe diagnostics output."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            key_str = str(key)
            if _is_sensitive_key(key_str):
                sanitized[key_str] = REDACTED
            else:
                sanitized[key_str] = redact_value(child)
        return sanitized
    if isinstance(value, tuple):
        return tuple(redact_value(child) for child in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_value(child) for child in value]
    return value


def build_html_artifact(
    raw_html: str,
    *,
    raw_html_opt_in: bool | None = None,
    snippet_chars: int = DEFAULT_HTML_SNIPPET_CHARS,
) -> HtmlArtifact:
    """Create redacted HTML artifact content with safe default truncation."""
    if snippet_chars <= 0:
        raise DiagnosticsError("snippet_chars must be > 0.")

    allow_raw = resolve_raw_html_opt_in(raw_html_opt_in)
    original = str(raw_html)
    chosen = original if allow_raw else original[:snippet_chars]
    redacted = redact_text(chosen)
    truncated = not allow_raw and len(original) > len(chosen)
    return HtmlArtifact(
        content=redacted,
        truncated=truncated,
        redacted=(redacted != chosen) or _contains_sensitive_markers(chosen),
        raw_html_opt_in=allow_raw,
    )


def write_html_artifact(
    artifacts_dir: str | Path,
    *,
    run_id: str,
    source_id: str,
    raw_html: str,
    raw_html_opt_in: bool | None = None,
    snippet_chars: int = DEFAULT_HTML_SNIPPET_CHARS,
) -> Path:
    """Write deterministic, redacted HTML artifact and return output path."""
    artifact = build_html_artifact(
        raw_html,
        raw_html_opt_in=raw_html_opt_in,
        snippet_chars=snippet_chars,
    )
    root = Path(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    suffix = "raw" if artifact.raw_html_opt_in else "snippet"
    path = root / f"{_slug(run_id)}_{_slug(source_id)}_{suffix}.html"
    body = artifact.content
    if artifact.truncated:
        body += "\n<!-- truncated: set XUI_DEBUG_RAW_HTML=1 to opt in to full redacted html -->\n"
    path.write_text(body, encoding="utf-8")
    return path


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _SENSITIVE_KEY_MARKERS)


def _replace_with_redacted(match: re.Match[str]) -> str:
    if match.lastindex and match.lastindex >= 2:
        return f"{match.group(1)}{REDACTED}"
    if match.lastindex == 1:
        return f"{match.group(1)}{REDACTED}"
    return REDACTED


def _contains_sensitive_markers(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in ("sessionid", "auth_token", "ct0", "authorization"))


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "unknown"
