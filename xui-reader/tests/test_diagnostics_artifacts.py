"""Diagnostics artifact redaction and html opt-in controls."""

from __future__ import annotations

from pathlib import Path

import pytest

from xui_reader.diagnostics.artifacts import (
    REDACTED,
    build_html_artifact,
    redact_text,
    redact_value,
    resolve_raw_html_opt_in,
    write_html_artifact,
)
from xui_reader.errors import DiagnosticsError


def test_redact_text_masks_common_auth_markers() -> None:
    raw = (
        'Authorization: Bearer abcdef123\n'
        'Set-Cookie: sessionid=secret-cookie;\n'
        '{"auth_token":"secret-token","message":"safe"}'
    )
    redacted = redact_text(raw)
    assert "abcdef123" not in redacted
    assert "secret-cookie" not in redacted
    assert "secret-token" not in redacted
    assert REDACTED in redacted


def test_redact_value_masks_sensitive_keys_recursively() -> None:
    payload = {
        "status": "ok",
        "storage_state": {"cookies": [{"name": "sessionid", "value": "top-secret"}]},
        "nested": {"auth_token": "hidden"},
    }
    redacted = redact_value(payload)
    assert redacted["status"] == "ok"
    assert redacted["storage_state"] == REDACTED
    assert redacted["nested"]["auth_token"] == REDACTED


def test_build_html_artifact_defaults_to_snippet_and_redaction() -> None:
    raw_html = "<html><body>" + ("A" * 5000) + " sessionid=secret </body></html>"
    artifact = build_html_artifact(raw_html, raw_html_opt_in=False, snippet_chars=200)
    assert artifact.raw_html_opt_in is False
    assert artifact.truncated is True
    assert len(artifact.content) < len(raw_html)
    assert "secret" not in artifact.content


def test_build_html_artifact_allows_full_capture_only_when_opted_in() -> None:
    raw_html = "<div>" + ("X" * 500) + " auth_token=super-secret </div>"
    artifact = build_html_artifact(raw_html, raw_html_opt_in=True, snippet_chars=100)
    assert artifact.raw_html_opt_in is True
    assert artifact.truncated is False
    assert len(artifact.content) >= 500
    assert "super-secret" not in artifact.content


def test_build_html_artifact_rejects_non_positive_snippet_chars() -> None:
    with pytest.raises(DiagnosticsError, match="snippet_chars"):
        build_html_artifact("<div>x</div>", snippet_chars=0)


def test_write_html_artifact_uses_deterministic_suffixes(tmp_path: Path) -> None:
    snippet_path = write_html_artifact(
        tmp_path,
        run_id="run-1",
        source_id="list:1",
        raw_html="<div>" + ("x" * 100) + "</div>",
        raw_html_opt_in=False,
        snippet_chars=10,
    )
    raw_path = write_html_artifact(
        tmp_path,
        run_id="run-1",
        source_id="list:1",
        raw_html="<div>" + ("x" * 20) + "</div>",
        raw_html_opt_in=True,
        snippet_chars=10,
    )

    assert snippet_path.name.endswith("_snippet.html")
    assert raw_path.name.endswith("_raw.html")
    assert "truncated" in snippet_path.read_text(encoding="utf-8")


def test_resolve_raw_html_opt_in_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XUI_DEBUG_RAW_HTML", "true")
    assert resolve_raw_html_opt_in(None) is True
    monkeypatch.setenv("XUI_DEBUG_RAW_HTML", "0")
    assert resolve_raw_html_opt_in(None) is False
