"""Doctor preflight source-selection behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from xui_reader.auth import AuthStatusResult
from xui_reader.config import RuntimeConfig
from xui_reader.diagnostics.doctor import run_doctor_preflight, select_doctor_smoke_sources
from xui_reader.errors import DiagnosticsError
from xui_reader.models import SourceKind, SourceRef


def test_select_doctor_smoke_sources_returns_guidance_when_no_sources() -> None:
    selection = select_doctor_smoke_sources(RuntimeConfig(), max_sources=2)

    assert selection.selected_sources == ()
    assert "No configured sources" in selection.warnings[0]
    assert any("Add at least one [[sources]] entry" in message for message in selection.guidance)


def test_select_doctor_smoke_sources_returns_guidance_when_all_sources_disabled() -> None:
    config = RuntimeConfig(
        sources=(
            SourceRef(source_id="list:1", kind=SourceKind.LIST, value="1", enabled=False),
            SourceRef(source_id="user:a", kind=SourceKind.USER, value="alice", enabled=False),
        )
    )

    selection = select_doctor_smoke_sources(config, max_sources=2)
    assert selection.selected_sources == ()
    assert "disabled" in selection.warnings[0]
    assert any("Enable one configured source" in message for message in selection.guidance)


def test_select_doctor_smoke_sources_prioritizes_list_then_user() -> None:
    config = RuntimeConfig(
        sources=(
            SourceRef(source_id="user:a", kind=SourceKind.USER, value="alice", enabled=True),
            SourceRef(source_id="list:1", kind=SourceKind.LIST, value="1", enabled=True),
            SourceRef(source_id="list:2", kind=SourceKind.LIST, value="2", enabled=True),
        )
    )

    selection = select_doctor_smoke_sources(config, max_sources=2)
    assert [source.source_id for source in selection.selected_sources] == ["list:1", "user:a"]


def test_run_doctor_preflight_reports_selected_sources() -> None:
    config = RuntimeConfig(
        sources=(
            SourceRef(source_id="list:1", kind=SourceKind.LIST, value="1", enabled=True),
            SourceRef(source_id="user:a", kind=SourceKind.USER, value="alice", enabled=True),
        )
    )

    report = run_doctor_preflight(
        config,
        max_sources=2,
        auth_probe=_auth_ok,
        smoke_runner=lambda _config, _source, _profile, _path, _limit: 3,
    )
    assert report.ok is True
    assert any("source_selection" in check for check in report.checks)
    assert report.details["selected_source_ids"] == "list:1,user:a"
    assert [section.name for section in report.sections] == [
        "config",
        "auth",
        "source_selection",
        "smoke",
    ]


def test_select_doctor_smoke_sources_rejects_invalid_max_sources() -> None:
    with pytest.raises(DiagnosticsError, match="max_sources"):
        select_doctor_smoke_sources(RuntimeConfig(), max_sources=0)


def test_run_doctor_preflight_captures_smoke_failure_context() -> None:
    config = RuntimeConfig(
        sources=(
            SourceRef(source_id="list:1", kind=SourceKind.LIST, value="1", enabled=True),
            SourceRef(source_id="user:a", kind=SourceKind.USER, value="alice", enabled=True),
        )
    )

    def smoke_runner(
        _config: RuntimeConfig,
        source: SourceRef,
        _profile: str | None,
        _path: str | Path | None,
        _limit: int,
    ) -> int:
        if source.source_id == "user:a":
            raise RuntimeError("selector mismatch")
        return 2

    report = run_doctor_preflight(
        config,
        max_sources=2,
        auth_probe=_auth_ok,
        smoke_runner=smoke_runner,
    )

    assert report.ok is False
    smoke_section = next(section for section in report.sections if section.name == "smoke")
    assert smoke_section.ok is False
    assert "selector mismatch" in smoke_section.details["failures"]


def _auth_ok(profile: str | None, _config_path: str | Path | None) -> AuthStatusResult:
    resolved = profile or "default"
    return AuthStatusResult(
        profile=resolved,
        storage_state_path=Path("/tmp/storage_state.json"),
        authenticated=True,
        status_code="authenticated",
        message="Authenticated session confirmed.",
    )
