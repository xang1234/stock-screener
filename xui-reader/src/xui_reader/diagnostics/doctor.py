"""Doctor preflight helpers and source selection rules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from xui_reader.auth import AuthStatusResult, probe_auth_status
from xui_reader.config import RuntimeConfig
from xui_reader.diagnostics.base import DiagnosticReport, DiagnosticSection
from xui_reader.diagnostics.artifacts import redact_text, write_html_artifact
from xui_reader.errors import DiagnosticsError
from xui_reader.models import SourceKind, SourceRef
from xui_reader.profiles import profiles_root

AuthProbeFn = Callable[[str | None, str | Path | None], AuthStatusResult]
SmokeRunnerFn = Callable[[RuntimeConfig, SourceRef, str | None, str | Path | None, int], int]


@dataclass(frozen=True)
class DoctorSourceSelection:
    selected_sources: tuple[SourceRef, ...]
    warnings: tuple[str, ...]
    guidance: tuple[str, ...]


def select_doctor_smoke_sources(config: RuntimeConfig, *, max_sources: int = 2) -> DoctorSourceSelection:
    """Select bounded smoke-test sources with actionable fallback guidance."""
    if max_sources <= 0:
        raise DiagnosticsError("max_sources must be > 0.")

    configured = tuple(config.sources)
    enabled = tuple(source for source in configured if source.enabled)
    if enabled:
        return DoctorSourceSelection(
            selected_sources=_prioritize_sources(enabled)[:max_sources],
            warnings=(),
            guidance=(),
        )

    if not configured:
        return DoctorSourceSelection(
            selected_sources=(),
            warnings=("No configured sources; skipping optional smoke checks.",),
            guidance=(
                "Add at least one [[sources]] entry in config.toml.",
                "After adding sources, run `xui doctor` again to include smoke checks.",
            ),
        )

    return DoctorSourceSelection(
        selected_sources=(),
        warnings=("All configured sources are disabled; skipping optional smoke checks.",),
        guidance=(
            "Enable one configured source (`enabled = true`) to run smoke checks.",
            "Then re-run `xui doctor` to validate collection paths.",
        ),
    )


def run_doctor_preflight(
    config: RuntimeConfig,
    *,
    profile_name: str | None = None,
    config_path: str | Path | None = None,
    max_sources: int = 2,
    smoke_limit: int = 20,
    auth_probe: AuthProbeFn | None = None,
    smoke_runner: SmokeRunnerFn | None = None,
) -> DiagnosticReport:
    """Run structured config/auth/source/smoke checks."""
    if smoke_limit <= 0:
        raise DiagnosticsError("smoke_limit must be > 0.")

    selected_profile = profile_name or config.app.default_profile
    probe = auth_probe or probe_auth_status
    if smoke_runner is None:
        # Lazy import avoids a diagnostics<->scheduler import cycle at module load time.
        from xui_reader.scheduler.read import run_source_smoke_check

        smoke = run_source_smoke_check
    else:
        smoke = smoke_runner

    sections: list[DiagnosticSection] = []

    sections.append(
        DiagnosticSection(
            name="config",
            ok=True,
            summary=f"Loaded config with {len(config.sources)} configured sources.",
            details={
                "configured_sources": str(len(config.sources)),
                "enabled_sources": str(sum(1 for source in config.sources if source.enabled)),
            },
        )
    )

    auth_ok = False
    auth_details: dict[str, str] = {}
    try:
        auth_result = probe(selected_profile, config_path)
        auth_ok = auth_result.authenticated
        auth_details = {
            "status_code": auth_result.status_code,
            "message": auth_result.message,
        }
        if auth_result.next_steps:
            auth_details["next_steps"] = "\n".join(auth_result.next_steps)
        sections.append(
            DiagnosticSection(
                name="auth",
                ok=auth_ok,
                summary=(
                    "Authenticated session confirmed."
                    if auth_ok
                    else f"Auth check failed ({auth_result.status_code})."
                ),
                details=auth_details,
            )
        )
    except Exception as exc:
        sections.append(
            DiagnosticSection(
                name="auth",
                ok=False,
                summary="Auth check raised an unexpected error.",
                details={"error": str(exc)},
            )
        )

    selection = select_doctor_smoke_sources(config, max_sources=max_sources)
    source_details: dict[str, str] = {
        "selected_source_ids": ",".join(source.source_id for source in selection.selected_sources),
        "selected_source_kinds": ",".join(source.kind.value for source in selection.selected_sources),
    }
    if selection.warnings:
        source_details["warnings"] = "\n".join(selection.warnings)
    if selection.guidance:
        source_details["guidance"] = "\n".join(selection.guidance)
    sections.append(
        DiagnosticSection(
            name="source_selection",
            ok=True,
            summary=f"Selected {len(selection.selected_sources)} source(s) for optional smoke checks.",
            details=source_details,
        )
    )

    smoke_errors: list[str] = []
    smoke_successes: list[str] = []
    if not selection.selected_sources:
        sections.append(
            DiagnosticSection(
                name="smoke",
                ok=True,
                summary="Skipped smoke checks: no eligible sources.",
                details={"reason": "No configured enabled sources."},
            )
        )
    elif not auth_ok:
        sections.append(
            DiagnosticSection(
                name="smoke",
                ok=False,
                summary="Skipped smoke checks: session is not authenticated.",
                details=auth_details,
            )
        )
    else:
        for source in selection.selected_sources:
            try:
                emitted = smoke(config, source, selected_profile, config_path, smoke_limit)
                smoke_successes.append(f"{source.source_id}:ok:{emitted}")
            except Exception as exc:
                try:
                    screenshot_path, html_path, selector_report_path = _write_smoke_failure_artifacts(
                        profile_name=selected_profile,
                        config_path=config_path,
                        source=source,
                        error=exc,
                    )
                except Exception:
                    screenshot_path, html_path, selector_report_path = (None, None, None)
                details: list[str] = []
                if screenshot_path:
                    details.append(f"screenshot={screenshot_path}")
                if html_path:
                    details.append(f"html={html_path}")
                if selector_report_path:
                    details.append(f"selector_report={selector_report_path}")
                suffix = f" ({', '.join(details)})" if details else ""
                smoke_errors.append(f"{source.source_id}: {exc}{suffix}")
        sections.append(
            DiagnosticSection(
                name="smoke",
                ok=len(smoke_errors) == 0,
                summary=(
                    f"Ran {len(selection.selected_sources)} smoke check(s): "
                    f"{len(smoke_successes)} passed, {len(smoke_errors)} failed."
                ),
                details={
                    "successes": "\n".join(smoke_successes),
                    "failures": "\n".join(smoke_errors),
                },
            )
        )

    checks = tuple(
        f"{'PASS' if section.ok else 'FAIL'} {section.name}: {section.summary}" for section in sections
    )
    details = {
        "selected_source_ids": ",".join(source.source_id for source in selection.selected_sources),
        "selected_source_kinds": ",".join(source.kind.value for source in selection.selected_sources),
        "profile": selected_profile,
    }
    if selection.guidance:
        details["guidance"] = "\n".join(selection.guidance)

    return DiagnosticReport(
        ok=all(section.ok for section in sections),
        checks=checks,
        details=details,
        sections=tuple(sections),
    )


def _prioritize_sources(sources: tuple[SourceRef, ...]) -> tuple[SourceRef, ...]:
    lists = [source for source in sources if source.kind is SourceKind.LIST]
    users = [source for source in sources if source.kind is SourceKind.USER]

    ordered: list[SourceRef] = []
    if lists:
        ordered.append(lists[0])
    if users:
        ordered.append(users[0])
    for source in sources:
        if source not in ordered:
            ordered.append(source)
    return tuple(ordered)


def _write_smoke_failure_artifacts(
    *,
    profile_name: str,
    config_path: str | Path | None,
    source: SourceRef,
    error: Exception,
) -> tuple[str | None, str | None, str | None]:
    artifacts_dir = profiles_root(config_path) / profile_name / "artifacts" / "doctor"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run_id = _new_run_id("doctor")
    html = _extract_dom_snapshot(error)
    html_path = write_html_artifact(
        artifacts_dir,
        run_id=run_id,
        source_id=source.source_id,
        raw_html=html,
    )

    selector_report_path = artifacts_dir / f"{_slug(run_id)}_{_slug(source.source_id)}_selector-report.json"
    status_matches = len(re.findall(r"/status/(\d+)", html))
    selector_report = {
        "schema_version": "v1",
        "run_id": run_id,
        "source_id": source.source_id,
        "source_kind": source.kind.value,
        "error": redact_text(str(error)),
        "dom_snapshot_chars": len(html),
        "status_id_matches": status_matches,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    selector_report_path.write_text(
        json.dumps(selector_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    screenshot_path = _write_screenshot_artifact(
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        source_id=source.source_id,
        error=error,
    )
    return (
        str(screenshot_path) if screenshot_path is not None else None,
        str(html_path),
        str(selector_report_path),
    )


def _write_screenshot_artifact(
    *,
    artifacts_dir: Path,
    run_id: str,
    source_id: str,
    error: Exception,
) -> Path | None:
    payload = getattr(error, "screenshot_png", None)
    if not isinstance(payload, (bytes, bytearray)):
        return None

    path = artifacts_dir / f"{_slug(run_id)}_{_slug(source_id)}.png"
    path.write_bytes(bytes(payload))
    return path


def _extract_dom_snapshot(error: Exception) -> str:
    candidate = getattr(error, "dom_snapshot", "")
    return candidate if isinstance(candidate, str) else ""


def _new_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{stamp}"


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "unknown"
