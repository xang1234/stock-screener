"""Doctor preflight helpers and source selection rules."""

from __future__ import annotations

from dataclasses import dataclass

from xui_reader.config import RuntimeConfig
from xui_reader.diagnostics.base import DiagnosticReport
from xui_reader.errors import DiagnosticsError
from xui_reader.models import SourceKind, SourceRef


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


def run_doctor_preflight(config: RuntimeConfig, *, max_sources: int = 2) -> DiagnosticReport:
    """Build an operator-facing preflight report."""
    selection = select_doctor_smoke_sources(config, max_sources=max_sources)
    checks: list[str] = [f"config.sources.total={len(config.sources)}"]
    checks.extend(selection.warnings)
    checks.append(f"doctor.smoke_sources.selected={len(selection.selected_sources)}")

    details: dict[str, str] = {
        "selected_source_ids": ",".join(source.source_id for source in selection.selected_sources),
        "selected_source_kinds": ",".join(source.kind.value for source in selection.selected_sources),
    }
    if selection.guidance:
        details["guidance"] = "\n".join(selection.guidance)

    return DiagnosticReport(ok=True, checks=tuple(checks), details=details)


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
