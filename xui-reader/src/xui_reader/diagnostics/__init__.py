"""Diagnostics contracts and helpers."""

from .artifacts import (
    DEFAULT_HTML_SNIPPET_CHARS,
    HtmlArtifact,
    build_html_artifact,
    redact_text,
    redact_value,
    resolve_raw_html_opt_in,
    write_html_artifact,
)
from .base import DiagnosticReport, DiagnosticSection, Doctor
from .doctor import DoctorSourceSelection, run_doctor_preflight, select_doctor_smoke_sources
from .events import (
    DEBUG_EVENT_COMPATIBILITY_NOTES,
    DEBUG_EVENT_SCHEMA_VERSION,
    JsonlEventLogger,
    build_debug_event,
    ensure_schema_compatible,
    validate_debug_event,
)

__all__ = [
    "DEFAULT_HTML_SNIPPET_CHARS",
    "DEBUG_EVENT_COMPATIBILITY_NOTES",
    "DEBUG_EVENT_SCHEMA_VERSION",
    "DiagnosticReport",
    "DiagnosticSection",
    "Doctor",
    "DoctorSourceSelection",
    "HtmlArtifact",
    "JsonlEventLogger",
    "build_debug_event",
    "build_html_artifact",
    "ensure_schema_compatible",
    "redact_text",
    "redact_value",
    "resolve_raw_html_opt_in",
    "run_doctor_preflight",
    "select_doctor_smoke_sources",
    "validate_debug_event",
    "write_html_artifact",
]
