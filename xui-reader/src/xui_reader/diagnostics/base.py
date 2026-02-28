"""Diagnostics interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class DiagnosticSection:
    name: str
    ok: bool
    summary: str
    details: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticReport:
    ok: bool
    checks: tuple[str, ...] = ()
    details: dict[str, str] = field(default_factory=dict)
    sections: tuple[DiagnosticSection, ...] = ()


class Doctor(Protocol):
    def run(self) -> DiagnosticReport:
        """Run diagnostics and return summarized results."""
