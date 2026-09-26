"""Common contract for market document adapters.

An adapter turns an issuer/question into bounded official document targets
and fetches them through the shared acquisition registry. It never runs an
unrestricted crawler, never uses a paid feed, and reports unavailable
routes as typed coverage rather than as "no exposure".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.domain.company_exposure.contracts import (
    CaptureResult,
    CoverageItem,
    DocumentTarget,
)


@dataclass(frozen=True, slots=True)
class DocumentQuery:
    """What to look for: document kinds and a bounded time window."""

    document_kinds: tuple[str, ...] = ("annual_report",)
    since_year: int | None = None
    max_documents: int = 4
    question_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AcquisitionLimits:
    max_documents: int = 12
    max_bytes_per_document: int = 25 * 1024 * 1024
    max_requests: int = 40


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    targets: tuple[DocumentTarget, ...] = ()
    coverage: tuple[CoverageItem, ...] = field(default_factory=tuple)


class MarketDocumentAdapter(Protocol):
    market: str

    def discover(
        self, issuer, query: DocumentQuery, limits: AcquisitionLimits, budget
    ) -> DiscoveryResult: ...

    def resolve_target(self, raw_metadata: dict) -> DocumentTarget: ...

    def fetch(self, target: DocumentTarget, budget) -> CaptureResult: ...
