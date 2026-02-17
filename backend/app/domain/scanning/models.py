"""Domain models for the scanning bounded context.

Pure value objects and enums that represent scanning concepts
independently of any infrastructure (ORM, HTTP, caching).
All dataclasses use frozen=True for immutability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..common.types import Score, Ticker


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ScreenerName(str, Enum):
    """Available stock screening methodologies."""

    MINERVINI = "minervini"
    CANSLIM = "canslim"
    IPO = "ipo"
    CUSTOM = "custom"
    VOLUME_BREAKTHROUGH = "volume_breakthrough"


class CompositeMethod(str, Enum):
    """How per-screener scores are combined into a composite score."""

    WEIGHTED_AVERAGE = "weighted_average"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"


class RatingCategory(str, Enum):
    """Human-readable rating derived from composite score."""

    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    WATCH = "Watch"
    PASS = "Pass"
    ERROR = "Error"


class PeerType(str, Enum):
    """Dimension used for peer grouping."""

    INDUSTRY = "industry"  # maps to ibd_industry_group
    SECTOR = "sector"  # maps to gics_sector


class ScanStatus(str, Enum):
    """Lifecycle states of a scan."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportFormat(str, Enum):
    """Supported export file formats."""

    CSV = "csv"


# ---------------------------------------------------------------------------
# Value Objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniverseSpec:
    """Pure domain representation of a scan universe.

    Uses tuple (not list) for symbols to ensure hashability
    and immutability of the frozen dataclass.
    """

    type: str  # "all", "exchange", "index", "custom", "test"
    exchange: str | None = None
    index: str | None = None
    symbols: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ScreenerOutputDomain:
    """Single screener's result for one stock.

    ``breakdown`` holds per-criterion detail (varies by screener).
    ``details`` holds the full analysis data (varies by screener).
    """

    screener_name: str
    score: Score  # 0-100, validated below
    passes: bool
    rating: str
    breakdown: dict[str, Any]
    details: dict[str, Any]

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 100.0):
            raise ValueError(f"Score must be 0-100, got {self.score}")


@dataclass(frozen=True)
class ScanResultItemDomain:
    """One stock's composite result from a multi-screener scan.

    ``extended_fields`` absorbs the 50+ screener-specific promoted fields
    (RS rating, stage, VCP pattern, etc.) that vary by screener and change
    frequently — keeping this type stable (Open/Closed Principle).
    """

    # Core
    symbol: Ticker
    composite_score: Score  # 0-100, validated below
    rating: str
    current_price: float | None

    # Per-screener
    screener_outputs: dict[str, ScreenerOutputDomain]

    # Meta
    screeners_run: list[str]
    composite_method: str
    screeners_passed: int
    screeners_total: int

    # Flexible screener-specific fields
    extended_fields: dict[str, Any] = field(default_factory=dict)

    # Errors
    data_errors: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.composite_score <= 100.0):
            raise ValueError(
                f"composite_score must be 0-100, got {self.composite_score}"
            )


@dataclass(frozen=True)
class ScanConfig:
    """Input configuration for initiating a scan.

    ``weights`` uses ScreenerName-typed keys to prevent misconfigured
    weight keys at the type level.
    """

    screeners: list[ScreenerName]
    composite_method: CompositeMethod
    criteria: dict[str, Any]
    universe: UniverseSpec
    weights: dict[ScreenerName, float] | None = None


@dataclass(frozen=True)
class FilterOptions:
    """Available categorical filter values for a scan's results.

    Used to populate frontend dropdown/checkbox filters.
    Each field contains the *sorted, de-duplicated* values
    actually present in the scan (not all possible values).
    """

    ibd_industries: tuple[str, ...]
    gics_sectors: tuple[str, ...]
    ratings: tuple[str, ...]


@dataclass(frozen=True)
class ResultPage:
    """Paginated result set with metadata.

    Centralises pagination math so ports return a single
    aggregate instead of ``(list, int)`` tuples.
    """

    items: tuple[ScanResultItemDomain, ...]
    total: int
    page: int
    per_page: int

    @property
    def total_pages(self) -> int:
        if self.per_page <= 0:
            return 0
        return (self.total + self.per_page - 1) // self.per_page


# ---------------------------------------------------------------------------
# Progress Reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressEvent:
    """Append-only progress snapshot for a running scan."""

    current: int
    total: int
    passed: int
    failed: int
    throughput: float | None = None
    eta_seconds: float | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ScreenerName",
    "CompositeMethod",
    "RatingCategory",
    "PeerType",
    "ScanStatus",
    "ExportFormat",
    "UniverseSpec",
    "ScreenerOutputDomain",
    "ScanResultItemDomain",
    "ScanConfig",
    "FilterOptions",
    "ResultPage",
    "ProgressEvent",
]
