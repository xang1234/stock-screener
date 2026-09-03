"""Provider-neutral value objects for options analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping


class CandidateKind(str, Enum):
    CURRENT = "current"
    CONTINUITY = "continuity"


class OptionSide(str, Enum):
    CALL = "call"
    PUT = "put"


class ObservationState(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    BUILDING_HISTORY = "building_history"
    INSUFFICIENT_QUALITY = "insufficient_quality"


class OptionsRunStatus(str, Enum):
    STAGED = "staged"
    RUNNING = "running"
    PUBLISHED = "published"
    FAILED_QUALITY = "failed_quality"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class OptionCandidateInput:
    symbol: str
    composite_score: float | None
    daily_dollar_volume: float | None
    spot_price: float | None
    dividend_yield: float | None = None
    price_closes: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("Candidate symbol is required")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "price_closes", tuple(self.price_closes))


@dataclass(frozen=True)
class OptionCandidate:
    symbol: str
    kind: CandidateKind
    composite_score: float | None
    daily_dollar_volume: float | None
    spot_price: float | None
    dividend_yield: float | None = None
    price_closes: tuple[float, ...] = ()
    candidate_rank: int | None = None
    leader_rank: int | None = None
    sessions_since_current: int = 0
    prior_best_rank: int | None = None

    @property
    def is_candidate(self) -> bool:
        return self.candidate_rank is not None

    @property
    def is_leader(self) -> bool:
        return self.leader_rank is not None


@dataclass(frozen=True)
class NormalizedOptionContract:
    side: OptionSide
    strike: float
    bid: float | None
    ask: float | None
    last_price: float | None
    volume: int | None
    open_interest: int | None
    implied_volatility: float | None
    last_trade_at: datetime | None
    contract_size: str | None
    multiplier: int | None
    delta: float | None = None


@dataclass(frozen=True)
class ChainObservation:
    symbol: str
    expiration: date
    source_spot_price: float
    fetched_at: datetime
    contracts: tuple[NormalizedOptionContract, ...]
    provider_spot_price: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "contracts", tuple(self.contracts))


@dataclass(frozen=True)
class MetricValue:
    available: bool
    value: float | None = None
    reason_codes: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    label: str | None = None

    def __post_init__(self) -> None:
        if self.available:
            if self.value is None or not math.isfinite(self.value):
                raise ValueError("Available metric values must be finite")
            if self.reason_codes:
                raise ValueError("Available metric values cannot have reason codes")
        elif self.value is not None:
            raise ValueError("Unavailable metric values cannot contain a value")


@dataclass(frozen=True)
class OptionsRunSummary:
    status: OptionsRunStatus
    expected_count: int
    completed_count: int
    core_valid_current_count: int
    failed_count: int
    retried_count: int
    coverage: float
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PublicationDecision:
    publish: bool
    current_count: int
    core_valid_current_count: int
    coverage: float
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class HistoryReadiness:
    short_history_available: bool
    iv_history_available: bool
    short_observation_count: int
    iv_observation_count: int
    lifetime_observation_count: int
    reason_codes: tuple[str, ...] = ()

