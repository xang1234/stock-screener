"""Typed reports and errors shared by the Market RS rollout workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

from app.services.static_rrg_history_contract import (
    STATIC_RRG_HISTORY_SCHEMA_VERSION,
)
from app.tasks.market_queues import normalize_market


def normalize_rollout_market(market: str) -> str:
    normalized = normalize_market(market)
    if normalized == "SHARED":
        raise ValueError("Market RS rollout requires an explicit market")
    return normalized


@dataclass(frozen=True)
class BackfillDateResult:
    as_of_date: date
    status: str
    market_rs_run_id: int | None
    group_market_rs_run_id: int | None
    eligible_symbol_count: int
    group_row_count: int
    reason_code: str | None = None
    diagnostics: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["as_of_date"] = self.as_of_date.isoformat()
        return payload


@dataclass(frozen=True)
class BackfillReport:
    market: str
    formula_version: str
    requested_start_date: date | None
    through_date: date
    first_valid_date: date | None
    candidate_count: int
    completed_count: int
    failed_count: int
    latest_run_id: int | None
    group_row_count: int
    results: tuple[BackfillDateResult, ...]
    validation_errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.failed_count == 0 and not self.validation_errors

    @property
    def failed_dates(self) -> tuple[date, ...]:
        return tuple(
            item.as_of_date for item in self.results if item.status == "failed"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "market": self.market,
            "formula_version": self.formula_version,
            "requested_start_date": (
                self.requested_start_date.isoformat()
                if self.requested_start_date is not None
                else None
            ),
            "through_date": self.through_date.isoformat(),
            "first_valid_date": (
                self.first_valid_date.isoformat()
                if self.first_valid_date is not None
                else None
            ),
            "candidate_count": self.candidate_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "latest_run_id": self.latest_run_id,
            "group_row_count": self.group_row_count,
            "failed_dates": [value.isoformat() for value in self.failed_dates],
            "validation_errors": list(self.validation_errors),
            "results": [item.to_dict() for item in self.results],
        }


@dataclass(frozen=True)
class ActivationValidationReport:
    market: str
    formula_version: str
    through_date: date
    first_valid_date: date | None
    candidate_count: int
    latest_market_rs_run_id: int | None
    latest_universe_hash: str | None
    feature_run_id: int | None
    feature_universe_hash: str | None
    static_bundle_sha256: str | None
    errors: tuple[str, ...]
    rrg_status: str | None = None
    rrg_history_schema_version: str = STATIC_RRG_HISTORY_SCHEMA_VERSION

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["through_date"] = self.through_date.isoformat()
        payload["first_valid_date"] = (
            self.first_valid_date.isoformat()
            if self.first_valid_date is not None
            else None
        )
        payload["ok"] = self.ok
        payload["errors"] = list(self.errors)
        return payload


class MarketRsActivationRejected(RuntimeError):
    def __init__(self, errors: tuple[str, ...] | list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors) or "Market RS activation rejected")


__all__ = [
    "ActivationValidationReport",
    "BackfillDateResult",
    "BackfillReport",
    "MarketRsActivationRejected",
    "normalize_rollout_market",
]
