"""Shared policy for publishing multi-market static site artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Mapping

from app.domain.markets import SUPPORTED_MARKET_CODES
from app.domain.relative_strength import LEGACY_RS_FORMULA_VERSION
from app.services.market_rs_result_contract import (
    MARKET_RS_REASON_BENCHMARK_ADJUSTED_ANCHOR_MISSING,
    MARKET_RS_REASON_CURRENT_ADJUSTED_PRICE_COVERAGE_BELOW_THRESHOLD,
    MARKET_RS_REASON_HISTORICAL_ADJUSTED_ANCHOR_GAP_ABOVE_THRESHOLD,
)

REQUIRED_STATIC_MARKETS = frozenset({"US"})
OPTIONAL_STATIC_MARKETS = frozenset(
    market for market in SUPPORTED_MARKET_CODES if market not in REQUIRED_STATIC_MARKETS
)
STATIC_MARKET_RS_NO_CURRENT_ARTIFACT_REASON_CODES = frozenset(
    {
        MARKET_RS_REASON_BENCHMARK_ADJUSTED_ANCHOR_MISSING,
        MARKET_RS_REASON_CURRENT_ADJUSTED_PRICE_COVERAGE_BELOW_THRESHOLD,
        MARKET_RS_REASON_HISTORICAL_ADJUSTED_ANCHOR_GAP_ABOVE_THRESHOLD,
    }
)
STATIC_NO_CURRENT_ARTIFACT_SNAPSHOT_REASONS = frozenset(
    {
        "group_rank_backfill_not_ready",
        "market_exposure_not_ready",
        "market_rs_not_ready",
    }
)


class StaticMarketRsArtifactState(Enum):
    READY = "ready"
    NO_CURRENT_ARTIFACT = "no_current_artifact"
    HARD_FAILURE = "hard_failure"


@dataclass(frozen=True)
class StaticMarketRsArtifactResult:
    status: str
    market: str
    as_of_date: str
    formula_version: str
    market_rs_run_id: object | None
    reason_code: str | None

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        market: str,
        as_of_date: date | str,
        formula_version: str,
    ) -> "StaticMarketRsArtifactResult | None":
        if not isinstance(payload, Mapping):
            return None

        expected_market = str(market).strip().upper()
        expected_as_of_date = _date_value(as_of_date)
        expected_formula = str(formula_version).strip()
        observed_market = payload.get("market")
        observed_as_of_date = _date_value(payload.get("as_of_date"))
        observed_formula = payload.get("formula_version")
        status = payload.get("status")
        reason_code = payload.get("reason_code")
        if (
            not expected_market
            or expected_as_of_date is None
            or not expected_formula
            or not isinstance(observed_market, str)
            or observed_market.strip().upper() != expected_market
            or observed_as_of_date != expected_as_of_date
            or not isinstance(observed_formula, str)
            or observed_formula.strip() != expected_formula
            or not isinstance(status, str)
        ):
            return None
        if reason_code is not None and not isinstance(reason_code, str):
            return None

        return cls(
            status=status.strip(),
            market=expected_market,
            as_of_date=expected_as_of_date,
            formula_version=observed_formula.strip(),
            market_rs_run_id=payload.get("market_rs_run_id"),
            reason_code=reason_code.strip() if reason_code is not None else None,
        )


@dataclass(frozen=True)
class StaticNoCurrentArtifactFailure:
    market: str
    reason: str
    snapshot: Mapping[str, Any]


def _date_value(value: object) -> str | None:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return None


def classify_static_market_rs_artifact_result(
    result: object,
    *,
    market: str,
    as_of_date: date | str,
    formula_version: str,
) -> StaticMarketRsArtifactState:
    parsed = StaticMarketRsArtifactResult.from_payload(
        result,
        market=market,
        as_of_date=as_of_date,
        formula_version=formula_version,
    )
    if parsed is None:
        return StaticMarketRsArtifactState.HARD_FAILURE
    if parsed.formula_version == LEGACY_RS_FORMULA_VERSION:
        if parsed.status == "selected":
            return StaticMarketRsArtifactState.READY
        return StaticMarketRsArtifactState.HARD_FAILURE
    if parsed.status == "completed" and parsed.market_rs_run_id is not None:
        return StaticMarketRsArtifactState.READY
    if (
        parsed.status == "failed"
        and parsed.reason_code in STATIC_MARKET_RS_NO_CURRENT_ARTIFACT_REASON_CODES
    ):
        return StaticMarketRsArtifactState.NO_CURRENT_ARTIFACT
    return StaticMarketRsArtifactState.HARD_FAILURE


def collect_static_no_current_artifact_failures(
    refresh_results: Mapping[str, Any],
    *,
    market: str | None,
) -> tuple[StaticNoCurrentArtifactFailure, ...]:
    feature_snapshots = refresh_results.get("feature_snapshots", {})
    if not isinstance(feature_snapshots, Mapping):
        return ()
    snapshots_by_market = {
        str(snapshot_market).strip().upper(): snapshot
        for snapshot_market, snapshot in feature_snapshots.items()
        if str(snapshot_market).strip()
    }
    selected_markets = (
        (str(market).strip().upper(),)
        if market is not None
        else tuple(snapshots_by_market)
    )
    failures: list[StaticNoCurrentArtifactFailure] = []
    for selected_market in selected_markets:
        snapshot = snapshots_by_market.get(selected_market)
        if not isinstance(snapshot, Mapping):
            continue
        reason = snapshot.get("reason")
        if not isinstance(reason, str):
            continue
        normalized_reason = reason.strip().lower()
        if normalized_reason in STATIC_NO_CURRENT_ARTIFACT_SNAPSHOT_REASONS:
            failures.append(
                StaticNoCurrentArtifactFailure(
                    market=selected_market,
                    reason=normalized_reason,
                    snapshot=snapshot,
                )
            )
    return tuple(failures)
