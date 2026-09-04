"""Project calculated options analytics into persistence-ready values."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

from app.domain.options_analytics.expiration import retain_contracts_for_persistence
from app.domain.options_analytics.metrics.aggregate import ChainMetrics
from app.domain.options_analytics.metrics.gex import estimate_contract_gex
from app.domain.options_analytics.metrics.history import HistoricalMetrics
from app.domain.options_analytics.models import (
    ChainObservation,
    NormalizedOptionContract,
    OptionCandidate,
    OptionSide,
)

from .analysis_models import OptionsMetricValues, OptionsStrikePoint


def _complete_total(
    contracts: Sequence[NormalizedOptionContract],
    field: str,
) -> int | None:
    values = tuple(getattr(contract, field) for contract in contracts)
    if not values or any(
        value is None or not math.isfinite(float(value)) or value < 0
        for value in values
    ):
        return None
    return sum(int(value) for value in values if value is not None)


def _aggregate_iv(contracts: Sequence[NormalizedOptionContract]) -> float | None:
    values = tuple(contract.implied_volatility for contract in contracts)
    if not values or any(
        value is None or not math.isfinite(float(value)) or value <= 0
        for value in values
    ):
        return None
    ivs = tuple(float(value) for value in values if value is not None)
    open_interests = tuple(contract.open_interest for contract in contracts)
    if all(
        value is not None and math.isfinite(float(value)) and value >= 0
        for value in open_interests
    ):
        total_open_interest = sum(
            int(value) for value in open_interests if value is not None
        )
        if total_open_interest > 0:
            return sum(
                iv * int(open_interest)
                for iv, open_interest in zip(ivs, open_interests, strict=True)
                if open_interest is not None
            ) / total_open_interest
    return sum(ivs) / len(ivs)


def dividend_assumption(candidate: OptionCandidate) -> tuple[float, str, str | None]:
    value = candidate.dividend_yield
    if value is None or not math.isfinite(float(value)) or float(value) < 0:
        return 0.0, "zero_assumption", "zero_dividend_assumption"
    if candidate.dividend_source == "zero_assumption":
        return 0.0, "zero_assumption", "zero_dividend_assumption"
    return float(value), "pinned_feature_run", None


def metric_values(
    metrics: ChainMetrics,
    observation: ChainObservation,
) -> OptionsMetricValues:
    def total(side: OptionSide, field: str) -> int | None:
        return _complete_total(
            tuple(
                contract
                for contract in observation.contracts
                if contract.side is side
            ),
            field,
        )

    return OptionsMetricValues(
        max_pain=metrics.max_pain.value,
        net_gex=metrics.net_gex.value,
        gamma_flip=metrics.gamma_flip.value,
        call_wall=metrics.call_wall.value,
        put_wall=metrics.put_wall.value,
        atm_iv=metrics.atm_iv.value,
        skew_25_delta=metrics.skew_25_delta.value,
        realized_volatility=metrics.realized_volatility.value,
        vrp=metrics.vrp.value,
        activity_intensity=metrics.activity.activity_intensity.value,
        call_open_interest=total(OptionSide.CALL, "open_interest"),
        put_open_interest=total(OptionSide.PUT, "open_interest"),
        call_volume=total(OptionSide.CALL, "volume"),
        put_volume=total(OptionSide.PUT, "volume"),
        volume_oi_ratio=metrics.activity.volume_oi_ratio.value,
        near_spot_volume_concentration=(
            metrics.activity.near_spot_volume_concentration.value
        ),
    )


def strike_points(
    observation: ChainObservation,
    *,
    as_of_date: date,
    risk_free_rate: float | None,
    dividend_yield: float,
) -> tuple[OptionsStrikePoint, ...]:
    retained = retain_contracts_for_persistence(
        observation.contracts,
        spot_price=observation.source_spot_price,
    )
    time_years = (observation.expiration - as_of_date).days / 365
    grouped: dict[float, dict[OptionSide, list[NormalizedOptionContract]]] = {}
    for contract in retained:
        grouped.setdefault(contract.strike, {}).setdefault(contract.side, []).append(
            contract
        )
    points: list[OptionsStrikePoint] = []
    for strike in sorted(grouped):
        point: dict[str, Any] = {"strike": strike}
        for side, contracts in grouped[strike].items():
            rows = tuple(contracts)
            prefix = "call" if side is OptionSide.CALL else "put"
            point[f"{prefix}_open_interest"] = _complete_total(
                rows, "open_interest"
            )
            point[f"{prefix}_volume"] = _complete_total(rows, "volume")
            point[f"{prefix}_iv"] = _aggregate_iv(rows)
            gex_values = []
            if risk_free_rate is not None:
                gex_values = [
                    estimate_contract_gex(
                        contract,
                        spot=observation.source_spot_price,
                        time_years=time_years,
                        rate=risk_free_rate,
                        dividend_yield=dividend_yield,
                    )
                    for contract in rows
                ]
            available_gex = [
                float(value.value)
                for value in gex_values
                if value.available and value.value is not None
            ]
            point[f"estimated_{prefix}_gex"] = (
                sum(available_gex) if available_gex else None
            )
        points.append(OptionsStrikePoint(**point))
    return tuple(points)


def metric_evidence(
    metrics: ChainMetrics,
    *,
    quality: dict[str, Any],
) -> dict[str, Any]:
    values = {
        "max_pain": metrics.max_pain,
        "net_gex": metrics.net_gex,
        "gamma_flip": metrics.gamma_flip,
        "call_wall": metrics.call_wall,
        "put_wall": metrics.put_wall,
        "atm_iv": metrics.atm_iv,
        "skew_25_delta": metrics.skew_25_delta,
        "realized_volatility": metrics.realized_volatility,
        "vrp": metrics.vrp,
        "activity_intensity": metrics.activity.activity_intensity,
        "volume_oi_ratio": metrics.activity.volume_oi_ratio,
        "near_spot_volume_concentration": metrics.activity.near_spot_volume_concentration,
    }
    evidence = {
        name: {
            "available": metric.available,
            "label": metric.label,
            "reason_codes": list(metric.reason_codes),
            "evidence": dict(metric.evidence),
        }
        for name, metric in values.items()
    }
    evidence["quality"] = quality
    return evidence


def historical_metric_evidence(metrics: HistoricalMetrics) -> dict[str, Any]:
    return {
        name: {
            "available": metric.available,
            "label": metric.label,
            "reason_codes": list(metric.reason_codes),
            "evidence": dict(metric.evidence),
        }
        for name, metric in vars(metrics).items()
    }


def unavailable_quality_evidence(candidate: OptionCandidate) -> dict[str, Any]:
    source_spot = candidate.spot_price
    if source_spot is None or not math.isfinite(float(source_spot)) or source_spot <= 0:
        source_spot = None
    return {
        "source_spot_price": source_spot,
        "provider_spot_price": None,
        "spot_disagreement_ratio": None,
        "latest_contract_trade_at": None,
        "days_to_expiration": None,
        "normalized_call_count": 0,
        "normalized_put_count": 0,
        "distinct_strike_count": 0,
        "open_interest_coverage": 0.0,
        "iv_coverage": 0.0,
        "volume_coverage": 0.0,
        "two_sided_quote_coverage": 0.0,
    }


def quality_evidence(
    observation: ChainObservation,
    *,
    as_of_date: date,
) -> dict[str, Any]:
    contracts = tuple(observation.contracts)
    retained = retain_contracts_for_persistence(
        contracts,
        spot_price=observation.source_spot_price,
    )
    trade_times = [
        contract.last_trade_at
        for contract in retained
        if contract.last_trade_at is not None
    ]
    latest_trade = max(trade_times) if trade_times else None
    total = len(contracts)
    provider_spot = observation.provider_spot_price
    if (
        provider_spot is None
        or not math.isfinite(float(provider_spot))
        or provider_spot <= 0
    ):
        provider_spot = None

    def coverage(predicate: Callable[[object], bool]) -> float:
        if total == 0:
            return 0.0
        return sum(bool(predicate(contract)) for contract in contracts) / total

    evidence: dict[str, Any] = {
        "source_spot_price": observation.source_spot_price,
        "provider_spot_price": provider_spot,
        "spot_disagreement_ratio": None,
        "latest_contract_trade_at": (
            latest_trade.isoformat() if latest_trade is not None else None
        ),
        "days_to_expiration": (observation.expiration - as_of_date).days,
        "normalized_call_count": sum(
            contract.side is OptionSide.CALL for contract in contracts
        ),
        "normalized_put_count": sum(
            contract.side is OptionSide.PUT for contract in contracts
        ),
        "distinct_strike_count": len(
            {
                contract.strike
                for contract in contracts
                if math.isfinite(float(contract.strike)) and contract.strike > 0
            }
        ),
        "open_interest_coverage": coverage(
            lambda contract: (
                contract.open_interest is not None and contract.open_interest >= 0
            )
        ),
        "iv_coverage": coverage(
            lambda contract: (
                contract.implied_volatility is not None
                and math.isfinite(float(contract.implied_volatility))
                and contract.implied_volatility > 0
            )
        ),
        "volume_coverage": coverage(
            lambda contract: contract.volume is not None and contract.volume >= 0
        ),
        "two_sided_quote_coverage": coverage(
            lambda contract: (
                contract.bid is not None
                and math.isfinite(float(contract.bid))
                and contract.bid >= 0
                and contract.ask is not None
                and math.isfinite(float(contract.ask))
                and contract.ask >= 0
            )
        ),
    }
    if provider_spot is not None and observation.source_spot_price > 0:
        evidence["spot_disagreement_ratio"] = (
            abs(float(provider_spot) - observation.source_spot_price)
            / observation.source_spot_price
        )
    return evidence


def quality_warnings(
    observation: ChainObservation,
    *,
    as_of_date: date,
    run_warnings: tuple[str, ...],
    recent_sessions: tuple[date, ...],
) -> tuple[str, ...]:
    evidence = quality_evidence(observation, as_of_date=as_of_date)
    warnings = list(run_warnings)
    disagreement = evidence.get("spot_disagreement_ratio")
    if disagreement is not None and disagreement > 0.02:
        warnings.append("provider_spot_disagreement")
    latest_trade = evidence.get("latest_contract_trade_at")
    if latest_trade is None or date.fromisoformat(latest_trade[:10]) not in set(
        recent_sessions
    ):
        warnings.append("stale_contract_trades")
    return tuple(warnings)
