"""Point-in-time assembly for the persisted opportunity-state projection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import TYPE_CHECKING

from app.analysis.patterns.config import SetupEngineParameters
from app.domain.scanning.default_filters import resolve_default_scan_filters
from app.domain.scanning.opportunity_state import (
    EvidenceValue,
    InvalidationEvidence,
    LeadershipEvidence,
    OpportunityEvidence,
    ProvenanceEvidence,
    RiskEvidence,
    StructureEvidence,
    TradabilityEvidence,
    TrendEvidence,
    evaluate_opportunity_state,
    normalize_event_date,
    serialize_opportunity_projection,
)
from app.services.security_master_service import SecurityMasterResolver

if TYPE_CHECKING:
    from app.scanners.base_screener import StockData


def read_liquidity_evidence(market: object, avg_dollar_volume: object) -> EvidenceValue:
    """Expose the existing local-currency gate without evaluating opportunity policy."""
    floor = resolve_default_scan_filters(_normalized_market(market)).get("minVolume")
    volume = _finite_float(avg_dollar_volume)
    available = floor is not None and volume is not None
    return EvidenceValue(volume >= floor if available else None, available)


def build_opportunity_projection(
    result: Mapping[str, object],
    stock_data: StockData,
    parameters: SetupEngineParameters,
) -> dict[str, object]:
    """Assemble current scan evidence and evaluate the policy exactly once."""
    setup = _mapping_or_none(result.get("setup_engine"))
    event = normalize_event_date(
        stock_data.next_earnings_date,
        key_present=stock_data.event_calendar_available,
    )
    market = _normalized_market(stock_data.market)
    as_of_date = _last_frame_date(stock_data.price_data)
    liquidity_floor = resolve_default_scan_filters(market).get("minVolume")
    invalidation_flags = _invalidation_flags(setup)
    pattern_primary, pattern_primary_available = _optional_text_field(
        setup,
        "pattern_primary",
    )

    evidence = OpportunityEvidence(
        provenance=ProvenanceEvidence(
            market=market,
            mic=SecurityMasterResolver.resolve_exchange_mic(market, stock_data.exchange),
            as_of_date=as_of_date,
            benchmark_symbol=_text_or_none(stock_data.benchmark_symbol),
            benchmark_as_of_date=_last_frame_date(stock_data.benchmark_data),
        ),
        leadership=LeadershipEvidence(
            benchmark_relative_return_65d=_number_from(setup, "rs_vs_spy_65d"),
            rs_rating_1m=_finite_float(result.get("rs_rating_1m")),
            rs_rating_3m=_finite_float(result.get("rs_rating_3m")),
            rs_line_new_high=_bool_from(setup, "rs_line_new_high"),
            rs_line_blue_dot=_bool_from(setup, "rs_line_blue_dot"),
        ),
        trend=TrendEvidence(
            stage=_integer_or_none(result.get("stage")),
            ma_alignment=_bool_or_none(result.get("ma_alignment")),
            invalidation=EvidenceValue(
                value=invalidation_flags,
                available=invalidation_flags is not None,
            ),
        ),
        structure=StructureEvidence(
            setup_payload_available=setup is not None,
            primary_pattern=EvidenceValue(
                value=pattern_primary,
                available=pattern_primary_available,
            ),
            squeeze=_bool_from(setup, "bb_squeeze"),
            tight_closes_count=_integer_from(setup, "tight_closes_count"),
            quiet_days_count=_integer_from(setup, "quiet_days_10d"),
            volume_vs_50d=_number_from(setup, "volume_vs_50d"),
            volume_dry_up_max=parameters.volume_vs_50d_max_for_ready,
        ),
        tradability=TradabilityEvidence(
            liquidity=read_liquidity_evidence(market, result.get("avg_dollar_volume")),
            feature_status=_text_from(result, "data_status"),
            is_scannable=_bool_or_none(result.get("is_scannable")),
        ),
        risk=RiskEvidence(
            event_risk=EvidenceValue(
                value=(
                    _event_is_inside_window(
                        event.value,
                        as_of_date,
                        parameters.earnings_soon_window_days,
                    )
                    if event.available
                    else None
                ),
                available=event.available,
            ),
            setup_ready=_bool_from(setup, "setup_ready"),
            in_early_zone=_bool_from(setup, "in_early_zone"),
            extended=_bool_from(setup, "extended_from_pivot"),
        ),
    )
    assessment = evaluate_opportunity_state(evidence).with_metrics(
        {"liquidity_floor_local": liquidity_floor}
    )
    return serialize_opportunity_projection(assessment)


def build_data_limited_projection(
    result: Mapping[str, object],
    stock_data: StockData,
    reason: str,
) -> dict[str, object]:
    """Build an explicit structured fallback while preserving row identity."""
    market = _normalized_market(stock_data.market)
    as_of_date = _last_frame_date(stock_data.price_data)
    event = normalize_event_date(
        stock_data.next_earnings_date,
        key_present=stock_data.event_calendar_available,
    )
    liquidity_floor = resolve_default_scan_filters(market).get("minVolume")

    evidence = OpportunityEvidence(
        provenance=ProvenanceEvidence(
            market=market,
            mic=SecurityMasterResolver.resolve_exchange_mic(market, stock_data.exchange),
            as_of_date=as_of_date,
            benchmark_symbol=_text_or_none(stock_data.benchmark_symbol),
            benchmark_as_of_date=_last_frame_date(stock_data.benchmark_data),
        ),
        leadership=LeadershipEvidence(None, None, None, None, None),
        trend=TrendEvidence(None, None, EvidenceValue(None, False)),
        structure=StructureEvidence(
            False,
            EvidenceValue(None, False),
            None,
            None,
            None,
            None,
            None,
        ),
        tradability=TradabilityEvidence(
            liquidity=read_liquidity_evidence(market, result.get("avg_dollar_volume")),
            feature_status=_text_from(result, "data_status"),
            is_scannable=_bool_or_none(result.get("is_scannable")),
        ),
        risk=RiskEvidence(EvidenceValue(None, event.available), None, None, None),
    )
    assessment = (
        evaluate_opportunity_state(evidence)
        .with_action_reasons((reason,))
        .with_metrics({"liquidity_floor_local": liquidity_floor})
    )
    return serialize_opportunity_projection(assessment)


def _mapping_or_none(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    if not all(isinstance(key, str) for key in value):
        return None
    return dict(value)


def _mapping_or_empty(value: object) -> dict[str, object]:
    return _mapping_or_none(value) or {}


def _normalized_market(value: object) -> str | None:
    normalized = str(value or "").strip().upper()
    return normalized or None


def _text_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _text_from(source: Mapping[str, object] | None, key: str) -> str | None:
    return _text_or_none(source.get(key)) if source is not None else None


def _optional_text_field(
    source: Mapping[str, object] | None,
    key: str,
) -> tuple[str | None, bool]:
    if source is None or key not in source:
        return None, False
    raw = source[key]
    if raw is None:
        return None, True
    normalized = _text_or_none(raw)
    return normalized, normalized is not None


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _number_from(source: Mapping[str, object] | None, key: str) -> float | None:
    return _finite_float(source.get(key)) if source is not None else None


def _bool_or_none(value: object) -> bool | None:
    return value if type(value) is bool else None


def _bool_from(source: Mapping[str, object] | None, key: str) -> bool | None:
    return _bool_or_none(source.get(key)) if source is not None else None


def _integer_or_none(value: object) -> int | None:
    numeric = _finite_float(value)
    if numeric is None or not numeric.is_integer():
        return None
    return int(numeric)


def _integer_from(source: Mapping[str, object] | None, key: str) -> int | None:
    return _integer_or_none(source.get(key)) if source is not None else None


def _last_frame_date(frame: object) -> date | None:
    if frame is None or getattr(frame, "empty", True):
        return None
    try:
        value = frame.index[-1]
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if callable(to_pydatetime):
        converted = to_pydatetime()
        if isinstance(converted, datetime):
            return converted.date()
        if isinstance(converted, date):
            return converted
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _event_is_inside_window(
    event_date: date | None,
    as_of_date: date | None,
    window_days: float,
) -> bool | None:
    if event_date is None:
        return False
    if as_of_date is None:
        return None
    days_until_event = (event_date - as_of_date).days
    return 0 <= days_until_event <= window_days


def _invalidation_flags(
    setup: Mapping[str, object] | None,
) -> tuple[InvalidationEvidence, ...] | None:
    if setup is None:
        return None
    explain = setup.get("explain")
    if not isinstance(explain, Mapping):
        return None
    raw_flags = explain.get("invalidation_flags")
    if not isinstance(raw_flags, list):
        return None

    flags: list[InvalidationEvidence] = []
    for raw_flag in raw_flags:
        if not isinstance(raw_flag, Mapping):
            return None
        code = _text_or_none(raw_flag.get("code"))
        is_hard = raw_flag.get("is_hard")
        if code is None or type(is_hard) is not bool:
            return None
        flags.append(InvalidationEvidence(code=code, is_hard=is_hard))
    return tuple(flags)


__all__ = ["build_data_limited_projection", "build_opportunity_projection", "read_liquidity_evidence"]
