"""Strict public contracts shared by live and static options reads."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class StaticOptionsSymbolReference(_StrictModel):
    key: str = Field(min_length=1)
    path: str = Field(min_length=1)


class StaticOptionsManifest(_StrictModel):
    schema_version: Literal["static-options-v1"]
    data_schema_version: Literal["options-analytics-v1"]
    calculation_version: Literal["options-analytics-v1"]
    published_run_id: int
    source_feature_run_id: int | None = None
    source_as_of_date: date
    market: Literal["US"]
    provider: str = Field(min_length=1)
    generated_at: datetime
    latest_observation_at: datetime | None = None
    coverage: float = Field(ge=0, le=1)
    stale: bool
    stale_relative_to_equity: bool
    equity_feature_run_id: int | None = None
    equity_as_of_date: date | None = None
    reason_codes: list[str]
    command_center_path: str = Field(min_length=1)
    symbols: dict[str, StaticOptionsSymbolReference] = Field(max_length=80)

    @model_validator(mode="after")
    def validate_manifest_consistency(self):
        if self.stale_relative_to_equity and (
            not self.stale or "stale_relative_to_equity" not in self.reason_codes
        ):
            raise ValueError("Invalid stale options metadata")
        if any(symbol != symbol.upper() or not symbol for symbol in self.symbols):
            raise ValueError("Options symbol-map keys must be uppercase")
        paths = [entry.path for entry in self.symbols.values()]
        if len(paths) != len(set(paths)):
            raise ValueError("Options symbol-detail paths must be unique")
        return self


class OptionsMetricResponse(_StrictModel):
    available: bool
    value: float | None = None
    label: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_availability(self):
        if self.available != (self.value is not None):
            raise ValueError("Metric availability must match value presence")
        return self


class OptionsMetricsResponse(_StrictModel):
    max_pain: OptionsMetricResponse
    net_gex: OptionsMetricResponse
    gamma_flip: OptionsMetricResponse
    call_wall: OptionsMetricResponse
    put_wall: OptionsMetricResponse
    atm_iv: OptionsMetricResponse
    skew_25_delta: OptionsMetricResponse
    realized_volatility: OptionsMetricResponse
    vrp: OptionsMetricResponse
    activity_intensity: OptionsMetricResponse
    call_put_volume_ratio: OptionsMetricResponse
    volume_oi_ratio: OptionsMetricResponse
    near_spot_volume_concentration: OptionsMetricResponse
    near_spot_open_interest_concentration: OptionsMetricResponse
    highest_contract_activity_ratio: OptionsMetricResponse


class OptionsHistoricalMetricsResponse(_StrictModel):
    iv_percentile: OptionsMetricResponse
    iv_rank: OptionsMetricResponse
    max_pain_change_5: OptionsMetricResponse
    net_gex_change_5: OptionsMetricResponse
    gamma_flip_change_5: OptionsMetricResponse
    atm_iv_change_5: OptionsMetricResponse
    skew_25_delta_change_5: OptionsMetricResponse
    realized_volatility_change_5: OptionsMetricResponse
    vrp_change_5: OptionsMetricResponse
    activity_intensity_change_5: OptionsMetricResponse


class OptionsQualityEvidenceResponse(_StrictModel):
    source_spot_price: float | None = Field(gt=0)
    provider_spot_price: float | None
    spot_disagreement_ratio: float | None = Field(ge=0)
    latest_contract_trade_at: datetime | None
    days_to_expiration: int | None = Field(ge=0)
    normalized_call_count: int = Field(ge=0)
    normalized_put_count: int = Field(ge=0)
    distinct_strike_count: int = Field(ge=0)
    open_interest_coverage: float = Field(ge=0, le=1)
    iv_coverage: float = Field(ge=0, le=1)
    volume_coverage: float = Field(ge=0, le=1)
    two_sided_quote_coverage: float = Field(ge=0, le=1)


class OptionsCommandCenterItemResponse(_StrictModel):
    symbol: str
    source_badges: list[str]
    candidate_rank: int | None = None
    leader_rank: int | None = None
    state: str
    core_valid: bool
    spot_price: float | None = None
    expiration: date | None = None
    observation_at: datetime | None = None
    call_open_interest: int | None = None
    put_open_interest: int | None = None
    call_volume: int | None = None
    put_volume: int | None = None
    activity_rank: int | None = None
    short_history_observation_count: int
    iv_history_observation_count: int
    lifetime_observation_count: int
    retry_count: int
    quality_evidence: OptionsQualityEvidenceResponse
    metrics: OptionsMetricsResponse
    historical_metrics: OptionsHistoricalMetricsResponse
    assumptions: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class OptionsRunMetadataResponse(_StrictModel):
    schema_version: str
    calculation_version: str
    run_id: int
    source_feature_run_id: int | None = None
    source_as_of_date: date
    market: str
    provider: str
    started_at: datetime
    published_at: datetime | None = None
    latest_observation_at: datetime | None = None
    expected_count: int
    current_count: int
    continuity_count: int
    completed_count: int
    core_valid_current_count: int
    failed_count: int
    retried_count: int
    coverage: float
    stale: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    assumptions: dict[str, Any] = Field(default_factory=dict)


class OptionsCommandCenterResponse(OptionsRunMetadataResponse):
    items: list[OptionsCommandCenterItemResponse] = Field(max_length=80)

    @classmethod
    def from_run(cls, run: Any, *, stale: bool = False):
        items = [
            _item_response(item)
            for item in run.items
            if item.candidate_kind == "current"
        ]
        items.sort(key=_item_order_key)
        return cls(
            **_run_metadata(run, stale=stale),
            items=items,
        )


class OptionsStrikePointResponse(_StrictModel):
    strike: float
    call_open_interest: int | None = None
    put_open_interest: int | None = None
    call_volume: int | None = None
    put_volume: int | None = None
    call_iv: float | None = None
    put_iv: float | None = None
    estimated_call_gex: float | None = None
    estimated_put_gex: float | None = None


class OptionsHistoryPointResponse(_StrictModel):
    as_of_date: date
    state: str
    max_pain: float | None = None
    net_gex: float | None = None
    gamma_flip: float | None = None
    atm_iv: float | None = None
    skew_25_delta: float | None = None
    realized_volatility: float | None = None
    vrp: float | None = None
    activity_intensity: float | None = None
    iv_percentile: float | None = None
    iv_rank: float | None = None
    max_pain_change_5: float | None = None
    net_gex_change_5: float | None = None
    gamma_flip_change_5: float | None = None
    atm_iv_change_5: float | None = None
    skew_25_delta_change_5: float | None = None
    realized_volatility_change_5: float | None = None
    vrp_change_5: float | None = None
    activity_intensity_change_5: float | None = None


class OptionsSymbolDetailResponse(OptionsRunMetadataResponse):
    item: OptionsCommandCenterItemResponse
    strike_points: list[OptionsStrikePointResponse]
    history: list[OptionsHistoryPointResponse]

    @classmethod
    def from_result(cls, result: Any, *, stale: bool = False):
        return cls(
            **_run_metadata(result.run, stale=stale),
            item=_item_response(result.item),
            strike_points=[
                OptionsStrikePointResponse.model_validate(
                    {
                        field: getattr(point, field)
                        for field in OptionsStrikePointResponse.model_fields
                    }
                )
                for point in sorted(
                    result.item.strike_points, key=lambda row: row.strike
                )
            ],
            history=[
                OptionsHistoryPointResponse(
                    as_of_date=row.as_of_date,
                    state=row.observation_state,
                    max_pain=row.max_pain,
                    net_gex=row.net_gex,
                    gamma_flip=row.gamma_flip,
                    atm_iv=row.atm_iv,
                    skew_25_delta=row.skew_25_delta,
                    realized_volatility=row.realized_volatility,
                    vrp=row.vrp,
                    activity_intensity=row.activity_intensity,
                    iv_percentile=row.iv_percentile,
                    iv_rank=row.iv_rank,
                    max_pain_change_5=row.max_pain_change_5,
                    net_gex_change_5=row.net_gex_change_5,
                    gamma_flip_change_5=row.gamma_flip_change_5,
                    atm_iv_change_5=row.atm_iv_change_5,
                    skew_25_delta_change_5=row.skew_25_delta_change_5,
                    realized_volatility_change_5=(row.realized_volatility_change_5),
                    vrp_change_5=row.vrp_change_5,
                    activity_intensity_change_5=(row.activity_intensity_change_5),
                )
                for row in result.history
            ],
        )


class OptionsRunDiagnosticsResponse(OptionsRunMetadataResponse):
    status: str
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_run(cls, run: Any):
        return cls(
            **_run_metadata(run),
            status=run.status,
            warnings=list(run.warnings_json or []),
            diagnostics=dict(run.diagnostics_json or {}),
        )


class OptionsRefreshRequest(_StrictModel):
    source_run_id: int | None = Field(default=None, gt=0)
    force: bool = False


class OptionsRefreshAcceptedResponse(_StrictModel):
    status: str = "accepted"
    task_id: str
    run_id: int | None = None
    source_run_id: int | None = None


_LABELS = {
    "max_pain": "Max Pain",
    "net_gex": "Estimated Net GEX",
    "gamma_flip": "Estimated Gamma Flip",
    "call_wall": "Estimated Call Wall",
    "put_wall": "Estimated Put Wall",
    "atm_iv": "ATM IV",
    "skew_25_delta": "25-Delta Skew",
    "realized_volatility": "Realized Volatility",
    "vrp": "Volatility Risk Premium",
    "activity_intensity": "Activity Intensity",
    "call_put_volume_ratio": "Call / Put Volume",
    "volume_oi_ratio": "Volume / Open Interest",
    "near_spot_volume_concentration": "Near-Spot Volume Concentration",
    "near_spot_open_interest_concentration": "Near-Spot Open Interest Concentration",
    "highest_contract_activity_ratio": "Highest Contract Activity Ratio",
}

_HISTORICAL_LABELS = {
    "iv_percentile": "ATM IV Percentile",
    "iv_rank": "ATM IV Rank",
    "max_pain_change_5": "5-Observation Max Pain Change",
    "net_gex_change_5": "5-Observation Net GEX Change",
    "gamma_flip_change_5": "5-Observation Gamma Flip Change",
    "atm_iv_change_5": "5-Observation ATM IV Change",
    "skew_25_delta_change_5": "5-Observation 25-Delta Skew Change",
    "realized_volatility_change_5": "5-Observation Realized Volatility Change",
    "vrp_change_5": "5-Observation Volatility Risk Premium Change",
    "activity_intensity_change_5": "5-Observation Activity Intensity Change",
}


def _metric(item: Any, name: str) -> OptionsMetricResponse:
    value = getattr(item, name)
    details = dict((item.evidence_json or {}).get(name) or {})
    available = value is not None and details.get("available", True)
    reasons = list(details.get("reason_codes") or [])
    if not available and not reasons:
        reasons = list(item.reasons_json or []) or ["metric_unavailable"]
    return OptionsMetricResponse(
        available=available,
        value=value if available else None,
        label=(details.get("label") or _LABELS.get(name) or _HISTORICAL_LABELS[name]),
        reason_codes=reasons,
        evidence=dict(details.get("evidence") or {}),
    )


def _item_response(item: Any) -> OptionsCommandCenterItemResponse:
    reasons = list(item.reasons_json or [])
    state = item.observation_state
    if state == "available" and not item.core_valid:
        state = "insufficient_quality"
    elif state == "available" and "building_history" in reasons:
        state = "building_history"
    badges = []
    if item.candidate_rank is not None:
        badges.append("candidate")
    if item.leader_rank is not None:
        badges.append("leader")
    return OptionsCommandCenterItemResponse(
        symbol=item.security_symbol,
        source_badges=badges,
        candidate_rank=item.candidate_rank,
        leader_rank=item.leader_rank,
        state=state,
        core_valid=bool(item.core_valid),
        spot_price=item.spot_price,
        expiration=item.expiration,
        observation_at=item.observation_at,
        call_open_interest=item.call_open_interest,
        put_open_interest=item.put_open_interest,
        call_volume=item.call_volume,
        put_volume=item.put_volume,
        activity_rank=item.activity_rank,
        short_history_observation_count=item.short_history_observation_count,
        iv_history_observation_count=item.iv_history_observation_count,
        lifetime_observation_count=item.lifetime_observation_count,
        retry_count=item.retry_count,
        quality_evidence=dict((item.evidence_json or {}).get("quality") or {}),
        metrics=OptionsMetricsResponse(
            **{name: _metric(item, name) for name in _LABELS}
        ),
        historical_metrics=OptionsHistoricalMetricsResponse(
            **{name: _metric(item, name) for name in _HISTORICAL_LABELS}
        ),
        assumptions=dict(item.assumptions_json or {}),
        warnings=list(item.warnings_json or []),
        reason_codes=reasons,
    )


def _run_metadata(run: Any, *, stale: bool = False) -> dict[str, Any]:
    observations = [item.observation_at for item in run.items if item.observation_at]
    return {
        "schema_version": run.schema_version,
        "calculation_version": run.calculation_version,
        "run_id": run.id,
        "source_feature_run_id": run.source_feature_run_id,
        "source_as_of_date": run.as_of_date,
        "market": run.market,
        "provider": run.provider,
        "started_at": run.created_at,
        "published_at": run.published_at,
        "latest_observation_at": max(observations) if observations else None,
        "expected_count": run.expected_count,
        "current_count": run.current_count,
        "continuity_count": run.continuity_count,
        "completed_count": run.completed_count,
        "core_valid_current_count": run.core_valid_current_count,
        "failed_count": run.failed_count,
        "retried_count": run.retried_count,
        "coverage": run.coverage,
        "stale": stale,
        "reason_codes": list(run.warnings_json or []),
        "assumptions": dict(run.assumptions_json or {}),
    }


def _item_order_key(item: OptionsCommandCenterItemResponse) -> tuple[int, str]:
    ranks = [
        rank for rank in (item.candidate_rank, item.leader_rank) if rank is not None
    ]
    return (min(ranks) if ranks else 10_000, item.symbol)
