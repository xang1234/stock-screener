"""Typed aggregate-only contract for portable options history."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.options_analytics.models import CandidateKind, ObservationState


class OptionsHistoryObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    external_source_feature_run_key: str = Field(min_length=1)
    as_of_date: date
    schema_version: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    published_at: datetime | None = None
    risk_free_rate: float | None = None
    run_assumptions: dict[str, object] = Field(default_factory=dict)
    symbol: str = Field(min_length=1)
    candidate_kind: CandidateKind
    candidate_rank: int | None = Field(default=None, ge=1)
    leader_rank: int | None = Field(default=None, ge=1)
    spot_price: float | None = None
    expiration: date | None = None
    observation_state: ObservationState
    core_valid: bool
    observation_at: datetime | None = None
    max_pain: float | None = None
    net_gex: float | None = None
    gamma_flip: float | None = None
    call_wall: float | None = None
    put_wall: float | None = None
    atm_iv: float | None = None
    skew_25_delta: float | None = None
    realized_volatility: float | None = None
    vrp: float | None = None
    activity_intensity: float | None = None
    activity_rank: int | None = Field(default=None, ge=1)
    call_open_interest: int | None = Field(default=None, ge=0)
    put_open_interest: int | None = Field(default=None, ge=0)
    call_volume: int | None = Field(default=None, ge=0)
    put_volume: int | None = Field(default=None, ge=0)
    volume_oi_ratio: float | None = Field(default=None, ge=0)
    near_spot_volume_concentration: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    short_history_observation_count: int = Field(ge=0)
    iv_history_observation_count: int = Field(ge=0)
    lifetime_observation_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    evidence: dict[str, object] = Field(default_factory=dict)
    assumptions: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        canonical = value.strip().upper()
        if not canonical:
            raise ValueError("symbol is required")
        return canonical

    @property
    def identity(self) -> tuple[str, str]:
        return self.external_source_feature_run_key, self.symbol

    @property
    def run_identity(self) -> tuple[date, str, str]:
        return self.as_of_date, self.schema_version, self.provider


class OptionsHistoryBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str
    calculation_version: str
    market: str
    exported_at: datetime
    observations: tuple[OptionsHistoryObservation, ...]
    payload_checksum: str = Field(min_length=64, max_length=64)


__all__ = ["OptionsHistoryBundle", "OptionsHistoryObservation"]
