"""Schemas for the Operations job console."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OperationsJobRow(BaseModel):
    task_id: str
    task_name: str
    queue: str | None = None
    market: str | None = None
    state: str
    worker: str | None = None
    age_seconds: float | None = None
    wait_reason: str | None = None
    heartbeat_lag_seconds: float | None = None
    cancel_strategy: str
    progress_mode: str = "indeterminate"
    percent: float | None = None
    current: int | None = None
    total: int | None = None
    message: str | None = None
    cancel_requested_at: str | None = None
    cancel_requested_strategy: str | None = None


class OperationsQueueSummary(BaseModel):
    queue: str
    depth: int = 0
    oldest_age_seconds: float | None = None
    # Responding workers that read this queue; None when no worker answered.
    consumers: int | None = None


class OperationsWorkerStatus(BaseModel):
    worker: str
    status: str
    queues: list[str] = Field(default_factory=list)
    active: int = 0
    reserved: int = 0
    scheduled: int = 0


class OperationsLeaseSnapshot(BaseModel):
    external_fetch_global: dict[str, Any] | None = None
    market_workload: dict[str, dict[str, Any] | None] = Field(default_factory=dict)
    # Held daily pipeline chain claims by market (task_id is the claim token).
    pipeline_chain: dict[str, dict[str, Any]] = Field(default_factory=dict)


class OperationsJobsResponse(BaseModel):
    jobs: list[OperationsJobRow] = Field(default_factory=list)
    queues: list[OperationsQueueSummary] = Field(default_factory=list)
    workers: list[OperationsWorkerStatus] = Field(default_factory=list)
    leases: OperationsLeaseSnapshot = Field(default_factory=OperationsLeaseSnapshot)
    generated_at: str


class OperationsReleaseChainRequest(BaseModel):
    # The claim token the operator saw (uuid4().hex at dispatch). Exact
    # format, because the release script matches ":<token>:" as a substring
    # of "daily_market_pipeline:<token>:<timestamp>": a short token such as
    # "34" would match the timestamp and free a claim it does not own.
    token: str = Field(pattern=r"^[0-9a-f]{32}$")


class OperationsCancelJobRequest(BaseModel):
    # The strategy the operator confirmed; a mismatch returns 409.
    expected_strategy: str | None = None


class OperationsCancelJobResponse(BaseModel):
    status: str
    cancel_strategy: str
    message: str


class CotOperationsPriceCounts(BaseModel):
    exact_or_proxy: int = Field(ge=0)
    partial: int = Field(ge=0)
    unavailable: int = Field(ge=0)


class CotOperationsResponse(BaseModel):
    generated_at: str
    latest_successful_run_id: int | None = None
    latest_failed_run_id: int | None = None
    latest_failed_status: str | None = None
    source_report_date: str | None = None
    source_retrieved_at: str | None = None
    expected_instrument_count: int = Field(ge=0)
    observed_instrument_count: int = Field(ge=0)
    validation_reason_codes: list[str] = Field(default_factory=list)
    price_counts: CotOperationsPriceCounts
    duration_seconds: float | None = Field(default=None, ge=0)
    retry_count: int = Field(ge=0)
    publication_age_days: int | None = None
    stale: bool
