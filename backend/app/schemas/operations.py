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


class OperationsQueueSummary(BaseModel):
    queue: str
    depth: int = 0
    oldest_age_seconds: float | None = None


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


class OperationsJobsResponse(BaseModel):
    jobs: list[OperationsJobRow] = Field(default_factory=list)
    queues: list[OperationsQueueSummary] = Field(default_factory=list)
    workers: list[OperationsWorkerStatus] = Field(default_factory=list)
    leases: OperationsLeaseSnapshot = Field(default_factory=OperationsLeaseSnapshot)
    generated_at: str


class OperationsCancelJobResponse(BaseModel):
    status: str
    cancel_strategy: str
    message: str

