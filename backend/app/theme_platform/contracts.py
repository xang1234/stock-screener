"""Typed contracts for high-traffic theme/cache payloads."""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class PipelineRunStatusPayload(TypedDict, total=False):
    run_id: str
    task_id: Optional[str]
    status: str
    current_step: Optional[str]
    step_number: int
    total_steps: int
    percent: float
    message: Optional[str]
    ingestion_result: Optional[dict[str, Any]]
    reprocessing_result: Optional[dict[str, Any]]
    extraction_result: Optional[dict[str, Any]]
    metrics_result: Optional[dict[str, Any]]
    alerts_result: Optional[dict[str, Any]]
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]


class MergeActionResult(TypedDict, total=False):
    success: bool
    error: str
    error_code: str
    source_name: str
    target_name: str
    constituents_merged: int
    mentions_merged: int
    warning: str
    idempotency_key: str
    merge_history_id: int
    idempotent_replay: bool


class WarmupHeartbeatState(TypedDict, total=False):
    status: str
    current: int
    total: int
    percent: float
    updated_at: str
    completed_at: str
    minutes: float


class WarmupStateSnapshot(TypedDict, total=False):
    status: str
    count: int
    total: int
    completed_at: str
    error: Optional[str]
