"""Presentation helpers for runtime activity status responses."""

from __future__ import annotations

from typing import Any

from .runtime_activity_contract import (
    ACTIVE_ACTIVITY_STATUSES,
    RUNTIME_STAGE_SEQUENCE,
    RuntimeActivityRecord,
    bootstrap_stage_metadata,
    stage_index,
)

WARNING_ACTIVITY_STATUSES = frozenset({"failed", "stale", "stuck"})


def _bootstrap_progress_percent(record: RuntimeActivityRecord | None) -> float:
    if record is None:
        return 0.0
    if record.stage_key not in RUNTIME_STAGE_SEQUENCE:
        return 0.0
    current_stage_index = stage_index(record.stage_key)
    raw_percent = record.percent
    if raw_percent is not None:
        stage_fraction = max(0.0, min(float(raw_percent), 100.0)) / 100.0
    elif record.status == "completed":
        stage_fraction = 1.0
    else:
        stage_fraction = 0.0
    return round(((current_stage_index + stage_fraction) / len(RUNTIME_STAGE_SEQUENCE)) * 100.0, 2)


def build_runtime_activity_status(
    *,
    bootstrap_status,
    bootstrap_run: dict[str, Any],
    market_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    enabled_markets = list(bootstrap_status.enabled_markets)
    primary_market = bootstrap_status.primary_market
    activity_records = [
        RuntimeActivityRecord.from_payload(payload)
        for payload in market_payloads
    ]
    active_markets = [
        record.market
        for record in activity_records
        if record.status in ACTIVE_ACTIVITY_STATUSES
    ]
    has_warning = any(
        record.status in WARNING_ACTIVITY_STATUSES
        for record in activity_records
    )
    summary_status = "warning" if has_warning else ("active" if active_markets else "idle")

    bootstrap_records = [
        record for record in activity_records if record.lifecycle == "bootstrap"
    ]
    primary_record = next(
        (record for record in bootstrap_records if record.market == primary_market),
        None,
    )
    secondary_active = [
        record.market
        for record in bootstrap_records
        if record.market != primary_market and record.active_bootstrap
    ]
    secondary_active_record = next(
        (
            record
            for record in bootstrap_records
            if record.market != primary_market and record.active_bootstrap
        ),
        None,
    )
    bootstrap_current = None
    bootstrap_total = None

    if bootstrap_status.bootstrap_state == "ready" and secondary_active_record:
        bootstrap_progress_mode = secondary_active_record.progress_mode or "indeterminate"
        bootstrap_percent = (
            _bootstrap_progress_percent(secondary_active_record)
            if bootstrap_progress_mode == "determinate"
            else None
        )
        bootstrap_stage = secondary_active_record.stage_label
        bootstrap_message = secondary_active_record.message or "Additional market loading continues."
        bootstrap_current = secondary_active_record.current
        bootstrap_total = secondary_active_record.total
    elif bootstrap_status.bootstrap_state == "ready":
        bootstrap_progress_mode = "determinate"
        bootstrap_percent = 100.0
        bootstrap_stage = primary_record.stage_label if primary_record else None
        bootstrap_message = (
            "Primary market is ready."
            if not secondary_active
            else "Primary market is ready while additional market loading continues."
        )
    elif any(record.progress_mode == "determinate" for record in bootstrap_records):
        focus_record = next(
            (record for record in bootstrap_records if record.status in ACTIVE_ACTIVITY_STATUSES),
            primary_record,
        )
        bootstrap_progress_mode = "determinate"
        bootstrap_percent = round(
            sum(_bootstrap_progress_percent(record) for record in bootstrap_records)
            / max(len(bootstrap_records), 1),
            2,
        )
        bootstrap_stage = focus_record.stage_label if focus_record else None
        bootstrap_message = focus_record.message if focus_record else "Bootstrap queued."
    else:
        bootstrap_progress_mode = "indeterminate"
        bootstrap_percent = None
        bootstrap_stage = primary_record.stage_label if primary_record else None
        bootstrap_message = (
            primary_record.message
            if primary_record is not None
            else "Bootstrap queued."
        )

    background_warning = None
    if len(enabled_markets) > 1 and (
        bootstrap_status.bootstrap_state == "running" or bool(secondary_active)
    ):
        background_warning = (
            "Additional enabled markets are still loading in the background."
        )

    return {
        "bootstrap": {
            "state": bootstrap_status.bootstrap_state,
            "app_ready": not bootstrap_status.bootstrap_required,
            "primary_market": primary_market,
            "enabled_markets": enabled_markets,
            "queue_state": bootstrap_run.get("queue_state") or "queued",
            "task_id": bootstrap_run.get("primary_task_id"),
            "market_task_ids": bootstrap_run.get("market_task_ids") or {},
            "current_stage": bootstrap_stage,
            "progress_mode": bootstrap_progress_mode,
            "percent": bootstrap_percent,
            "current": bootstrap_current,
            "total": bootstrap_total,
            "message": bootstrap_message,
            "background_warning": background_warning,
            "stages": bootstrap_stage_metadata(),
        },
        "summary": {
            "active_market_count": len(active_markets),
            "active_markets": active_markets,
            "status": summary_status,
        },
        "markets": market_payloads,
    }
