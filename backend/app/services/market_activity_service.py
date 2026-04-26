"""Runtime market activity persistence and aggregation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from ..models.app_settings import AppSetting
from ..services.runtime_preferences_service import get_runtime_bootstrap_status
from ..wiring.bootstrap import get_data_fetch_lock

RUNTIME_ACTIVITY_CATEGORY = "runtime_activity"
MARKET_ACTIVITY_KEY_PREFIX = "runtime.activity.market."

STAGE_SEQUENCE = (
    "universe",
    "prices",
    "fundamentals",
    "breadth",
    "groups",
    "scan",
)

STAGE_LABELS = {
    "universe": "Universe Refresh",
    "prices": "Price Refresh",
    "fundamentals": "Fundamentals Refresh",
    "breadth": "Breadth Calculation",
    "groups": "Group Rankings",
    "snapshot": "Feature Snapshot",
    "scan": "Scan",
}

DEFAULT_LIFECYCLE_BY_STAGE = {
    "universe": "weekly_refresh",
    "prices": "daily_refresh",
    "fundamentals": "weekly_refresh",
    "breadth": "daily_refresh",
    "groups": "daily_refresh",
    "snapshot": "daily_refresh",
    "scan": "daily_refresh",
}

ACTIVE_STATUSES = {"queued", "running"}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _activity_key(market: str) -> str:
    return f"{MARKET_ACTIVITY_KEY_PREFIX}{str(market).upper()}"


def _stage_label(stage_key: str | None) -> str | None:
    if stage_key is None:
        return None
    return STAGE_LABELS.get(stage_key, str(stage_key).replace("_", " ").title())


def _default_lifecycle(stage_key: str | None) -> str:
    if stage_key is None:
        return "daily_refresh"
    return DEFAULT_LIFECYCLE_BY_STAGE.get(stage_key, "daily_refresh")


def _get_setting(db: Session, key: str) -> AppSetting | None:
    return db.query(AppSetting).filter(AppSetting.key == key).first()


def _load_market_activity(db: Session, market: str) -> dict[str, Any] | None:
    setting = _get_setting(db, _activity_key(market))
    if setting is None:
        return None
    try:
        payload = json.loads(setting.value)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _save_market_activity(
    db: Session,
    market: str,
    payload: dict[str, Any],
    *,
    preserve_existing_statuses: set[str] | None = None,
) -> dict[str, Any]:
    key = _activity_key(market)
    setting = _get_setting(db, key)
    existing_payload = None
    if setting is not None:
        try:
            existing_payload = json.loads(setting.value)
        except (json.JSONDecodeError, TypeError):
            existing_payload = None
    if preserve_existing_statuses and isinstance(existing_payload, dict):
        existing_status = existing_payload.get("status")
        if existing_status in preserve_existing_statuses:
            payload_status = payload.get("status")
            same_task = existing_payload.get("task_id") == payload.get("task_id")
            same_stage = existing_payload.get("stage_key") == payload.get("stage_key")
            same_owner = same_task and same_stage

            if existing_status == "running":
                if payload_status == "queued" or (
                    payload_status != "failed" and not same_owner
                ):
                    return existing_payload
                if payload_status == "failed" and not same_owner:
                    return existing_payload
            elif existing_status == "completed":
                if payload_status == "failed":
                    pass
                else:
                    incoming_new_cycle = payload_status in {"queued", "running"} and not same_owner
                    if not incoming_new_cycle:
                        return existing_payload
            elif existing_status == "failed":
                incoming_new_cycle = payload_status in {"queued", "running"} and not same_owner
                if incoming_new_cycle:
                    existing_stage_index = _stage_index(existing_payload.get("stage_key"))
                    payload_stage_index = _stage_index(payload.get("stage_key"))
                    lifecycle_changed = existing_payload.get("lifecycle") != payload.get("lifecycle")
                    incoming_new_cycle = lifecycle_changed or payload_stage_index <= existing_stage_index
                if payload_status == "failed" and same_owner:
                    pass
                elif not incoming_new_cycle:
                    return existing_payload
    encoded = json.dumps(payload)
    if setting is None:
        setting = AppSetting(
            key=key,
            value=encoded,
            category=RUNTIME_ACTIVITY_CATEGORY,
            description=f"Latest runtime activity state for {market.upper()}",
        )
        db.add(setting)
    else:
        setting.value = encoded
        setting.category = RUNTIME_ACTIVITY_CATEGORY
        setting.description = f"Latest runtime activity state for {market.upper()}"
    db.commit()
    return payload


def _resolve_progress_percent(
    percent: float | None,
    current: int | None,
    total: int | None,
) -> float | None:
    if percent is not None:
        return float(percent)
    if current is None or total in (None, 0):
        return None
    return round((float(current) / float(total)) * 100.0, 1)


def _progress_mode(
    status: str | None,
    percent: float | None,
    current: int | None = None,
    total: int | None = None,
) -> str:
    if _resolve_progress_percent(percent, current, total) is not None or status in {"completed", "idle"}:
        return "determinate"
    return "indeterminate"


def _stage_index(stage_key: str | None) -> int:
    if stage_key in STAGE_SEQUENCE:
        return STAGE_SEQUENCE.index(stage_key)
    return len(STAGE_SEQUENCE)


def _activity_payload(
    *,
    market: str,
    stage_key: str | None,
    lifecycle: str | None,
    status: str,
    task_name: str | None = None,
    task_id: str | None = None,
    percent: float | None = None,
    current: int | None = None,
    total: int | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    stage_label = _stage_label(stage_key)
    resolved_lifecycle = lifecycle or _default_lifecycle(stage_key)
    resolved_message = message
    resolved_percent = _resolve_progress_percent(percent, current, total)
    if not resolved_message and stage_label:
        action = {
            "queued": "Queued",
            "running": "Running",
            "completed": "Completed",
            "failed": "Failed",
        }.get(status, "Updated")
        resolved_message = f"{action} {stage_label.lower()}"
    return {
        "market": str(market).upper(),
        "lifecycle": resolved_lifecycle,
        "stage_key": stage_key,
        "stage_label": stage_label,
        "status": status,
        "progress_mode": _progress_mode(status, resolved_percent, current, total),
        "percent": resolved_percent,
        "current": current,
        "total": total,
        "message": resolved_message,
        "task_name": task_name,
        "task_id": task_id,
        "updated_at": _utcnow_iso(),
    }


def mark_market_activity_queued(
    db: Session,
    *,
    market: str,
    stage_key: str,
    lifecycle: str | None = None,
    task_name: str | None = None,
    task_id: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    return _save_market_activity(
        db,
        market,
        _activity_payload(
            market=market,
            stage_key=stage_key,
            lifecycle=lifecycle,
            status="queued",
            task_name=task_name,
            task_id=task_id,
            message=message,
        ),
        preserve_existing_statuses={"running", "completed", "failed"},
    )


def mark_market_activity_started(
    db: Session,
    *,
    market: str,
    stage_key: str,
    lifecycle: str | None = None,
    task_name: str | None = None,
    task_id: str | None = None,
    percent: float | None = None,
    current: int | None = None,
    total: int | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    return _save_market_activity(
        db,
        market,
        _activity_payload(
            market=market,
            stage_key=stage_key,
            lifecycle=lifecycle,
            status="running",
            task_name=task_name,
            task_id=task_id,
            percent=percent,
            current=current,
            total=total,
            message=message,
        ),
        preserve_existing_statuses={"running", "completed", "failed"},
    )


def mark_market_activity_progress(
    db: Session,
    *,
    market: str,
    stage_key: str,
    lifecycle: str | None = None,
    task_name: str | None = None,
    task_id: str | None = None,
    percent: float | None = None,
    current: int | None = None,
    total: int | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    existing = _load_market_activity(db, market)
    if isinstance(existing, dict):
        existing_task_id = existing.get("task_id")
        existing_stage_key = existing.get("stage_key")
        existing_status = existing.get("status")
        if existing_status in {"completed", "failed"}:
            return existing
        if existing_task_id and task_id and existing_task_id != task_id:
            return existing
        if existing_stage_key and existing_stage_key != stage_key:
            return existing
        stage_key = existing_stage_key or stage_key
        lifecycle = lifecycle or existing.get("lifecycle")
        task_name = task_name or existing.get("task_name")
        task_id = task_id or existing_task_id
        message = message or existing.get("message")

    return _save_market_activity(
        db,
        market,
        _activity_payload(
            market=market,
            stage_key=stage_key,
            lifecycle=lifecycle,
            status="running",
            task_name=task_name,
            task_id=task_id,
            percent=percent,
            current=current,
            total=total,
            message=message,
        ),
        preserve_existing_statuses={"running", "completed", "failed"},
    )


def mark_market_activity_completed(
    db: Session,
    *,
    market: str,
    stage_key: str,
    lifecycle: str | None = None,
    task_name: str | None = None,
    task_id: str | None = None,
    percent: float | None = 100.0,
    current: int | None = None,
    total: int | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    return _save_market_activity(
        db,
        market,
        _activity_payload(
            market=market,
            stage_key=stage_key,
            lifecycle=lifecycle,
            status="completed",
            task_name=task_name,
            task_id=task_id,
            percent=percent,
            current=current,
            total=total,
            message=message,
        ),
        preserve_existing_statuses={"running", "completed", "failed"},
    )


def mark_market_activity_failed(
    db: Session,
    *,
    market: str,
    stage_key: str,
    lifecycle: str | None = None,
    task_name: str | None = None,
    task_id: str | None = None,
    percent: float | None = None,
    current: int | None = None,
    total: int | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    return _save_market_activity(
        db,
        market,
        _activity_payload(
            market=market,
            stage_key=stage_key,
            lifecycle=lifecycle,
            status="failed",
            task_name=task_name,
            task_id=task_id,
            percent=percent,
            current=current,
            total=total,
            message=message,
        ),
        preserve_existing_statuses={"running", "completed", "failed"},
    )


def mark_current_market_activity_failed(
    db: Session,
    *,
    market: str,
    lifecycle: str | None = None,
    task_name: str | None = None,
    task_id: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    existing = _load_market_activity(db, market)
    stage_key = "scan"
    resolved_task_name = task_name
    resolved_task_id = task_id
    if isinstance(existing, dict) and existing.get("status") in {"queued", "running"}:
        stage_key = existing.get("stage_key") or stage_key
        lifecycle = lifecycle or existing.get("lifecycle")
        resolved_task_name = resolved_task_name or existing.get("task_name")
        resolved_task_id = resolved_task_id or existing.get("task_id")
    return mark_market_activity_failed(
        db,
        market=market,
        stage_key=stage_key,
        lifecycle=lifecycle,
        task_name=resolved_task_name,
        task_id=resolved_task_id,
        message=message,
    )


def _overlay_live_progress(record: dict[str, Any], market: str) -> dict[str, Any]:
    if record.get("status") != "running":
        return record
    if record.get("stage_key") != "prices":
        return record
    try:
        current_task = get_data_fetch_lock().get_current_task(market=market)
    except Exception:
        current_task = None
    if not current_task or current_task.get("task_id") != record.get("task_id"):
        return record
    merged = dict(record)
    if merged.get("percent") is None and current_task.get("progress") is not None:
        merged["percent"] = float(current_task["progress"])
    if merged.get("current") is None and current_task.get("current") is not None:
        merged["current"] = current_task["current"]
    if merged.get("total") is None and current_task.get("total") is not None:
        merged["total"] = current_task["total"]
    if current_task.get("last_heartbeat") is not None:
        merged["updated_at"] = current_task["last_heartbeat"]
    merged["percent"] = _resolve_progress_percent(
        merged.get("percent"),
        merged.get("current"),
        merged.get("total"),
    )
    merged["progress_mode"] = _progress_mode(
        merged.get("status"),
        merged.get("percent"),
        merged.get("current"),
        merged.get("total"),
    )
    return merged


def _bootstrap_progress_percent(record: dict[str, Any] | None) -> float:
    if record is None:
        return 0.0
    stage_key = record.get("stage_key")
    if stage_key not in STAGE_SEQUENCE:
        return 0.0
    stage_index = STAGE_SEQUENCE.index(stage_key)
    raw_percent = record.get("percent")
    if raw_percent is not None:
        stage_fraction = max(0.0, min(float(raw_percent), 100.0)) / 100.0
    elif record.get("status") == "completed":
        stage_fraction = 1.0
    else:
        stage_fraction = 0.0
    return round(((stage_index + stage_fraction) / len(STAGE_SEQUENCE)) * 100.0, 2)


def _queued_bootstrap_market_payload(market: str, primary_market: str) -> dict[str, Any]:
    queued_for_primary = str(market).upper() != str(primary_market).upper()
    message = (
        f"Queued until {primary_market.upper()} is ready."
        if queued_for_primary
        else "Bootstrap queued."
    )
    return {
        "market": str(market).upper(),
        "lifecycle": "bootstrap",
        "stage_key": "universe",
        "stage_label": _stage_label("universe"),
        "status": "queued",
        "progress_mode": "indeterminate",
        "percent": None,
        "current": None,
        "total": None,
        "message": message,
        "task_name": None,
        "task_id": None,
        "updated_at": None,
    }


def _idle_market_payload(market: str, record: dict[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {
            "market": str(market).upper(),
            "lifecycle": "idle",
            "stage_key": None,
            "stage_label": None,
            "status": "idle",
            "progress_mode": "determinate",
            "percent": None,
            "current": None,
            "total": None,
            "message": "Idle",
            "task_name": None,
            "task_id": None,
            "updated_at": None,
        }
    payload = dict(record)
    if payload.get("status") == "completed":
        payload["lifecycle"] = "idle"
    payload["percent"] = _resolve_progress_percent(
        payload.get("percent"),
        payload.get("current"),
        payload.get("total"),
    )
    payload["progress_mode"] = _progress_mode(
        payload.get("status"),
        payload.get("percent"),
        payload.get("current"),
        payload.get("total"),
    )
    return payload


def _market_payload(
    *,
    market: str,
    record: dict[str, Any] | None,
    bootstrap_state: str,
    bootstrap_required: bool,
    primary_market: str,
) -> dict[str, Any]:
    if record is None and bootstrap_state == "running" and bootstrap_required:
        return _queued_bootstrap_market_payload(market, primary_market)
    if record is None:
        return _idle_market_payload(market, None)
    return _idle_market_payload(market, _overlay_live_progress(record, market))


def get_runtime_activity_status(db: Session) -> dict[str, Any]:
    bootstrap_status = get_runtime_bootstrap_status(db)
    enabled_markets = list(bootstrap_status.enabled_markets)
    primary_market = bootstrap_status.primary_market

    market_payloads = []
    for market in enabled_markets:
        record = _load_market_activity(db, market)
        market_payloads.append(
            _market_payload(
                market=market,
                record=record,
                bootstrap_state=bootstrap_status.bootstrap_state,
                bootstrap_required=bootstrap_status.bootstrap_required,
                primary_market=primary_market,
            )
        )

    active_markets = [
        payload["market"]
        for payload in market_payloads
        if payload.get("status") in ACTIVE_STATUSES
    ]
    has_failed = any(payload.get("status") == "failed" for payload in market_payloads)
    summary_status = "warning" if has_failed else ("active" if active_markets else "idle")

    primary_payload = next(
        (payload for payload in market_payloads if payload["market"] == primary_market),
        None,
    )
    secondary_active = [
        payload["market"]
        for payload in market_payloads
        if payload["market"] != primary_market and payload.get("status") in ACTIVE_STATUSES
    ]

    if bootstrap_status.bootstrap_state == "ready":
        bootstrap_progress_mode = "determinate"
        bootstrap_percent = 100.0
        bootstrap_stage = primary_payload.get("stage_label") if primary_payload else None
        bootstrap_message = (
            "Primary market is ready."
            if not secondary_active
            else "Primary market is ready while additional market loading continues."
        )
    elif any(payload.get("progress_mode") == "determinate" for payload in market_payloads):
        focus_payload = next(
            (payload for payload in market_payloads if payload.get("status") in ACTIVE_STATUSES),
            primary_payload,
        )
        bootstrap_progress_mode = "determinate"
        bootstrap_percent = round(
            sum(_bootstrap_progress_percent(payload) for payload in market_payloads)
            / max(len(market_payloads), 1),
            2,
        )
        bootstrap_stage = focus_payload.get("stage_label") if focus_payload else None
        bootstrap_message = focus_payload.get("message") if focus_payload else "Bootstrap queued."
    else:
        bootstrap_progress_mode = "indeterminate"
        bootstrap_percent = None
        bootstrap_stage = primary_payload.get("stage_label") if primary_payload else None
        bootstrap_message = (
            primary_payload.get("message")
            if primary_payload is not None
            else "Bootstrap queued."
        )

    background_warning = None
    if len(enabled_markets) > 1 and (
        bootstrap_status.bootstrap_state == "running" or bool(secondary_active)
    ):
        background_warning = (
            "Bootstrap remains active until every enabled market has a published scan."
        )

    return {
        "bootstrap": {
            "state": bootstrap_status.bootstrap_state,
            "app_ready": not bootstrap_status.bootstrap_required,
            "primary_market": primary_market,
            "enabled_markets": enabled_markets,
            "current_stage": bootstrap_stage,
            "progress_mode": bootstrap_progress_mode,
            "percent": bootstrap_percent,
            "message": bootstrap_message,
            "background_warning": background_warning,
        },
        "summary": {
            "active_market_count": len(active_markets),
            "active_markets": active_markets,
            "status": summary_status,
        },
        "markets": market_payloads,
    }
