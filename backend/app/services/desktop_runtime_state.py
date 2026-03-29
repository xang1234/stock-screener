"""Shared state helpers for the desktop setup/update runtime."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.app_settings import AppSetting
from app.models.industry import IBDGroupRank
from app.models.market_breadth import MarketBreadth
from app.models.stock import StockFundamental, StockPrice
from app.models.stock_universe import StockUniverse


DESKTOP_CATEGORY = "desktop"
SETUP_STATE_KEY = "desktop_setup_state"
UPDATE_STATE_KEY = "desktop_update_state"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_json_setting(default: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(default))


def load_json_setting(db: Session, *, key: str, default: dict[str, Any]) -> dict[str, Any]:
    setting = db.query(AppSetting).filter(AppSetting.key == key).first()
    if setting is None:
        return _default_json_setting(default)

    try:
        loaded = json.loads(setting.value)
    except json.JSONDecodeError:
        return _default_json_setting(default)

    merged = _default_json_setting(default)
    merged.update(loaded if isinstance(loaded, dict) else {})
    return merged


def save_json_setting(
    db: Session,
    *,
    key: str,
    payload: dict[str, Any],
    description: str,
) -> None:
    setting = db.query(AppSetting).filter(AppSetting.key == key).first()
    serialized = json.dumps(payload, sort_keys=True)
    if setting is None:
        setting = AppSetting(
            key=key,
            value=serialized,
            category=DESKTOP_CATEGORY,
            description=description,
        )
        db.add(setting)
    else:
        setting.value = serialized
        setting.category = DESKTOP_CATEGORY
        setting.description = description
    db.commit()


def local_data_present(db: Session) -> bool:
    return any(
        (
            db.query(func.count(StockUniverse.id)).scalar() or 0,
            db.query(func.count(StockPrice.id)).scalar() or 0,
            db.query(func.count(StockFundamental.id)).scalar() or 0,
            db.query(func.count(MarketBreadth.id)).scalar() or 0,
            db.query(func.count(IBDGroupRank.id)).scalar() or 0,
        )
    )


def build_data_status(db: Session) -> dict[str, Any]:
    latest_price_date = db.query(func.max(StockPrice.date)).scalar()
    latest_breadth_date = db.query(func.max(MarketBreadth.date)).scalar()
    latest_groups_date = db.query(func.max(IBDGroupRank.date)).scalar()
    latest_fundamentals_at = db.query(func.max(StockFundamental.updated_at)).scalar()
    latest_universe_at = db.query(func.max(StockUniverse.updated_at)).scalar()

    setup_state = load_json_setting(db, key=SETUP_STATE_KEY, default=default_setup_state())

    def _iso_date(value: date | None) -> str | None:
        return value.isoformat() if value else None

    def _iso_datetime(value: datetime | None) -> str | None:
        return value.isoformat() if value else None

    return {
        "local_data_present": local_data_present(db),
        "starter_baseline_active": bool(setup_state.get("starter_baseline_active")),
        "setup_completed_at": setup_state.get("completed_at"),
        "prices": {
            "ready": latest_price_date is not None,
            "last_success_at": _iso_date(latest_price_date),
            "message": None if latest_price_date else "No local price data loaded yet.",
        },
        "breadth": {
            "ready": latest_breadth_date is not None,
            "last_success_at": _iso_date(latest_breadth_date),
            "message": None if latest_breadth_date else "No market breadth baseline loaded yet.",
        },
        "groups": {
            "ready": latest_groups_date is not None,
            "last_success_at": _iso_date(latest_groups_date),
            "message": None if latest_groups_date else "No group ranking baseline loaded yet.",
        },
        "fundamentals": {
            "ready": latest_fundamentals_at is not None,
            "last_success_at": _iso_datetime(latest_fundamentals_at),
            "message": None if latest_fundamentals_at else "No local fundamentals cache has been populated yet.",
        },
        "universe": {
            "ready": latest_universe_at is not None,
            "last_success_at": _iso_datetime(latest_universe_at),
            "message": None if latest_universe_at else "No local stock universe has been loaded yet.",
        },
    }


def default_setup_state() -> dict[str, Any]:
    return {
        "status": "idle",
        "mode": None,
        "job_id": None,
        "message": "Desktop setup has not started",
        "current_step": None,
        "started_at": None,
        "completed_at": None,
        "current": 0,
        "total": 0,
        "percent": 0.0,
        "steps": [],
        "warnings": [],
        "error": None,
        "starter_baseline_active": False,
        "app_ready": False,
    }


def default_update_state() -> dict[str, Any]:
    return {
        "status": "idle",
        "scope": None,
        "triggered_by": None,
        "job_id": None,
        "message": "Automatic updates are idle",
        "current_step": None,
        "started_at": None,
        "completed_at": None,
        "last_success_at": None,
        "current": 0,
        "total": 0,
        "percent": 0.0,
        "steps": [],
        "warnings": [],
        "error": None,
    }

