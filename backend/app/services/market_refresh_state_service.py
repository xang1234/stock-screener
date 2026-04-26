"""Observed per-market price refresh state stored in AppSetting."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.models.app_settings import AppSetting
from app.tasks.market_queues import normalize_market

MARKET_REFRESH_STATE_CATEGORY = "market_refresh_state"
MARKET_REFRESH_STATE_KEY_PREFIX = "market.refresh_state."


def _state_key(market: str | None) -> str:
    return f"{MARKET_REFRESH_STATE_KEY_PREFIX}{normalize_market(market)}"


def get_market_refresh_state(db: Session, market: str | None) -> dict[str, Any] | None:
    setting = db.query(AppSetting).filter(AppSetting.key == _state_key(market)).first()
    if setting is None:
        return None
    try:
        payload = json.loads(setting.value)
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def record_market_refresh_success(
    db: Session,
    *,
    market: str,
    trading_day: date,
    success_rate: float,
    status: str = "completed",
) -> dict[str, Any]:
    market_code = normalize_market(market)
    payload: dict[str, Any] = {
        "market": market_code,
        "status": status,
        "last_refreshed_trading_day": trading_day.isoformat(),
        "success_rate": float(success_rate),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    key = _state_key(market_code)
    encoded = json.dumps(payload)
    setting = db.query(AppSetting).filter(AppSetting.key == key).first()
    if setting is None:
        setting = AppSetting(
            key=key,
            value=encoded,
            category=MARKET_REFRESH_STATE_CATEGORY,
            description=f"Observed successful price refresh state for {market_code}",
        )
        db.add(setting)
    else:
        setting.value = encoded
        setting.category = MARKET_REFRESH_STATE_CATEGORY
        setting.description = f"Observed successful price refresh state for {market_code}"
    db.commit()
    return payload
