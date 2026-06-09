"""Persisted local runtime preferences and bootstrap status helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..domain.markets.catalog import get_market_catalog
from ..models.app_settings import AppSetting
from .bootstrap_readiness_service import (
    BootstrapReadiness,
    BootstrapReadinessService,
    MarketBootstrapReadiness,
)

RUNTIME_SETTINGS_CATEGORY = "runtime"
PRIMARY_MARKET_KEY = "runtime.primary_market"
ENABLED_MARKETS_KEY = "runtime.enabled_markets"
BOOTSTRAP_STATE_KEY = "runtime.bootstrap_state"
BOOTSTRAP_STARTED_AT_KEY = "runtime.bootstrap_started_at"

DEFAULT_PRIMARY_MARKET = "US"
DEFAULT_ENABLED_MARKETS = ("US",)
DEFAULT_BOOTSTRAP_STATE = "not_started"
VALID_BOOTSTRAP_STATES = {"not_started", "running", "ready", "failed"}


@dataclass(frozen=True)
class RuntimePreferences:
    primary_market: str
    enabled_markets: list[str]
    bootstrap_state: str
    bootstrap_started_at: datetime | None = None


@dataclass(frozen=True)
class RuntimeBootstrapStatus:
    bootstrap_required: bool
    empty_system: bool
    primary_market: str
    enabled_markets: list[str]
    bootstrap_state: str
    supported_markets: list[str]


def _get_setting(db: Session, key: str) -> AppSetting | None:
    return db.query(AppSetting).filter(AppSetting.key == key).first()


def _upsert_setting(db: Session, *, key: str, value: str, description: str) -> None:
    setting = _get_setting(db, key)
    if setting is None:
        setting = AppSetting(
            key=key,
            value=value,
            category=RUNTIME_SETTINGS_CATEGORY,
            description=description,
        )
        db.add(setting)
        return

    setting.value = value
    setting.category = RUNTIME_SETTINGS_CATEGORY
    setting.description = description


def get_bootstrap_readiness_service() -> BootstrapReadinessService:
    return BootstrapReadinessService()


def _normalize_supported_market(value: str | None) -> str:
    catalog = get_market_catalog()
    try:
        return catalog.get(value).code
    except ValueError as exc:
        raise ValueError(
            f"Unsupported market {value!r}. Supported: {catalog.supported_market_codes()}"
        ) from exc


def _normalize_enabled_markets(markets: list[str] | tuple[str, ...] | None) -> list[str]:
    raw_markets = list(markets or DEFAULT_ENABLED_MARKETS)
    normalized: list[str] = []
    for market in raw_markets:
        canonical = _normalize_supported_market(market)
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized or [DEFAULT_PRIMARY_MARKET]


def _normalize_bootstrap_state(value: str | None) -> str:
    if not value:
        return DEFAULT_BOOTSTRAP_STATE
    normalized = str(value).strip().lower()
    if normalized in VALID_BOOTSTRAP_STATES:
        return normalized
    return DEFAULT_BOOTSTRAP_STATE


def _parse_bootstrap_started_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _write_bootstrap_started_at(db: Session) -> None:
    _upsert_setting(
        db,
        key=BOOTSTRAP_STARTED_AT_KEY,
        value=datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp for the current local bootstrap attempt.",
    )


def _ensure_bootstrap_started_at(db: Session) -> None:
    existing = _get_setting(db, BOOTSTRAP_STARTED_AT_KEY)
    if _parse_bootstrap_started_at(existing.value if existing else None) is not None:
        return
    _write_bootstrap_started_at(db)


def get_runtime_preferences(db: Session) -> RuntimePreferences:
    primary_setting = _get_setting(db, PRIMARY_MARKET_KEY)
    primary_market = _normalize_supported_market(
        primary_setting.value if primary_setting and primary_setting.value else DEFAULT_PRIMARY_MARKET
    )

    enabled_setting = _get_setting(db, ENABLED_MARKETS_KEY)
    enabled_markets_payload: list[str] | tuple[str, ...] | None = None
    if enabled_setting and enabled_setting.value:
        try:
            parsed = json.loads(enabled_setting.value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            enabled_markets_payload = [str(item).upper() for item in parsed if item]

    enabled_markets = _normalize_enabled_markets(enabled_markets_payload)
    if primary_market not in enabled_markets:
        enabled_markets = [primary_market, *enabled_markets]

    bootstrap_setting = _get_setting(db, BOOTSTRAP_STATE_KEY)
    bootstrap_state = _normalize_bootstrap_state(
        bootstrap_setting.value if bootstrap_setting else None
    )
    bootstrap_started_at_setting = _get_setting(db, BOOTSTRAP_STARTED_AT_KEY)
    bootstrap_started_at = _parse_bootstrap_started_at(
        bootstrap_started_at_setting.value if bootstrap_started_at_setting else None
    )

    return RuntimePreferences(
        primary_market=primary_market,
        enabled_markets=enabled_markets,
        bootstrap_state=bootstrap_state,
        bootstrap_started_at=bootstrap_started_at,
    )


def save_runtime_preferences(
    db: Session,
    *,
    primary_market: str,
    enabled_markets: list[str],
    bootstrap_state: str | None = None,
) -> RuntimePreferences:
    normalized_primary = _normalize_supported_market(primary_market)
    normalized_enabled = _normalize_enabled_markets(enabled_markets)
    if normalized_primary not in normalized_enabled:
        normalized_enabled = [normalized_primary, *normalized_enabled]

    _upsert_setting(
        db,
        key=PRIMARY_MARKET_KEY,
        value=normalized_primary,
        description="Primary market used for local-first bootstrap and resume behavior.",
    )
    _upsert_setting(
        db,
        key=ENABLED_MARKETS_KEY,
        value=json.dumps(normalized_enabled),
        description="Enabled markets for the local-default runtime scheduler.",
    )
    if bootstrap_state is not None:
        normalized_bootstrap_state = _normalize_bootstrap_state(bootstrap_state)
        existing_bootstrap_setting = _get_setting(db, BOOTSTRAP_STATE_KEY)
        existing_bootstrap_state = _normalize_bootstrap_state(
            existing_bootstrap_setting.value if existing_bootstrap_setting else None
        )
        _upsert_setting(
            db,
            key=BOOTSTRAP_STATE_KEY,
            value=normalized_bootstrap_state,
            description="Current local bootstrap orchestration state.",
        )
        if normalized_bootstrap_state == "running":
            if existing_bootstrap_state == "running":
                _ensure_bootstrap_started_at(db)
            else:
                _write_bootstrap_started_at(db)
    db.commit()
    return get_runtime_preferences(db)


def set_bootstrap_state(db: Session, bootstrap_state: str) -> RuntimePreferences:
    prefs = get_runtime_preferences(db)
    return save_runtime_preferences(
        db,
        primary_market=prefs.primary_market,
        enabled_markets=prefs.enabled_markets,
        bootstrap_state=bootstrap_state,
    )


def is_market_enabled(db: Session, market: str | None) -> bool:
    if market is None:
        return True
    return _normalize_supported_market(market) in get_runtime_preferences(db).enabled_markets


def is_market_enabled_now(market: str | None) -> bool:
    """Process-safe helper for task code that only has a market label."""
    if market is None:
        return True
    from ..database import SessionLocal

    db = SessionLocal()
    try:
        return is_market_enabled(db, market)
    finally:
        db.close()


def get_runtime_bootstrap_status(db: Session) -> RuntimeBootstrapStatus:
    prefs = get_runtime_preferences(db)
    enabled_markets = list(prefs.enabled_markets)
    readiness = get_bootstrap_readiness_service().evaluate(
        db,
        enabled_markets=enabled_markets,
        bootstrap_started_at=prefs.bootstrap_started_at,
    )
    empty_system = readiness.empty_system

    bootstrap_state = prefs.bootstrap_state
    all_markets_ready = (
        readiness.ready
        if bootstrap_state in {"running", "ready", "failed"}
        else False
    )
    if all_markets_ready:
        bootstrap_state = "ready"
    elif empty_system and bootstrap_state == "ready":
        bootstrap_state = "not_started"
    elif bootstrap_state == "ready" and not all_markets_ready:
        bootstrap_state = "running"

    bootstrap_required = empty_system or (
        bootstrap_state in {"running", "failed"} and not all_markets_ready
    )

    return RuntimeBootstrapStatus(
        bootstrap_required=bootstrap_required,
        empty_system=empty_system,
        primary_market=prefs.primary_market,
        enabled_markets=enabled_markets,
        bootstrap_state=bootstrap_state,
        supported_markets=get_market_catalog().supported_market_codes(),
    )
