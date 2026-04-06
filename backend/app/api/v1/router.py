"""Main API router with runtime-aware feature gating."""

from __future__ import annotations

from importlib import import_module

from fastapi import APIRouter, Depends

from ...config import settings
from ...services.server_auth import require_server_session

router = APIRouter()


def _include(
    module_name: str,
    *,
    prefix: str = "",
    tags: list[str] | None = None,
    protected: bool = True,
) -> None:
    module = import_module(f"{__package__}.{module_name}")
    dependencies = [Depends(require_server_session)] if protected else None
    router.include_router(
        module.router,
        prefix=prefix,
        tags=tags or [module_name],
        dependencies=dependencies,
    )


_include("app_runtime", protected=False)
_include("auth", prefix="/auth", tags=["auth"], protected=False)
_include("stocks", prefix="/stocks", tags=["stocks"])
_include("technical", prefix="/technical", tags=["technical"])
_include("scans", prefix="/scans", tags=["scans"])
_include("universe", prefix="/universe", tags=["universe"])
_include("breadth", prefix="/breadth", tags=["breadth"])
_include("groups", prefix="/groups", tags=["groups"])
_include("market_scan", prefix="/market-scan", tags=["market-scan"])
_include("user_watchlists", prefix="/user-watchlists", tags=["user-watchlists"])
_include("validation", prefix="/validation", tags=["validation"])
_include("digest", prefix="/digest", tags=["digest"])
_include("strategy_profiles", prefix="/strategy-profiles", tags=["strategy-profiles"])
_include("ticker_validation", prefix="/ticker-validation", tags=["ticker-validation"])
_include("filter_presets", prefix="/filter-presets", tags=["filter-presets"])
_include("features", prefix="/features", tags=["features"])

if not settings.desktop_mode:
    _include("cache", tags=["cache"])
    _include("fundamentals", tags=["fundamentals"])
    _include("data_fetch_status", tags=["data-fetch"])

if settings.feature_themes:
    _include("themes", prefix="/themes", tags=["themes"])
    _include("user_themes", prefix="/user-themes", tags=["user-themes"])

if settings.feature_tasks:
    _include("tasks", prefix="/tasks", tags=["tasks"])

if settings.feature_chatbot:
    _include("chatbot", prefix="/chatbot", tags=["chatbot"])
    _include("chatbot_folders", prefix="/chatbot/folders", tags=["chatbot-folders"])
    _include("prompt_presets", prefix="/prompt-presets", tags=["prompt-presets"])

if settings.feature_themes or settings.feature_chatbot:
    _include("config", tags=["config"], protected=False)
