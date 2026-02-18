"""Pydantic schemas for cache management API endpoints."""

from typing import List, Literal, Optional

from pydantic import BaseModel


class CacheStatsResponse(BaseModel):
    """Cache statistics response model."""

    redis_connected: bool
    market_status: str
    spy_cache: dict
    price_cache: dict
    redis_memory: Optional[dict] = None


class CacheWarmRequest(BaseModel):
    """Request model for cache warming."""

    symbols: Optional[List[str]] = None
    count: Optional[int] = None
    force_refresh: bool = False


class CacheInvalidateRequest(BaseModel):
    """Request model for cache invalidation."""

    symbol: Optional[str] = None


class DashboardStatsResponse(BaseModel):
    """Dashboard cache statistics response model."""

    fundamentals: dict
    prices: dict
    market_status: dict
    timestamp: str


class StalenessStatusResponse(BaseModel):
    """Staleness status response model."""

    stale_intraday_count: int
    stale_symbols: List[str]
    market_is_open: bool
    current_time_et: str
    has_stale_data: bool


class ForceRefreshRequest(BaseModel):
    """Request model for force refresh."""

    symbols: Optional[List[str]] = None  # None means refresh all stale symbols
    refresh_all: bool = False  # True = refresh ALL cached symbols, not just stale ones


class SmartRefreshRequest(BaseModel):
    """Request model for smart refresh."""

    mode: Literal["auto", "full"] = "auto"


class CacheHealthResponse(BaseModel):
    """Cache health status response model."""

    status: str  # fresh, updating, stuck, partial, stale, error
    spy_last_date: Optional[str] = None
    expected_date: Optional[str] = None
    message: str
    can_refresh: bool
    can_force_cancel: Optional[bool] = None
    task_running: Optional[dict] = None
    last_warmup: Optional[dict] = None


class SmartRefreshResponse(BaseModel):
    """Smart refresh response model."""

    status: str  # queued, already_running
    task_id: Optional[str] = None
    message: str
