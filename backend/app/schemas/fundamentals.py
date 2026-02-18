"""Pydantic schemas for fundamentals data management API endpoints."""

from typing import Optional

from pydantic import BaseModel


class FundamentalsCacheStats(BaseModel):
    """Fundamentals cache statistics response model."""

    total_checked: int
    redis_cached: int
    db_cached: int
    fresh: int
    stale: int
    redis_hit_rate: float
    db_hit_rate: float
    last_refresh_date: Optional[str]
    timestamp: str
