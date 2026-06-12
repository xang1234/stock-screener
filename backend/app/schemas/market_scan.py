"""Market Scan schemas"""
from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import datetime

from .scanning import ScanResultItem


class WatchlistSymbolBase(BaseModel):
    """Base schema for watchlist symbol"""
    symbol: str
    display_name: Optional[str] = None
    notes: Optional[str] = None


class WatchlistSymbolCreate(WatchlistSymbolBase):
    """Schema for creating a new watchlist symbol"""
    position: Optional[int] = None


class WatchlistSymbolUpdate(BaseModel):
    """Schema for updating a watchlist symbol"""
    display_name: Optional[str] = None
    notes: Optional[str] = None
    position: Optional[int] = None


class WatchlistSymbolResponse(WatchlistSymbolBase):
    """Response schema for a watchlist symbol"""
    id: int
    list_name: str
    position: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class WatchlistResponse(BaseModel):
    """Response schema for a complete watchlist"""
    list_name: str
    symbols: List[WatchlistSymbolResponse]
    total: int


class ReorderRequest(BaseModel):
    """Request body for reordering symbols"""
    symbol_ids: List[int]


class KeyMarketHistoryPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    close: Optional[float] = None


class KeyMarketEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    display_name: Optional[str] = None
    currency: Optional[str] = None
    latest_close: Optional[float] = None
    latest_date: Optional[str] = None
    change_1d: Optional[float] = None
    history: List[KeyMarketHistoryPoint]


class DailySnapshotFreshness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scan_id: Optional[str] = None
    scan_as_of_date: Optional[str] = None
    scan_published_at: Optional[str] = None
    breadth_latest_date: Optional[str] = None
    groups_latest_date: Optional[str] = None


class DailySnapshotTopCandidates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_dollar_volume: Optional[float] = None
    rows: List[ScanResultItem]


class DailySnapshotLeadersCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_group_rank: int
    min_rs_rating: int
    min_dollar_volume: Optional[float] = None


class DailySnapshotLeaders(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criteria: DailySnapshotLeadersCriteria
    rows: List[ScanResultItem]


class DailySnapshotTopGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    industry_group: str
    rank: Optional[float] = None
    rank_change_1w: Optional[float] = None
    rank_change_1m: Optional[float] = None
    top_symbol: Optional[str] = None
    top_symbol_name: Optional[str] = None
    top_rs_rating: Optional[float] = None


class DailySnapshotResponse(BaseModel):
    """Aggregated Daily Snapshot payload (one market, one request).

    ``extra="forbid"`` on the snapshot-specific models keeps this schema in
    lockstep with ``build_daily_snapshot_payload``: any payload drift fails
    validation on the cache-miss path instead of silently changing the
    public contract.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    generated_at: str
    market: str
    market_display_name: str
    scan_id: Optional[str] = None
    freshness: DailySnapshotFreshness
    key_markets: List[KeyMarketEntry]
    top_candidates: DailySnapshotTopCandidates
    leaders: DailySnapshotLeaders
    top_groups: List[DailySnapshotTopGroup]
