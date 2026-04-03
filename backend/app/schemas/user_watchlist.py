"""Schemas for User-defined Watchlists"""
from pydantic import BaseModel
from typing import Literal, Optional, List, Dict
from datetime import datetime

from .common import PriceChangeBounds


# ================= Watchlist Schemas =================

class WatchlistBase(BaseModel):
    """Base schema for a watchlist"""
    name: str
    description: Optional[str] = None
    color: Optional[str] = None


class WatchlistCreate(WatchlistBase):
    """Schema for creating a new watchlist"""
    pass


class WatchlistUpdate(BaseModel):
    """Schema for updating a watchlist"""
    name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    position: Optional[int] = None


class WatchlistResponse(WatchlistBase):
    """Response schema for a watchlist"""
    id: int
    position: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ================= Item Schemas =================

class WatchlistItemBase(BaseModel):
    """Base schema for a watchlist item"""
    symbol: str
    display_name: Optional[str] = None
    notes: Optional[str] = None


class WatchlistItemCreate(WatchlistItemBase):
    """Schema for adding an item to a watchlist"""
    pass


class WatchlistItemUpdate(BaseModel):
    """Schema for updating an item"""
    display_name: Optional[str] = None
    notes: Optional[str] = None
    position: Optional[int] = None


class WatchlistItemResponse(WatchlistItemBase):
    """Response schema for an item"""
    id: int
    watchlist_id: int
    position: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ================= Stock Data Response (with sparklines/price changes) =================

class WatchlistStockData(BaseModel):
    """Stock with computed market data for display in watchlist table"""
    id: int
    symbol: str
    display_name: Optional[str] = None
    watchlist_id: int
    position: int

    # Company info
    company_name: Optional[str] = None
    ibd_industry: Optional[str] = None

    # Sparkline data (30-day)
    rs_data: Optional[List[float]] = None
    rs_trend: Optional[int] = None
    price_data: Optional[List[float]] = None
    price_trend: Optional[int] = None

    # Price changes for bar chart (percentage)
    change_1d: Optional[float] = None
    change_5d: Optional[float] = None
    change_2w: Optional[float] = None
    change_1m: Optional[float] = None
    change_3m: Optional[float] = None
    change_6m: Optional[float] = None
    change_12m: Optional[float] = None


class WatchlistDataResponse(BaseModel):
    """Complete watchlist data for display"""
    id: int
    name: str
    description: Optional[str] = None
    color: Optional[str] = None
    items: List[WatchlistStockData]
    price_change_bounds: Dict[str, PriceChangeBounds]


class WatchlistListResponse(BaseModel):
    """List of watchlists for selector"""
    watchlists: List[WatchlistResponse]
    total: int


# ================= Reorder Schemas =================

class ReorderWatchlistsRequest(BaseModel):
    """Request body for reordering watchlists"""
    watchlist_ids: List[int]


class ReorderItemsRequest(BaseModel):
    """Request body for reordering items"""
    item_ids: List[int]


# ================= Bulk Add Schema =================

class BulkAddItemsRequest(BaseModel):
    """Request body for adding multiple symbols at once"""
    symbols: List[str]


class WatchlistImportRequest(BaseModel):
    """Request body for importing symbols from pasted text or CSV."""

    content: str
    format: Optional[Literal["auto", "text", "csv"]] = "auto"


class WatchlistImportResult(BaseModel):
    """Import result with partial-success details."""

    requested_count: int
    added: List[str]
    skipped_existing: List[str]
    invalid_symbols: List[str]
    added_items: List[WatchlistItemResponse]
