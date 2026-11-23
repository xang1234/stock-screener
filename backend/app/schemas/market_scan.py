"""Market Scan schemas"""
from datetime import datetime
from typing import Annotated, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain.markets.market_cycle import CycleState
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


class KeyMarketDateRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: Optional[str] = None
    max: Optional[str] = None


class KeyMarketDateMismatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    latest_date: Optional[str] = None
    status: Literal["missing", "stale", "future"]


class DailySnapshotFreshness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scan_id: Optional[str] = None
    scan_as_of_date: Optional[str] = None
    scan_published_at: Optional[str] = None
    snapshot_as_of_date: Optional[str] = None
    market_timezone: Optional[str] = None
    breadth_latest_date: Optional[str] = None
    groups_latest_date: Optional[str] = None
    exposure_latest_date: Optional[str] = None
    key_markets_latest_date: Optional[str] = None
    key_markets_date_range: Optional[KeyMarketDateRange] = None
    key_markets_mismatched_symbols: List[KeyMarketDateMismatch] = Field(default_factory=list)
    date_coherence_status: str = "unknown"


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


NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]


class DailySnapshotActionStateCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exit_risk: NonNegativeInt
    deteriorating: NonNegativeInt
    event_risk: NonNegativeInt
    extended: NonNegativeInt
    data_limited: NonNegativeInt
    setup_ready: NonNegativeInt
    watch: NonNegativeInt


class DailySnapshotCorrectionSurvivors(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    complete: bool
    count: NonNegativeInt
    counts_by_action_state: DailySnapshotActionStateCounts
    rows: list[ScanResultItem]

    @model_validator(mode="after")
    def _validate_count_partition(self):
        state_total = sum(self.counts_by_action_state.model_dump().values())
        if self.count != state_total:
            raise ValueError("count must equal the sum of counts_by_action_state")
        return self


class DailySnapshotTopGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    industry_group: str
    rank: Optional[float] = None
    avg_rs_rating: Optional[float] = None
    avg_rs_rating_1d: Optional[float] = None
    avg_rs_rating_1w: Optional[float] = None
    avg_rs_rating_1m: Optional[float] = None
    avg_rs_rating_3m: Optional[float] = None
    avg_rs_rating_6m: Optional[float] = None
    rank_change_1w: Optional[float] = None
    rank_change_1m: Optional[float] = None
    top_symbol: Optional[str] = None
    top_symbol_name: Optional[str] = None
    top_rs_rating: Optional[float] = None
    rs_formula_version: Optional[str] = None
    market_rs_run_id: Optional[int] = None
    rs_as_of_date: Optional[str] = None
    rs_universe_size: Optional[int] = None


class MarketHealthExposureHistoryPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    exposure_score: float
    stance: str
    cycle_state: Optional[CycleState] = None
    rally_day: Optional[int] = None
    follow_through: bool = False
    failed_ftd: bool = False


class MarketHealthBreadthDivergence(BaseModel):
    """Index near its high with weak participation; ``flagged`` None = unknown."""
    model_config = ConfigDict(extra="forbid")

    index_off_high_pct: Optional[float] = None
    above_sma200_pct: Optional[float] = None
    new_high_52week_count: Optional[int] = None
    new_low_52week_count: Optional[int] = None
    flagged: Optional[bool] = None


class MarketHealthFtdQuality(BaseModel):
    """How many of the FTD session's 4% gainers closed near their highs."""
    model_config = ConfigDict(extra="forbid")

    ftd_date: str
    stocks_up_4pct: Optional[int] = None
    up_4pct_near_high_count: Optional[int] = None
    near_high_share: Optional[float] = None
    new_high_52week_count: Optional[int] = None
    new_low_52week_count: Optional[int] = None


class MarketHealthExposure(BaseModel):
    """0-100 recommended-exposure score + the inputs that justify it."""
    model_config = ConfigDict(extra="forbid")

    market: str
    date: str
    exposure_score: float
    stance: str
    distribution_day_count: int
    follow_through_day: bool
    follow_through_date: Optional[str] = None
    cycle_state: Optional[CycleState] = None
    rally_day: Optional[int] = None
    trend: Optional[str] = None
    vix: Optional[float] = None
    benchmark_symbol: Optional[str] = None
    components: Optional[Dict[str, float]] = None
    breadth_divergence: Optional[MarketHealthBreadthDivergence] = None
    ftd_quality: Optional[MarketHealthFtdQuality] = None
    history: List[MarketHealthExposureHistoryPoint]


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
    market_health_exposure: Optional[MarketHealthExposure] = None
    correction_survivors: DailySnapshotCorrectionSurvivors
    top_candidates: DailySnapshotTopCandidates
    leaders: DailySnapshotLeaders
    top_groups: List[DailySnapshotTopGroup]
