from pydantic import BaseModel
from typing import List, Optional


class StrikeExposure(BaseModel):
    strike: float
    call_gex: float
    put_gex: float
    total_gex: float
    dex: float
    vex: float
    cex: float
    oi: int
    iv_avg: Optional[float]


class KeyLevels(BaseModel):
    call_wall: Optional[float]
    put_wall: Optional[float]
    zero_gamma: Optional[float]


class NetExposures(BaseModel):
    net_dex: float
    net_vex: float
    net_cex: float


class IvSmilePoint(BaseModel):
    strike: float
    iv: float


class IvSmile(BaseModel):
    calls: List[IvSmilePoint]
    puts: List[IvSmilePoint]


class UnusualVolumeContract(BaseModel):
    strike: float
    type: str
    volume: int
    open_interest: int
    ratio: float


class OptionsMetricsResponse(BaseModel):
    ticker: Optional[str] = None
    expiration: Optional[str] = None
    key_levels: KeyLevels
    net: NetExposures
    ivr: Optional[float]
    skew: Optional[float]
    strikes: List[StrikeExposure]
    volume_put_call_ratio: Optional[float] = None
    open_interest_put_call_ratio: Optional[float] = None
    total_call_oi: Optional[int] = None
    total_put_oi: Optional[int] = None
    call_premium_notional: Optional[float] = None
    put_premium_notional: Optional[float] = None
    underlying_price: Optional[float] = None
    historical_volatility: Optional[float] = None
    current_atm_iv: Optional[float] = None
    volatility_risk_premium: Optional[float] = None
    expected_move: Optional[float] = None
    atm_strike: Optional[float] = None
    total_call_gex: Optional[float] = None
    total_put_gex: Optional[float] = None
    total_gex: Optional[float] = None
    call_wall: Optional[float] = None
    put_wall: Optional[float] = None
    call_wall_gex: Optional[float] = None
    put_wall_gex: Optional[float] = None
    iv_smile: Optional[IvSmile] = None
    unusual_volume: List[UnusualVolumeContract] = []
    next_earnings_date: Optional[str] = None
    max_pain_strike: Optional[float] = None
    max_pain_distance_pct: Optional[float] = None
    greeks_methodology: Optional[str] = None
    computed_at: Optional[str] = None
    schema_version: Optional[int] = None
    # Set when live yfinance data reports zero open interest across every
    # strike (a known off-hours/pre-market data-staleness pattern, not a
    # real market condition) and a persisted OptionsMetricsSnapshot with
    # real open interest was substituted instead. Absent/false on a normal
    # live response.
    data_source: Optional[str] = None
    is_stale_fallback: Optional[bool] = None
    snapshot_trading_date: Optional[str] = None
    snapshot_fetched_at: Optional[str] = None
