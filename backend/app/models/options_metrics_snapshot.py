"""Daily persisted options-metrics snapshots.

Historically the richer options-analytics payload (IV Rank, skew, DEX/VEX/CEX
net exposures, PCR, VRP, expected move, IV smile, unusual volume, max pain,
next earnings) only ever lived in the `options_metrics:{symbol}` Redis key
(7-day TTL) -- only the narrower GEX walls (gex_snapshots) and max-pain strike
(max_pain_snapshots) were ever persisted to Postgres. This table gives that
data a durable home for historical correlation against the stock scanner's
screener scores, instead of evaporating after 7 days.

Two different code paths write rows here, at two different levels of
richness -- both are legitimate, not a bug to reconcile:

  - The nightly universe-wide batch (`analyze_options_exposure` in
    tasks/options_analysis_tasks.py) reuses the option chain it already
    fetched to compute a CHEAP payload (key levels, net exposures, IVR,
    skew, per-strike GEX) -- it deliberately avoids a second yfinance sweep
    across the whole universe. Rows written from here have
    source="batch_abbreviated" and schema_version=None (PCR/VRP/expected
    move/IV smile/unusual volume/max pain/next earnings are left null).

  - The on-demand `POST /v1/options/metrics` endpoint computes the FULL
    payload via `calculate_options_metrics()` (its own live fetch) whenever
    a user actually opens a symbol's Options Analytics dashboard. Rows
    written from here have source="live_full" and a populated
    schema_version, with every field populated.

Consumers that need the full payload should filter on
`schema_version IS NOT NULL` (or `source = 'live_full'`) rather than
assuming every row has every field -- coverage of the full payload is
whatever set of symbols users have actually viewed within the persistence
window, not the whole universe.
"""

from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Float, Index, Integer, JSON, String, UniqueConstraint

from ..database import Base


class OptionsMetricsSnapshot(Base):
    """One row per ticker per trading day: that day's options-metrics payload.

    Natural key is (ticker, trading_date, expiration), matching the
    convention used by MaxPainSnapshot/GexSnapshot. Unlike those two tables,
    this one is new with no legacy duplicate-row risk, so the natural key is
    enforced with a real DB-level UNIQUE constraint (see iv_history.py for
    the precedent -- gex/max_pain predate that decision and rely on
    app-level check-then-write only via options_snapshot_upsert.upsert_snapshot()).
    """

    __tablename__ = "options_metrics_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), index=True, nullable=False)
    status = Column(String(20), default="OK", nullable=False)  # OK, FAILED
    error = Column(String(1000), nullable=True)

    # Which code path wrote this row -- see module docstring.
    source = Column(String(20), nullable=True)  # batch_abbreviated, live_full
    schema_version = Column(Integer, nullable=True)

    expiration = Column(Date, nullable=True)
    underlying_price = Column(Float, nullable=True)

    # Key gamma levels
    call_wall = Column(Float, nullable=True)
    call_wall_gex = Column(Float, nullable=True)
    put_wall = Column(Float, nullable=True)
    put_wall_gex = Column(Float, nullable=True)
    zero_gamma = Column(Float, nullable=True)
    total_call_gex = Column(Float, nullable=True)
    total_put_gex = Column(Float, nullable=True)
    total_gex = Column(Float, nullable=True)

    # Second-order net exposures
    net_dex = Column(Float, nullable=True)
    net_vex = Column(Float, nullable=True)
    net_cex = Column(Float, nullable=True)

    # Volatility & skew
    ivr = Column(Float, nullable=True)
    skew = Column(Float, nullable=True)
    historical_volatility = Column(Float, nullable=True)
    current_atm_iv = Column(Float, nullable=True)
    volatility_risk_premium = Column(Float, nullable=True)
    expected_move = Column(Float, nullable=True)
    atm_strike = Column(Float, nullable=True)

    # Flow / positioning (live_full only)
    volume_put_call_ratio = Column(Float, nullable=True)
    open_interest_put_call_ratio = Column(Float, nullable=True)
    total_call_oi = Column(Integer, nullable=True)
    total_put_oi = Column(Integer, nullable=True)
    call_premium_notional = Column(Float, nullable=True)
    put_premium_notional = Column(Float, nullable=True)

    # Max pain (live_full only -- also separately tracked in max_pain_snapshots)
    max_pain_strike = Column(Float, nullable=True)
    max_pain_distance_pct = Column(Float, nullable=True)

    # Event risk (live_full only)
    next_earnings_date = Column(Date, nullable=True)

    greeks_methodology = Column(String(50), nullable=True)

    # Full-fidelity detail, kept as JSON rather than flattened -- this is the
    # actual "goldmine" part: per-strike GEX/DEX/VEX/CEX, the IV smile curve,
    # and any unusual-volume contracts flagged that day.
    strikes_json = Column(JSON, nullable=True)
    iv_smile_json = Column(JSON, nullable=True)
    unusual_volume_json = Column(JSON, nullable=True)

    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    # US/Eastern trading day this snapshot belongs to -- same convention as
    # gex_snapshots/max_pain_snapshots (see options_snapshot_upsert.trading_date_for).
    trading_date = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    batch_id = Column(String(50), nullable=True, index=True)  # set only for batch_abbreviated rows

    __table_args__ = (
        UniqueConstraint(
            "ticker", "trading_date", "expiration",
            name="uq_options_metrics_snapshots_ticker_date_expiration",
        ),
        Index("ix_options_metrics_snapshots_ticker_trading_date", "ticker", "trading_date"),
        Index("ix_options_metrics_snapshots_batch_fetched", "batch_id", "fetched_at"),
    )
