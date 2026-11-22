"""Market exposure / health model.

One row per ``(date, market)`` holding a transparent 0-100 recommended-exposure
score plus the inputs that justify it (distribution days, market cycle,
MA/trend, VIX, breadth). Computed daily by the market pipeline and read by the
Daily Snapshot payloads (live + static). Partitioned by ``market`` exactly like
``MarketBreadth`` — same calendar dates, independent universes.
"""
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    false,
)
from sqlalchemy.sql import func
from ..database import Base
from .types import JsonColumn


class MarketExposure(Base):
    __tablename__ = "market_exposure"

    # Indexes are declared once in __table_args__ to match the migration exactly
    # (no per-column index=True, which would create extra ix_* indexes the
    # migration doesn't and that the composite/PK/unique already cover).
    id = Column(Integer, primary_key=True)
    market = Column(String(8), nullable=False, default="US", server_default="US")
    date = Column(Date, nullable=False)

    # Headline
    exposure_score = Column(Float, nullable=False)  # 0-100
    stance = Column(String(32), nullable=False)  # band label

    # Trend components computed from the market's benchmark OHLCV (SPY for US,
    # ^HSI for HK, ^N225 for JP, ...). benchmark_symbol records which one.
    benchmark_price = Column(Float, nullable=True)
    benchmark_ma50 = Column(Float, nullable=True)
    benchmark_ma200 = Column(Float, nullable=True)
    trend = Column(String(20), nullable=True)  # bullish | neutral | bearish

    distribution_day_count = Column(Integer, nullable=False, default=0, server_default="0")

    # Market cycle (app.domain.markets.market_cycle). follow_through_* hold the
    # FTD confirming the current uptrend while its rally low holds.
    follow_through_day = Column(Boolean, nullable=False, default=False, server_default=false())
    follow_through_date = Column(Date, nullable=True)
    cycle_state = Column(String(16), nullable=True)  # uptrend | correction | rally_attempt
    rally_day = Column(Integer, nullable=True)
    rally_low_date = Column(Date, nullable=True)
    failed_ftd_date = Column(Date, nullable=True)

    # Volatility + breadth inputs (transparency)
    vix = Column(Float, nullable=True)  # None for non-US / when unavailable
    net_4pct = Column(Integer, nullable=True)  # stocks_up_4pct - stocks_down_4pct

    # Per-component score contributions for the UI "why" ({label: delta}).
    # JsonColumn = JSON on SQLite (tests), JSONB on Postgres — matches the migration.
    components = Column(JsonColumn, nullable=True)
    benchmark_symbol = Column(String(16), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("date", "market", name="uix_exposure_date_market"),
        Index("idx_exposure_date", "date"),
        Index("idx_exposure_market_date", "market", "date"),
    )

    def __repr__(self):
        return (
            f"<MarketExposure(market={self.market}, date={self.date}, "
            f"score={self.exposure_score}, stance={self.stance!r})>"
        )
