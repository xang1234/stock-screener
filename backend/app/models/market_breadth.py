"""Market breadth indicator models"""

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from ..database import Base


class MarketBreadth(Base):
    """
    Daily market breadth indicators based on StockBee methodology.

    Rows are partitioned by ``market`` — HK/JP/TW/IN share the same calendar
    dates as US but have independent universes, so the UNIQUE key is
    ``(date, market)``. Rolling-ratio history is computed per market.
    """

    __tablename__ = "market_breadth"

    id = Column(Integer, primary_key=True, index=True)
    market = Column(String(8), nullable=False, default="US", index=True)
    date = Column(Date, nullable=False, index=True)

    # Daily movers (4%+ threshold)
    stocks_up_4pct = Column(Integer, default=0, nullable=False)
    stocks_down_4pct = Column(Integer, default=0, nullable=False)

    # Multi-day ratios (up/down over period)
    ratio_5day = Column(
        Float, nullable=True
    )  # Nullable for edge cases (denominator = 0)
    ratio_10day = Column(Float, nullable=True)

    # Quarterly movers (65 trading sessions, 25% threshold)
    stocks_up_25pct_quarter = Column(Integer, default=0, nullable=False)
    stocks_down_25pct_quarter = Column(Integer, default=0, nullable=False)

    # Monthly movers (exactly 20 trading sessions, 25% threshold)
    stocks_up_25pct_month = Column(Integer, default=0, nullable=False)
    stocks_down_25pct_month = Column(Integer, default=0, nullable=False)

    # Monthly extreme movers (exactly 20 trading sessions, 50% threshold)
    stocks_up_50pct_month = Column(Integer, default=0, nullable=False)
    stocks_down_50pct_month = Column(Integer, default=0, nullable=False)

    # 34-day movers (13% threshold - IBD-style)
    stocks_up_13pct_34days = Column(Integer, default=0, nullable=False)
    stocks_down_13pct_34days = Column(Integer, default=0, nullable=False)

    # Broad-universe context metrics
    advancing_count = Column(Integer, nullable=True)
    declining_count = Column(Integer, nullable=True)
    unchanged_count = Column(Integer, nullable=True)
    new_high_52week_count = Column(Integer, nullable=True)
    new_low_52week_count = Column(Integer, nullable=True)
    t2108_count = Column(Integer, nullable=True)
    t2108_pct = Column(Float, nullable=True)
    atr_10x_extension_count = Column(Integer, nullable=True)
    above_sma200_count = Column(Integer, nullable=True)
    above_sma200_pct = Column(Float, nullable=True)
    down_20pct_from_high_count = Column(Integer, nullable=True)
    down_20pct_from_high_pct = Column(Float, nullable=True)
    up_4pct_near_high_count = Column(Integer, nullable=True)

    # Explicit metric-family denominators
    broad_universe_count = Column(Integer, nullable=True)
    advance_decline_eligible_count = Column(Integer, nullable=True)
    stockbee_daily_eligible_count = Column(Integer, nullable=True)
    stockbee_month_eligible_count = Column(Integer, nullable=True)
    stockbee_34day_eligible_count = Column(Integer, nullable=True)
    stockbee_quarter_eligible_count = Column(Integer, nullable=True)
    t2108_eligible_count = Column(Integer, nullable=True)
    high_low_52week_eligible_count = Column(Integer, nullable=True)
    atr_extension_eligible_count = Column(Integer, nullable=True)
    sma200_eligible_count = Column(Integer, nullable=True)
    off_high_eligible_count = Column(Integer, nullable=True)

    # Metadata
    # Deprecated compatibility alias; current writers set this equal to
    # broad_universe_count.
    total_stocks_scanned = Column(Integer, default=0, nullable=False)
    eligibility_signature = Column(String(64), nullable=True)
    stockbee_eligibility_signature = Column(String(64), nullable=True)
    contributor_calculation_signature = Column(String(64), nullable=True)
    calculation_revision = Column(Integer, nullable=True)
    calculation_duration_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("date", "market", name="uix_breadth_date_market"),
        Index("idx_breadth_date", "date"),
        Index("idx_breadth_market_date", "market", "date"),
    )

    def __repr__(self):
        return (
            f"<MarketBreadth(market={self.market}, date={self.date}, "
            f"up_4pct={self.stocks_up_4pct}, down_4pct={self.stocks_down_4pct})>"
        )
