"""Industry and sector models"""
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, ForeignKey, Index, UniqueConstraint
from sqlalchemy.sql import func
from ..database import Base


class Industry(Base):
    """Industry and sector master list (GICS classification)"""

    __tablename__ = "industries"

    id = Column(Integer, primary_key=True, index=True)

    # GICS hierarchy
    sector_name = Column(String(100), nullable=False, index=True)
    industry_group = Column(String(100))
    industry = Column(String(100), nullable=False, index=True)
    sub_industry = Column(String(100))

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("sector_name", "industry", name="uix_sector_industry"),
        Index("idx_sector", "sector_name"),
    )


class IndustryPerformance(Base):
    """Industry group performance metrics"""

    __tablename__ = "industry_performance"

    id = Column(Integer, primary_key=True, index=True)
    industry = Column(String(100), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)

    # Performance metrics
    group_rs = Column(Float)  # Relative strength rating (0-100)
    leadership_score = Column(Float)  # Composite leadership score (0-100)

    # Group composition
    num_stocks = Column(Integer)
    stage2_count = Column(Integer)
    stage2_pct = Column(Float)
    high_rs_count = Column(Integer)  # Stocks with RS > 80
    high_rs_pct = Column(Float)

    # Volume trend
    volume_trend = Column(String(20))  # increasing, decreasing, neutral

    # Timestamp
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("industry", "date", name="uix_industry_date"),
        Index("idx_industry_date", "industry", "date"),
    )


class SectorRotation(Base):
    """Sector rotation tracking"""

    __tablename__ = "sector_rotation"

    id = Column(Integer, primary_key=True, index=True)
    sector = Column(String(100), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)

    # Performance
    rs_rating = Column(Float)  # 0-100
    rs_change = Column(Float)  # Change vs previous month

    # Rotation status
    status = Column(String(20))  # Leading, Weakening, Lagging, Improving

    # Trend
    trend = Column(String(20))  # bullish, bearish, neutral

    # Timestamp
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("sector", "date", name="uix_sector_date"),
        Index("idx_sector_date", "sector", "date"),
    )


class IBDIndustryGroup(Base):
    """IBD Industry Group Classifications"""

    __tablename__ = "ibd_industry_groups"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    industry_group = Column(String(100), nullable=False, index=True)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_ibd_industry_group", "industry_group"),
    )


class IBDGroupPeerCache(Base):
    """Cache of peer metrics for IBD industry groups (per scan)"""

    __tablename__ = "ibd_group_peer_cache"

    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(String(36), ForeignKey("scans.scan_id"), nullable=False, index=True)
    industry_group = Column(String(100), nullable=False, index=True)

    # Aggregate metrics for all stocks in this group (from scan universe)
    total_stocks = Column(Integer)  # Total stocks in this group
    avg_rs_1m = Column(Float)
    avg_rs_3m = Column(Float)
    avg_rs_12m = Column(Float)
    avg_minervini_score = Column(Float)
    avg_composite_score = Column(Float)

    # Top performers in group
    top_symbol = Column(String(20))  # Best performer by composite score
    top_score = Column(Float)

    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("scan_id", "industry_group", name="uix_scan_group"),
        Index("idx_scan_group", "scan_id", "industry_group"),
    )


class IBDGroupRank(Base):
    """Daily Industry Group Rankings based on average RS rating.

    Rankings are partitioned by `market` — HK/JP/TW/IN use their own
    classification schemas (EM Industry / TSE 33-Sector / local index)
    instead of IBD's US-only taxonomy, so groups named identically in
    different markets are distinct rows.
    """

    __tablename__ = "ibd_group_ranks"

    id = Column(Integer, primary_key=True, index=True)
    market = Column(String(8), nullable=False, default="US", index=True)
    industry_group = Column(String(100), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)

    # Core ranking metrics
    rank = Column(Integer, nullable=False)  # 1 = best, higher = worse
    avg_rs_rating = Column(Float, nullable=False)  # Average RS of all stocks in group
    median_rs_rating = Column(Float)  # Median RS of stocks in group
    weighted_avg_rs_rating = Column(Float)  # Market-cap weighted average RS
    rs_std_dev = Column(Float)  # Dispersion of RS ratings (std dev)

    # Composition stats
    num_stocks = Column(Integer, default=0)  # Number of stocks with valid RS
    num_stocks_rs_above_80 = Column(Integer, default=0)  # High RS count

    # Top performer in group
    top_symbol = Column(String(20))
    top_rs_rating = Column(Float)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "industry_group", "date", "market", name="uix_ibd_group_rank_market_date"
        ),
        Index("idx_ibd_group_rank_date", "industry_group", "date"),
        Index("idx_ibd_group_rank_market_date", "market", "date"),
        Index("idx_ibd_rank_by_date", "date", "rank"),
    )
