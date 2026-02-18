"""Scan and scan result models"""
import logging
from sqlalchemy import Column, Integer, String, Float, BigInteger, DateTime, JSON, Index, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..database import Base

logger = logging.getLogger(__name__)


class Scan(Base):
    """Scan metadata and configuration"""

    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(String(36), nullable=False, unique=True, index=True)  # UUID

    # Scan configuration
    criteria = Column(JSON)  # Scan criteria configuration
    universe = Column(String(50))  # Legacy: "test", "all", "custom" — kept for backward compat

    # Structured universe fields (populated by UniverseDefinition)
    universe_key = Column(String(128), index=True)       # Canonical key for retention grouping
    universe_type = Column(String(20), index=True)       # UniverseType enum value
    universe_exchange = Column(String(20), nullable=True, index=True)  # For exchange-scoped scans
    universe_index = Column(String(20), nullable=True, index=True)     # For index-scoped scans
    universe_symbols = Column(JSON, nullable=True)       # Symbol list for custom/test

    # Multi-screener configuration
    screener_types = Column(JSON, default=lambda: ["minervini"])  # List of screener names
    composite_method = Column(String(50), default="weighted_average")  # How to combine scores

    # Results summary
    total_stocks = Column(Integer)
    passed_stocks = Column(Integer)

    # Status
    status = Column(String(20), default="running")  # running, completed, failed
    task_id = Column(String(100), nullable=True)  # Celery task ID for real-time progress

    # Idempotency
    idempotency_key = Column(String(64), nullable=True, unique=True, index=True)

    # Feature Store binding (nullable — legacy scans predate feature store)
    feature_run_id = Column(
        Integer,
        ForeignKey("feature_runs.id"),
        nullable=True,
        index=True,
    )
    feature_run = relationship("FeatureRun", foreign_keys=[feature_run_id], lazy="select")

    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))

    def get_universe_definition(self):
        """
        Reconstruct a UniverseDefinition from the structured DB fields.

        Falls back to from_legacy() for pre-migration rows where
        universe_type is NULL.

        Returns:
            UniverseDefinition instance
        """
        from ..schemas.universe import (
            Exchange,
            IndexName,
            UniverseDefinition,
            UniverseType,
        )

        if self.universe_type is None:
            # Pre-migration row: parse from legacy string
            try:
                return UniverseDefinition.from_legacy(
                    self.universe or "all",
                    self.universe_symbols,
                )
            except Exception as e:
                logger.warning(
                    f"Could not parse legacy universe '{self.universe}' "
                    f"for scan {self.scan_id}: {e}"
                )
                return UniverseDefinition(type=UniverseType.ALL)

        return UniverseDefinition(
            type=UniverseType(self.universe_type),
            exchange=Exchange(self.universe_exchange) if self.universe_exchange else None,
            index=IndexName(self.universe_index) if self.universe_index else None,
            symbols=self.universe_symbols,
        )


class ScanResult(Base):
    """Individual stock results from a scan"""

    __tablename__ = "scan_results"

    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(String(36), ForeignKey("scans.scan_id"), nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)

    # Scores
    composite_score = Column(Float, index=True)
    minervini_score = Column(Float)
    canslim_score = Column(Float)
    ipo_score = Column(Float)
    custom_score = Column(Float)
    volume_breakthrough_score = Column(Float)

    # Rating
    rating = Column(String(20))  # Strong Buy, Buy, Watch, Pass

    # Current data
    price = Column(Float)
    volume = Column(BigInteger)
    market_cap = Column(BigInteger, index=True)  # Market capitalization

    # Phase 3.3: Extracted JSON fields for indexing
    stage = Column(Integer, index=True)  # Weinstein stage (1-4)
    rs_rating = Column(Float, index=True)  # RS Rating (0-100)

    # Multi-period RS ratings for granular filtering
    rs_rating_1m = Column(Float, index=True)   # 1-month RS (0-100)
    rs_rating_3m = Column(Float, index=True)   # 3-month RS (0-100)
    rs_rating_12m = Column(Float, index=True)  # 12-month RS (0-100)

    # Growth metrics (extracted for filtering/sorting)
    eps_growth_qq = Column(Float, index=True)  # EPS growth quarter-over-quarter (%)
    sales_growth_qq = Column(Float, index=True)  # Sales growth quarter-over-quarter (%)
    eps_growth_yy = Column(Float, index=True)  # EPS growth year-over-year (%)
    sales_growth_yy = Column(Float, index=True)  # Sales growth year-over-year (%)

    # Valuation metrics (extracted for filtering/sorting)
    peg_ratio = Column(Float, index=True)  # PEG ratio (P/E to growth)

    # Volatility metrics
    adr_percent = Column(Float, index=True)  # Average Daily Range as percentage

    # EPS Rating (IBD-style 0-99 percentile)
    eps_rating = Column(Integer, index=True)  # EPS Rating for filtering

    # Industry classifications (Phase 4: Industry Groups & Peers)
    ibd_industry_group = Column(String(100), index=True)  # IBD industry group
    ibd_group_rank = Column(Integer, index=True)  # IBD group rank (1=best)
    gics_sector = Column(String(100), index=True)  # GICS sector
    gics_industry = Column(String(100))  # GICS industry (detailed)

    # RS Sparkline data (30-day stock/SPY ratio trend)
    rs_sparkline_data = Column(JSON)  # Array of 30 normalized RS ratio values
    rs_trend = Column(Integer, index=True)  # -1=declining, 0=flat, 1=improving

    # Price Sparkline data (30-day normalized price trend)
    price_sparkline_data = Column(JSON)  # Array of 30 normalized price values
    price_change_1d = Column(Float, index=True)  # 1-day percentage change
    price_trend = Column(Integer, index=True)  # -1=down, 0=flat, 1=up overall

    # Performance filters (price change percentages)
    perf_week = Column(Float, index=True)     # 5-day % change
    perf_month = Column(Float, index=True)    # 21-day % change

    # Qullamaggie extended performance metrics
    perf_3m = Column(Float, index=True)       # 67-day % change (Qullamaggie requires >=50%)
    perf_6m = Column(Float, index=True)       # 126-day % change (Qullamaggie requires >=150%)

    # Episodic Pivot metrics
    gap_percent = Column(Float, index=True)   # Gap up % (Episodic Pivot requires >=10%)
    volume_surge = Column(Float, index=True)  # Volume ratio vs 50-day avg (requires >=2.0)

    # EMA distances (% above/below EMA)
    ema_10_distance = Column(Float, index=True)
    ema_20_distance = Column(Float, index=True)
    ema_50_distance = Column(Float, index=True)

    # 52-week distances (promoted from details JSON for fast filtering)
    week_52_high_distance = Column(Float, index=True)  # % below 52-week high
    week_52_low_distance = Column(Float, index=True)   # % above 52-week low

    # IPO date for filtering by stock age
    ipo_date = Column(String(10), index=True)  # Format: YYYY-MM-DD

    # Beta and Beta-Adjusted RS metrics (Matt Caruso screening)
    beta = Column(Float, index=True)  # 252-day rolling beta
    beta_adj_rs = Column(Float, index=True)  # Beta-adjusted RS (weighted)
    beta_adj_rs_1m = Column(Float, index=True)  # Beta-adjusted RS (1-month)
    beta_adj_rs_3m = Column(Float, index=True)  # Beta-adjusted RS (3-month)
    beta_adj_rs_12m = Column(Float, index=True)  # Beta-adjusted RS (12-month)

    # Detailed breakdown
    details = Column(JSON)  # Full score breakdown and metrics

    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_scan_result_score", "scan_id", "composite_score"),
        Index("idx_symbol_scan", "symbol", "scan_id"),
        # Phase 3.3: Composite indexes for filtering
        Index("idx_scan_stage", "scan_id", "stage"),
        Index("idx_scan_rs_rating", "scan_id", "rs_rating"),
        # Multi-period RS indexes
        Index("idx_scan_rs_1m", "scan_id", "rs_rating_1m"),
        Index("idx_scan_rs_3m", "scan_id", "rs_rating_3m"),
        Index("idx_scan_rs_12m", "scan_id", "rs_rating_12m"),
        # Growth metrics indexes
        Index("idx_scan_eps_growth", "scan_id", "eps_growth_qq"),
        Index("idx_scan_sales_growth", "scan_id", "sales_growth_qq"),
        Index("idx_scan_eps_growth_yy", "scan_id", "eps_growth_yy"),
        Index("idx_scan_sales_growth_yy", "scan_id", "sales_growth_yy"),
        # Valuation metrics indexes
        Index("idx_scan_peg", "scan_id", "peg_ratio"),
        # Phase 4: Industry group indexes
        Index("idx_scan_ibd_group", "scan_id", "ibd_industry_group"),
        Index("idx_scan_ibd_group_rank", "scan_id", "ibd_group_rank"),
        Index("idx_scan_gics_sector", "scan_id", "gics_sector"),
        # RS Sparkline trend index
        Index("idx_scan_rs_trend", "scan_id", "rs_trend"),
        # Price Sparkline indexes
        Index("idx_scan_price_trend", "scan_id", "price_trend"),
        Index("idx_scan_price_change_1d", "scan_id", "price_change_1d"),
        # Volume and Market Cap indexes
        Index("idx_scan_volume", "scan_id", "volume"),
        Index("idx_scan_market_cap", "scan_id", "market_cap"),
        # EPS Rating index
        Index("idx_scan_eps_rating", "scan_id", "eps_rating"),
        # Performance filters indexes
        Index("idx_scan_perf_week", "scan_id", "perf_week"),
        Index("idx_scan_perf_month", "scan_id", "perf_month"),
        # EMA distance indexes
        Index("idx_scan_ema_10", "scan_id", "ema_10_distance"),
        Index("idx_scan_ema_20", "scan_id", "ema_20_distance"),
        Index("idx_scan_ema_50", "scan_id", "ema_50_distance"),
        # 52-week distance indexes
        Index("idx_scan_52w_high", "scan_id", "week_52_high_distance"),
        Index("idx_scan_52w_low", "scan_id", "week_52_low_distance"),
        # IPO date index
        Index("idx_scan_ipo_date", "scan_id", "ipo_date"),
        # Beta and Beta-Adjusted RS indexes
        Index("idx_scan_beta", "scan_id", "beta"),
        Index("idx_scan_beta_adj_rs", "scan_id", "beta_adj_rs"),
        Index("idx_scan_beta_adj_rs_1m", "scan_id", "beta_adj_rs_1m"),
        Index("idx_scan_beta_adj_rs_3m", "scan_id", "beta_adj_rs_3m"),
        Index("idx_scan_beta_adj_rs_12m", "scan_id", "beta_adj_rs_12m"),
        # Qullamaggie screening metrics indexes
        Index("idx_scan_perf_3m", "scan_id", "perf_3m"),
        Index("idx_scan_perf_6m", "scan_id", "perf_6m"),
        Index("idx_scan_gap_percent", "scan_id", "gap_percent"),
        Index("idx_scan_volume_surge", "scan_id", "volume_surge"),
    )
