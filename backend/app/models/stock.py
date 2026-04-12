"""Stock-related database models"""
from sqlalchemy import Column, Integer, String, Float, BigInteger, Date, DateTime, Index, JSON, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from ..database import Base


class StockPrice(Base):
    """Historical price data for stocks (OHLCV)"""

    __tablename__ = "stock_prices"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)
    adj_close = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uix_symbol_date"),
        Index("idx_symbol_date", "symbol", "date"),
    )


class StockFundamental(Base):
    """Fundamental data for stocks"""

    __tablename__ = "stock_fundamentals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)

    # Market data
    market_cap = Column(BigInteger)
    shares_outstanding = Column(BigInteger)
    avg_volume = Column(BigInteger)
    relative_volume = Column(Float)

    # Valuation metrics
    pe_ratio = Column(Float)
    forward_pe = Column(Float)
    peg_ratio = Column(Float)
    price_to_book = Column(Float)
    price_to_sales = Column(Float)
    price_to_cash = Column(Float)
    price_to_fcf = Column(Float)
    ev_ebitda = Column(Float)
    ev_sales = Column(Float)
    target_price = Column(Float)

    # Earnings/Growth metrics
    eps_current = Column(Float)
    eps_next_y = Column(Float)
    eps_next_5y = Column(Float)
    eps_next_q = Column(Float)
    eps_growth_quarterly = Column(Float)
    eps_growth_annual = Column(Float)
    eps_growth_yy = Column(Float)

    # Revenue metrics
    revenue_current = Column(BigInteger)
    revenue_growth = Column(Float)
    sales_past_5y = Column(Float)
    sales_growth_yy = Column(Float)
    sales_growth_qq = Column(Float)  # Sales growth quarter-over-quarter (%)

    # Quarter metadata (consolidated from QuarterlyData)
    recent_quarter_date = Column(String(50))  # e.g., "2024-Q4"
    previous_quarter_date = Column(String(50))  # e.g., "2024-Q3"

    # Profitability metrics
    profit_margin = Column(Float)
    operating_margin = Column(Float)
    gross_margin = Column(Float)
    roe = Column(Float)
    roa = Column(Float)
    roic = Column(Float)

    # Financial health
    current_ratio = Column(Float)
    quick_ratio = Column(Float)
    debt_to_equity = Column(Float)
    lt_debt_to_equity = Column(Float)

    # Ownership & sentiment
    insider_ownership = Column(Float)
    insider_transactions = Column(Float)
    institutional_ownership = Column(Float)
    institutional_transactions = Column(Float)
    institutional_change = Column(Float)  # Legacy field
    short_float = Column(Float)
    short_ratio = Column(Float)
    short_interest = Column(BigInteger)

    # Technical indicators
    beta = Column(Float)
    rsi_14 = Column(Float)
    atr_14 = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    volatility_week = Column(Float)
    volatility_month = Column(Float)

    # Performance metrics
    perf_week = Column(Float)
    perf_month = Column(Float)
    perf_quarter = Column(Float)
    perf_half_year = Column(Float)
    perf_year = Column(Float)
    perf_ytd = Column(Float)

    # Dividend metrics
    dividend_ttm = Column(Float)
    dividend_yield = Column(Float)
    payout_ratio = Column(Float)

    # 52-week range
    week_52_high = Column(Float)
    week_52_high_distance = Column(Float)
    week_52_low = Column(Float)
    week_52_low_distance = Column(Float)

    # Company info
    sector = Column(String(100))
    industry = Column(String(100))
    country = Column(String(50))
    employees = Column(Integer)
    ipo_date = Column(Date)  # First public trading date (from yfinance firstTradeDateEpochUtc)

    # Company descriptions
    description_yfinance = Column(String)  # From yfinance longBusinessSummary
    description_finviz = Column(String)    # From finviz ticker_description()

    # Analyst recommendations
    recommendation = Column(Float)

    # EPS Rating (IBD-style 0-99 percentile)
    eps_5yr_cagr = Column(Float)          # 5-year compound annual growth rate
    eps_q1_yoy = Column(Float)            # Most recent quarter YoY growth
    eps_q2_yoy = Column(Float)            # Prior quarter YoY growth
    eps_raw_score = Column(Float)         # Raw composite score (before percentile)
    eps_rating = Column(Integer)          # 0-99 percentile rank
    eps_years_available = Column(Integer) # Data completeness tracking (1-5)

    # Data source tracking
    data_source = Column(String(20))
    data_source_timestamp = Column(DateTime(timezone=True))
    finviz_snapshot_revision = Column(String(128), index=True)
    finviz_snapshot_at = Column(DateTime(timezone=True))
    yahoo_profile_refreshed_at = Column(DateTime(timezone=True))
    yahoo_statements_refreshed_at = Column(DateTime(timezone=True))
    technicals_refreshed_at = Column(DateTime(timezone=True))

    # Field-level quality metadata (T2)
    # Market-aware 0-100 score. Indexed for quality-tier filtering by
    # scanners/ranking logic. NULL means "not yet computed" — treat as unknown.
    field_completeness_score = Column(Integer, index=True)
    # {field_name: provider_name} for every populated field. JSONB in
    # production (PG) so T4 can filter on key paths efficiently; tests use
    # SQLite which falls back to the JSON variant.
    field_provenance = Column(JSONB().with_variant(JSON(), "sqlite"))

    # Metadata
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class StockTechnical(Base):
    """Technical indicators for stocks"""

    __tablename__ = "stock_technicals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)

    # Current price
    current_price = Column(Float)

    # Moving averages
    ma_50 = Column(Float)
    ma_150 = Column(Float)
    ma_200 = Column(Float)
    ma_200_month_ago = Column(Float)  # For trend calculation

    # Relative strength
    rs_rating = Column(Float)  # 0-100

    # 52-week range
    high_52w = Column(Float)
    low_52w = Column(Float)

    # Volume
    avg_volume_50d = Column(BigInteger)
    current_volume = Column(BigInteger)

    # Weinstein stage
    stage = Column(Integer)  # 1-4

    # VCP
    vcp_score = Column(Float)  # 0-100

    # Metadata
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class StockIndustry(Base):
    """Stock-to-industry classification mapping"""

    __tablename__ = "stock_industry"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)

    # GICS classification
    sector = Column(String(100), index=True)  # e.g., "Information Technology"
    industry_group = Column(String(100))  # e.g., "Software & Services"
    industry = Column(String(100), index=True)  # e.g., "Application Software"
    sub_industry = Column(String(100))  # e.g., "Application Software"

    # Metadata
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("idx_sector_industry", "sector", "industry"),)
