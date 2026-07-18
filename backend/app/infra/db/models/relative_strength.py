from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    SmallInteger,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class MarketRsRun(Base):
    __tablename__ = "market_rs_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market = Column(String(8), nullable=False)
    as_of_date = Column(Date, nullable=False)
    formula_version = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False)
    benchmark_symbol = Column(String(32), nullable=False)
    benchmark_as_of_date = Column(Date, nullable=False)
    universe_hash = Column(String(64), nullable=False)
    expected_symbol_count = Column(Integer, nullable=False)
    eligible_symbol_count = Column(Integer, nullable=False, default=0)
    excluded_symbol_count = Column(Integer, nullable=False, default=0)
    diagnostics_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)

    rows = relationship(
        "StockRsSnapshot",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "market", "as_of_date", "formula_version", name="uq_market_rs_run"
        ),
        CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_market_rs_run_status",
        ),
        Index(
            "ix_market_rs_run_lookup",
            "market",
            "formula_version",
            "as_of_date",
            "status",
        ),
    )


class StockRsSnapshot(Base):
    __tablename__ = "stock_rs_snapshots"

    run_id = Column(
        Integer,
        ForeignKey("market_rs_runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    symbol = Column(String(20), primary_key=True)
    overall_rs = Column(SmallInteger, nullable=False)
    rs_1m = Column(SmallInteger, nullable=False)
    rs_3m = Column(SmallInteger, nullable=False)
    rs_6m = Column(SmallInteger, nullable=False)
    rs_9m = Column(SmallInteger, nullable=False)
    rs_12m = Column(SmallInteger, nullable=False)
    weighted_composite = Column(Float, nullable=False)
    excess_return_1m = Column(Float, nullable=False)
    excess_return_3m = Column(Float, nullable=False)
    excess_return_6m = Column(Float, nullable=False)
    excess_return_9m = Column(Float, nullable=False)
    excess_return_12m = Column(Float, nullable=False)

    run = relationship("MarketRsRun", back_populates="rows")

    __table_args__ = (
        CheckConstraint(
            "overall_rs BETWEEN 1 AND 99 AND rs_1m BETWEEN 1 AND 99 "
            "AND rs_3m BETWEEN 1 AND 99 AND rs_6m BETWEEN 1 AND 99 "
            "AND rs_9m BETWEEN 1 AND 99 AND rs_12m BETWEEN 1 AND 99",
            name="ck_stock_rs_rating_range",
        ),
        Index("ix_stock_rs_symbol_run", "symbol", "run_id"),
    )


class MarketRsFormulaPointer(Base):
    __tablename__ = "market_rs_formula_pointers"

    market = Column(String(8), primary_key=True)
    formula_version = Column(String(64), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
