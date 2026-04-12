"""FX rate log — historical record of exchange rates used for USD normalisation.

Rows are append-only per ``(from_currency, as_of_date, source)``. Each row
represents "at this date, from this source, the rate used to convert
``from_currency`` → USD was ``rate``".

This is the *log*; the per-fundamentals-row FX snapshot lives in
``StockFundamental.fx_metadata`` (JSONB). The table supports time-series
auditing; the row snapshot supports single-row replay without a join.
"""
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, UniqueConstraint, Index
from sqlalchemy.sql import func

from ..database import Base


class FXRate(Base):
    """Daily FX rate to USD from a named source."""

    __tablename__ = "fx_rates"

    id = Column(Integer, primary_key=True, index=True)
    from_currency = Column(String(8), nullable=False)  # e.g. "HKD"
    # ``to_currency`` is effectively always "USD" for T3, but modelled
    # explicitly so future cross-currency needs don't require a migration.
    to_currency = Column(String(8), nullable=False, default="USD")
    as_of_date = Column(Date, nullable=False)
    rate = Column(Float, nullable=False)
    source = Column(String(32), nullable=False)  # "yfinance", "identity", ...
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "from_currency", "to_currency", "as_of_date", "source",
            name="uq_fx_rates_currency_date_source",
        ),
        Index("ix_fx_rates_lookup", "from_currency", "to_currency", "as_of_date"),
    )
