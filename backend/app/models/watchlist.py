"""Watchlist model"""
from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from ..database import Base


class Watchlist(Base):
    """User watchlist for tracking stocks"""

    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)

    # User notes
    notes = Column(Text)

    # Timestamp
    added_at = Column(DateTime(timezone=True), server_default=func.now())
