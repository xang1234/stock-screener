from sqlalchemy import Column, Integer, String, Float, Date
from ..database import Base


class SignalArchive(Base):
    __tablename__ = "signal_archive"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    signal_date = Column(Date, nullable=False)
    entry_price = Column(Float)
    stop_loss = Column(Float)
    target_price = Column(Float)
    screener = Column(String(50))
    composite_score = Column(Float)
    sector = Column(String(100))
    stage = Column(Integer)
    outcome = Column(String(20), nullable=True)
    outcome_date = Column(Date, nullable=True)
    outcome_price = Column(Float, nullable=True)
    pct_return = Column(Float, nullable=True)
    days_held = Column(Integer, nullable=True)
