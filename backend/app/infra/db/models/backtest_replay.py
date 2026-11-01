from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    SmallInteger,
    String,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from app.database import Base

# Tags every row the replay writes and the Backtest page reads. Change the pick
# rule and this value together, so old and new picks never mix.
DEFINITION_VERSION = "minervini-top10-v1"


class BacktestReplayPick(Base):
    """One session's replayed pick, scored point-in-time from stored prices and Market RS."""

    __tablename__ = "backtest_replay_picks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market = Column(String(8), nullable=False)
    as_of_date = Column(Date, nullable=False)
    rank = Column(SmallInteger, nullable=False)
    symbol = Column(String(20), nullable=False)
    minervini_score = Column(Float, nullable=False)
    rs_rating = Column(SmallInteger, nullable=False)
    stage = Column(SmallInteger, nullable=True)
    definition_version = Column(String(64), nullable=False)
    market_rs_run_id = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "market", "as_of_date", "definition_version", "rank",
            name="uq_backtest_replay_pick_rank",
        ),
        Index("ix_backtest_replay_picks_window", "market", "definition_version", "as_of_date"),
    )
