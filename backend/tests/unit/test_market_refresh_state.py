"""Observed market refresh state persistence."""

from __future__ import annotations

from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base

import app.models.app_settings  # noqa: F401


def test_market_refresh_state_round_trips_successful_trading_day() -> None:
    from app.services.market_refresh_state_service import (
        get_market_refresh_state,
        record_market_refresh_success,
    )

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        record_market_refresh_success(
            session,
            market="hk",
            trading_day=date(2026, 3, 16),
            success_rate=0.98,
        )

        state = get_market_refresh_state(session, "HK")
    finally:
        session.close()
        engine.dispose()

    assert state is not None
    assert state["market"] == "HK"
    assert state["status"] == "completed"
    assert state["last_refreshed_trading_day"] == "2026-03-16"
    assert state["success_rate"] == 0.98

