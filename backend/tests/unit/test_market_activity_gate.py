"""Tests for market scan gating decisions."""

from __future__ import annotations

import pytest

from app.services.market_activity_gate import (
    MarketActivityGate,
    MarketGateAllowed,
    MarketGateConflict,
)


class _Session:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _gate_with_activity(activity: dict[str, object], session: _Session | None = None) -> MarketActivityGate:
    resolved_session = session or _Session()
    return MarketActivityGate(
        session_factory=lambda: resolved_session,
        runtime_activity_reader=lambda _: activity,
    )


def test_market_activity_gate_allows_when_no_matching_activity():
    gate = _gate_with_activity(
        {
            "markets": [
                {"market": "HK", "stage_key": "breadth", "stage_label": "Breadth", "status": "running"},
                {"market": "US", "stage_key": "prices", "stage_label": "Price Refresh", "status": "running"},
            ]
        }
    )

    result = gate.check("hk")

    assert isinstance(result, MarketGateAllowed)
    assert result.market.code == "HK"


def test_market_activity_gate_conflicts_on_active_prices_or_fundamentals():
    gate = _gate_with_activity(
        {
            "markets": [
                {
                    "market": "HK",
                    "lifecycle": "daily_refresh",
                    "stage_key": "prices",
                    "stage_label": "Price Refresh",
                    "status": "running",
                },
                {
                    "market": "HK",
                    "lifecycle": "weekly_refresh",
                    "stage_key": "fundamentals",
                    "stage_label": "Fundamentals Refresh",
                    "status": "queued",
                },
            ]
        }
    )

    result = gate.check("HK")

    assert isinstance(result, MarketGateConflict)
    assert result.detail["code"] == "market_refresh_active"
    assert result.detail["market"] == "HK"
    assert result.detail["active_stages"] == ["fundamentals", "prices"]
    assert result.detail["lifecycle"] == "daily_refresh"
    assert "price refresh" in str(result.detail["message"]).lower()


def test_market_activity_gate_closes_session_after_read():
    session = _Session()
    gate = _gate_with_activity({"markets": []}, session=session)

    gate.check("US")

    assert session.closed is True


def test_market_activity_gate_rejects_unknown_market_before_reading_activity():
    read_called = False

    def reader(_):
        nonlocal read_called
        read_called = True
        return {"markets": []}

    gate = MarketActivityGate(
        session_factory=_Session,
        runtime_activity_reader=reader,
    )

    with pytest.raises(ValueError):
        gate.check("UK")

    assert read_called is False

