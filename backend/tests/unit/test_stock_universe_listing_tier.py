from __future__ import annotations

from app.models.stock_universe import (
    UNIVERSE_EVENT_LISTING_TIER_CHANGED,
    UNIVERSE_EVENT_STATUS_CHANGED,
    StockUniverse,
    StockUniverseStatusEvent,
)
from app.services.stock_universe_service import StockUniverseService


def test_stock_universe_listing_tier_is_nullable_row_metadata() -> None:
    row = StockUniverse(symbol="0700.HK", market="HK")

    assert row.listing_tier is None

    row.listing_tier = "main_board"

    assert row.listing_tier == "main_board"


def test_status_event_defaults_to_status_changed_event_type() -> None:
    event = StockUniverseService._build_status_event_record(
        symbol="0700.HK",
        old_status=None,
        new_status="active",
        trigger_source="test",
        reason="created",
    )

    assert event.event_type == UNIVERSE_EVENT_STATUS_CHANGED


def test_status_event_can_record_listing_tier_change_without_status_transition() -> None:
    assert StockUniverseStatusEvent.new_status.property.columns[0].nullable is True

    event = StockUniverseService._build_metadata_event_record(
        symbol="0700.HK",
        event_type=UNIVERSE_EVENT_LISTING_TIER_CHANGED,
        trigger_source="test",
        reason="listing tier changed",
        payload={"previous": None, "current": "main_board"},
    )

    assert event.event_type == "listing_tier_changed"
    assert event.old_status is None
    assert event.new_status is None
