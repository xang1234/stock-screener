"""Tests for exact scan signature helpers."""

from app.domain.scanning.signature import (
    build_scan_signature_payload,
    hash_scan_signature,
    hash_universe_symbols,
)
from app.schemas.universe import UniverseType


def test_signature_payload_normalizes_screeners_and_universe_type():
    payload = build_scan_signature_payload(
        universe_type=UniverseType.ALL,
        screeners=["canslim", "minervini", "canslim"],
        composite_method="Weighted_Average",
        criteria={"b": 2, "a": 1},
    )

    assert payload["universe_type"] == "all"
    assert payload["screeners"] == ["canslim", "minervini"]
    assert payload["composite_method"] == "weighted_average"
    assert list(payload["criteria"].keys()) == ["a", "b"]


def test_universe_hash_is_order_insensitive():
    assert hash_universe_symbols(["AAPL", "MSFT"]) == hash_universe_symbols(["MSFT", "AAPL"])


def test_signature_hash_changes_when_criteria_changes():
    left = build_scan_signature_payload(
        universe_type="all",
        screeners=["minervini"],
        composite_method="weighted_average",
        criteria={"include_vcp": True},
    )
    right = build_scan_signature_payload(
        universe_type="all",
        screeners=["minervini"],
        composite_method="weighted_average",
        criteria={"include_vcp": False},
    )

    assert hash_scan_signature(left) != hash_scan_signature(right)
