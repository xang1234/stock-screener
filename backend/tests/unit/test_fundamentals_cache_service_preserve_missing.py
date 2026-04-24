from __future__ import annotations

from types import SimpleNamespace

from app.services.fundamentals_cache_service import _assign_if_present


def test_assign_if_present_writes_when_value_is_non_none():
    record = SimpleNamespace(market_cap=None)
    _assign_if_present(record, "market_cap", {"market_cap": 123_000_000}, "market_cap")
    assert record.market_cap == 123_000_000


def test_assign_if_present_preserves_prior_value_when_key_missing():
    record = SimpleNamespace(market_cap=999_000_000)
    _assign_if_present(record, "market_cap", {"shares_outstanding": 5}, "market_cap")
    assert record.market_cap == 999_000_000


def test_assign_if_present_preserves_prior_value_when_value_is_none():
    record = SimpleNamespace(market_cap=999_000_000)
    _assign_if_present(record, "market_cap", {"market_cap": None}, "market_cap")
    assert record.market_cap == 999_000_000


def test_assign_if_present_allows_writing_zero():
    record = SimpleNamespace(avg_volume=None)
    _assign_if_present(record, "avg_volume", {"avg_volume": 0}, "avg_volume")
    assert record.avg_volume == 0
