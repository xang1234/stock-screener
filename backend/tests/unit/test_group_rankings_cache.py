"""Tests for the epoch-invalidated group-rankings Redis cache."""

import json
from datetime import date
from unittest.mock import MagicMock, patch

from app.services.group_rankings_cache import (
    TTL_SECONDS,
    bump_group_rankings_epoch,
    cached_group_payload,
)

MODULE = "app.services.group_rankings_cache"


def _fake_redis(store=None):
    store = store if store is not None else {}
    client = MagicMock()
    client.get.side_effect = lambda key: store.get(key)
    client.setex.side_effect = lambda key, ttl, value: store.__setitem__(key, value)
    client.incr.side_effect = lambda key: store.__setitem__(
        key, str(int(store.get(key) or 0) + 1)
    )
    return client, store


def test_miss_computes_and_caches():
    client, store = _fake_redis()
    compute = MagicMock(return_value=[{"rank": 1, "date": date(2026, 6, 12)}])

    with patch(f"{MODULE}.get_redis_client", return_value=client):
        result = cached_group_payload(
            market="us", name="rankings", params="limit=197", compute=compute
        )

    assert result == [{"rank": 1, "date": date(2026, 6, 12)}]
    compute.assert_called_once()
    cached_keys = [k for k in store if "rankings" in k]
    assert cached_keys == ["groups:US:e0:rankings:limit=197"]
    # Dates serialize to ISO strings for the JSON round-trip
    assert json.loads(store[cached_keys[0]]) == [{"rank": 1, "date": "2026-06-12"}]


def test_hit_skips_compute():
    client, store = _fake_redis(
        {"groups:US:e0:rankings:limit=197": json.dumps([{"rank": 1}])}
    )
    compute = MagicMock()

    with patch(f"{MODULE}.get_redis_client", return_value=client):
        result = cached_group_payload(
            market="US", name="rankings", params="limit=197", compute=compute
        )

    assert result == [{"rank": 1}]
    compute.assert_not_called()


def test_epoch_bump_invalidates():
    client, store = _fake_redis(
        {"groups:US:e0:rankings:limit=197": json.dumps([{"rank": 1}])}
    )
    compute = MagicMock(return_value=[{"rank": 2}])

    with patch(f"{MODULE}.get_redis_client", return_value=client):
        bump_group_rankings_epoch("US")
        result = cached_group_payload(
            market="US", name="rankings", params="limit=197", compute=compute
        )

    assert result == [{"rank": 2}]
    compute.assert_called_once()
    assert "groups:US:e1:rankings:limit=197" in store


def test_empty_results_not_cached():
    client, store = _fake_redis()
    compute = MagicMock(return_value=[])

    with patch(f"{MODULE}.get_redis_client", return_value=client):
        result = cached_group_payload(
            market="US", name="rankings", params="limit=197", compute=compute
        )

    assert result == []
    assert not [k for k in store if "rankings" in k]


def test_redis_down_degrades_to_compute():
    compute = MagicMock(return_value=[{"rank": 1}])

    with patch(f"{MODULE}.get_redis_client", side_effect=ConnectionError("down")):
        result = cached_group_payload(
            market="US", name="rankings", params="limit=197", compute=compute
        )
        bump_group_rankings_epoch("US")  # must not raise

    assert result == [{"rank": 1}]
    compute.assert_called_once()


def test_cache_writes_use_ttl():
    client, _ = _fake_redis()
    with patch(f"{MODULE}.get_redis_client", return_value=client):
        cached_group_payload(
            market="US", name="movers", params="period=1w:limit=10",
            compute=lambda: {"gainers": [{"rank": 1}], "losers": []},
            should_cache=lambda v: bool(v.get("gainers") or v.get("losers")),
        )
    _, ttl, _ = client.setex.call_args[0]
    assert ttl == TTL_SECONDS
