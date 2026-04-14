"""Per-market warmup heartbeat (bead asia.9.2 — fix for 9.1 inherited issue #2).

Pre-9.2: WARMUP_HEARTBEAT_KEY was a single global Redis key. With 9.1's
per-market parallelism, simultaneous US+HK refreshes both wrote the same
key — the dashboard "X/Y progress" was whichever task wrote last. These
tests lock in the per-market scoping fix.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.services.cache.price_cache_warmup import PriceCacheWarmupStore, scoped_heartbeat_key


def _make_store():
    redis_client = MagicMock()
    logger = MagicMock()
    store = PriceCacheWarmupStore(
        logger=logger,
        redis_client=redis_client,
        metadata_key="cache:warmup:metadata",
        heartbeat_key="cache:warmup:heartbeat",
    )
    return store, redis_client


class TestScopedKey:
    @pytest.mark.parametrize("market,expected", [
        ("US", "cache:warmup:heartbeat:us"),
        ("HK", "cache:warmup:heartbeat:hk"),
        ("hk", "cache:warmup:heartbeat:hk"),
        (" JP ", "cache:warmup:heartbeat:jp"),
        (None, "cache:warmup:heartbeat:shared"),
    ])
    def test_scoped_key(self, market, expected):
        assert scoped_heartbeat_key("cache:warmup:heartbeat", market) == expected


class TestUpdateHeartbeatPerMarket:
    def test_us_and_hk_write_independent_keys(self):
        store, redis_client = _make_store()

        store.update_warmup_heartbeat(current=10, total=100, market="US")
        store.update_warmup_heartbeat(current=5, total=50, market="HK")

        # Two separate setex calls, two separate keys
        keys_written = [c.args[0] for c in redis_client.setex.call_args_list]
        assert "cache:warmup:heartbeat:us" in keys_written
        assert "cache:warmup:heartbeat:hk" in keys_written

    def test_payload_includes_market_field(self):
        store, redis_client = _make_store()
        store.update_warmup_heartbeat(current=42, total=100, market="JP")

        payload_str = redis_client.setex.call_args.args[2]
        payload = json.loads(payload_str)
        assert payload["market"] == "jp"
        assert payload["current"] == 42

    def test_no_market_uses_shared_key(self):
        store, redis_client = _make_store()
        store.update_warmup_heartbeat(current=1, total=1)
        assert redis_client.setex.call_args.args[0] == "cache:warmup:heartbeat:shared"


class TestReadHeartbeatPerMarket:
    def test_get_heartbeat_info_reads_market_specific_key(self):
        store, redis_client = _make_store()
        redis_client.get.return_value = json.dumps(
            {"current": 42, "total": 100, "percent": 42.0, "status": "running",
             "updated_at": "2026-04-14T10:00:00", "market": "hk"}
        ).encode()

        store.get_heartbeat_info(market="HK")
        assert redis_client.get.call_args.args[0] == "cache:warmup:heartbeat:hk"

    def test_get_task_progress_per_market(self):
        store, redis_client = _make_store()
        redis_client.get.return_value = json.dumps(
            {"current": 7, "total": 10, "percent": 70.0}
        ).encode()

        progress = store.get_task_progress(market="TW")
        assert progress == {"current": 7, "total": 10, "progress": 70.0}
        assert redis_client.get.call_args.args[0] == "cache:warmup:heartbeat:tw"


class TestCompletePerMarket:
    def test_complete_writes_terminal_state_to_market_key(self):
        store, redis_client = _make_store()
        # First call (in get_heartbeat_info) returns prior state
        redis_client.get.return_value = json.dumps({
            "current": 100, "total": 100, "percent": 99.0, "status": "running",
            "updated_at": "2026-04-14T10:00:00",
        }).encode()

        store.complete_warmup_heartbeat("completed", market="US")

        # setex was called on the US key with terminal status
        setex_call = redis_client.setex.call_args
        assert setex_call.args[0] == "cache:warmup:heartbeat:us"
        payload = json.loads(setex_call.args[2])
        assert payload["status"] == "completed"
        assert payload["percent"] == 100.0
        assert payload["market"] == "us"
