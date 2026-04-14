"""Unit tests for PriceCacheWarmupStore payload handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from app.services.cache.price_cache_warmup import PriceCacheWarmupStore


class _FakeRedis:
    def __init__(self, payload: str | None = None):
        self.payload = payload
        self.last_set_key = None
        self.last_set_ttl = None
        self.last_set_value = None

    def get(self, _key):
        return self.payload

    def setex(self, key, ttl, value):
        self.last_set_key = key
        self.last_set_ttl = ttl
        self.last_set_value = value
        self.payload = value


def test_get_warmup_metadata_ignores_non_mapping_payload():
    logger = MagicMock()
    redis_client = _FakeRedis(payload=json.dumps(["unexpected", "shape"]))
    store = PriceCacheWarmupStore(
        logger=logger,
        redis_client=redis_client,
        metadata_key="cache:warmup:metadata",
        heartbeat_key="cache:warmup:heartbeat",
    )

    assert store.get_warmup_metadata() is None
    logger.warning.assert_called_once()


def test_complete_warmup_heartbeat_preserves_progress_fields():
    logger = MagicMock()
    redis_client = _FakeRedis(
        payload=json.dumps(
            {
                "status": "running",
                "current": 3,
                "total": 10,
                "percent": 30.0,
                "updated_at": "2026-04-08T00:00:00",
            }
        )
    )
    store = PriceCacheWarmupStore(
        logger=logger,
        redis_client=redis_client,
        metadata_key="cache:warmup:metadata",
        heartbeat_key="cache:warmup:heartbeat",
    )

    store.complete_warmup_heartbeat("completed")

    # Bead asia.9.2: heartbeat key is now per-market scoped; with no market
    # explicitly passed, the key is suffixed with ":shared".
    assert redis_client.last_set_key == "cache:warmup:heartbeat:shared"
    assert redis_client.last_set_ttl == 3600
    saved_payload = json.loads(redis_client.last_set_value)
    assert saved_payload["status"] == "completed"
    assert saved_payload["current"] == 3
    assert saved_payload["total"] == 10
    assert saved_payload["percent"] == 100.0
