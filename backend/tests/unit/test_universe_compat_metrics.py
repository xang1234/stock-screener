"""Unit tests for legacy universe compatibility telemetry counters."""

from __future__ import annotations

from unittest.mock import patch

from app.services import universe_compat_metrics as metrics


class _FakePipeline:
    def __init__(self, client: "_FakeRedis") -> None:
        self._client = client
        self._ops: list[tuple[str, tuple]] = []

    def incr(self, key: str):
        self._ops.append(("incr", (key,)))
        return self

    def set(self, key: str, value):
        self._ops.append(("set", (key, value)))
        return self

    def execute(self):
        results = []
        for op, args in self._ops:
            if op == "incr":
                key = args[0]
                self._client.store[key] = int(self._client.store.get(key, 0)) + 1
                results.append(self._client.store[key])
            elif op == "set":
                key, value = args
                self._client.store[key] = value
                results.append(True)
        return results


class _FakeRedis:
    """Minimal Redis stand-in covering the ops the metrics module uses."""

    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    def pipeline(self, transaction: bool = True):
        return _FakePipeline(self)

    def get(self, key: str):
        value = self.store.get(key)
        if value is None:
            return None
        return str(value).encode()

    def scan_iter(self, match: str):
        prefix = match.rstrip("*")
        for key in list(self.store.keys()):
            if key.startswith(prefix):
                yield key.encode()


def test_record_increments_total_and_per_value():
    fake = _FakeRedis()
    with patch.object(metrics, "get_redis_client", return_value=fake):
        metrics.record_legacy_universe_usage("nyse")
        metrics.record_legacy_universe_usage("nyse")
        metrics.record_legacy_universe_usage("all")

    assert int(fake.store[metrics.LEGACY_TOTAL_KEY]) == 3
    assert int(fake.store[f"{metrics.LEGACY_VALUE_KEY_PREFIX}nyse"]) == 2
    assert int(fake.store[f"{metrics.LEGACY_VALUE_KEY_PREFIX}all"]) == 1
    assert metrics.LEGACY_LAST_SEEN_KEY in fake.store


def test_record_buckets_empty_and_oversized_values_under_unknown():
    fake = _FakeRedis()
    with patch.object(metrics, "get_redis_client", return_value=fake):
        metrics.record_legacy_universe_usage("")
        metrics.record_legacy_universe_usage(None)
        metrics.record_legacy_universe_usage("x" * 64)  # oversized
        metrics.record_legacy_universe_usage("has space")  # whitespace rejected

    assert int(fake.store[f"{metrics.LEGACY_VALUE_KEY_PREFIX}unknown"]) == 4
    assert int(fake.store[metrics.LEGACY_TOTAL_KEY]) == 4


def test_record_is_noop_when_redis_unavailable():
    with patch.object(metrics, "get_redis_client", return_value=None):
        metrics.record_legacy_universe_usage("nyse")  # must not raise


def test_record_is_noop_when_pipeline_raises():
    class _ExplodingRedis(_FakeRedis):
        def pipeline(self, transaction: bool = True):
            raise RuntimeError("connection lost")

    with patch.object(metrics, "get_redis_client", return_value=_ExplodingRedis()):
        metrics.record_legacy_universe_usage("nyse")  # swallowed


def test_get_legacy_universe_counts_returns_snapshot():
    fake = _FakeRedis()
    with patch.object(metrics, "get_redis_client", return_value=fake):
        metrics.record_legacy_universe_usage("nyse")
        metrics.record_legacy_universe_usage("sp500")
        metrics.record_legacy_universe_usage("sp500")

        snapshot = metrics.get_legacy_universe_counts()

    assert snapshot["total"] == 3
    assert snapshot["by_value"] == {"nyse": 1, "sp500": 2}
    assert isinstance(snapshot["last_seen_ts"], int)


def test_get_legacy_universe_counts_empty_when_redis_missing():
    with patch.object(metrics, "get_redis_client", return_value=None):
        assert metrics.get_legacy_universe_counts() == {}
