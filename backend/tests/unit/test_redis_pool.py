from __future__ import annotations

import importlib
from unittest.mock import patch


def test_get_redis_pool_short_circuits_when_disabled(monkeypatch):
    import app.services.redis_pool as module

    module = importlib.reload(module)
    monkeypatch.setattr(module.settings, "redis_enabled", False)
    module.reset_pool()

    with patch.object(module, "ConnectionPool", side_effect=AssertionError("should not connect")):
        with patch.object(module, "logger") as mock_logger:
            assert module.get_redis_pool() is None
            assert module.get_redis_pool() is None

    mock_logger.info.assert_called_once_with(
        "Redis disabled via configuration; using non-Redis fallback paths"
    )


def test_get_bulk_redis_pool_short_circuits_when_disabled(monkeypatch):
    import app.services.redis_pool as module

    module = importlib.reload(module)
    monkeypatch.setattr(module.settings, "redis_enabled", False)
    module.reset_pool()

    with patch.object(module, "ConnectionPool", side_effect=AssertionError("should not connect")):
        with patch.object(module, "logger") as mock_logger:
            assert module.get_bulk_redis_pool() is None
            assert module.get_bulk_redis_pool() is None

    mock_logger.info.assert_called_once_with(
        "Redis disabled via configuration; using non-Redis fallback paths"
    )
