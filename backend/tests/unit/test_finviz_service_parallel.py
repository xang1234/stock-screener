"""Parallel finviz batch and per-market wiring tests."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from app.services.finviz_service import FinvizService


class _RecordingLimiter:
    """Stand-in for ``RedisRateLimiter`` that records the keys it serialised on."""

    def __init__(self):
        self.market_calls: list[tuple[str, str]] = []
        self.legacy_calls: list[tuple[str, float]] = []
        self.concurrent_max = 0
        self._inflight = 0
        self._lock = threading.Lock()

    def wait_for_market(self, provider, market, timeout_s=120.0):
        with self._lock:
            self.market_calls.append((provider, market))
            self._inflight += 1
            self.concurrent_max = max(self.concurrent_max, self._inflight)
        # Tiny delay so threads can actually overlap during the test.
        import time
        time.sleep(0.005)
        with self._lock:
            self._inflight -= 1
        return 0.0

    def wait(self, key, min_interval_s=0.0, timeout_s=120.0):
        self.legacy_calls.append((key, min_interval_s))
        return 0.0


@pytest.fixture(autouse=True)
def _disable_breaker():
    """Ensure the circuit breaker stays closed throughout these tests."""
    with patch("app.services.provider_circuit_breaker.settings") as s:
        s.circuit_breaker_enabled = False
        s.circuit_breaker_threshold = 100
        for market in ("us", "hk", "in", "jp", "tw"):
            setattr(s, f"circuit_breaker_cooldown_{market}", 60)
        # Reset the singleton so it picks up the patched settings.
        import app.services.provider_circuit_breaker as cb_mod
        cb_mod._default_breaker = None
        yield
        cb_mod._default_breaker = None


class TestPerMarketRouting:
    def test_market_kwarg_routes_to_per_market_key(self):
        limiter = _RecordingLimiter()
        service = FinvizService(rate_limiter=limiter)

        with patch("finvizfinance.quote.finvizfinance") as mock_finviz:
            stock = MagicMock()
            stock.flag = True
            stock.ticker_fundament.return_value = {}
            stock.ticker_description.return_value = "desc"
            mock_finviz.return_value = stock
            service.get_finviz_only_fields("AAPL", market="US")

        assert limiter.market_calls == [("finviz", "US")]
        assert limiter.legacy_calls == []

    def test_no_market_falls_back_to_legacy_key(self):
        from app.config import settings as settings_singleton

        limiter = _RecordingLimiter()
        service = FinvizService(rate_limiter=limiter)

        # ``settings`` is imported lazily inside ``_rate_limited_call``;
        # patch the singleton attribute so the deferred import sees it.
        with patch("finvizfinance.quote.finvizfinance") as mock_finviz, \
             patch.object(settings_singleton, "finviz_rate_limit_interval", 0.5):
            stock = MagicMock()
            stock.flag = True
            stock.ticker_fundament.return_value = {}
            stock.ticker_description.return_value = "desc"
            mock_finviz.return_value = stock
            service.get_finviz_only_fields("AAPL")

        assert limiter.market_calls == []
        assert len(limiter.legacy_calls) == 1
        assert limiter.legacy_calls[0][0] == "finviz"
        assert limiter.legacy_calls[0][1] == 0.5


class TestParallelBatch:
    def test_batch_uses_threadpool_when_workers_gt_1(self):
        limiter = _RecordingLimiter()
        service = FinvizService(rate_limiter=limiter)

        symbols = [f"S{i}" for i in range(8)]
        with patch("finvizfinance.quote.finvizfinance") as mock_finviz:
            stock = MagicMock()
            stock.flag = True
            stock.ticker_fundament.return_value = {}
            stock.ticker_description.return_value = "desc"
            mock_finviz.return_value = stock
            results = service.get_finviz_only_fields_batch(
                symbols, max_workers=4, market="US",
            )

        assert set(results.keys()) == set(symbols)
        # Verified parallelism: at some point >1 thread was inside the limiter.
        assert limiter.concurrent_max >= 2, (
            f"Expected concurrent threads, only saw {limiter.concurrent_max}"
        )
        # All threads serialised on the same per-market key.
        keys = {(p, m) for p, m in limiter.market_calls}
        assert keys == {("finviz", "US")}

    def test_batch_with_workers_1_is_sequential(self):
        limiter = _RecordingLimiter()
        service = FinvizService(rate_limiter=limiter)

        symbols = ["A", "B", "C"]
        with patch("finvizfinance.quote.finvizfinance") as mock_finviz:
            stock = MagicMock()
            stock.flag = True
            stock.ticker_fundament.return_value = {}
            stock.ticker_description.return_value = "desc"
            mock_finviz.return_value = stock
            service.get_finviz_only_fields_batch(symbols, max_workers=1, market="US")

        assert limiter.concurrent_max == 1

    def test_batch_resolves_workers_via_policy_when_none(self):
        limiter = _RecordingLimiter()
        service = FinvizService(rate_limiter=limiter)

        symbols = ["A", "B"]
        with patch("finvizfinance.quote.finvizfinance") as mock_finviz, \
             patch("app.services.rate_budget_policy.get_rate_budget_policy") as get_policy:
            mock_policy = MagicMock()
            mock_policy.get_provider_workers.return_value = 2
            get_policy.return_value = mock_policy
            stock = MagicMock()
            stock.flag = True
            stock.ticker_fundament.return_value = {}
            stock.ticker_description.return_value = "desc"
            mock_finviz.return_value = stock
            service.get_finviz_only_fields_batch(symbols, market="HK")

        mock_policy.get_provider_workers.assert_called_with("finviz", "HK")


class TestRateBudgetPolicyWorkers:
    def test_get_provider_workers_defaults(self):
        from app.services.rate_budget_policy import RateBudgetPolicy

        policy = RateBudgetPolicy()
        with patch("app.services.rate_budget_policy.settings") as s:
            # Make all settings overrides None so policy uses defaults.
            for attr in ("finviz_workers_us", "finviz_workers_hk",
                         "finviz_workers_jp", "finviz_workers_tw",
                         "finviz_workers_in"):
                setattr(s, attr, None)
            # Defaults: US=4, others=2 per _DEFAULT_PROVIDER_WORKERS.
            assert policy.get_provider_workers("finviz", "US") == 4
            assert policy.get_provider_workers("finviz", "HK") == 2

    def test_get_provider_workers_honours_settings_override(self):
        from app.services.rate_budget_policy import RateBudgetPolicy

        policy = RateBudgetPolicy()
        with patch("app.services.rate_budget_policy.settings") as s:
            s.finviz_workers_us = 8
            assert policy.get_provider_workers("finviz", "US") == 8

    def test_get_provider_workers_unknown_provider_returns_one(self):
        from app.services.rate_budget_policy import RateBudgetPolicy

        policy = RateBudgetPolicy()
        with patch("app.services.rate_budget_policy.settings") as s:
            s.bogus_workers_us = None
            assert policy.get_provider_workers("bogus", "US") == 1
