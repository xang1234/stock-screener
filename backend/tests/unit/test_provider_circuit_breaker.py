"""Unit tests for ProviderCircuitBreaker.

Tests the state machine semantics (closed → open → half_open → closed/open)
using the in-process fallback path (Redis unavailable). The Lua-script
Redis path is exercised in integration smoke; here we focus on the logic
that is portable across both paths.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.services.provider_circuit_breaker import (
    CircuitOpenError,
    ProviderCircuitBreaker,
)


def _no_redis() -> ProviderCircuitBreaker:
    """Return a breaker pinned to the in-process fallback path."""
    return ProviderCircuitBreaker(redis_client_factory=lambda: None)


def _fake_clock(monkeypatch, start: float = 1000.0) -> SimpleNamespace:
    """Pin the breaker module's ``time.time()``; advance it via ``clock.now``."""
    clock = SimpleNamespace(now=start)
    monkeypatch.setattr(
        "app.services.provider_circuit_breaker.time",
        SimpleNamespace(time=lambda: clock.now),
    )
    return clock


class TestStateMachine:
    def test_starts_closed(self):
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 3
            s.circuit_breaker_cooldown_us = 60
            breaker = _no_redis()
            assert breaker.check("finviz", "US") == "closed"

    def test_trips_to_open_at_threshold(self):
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 3
            s.circuit_breaker_cooldown_us = 60
            breaker = _no_redis()
            for _ in range(3):
                breaker.record_429("finviz", "US")
            assert breaker.check("finviz", "US") == "open"

    def test_below_threshold_stays_closed(self):
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 3
            s.circuit_breaker_cooldown_us = 60
            breaker = _no_redis()
            breaker.record_429("finviz", "US")
            breaker.record_429("finviz", "US")
            assert breaker.check("finviz", "US") == "closed"

    def test_success_resets_consecutive_counter(self):
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 3
            s.circuit_breaker_cooldown_us = 60
            breaker = _no_redis()
            breaker.record_429("finviz", "US")
            breaker.record_429("finviz", "US")
            breaker.record_success("finviz", "US")
            # Two more 429s should not trip — counter was reset.
            breaker.record_429("finviz", "US")
            breaker.record_429("finviz", "US")
            assert breaker.check("finviz", "US") == "closed"

    def test_open_promotes_to_half_open_after_cooldown(self):
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 2
            s.circuit_breaker_cooldown_us = 0  # immediate cooldown
            breaker = _no_redis()
            breaker.record_429("finviz", "US")
            breaker.record_429("finviz", "US")
            assert breaker.check("finviz", "US") == "half_open_probe"

    def test_half_open_only_admits_one_probe(self, monkeypatch):
        clock = _fake_clock(monkeypatch)
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 2
            s.circuit_breaker_cooldown_us = 60
            breaker = _no_redis()
            breaker.record_429("finviz", "US")
            breaker.record_429("finviz", "US")
            clock.now = 1060.0
            assert breaker.check("finviz", "US") == "half_open_probe"
            # Second check during half-open without success/429 should refuse.
            assert breaker.check("finviz", "US") == "open"

    def test_unreported_probe_expires_after_cooldown(self, monkeypatch):
        # Matches the Redis probe key's ``EX cooldown``: a probe whose caller
        # never reports back must not wedge the in-process breaker open.
        clock = _fake_clock(monkeypatch)
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 2
            s.circuit_breaker_cooldown_us = 60
            breaker = _no_redis()
            breaker.record_429("finviz", "US")
            breaker.record_429("finviz", "US")
            # Take the probe off the open boundary, so expiry is measured from
            # the probe, not from opened_at.
            clock.now = 1080.0
            assert breaker.check("finviz", "US") == "half_open_probe"
            clock.now = 1139.0
            assert breaker.check("finviz", "US") == "open"
            clock.now = 1140.0
            assert breaker.check("finviz", "US") == "half_open_probe"
            # The re-issued probe holds its own lease, which also expires.
            clock.now = 1199.0
            assert breaker.check("finviz", "US") == "open"
            clock.now = 1200.0
            assert breaker.check("finviz", "US") == "half_open_probe"
            breaker.record_success("finviz", "US")
            assert breaker.check("finviz", "US") == "closed"

    def test_zero_cooldown_still_admits_one_probe_at_a_time(self, monkeypatch):
        clock = _fake_clock(monkeypatch)
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 2
            s.circuit_breaker_cooldown_us = 0
            breaker = _no_redis()
            breaker.record_429("finviz", "US")
            breaker.record_429("finviz", "US")
            assert breaker.check("finviz", "US") == "half_open_probe"
            assert breaker.check("finviz", "US") == "open"
            clock.now += 1
            assert breaker.check("finviz", "US") == "half_open_probe"

    def test_reports_while_open_leave_the_cooldown_alone(self, monkeypatch):
        # Matches Redis: a 429 or success from a call that started before the
        # trip neither pushes opened_at forward nor clears the count.
        clock = _fake_clock(monkeypatch)
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 2
            s.circuit_breaker_cooldown_us = 60
            breaker = _no_redis()
            breaker.record_429("finviz", "US")
            breaker.record_429("finviz", "US")
            entry = breaker._fallback_state[breaker.state_key("finviz", "US")]
            clock.now = 1040.0
            breaker.record_success("finviz", "US")
            assert entry["consecutive_429"] == 2
            breaker.record_429("finviz", "US")
            clock.now = 1060.0
            assert breaker.check("finviz", "US") == "half_open_probe"

    def test_failed_probe_restarts_cooldown_from_the_failure(self, monkeypatch):
        clock = _fake_clock(monkeypatch)
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 2
            s.circuit_breaker_cooldown_us = 60
            breaker = _no_redis()
            breaker.record_429("finviz", "US")
            breaker.record_429("finviz", "US")
            clock.now = 1060.0
            assert breaker.check("finviz", "US") == "half_open_probe"
            clock.now = 1090.0
            assert breaker.record_429("finviz", "US") is True
            clock.now = 1149.0
            assert breaker.check("finviz", "US") == "open"
            clock.now = 1150.0
            assert breaker.check("finviz", "US") == "half_open_probe"

    def test_half_open_success_closes_circuit(self):
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 2
            s.circuit_breaker_cooldown_us = 0
            breaker = _no_redis()
            breaker.record_429("finviz", "US")
            breaker.record_429("finviz", "US")
            assert breaker.check("finviz", "US") == "half_open_probe"
            breaker.record_success("finviz", "US")
            assert breaker.check("finviz", "US") == "closed"

    def test_half_open_failure_reopens(self):
        # Use a non-zero cooldown so we can observe the "open" gate before
        # the auto-promotion path fires again.
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 2
            s.circuit_breaker_cooldown_us = 0  # first promotion immediate
            breaker = _no_redis()
            breaker.record_429("finviz", "US")
            breaker.record_429("finviz", "US")
            assert breaker.check("finviz", "US") == "half_open_probe"
            # Once the probe fires, raise the cooldown so a subsequent 429
            # actually pauses the breaker rather than instantly re-promoting.
            s.circuit_breaker_cooldown_us = 60
            breaker.record_429("finviz", "US")
            assert breaker.check("finviz", "US") == "open"

    def test_disabled_breaker_always_closed(self):
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = False
            s.circuit_breaker_threshold = 1
            s.circuit_breaker_cooldown_us = 60
            breaker = _no_redis()
            for _ in range(10):
                breaker.record_429("finviz", "US")
            assert breaker.check("finviz", "US") == "closed"


class TestPerMarketIsolation:
    def test_cn_cooldown_uses_market_specific_setting(self):
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 2
            s.circuit_breaker_cooldown_cn = 17
            breaker = _no_redis()
            assert breaker.cooldown_s("CN") == 17

    def test_different_markets_dont_share_state(self):
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 2
            s.circuit_breaker_cooldown_us = 60
            s.circuit_breaker_cooldown_hk = 60
            breaker = _no_redis()
            # Trip US.
            breaker.record_429("finviz", "US")
            breaker.record_429("finviz", "US")
            assert breaker.check("finviz", "US") == "open"
            # HK still closed.
            assert breaker.check("finviz", "HK") == "closed"

    def test_different_providers_dont_share_state(self):
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 2
            s.circuit_breaker_cooldown_us = 60
            breaker = _no_redis()
            breaker.record_429("yfinance", "US")
            breaker.record_429("yfinance", "US")
            assert breaker.check("yfinance", "US") == "open"
            assert breaker.check("finviz", "US") == "closed"


class TestRaiseIfOpen:
    def test_raises_when_open(self):
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 1
            s.circuit_breaker_cooldown_us = 60
            breaker = _no_redis()
            breaker.record_429("finviz", "US")
            with pytest.raises(CircuitOpenError) as exc_info:
                breaker.raise_if_open("finviz", "US")
            assert exc_info.value.provider == "finviz"
            assert exc_info.value.market == "US"

    def test_returns_when_closed(self):
        with patch("app.services.provider_circuit_breaker.settings") as s:
            s.circuit_breaker_enabled = True
            s.circuit_breaker_threshold = 3
            s.circuit_breaker_cooldown_us = 60
            breaker = _no_redis()
            assert breaker.raise_if_open("finviz", "US") == "closed"


class TestKeyHelpers:
    def test_state_key_format(self):
        assert ProviderCircuitBreaker.state_key("finviz", "US") == "circuit:finviz:us"
        assert ProviderCircuitBreaker.state_key("yfinance", "HK") == "circuit:yfinance:hk"
        assert ProviderCircuitBreaker.state_key("finviz", None) == "circuit:finviz:shared"

    def test_probe_key_distinct_from_state(self):
        assert ProviderCircuitBreaker.probe_key("finviz", "US") != ProviderCircuitBreaker.state_key("finviz", "US")
