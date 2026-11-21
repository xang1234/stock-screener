"""The fundamentals path pulses the circuit breaker once per bad batch, not per symbol."""
from __future__ import annotations

import time
from types import SimpleNamespace

import yfinance as yf

import app.services.provider_circuit_breaker as breaker_module
import app.services.rate_budget_policy as policy_module
from app.config import settings
from app.services.bulk_data_fetcher import BulkDataFetcher
from app.services.provider_circuit_breaker import ProviderCircuitBreaker


def _fetch_fundamentals(monkeypatch, symbols, rate_limited, batch_size, breaker=None):
    monkeypatch.setattr(settings, "circuit_breaker_enabled", True)
    monkeypatch.setattr(settings, "circuit_breaker_threshold", 3)
    breaker = breaker or ProviderCircuitBreaker(redis_client_factory=lambda: None)
    monkeypatch.setattr(breaker_module, "get_circuit_breaker", lambda: breaker)

    policy_429s = []
    monkeypatch.setattr(
        policy_module,
        "get_rate_budget_policy",
        lambda: SimpleNamespace(
            get_backoff_params=lambda *_: {"base_s": 60, "max_s": 480, "factor": 2.0},
            record_429=lambda *args: policy_429s.append(args),
        ),
    )
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    fetched_batches = []

    def fake_tickers(symbols_str, **_kwargs):
        batch = symbols_str.split()
        fetched_batches.append(batch)
        return SimpleNamespace(tickers={s: _Ticker(s, rate_limited) for s in batch})

    monkeypatch.setattr(yf, "Tickers", fake_tickers)
    no_wait = SimpleNamespace(wait_for_market=lambda *a, **k: None, wait=lambda *a, **k: None)
    results = BulkDataFetcher(rate_limiter=no_wait).fetch_batch_fundamentals(
        symbols,
        batch_size=batch_size,
        include_quarterly=False,
        delay_per_ticker=0,
        market="US",
    )
    return results, fetched_batches, policy_429s


class _Ticker:
    def __init__(self, symbol, rate_limited):
        self._rate_limited = symbol in rate_limited

    @property
    def info(self):
        if self._rate_limited:
            raise RuntimeError("429 Too Many Requests")
        return {"marketCap": 1_000_000}


def test_scattered_429s_do_not_open_breaker(monkeypatch):
    symbols = [f"S{i:02d}" for i in range(50)]
    results, fetched_batches, policy_429s = _fetch_fundamentals(
        monkeypatch, symbols, rate_limited={"S00", "S01", "S02"}, batch_size=25
    )

    assert fetched_batches == [symbols[:25], symbols[25:]]
    assert [s for s, r in results.items() if r.get("error") == "circuit_open"] == []
    assert all(results[s]["market_cap"] == 1_000_000 for s in symbols[25:])
    assert len(policy_429s) == 3  # rate budget still counts every 429


def test_consecutive_rate_limited_batches_open_breaker(monkeypatch):
    symbols = [f"S{i:02d}" for i in range(40)]
    results, fetched_batches, _ = _fetch_fundamentals(
        monkeypatch, symbols, rate_limited=set(symbols[:30]), batch_size=10
    )

    assert fetched_batches == [symbols[:10], symbols[10:20], symbols[20:30]]
    assert all(results[s]["error"] == "circuit_open" for s in symbols[30:])


def test_half_open_probe_with_one_429_reopens_instead_of_sticking(monkeypatch):
    monkeypatch.setattr(settings, "circuit_breaker_enabled", True)
    monkeypatch.setattr(settings, "circuit_breaker_threshold", 3)
    breaker = ProviderCircuitBreaker(redis_client_factory=lambda: None)
    for _ in range(3):
        breaker.record_429("yfinance", "US")
    entry = breaker._fallback_state[breaker.state_key("yfinance", "US")]
    entry["opened_at"] -= breaker.cooldown_s("US")  # cooldown elapsed: next check probes

    symbols = ["S00", "S01", "S02", "S03"]
    _fetch_fundamentals(monkeypatch, symbols, rate_limited={"S00"}, batch_size=4, breaker=breaker)

    # The failed probe reopened the circuit, so after another cooldown it
    # probes again. A stuck half-open circuit would answer "open" forever.
    entry["opened_at"] -= breaker.cooldown_s("US")
    assert breaker.check("yfinance", "US") == "half_open_probe"
