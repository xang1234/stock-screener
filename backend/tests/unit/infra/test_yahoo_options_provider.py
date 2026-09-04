from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.domain.options_analytics.expiration import select_monthly_expiration
from app.domain.options_analytics.models import OptionSide
from app.infra.providers.yahoo_options import (
    OptionsSchemaError,
    ThrottledOptionsProviderError,
    YahooOptionsProvider,
)
from app.domain.options_analytics.ports import TransientOptionsProviderError

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "options" / "yahoo_chain_normalized_source.json"
NOW = datetime(2026, 9, 4, 1, 2, 3, tzinfo=timezone.utc)


class _Calendar:
    def is_session(self, value: date) -> bool:
        return value.weekday() < 5


class _Ticker:
    def __init__(self, *, options=(), calls=None, puts=None, fast_info=None, history=None):
        self.options = options
        self._calls = calls
        self._puts = puts
        self.fast_info = fast_info or {}
        self._history = history
        self.chain_calls = 0

    def option_chain(self, _expiration):
        self.chain_calls += 1
        return SimpleNamespace(calls=self._calls, puts=self._puts)

    def history(self, **_kwargs):
        return self._history


def _frames():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return pd.DataFrame(payload["calls"]), pd.DataFrame(payload["puts"])


def test_expiration_discovery_normalizes_dates_for_domain_monthly_selection() -> None:
    ticker = _Ticker(options=("2026-09-11", "2026-09-18", "2026-10-16"))
    waits = []
    provider = YahooOptionsProvider(
        ticker_factory=lambda _symbol: ticker,
        rate_limiter=lambda: waits.append("wait"),
        clock=lambda: NOW,
    )

    listed = provider.list_expirations("AAPL")
    selected = select_monthly_expiration(
        as_of_date=date(2026, 9, 4),
        listed_expirations=listed,
        calendar=_Calendar(),
    )

    assert listed == (date(2026, 9, 11), date(2026, 9, 18), date(2026, 10, 16))
    assert selected == date(2026, 9, 18)
    assert waits == ["wait"]


def test_chain_normalization_preserves_zero_null_time_and_multiplier_meaning() -> None:
    calls, puts = _frames()
    ticker = _Ticker(calls=calls, puts=puts, fast_info={"last_price": 101.5})
    provider = YahooOptionsProvider(
        ticker_factory=lambda _symbol: ticker,
        rate_limiter=lambda: None,
        clock=lambda: NOW,
    )

    observation = provider.fetch_chain(
        "aapl", date(2026, 9, 18), source_spot_price=100.0
    )

    call, put = observation.contracts
    assert observation.symbol == "AAPL"
    assert observation.source_spot_price == 100.0
    assert observation.provider_spot_price == 101.5
    assert observation.fetched_at == NOW
    assert call.side is OptionSide.CALL
    assert call.volume == 0
    assert call.open_interest == 1250
    assert call.implied_volatility == 0.285
    assert call.last_trade_at == datetime(2026, 9, 3, 19, 45, tzinfo=timezone.utc)
    assert call.multiplier == 100
    assert put.side is OptionSide.PUT
    assert put.last_trade_at == datetime(2026, 9, 3, 23, 44, tzinfo=timezone.utc)
    assert put.multiplier is None


def test_non_finite_cells_become_missing_but_numeric_zero_survives() -> None:
    calls, puts = _frames()
    calls.loc[0, "bid"] = float("nan")
    puts["openInterest"] = puts["openInterest"].astype(float)
    puts.loc[0, "openInterest"] = float("inf")
    provider = YahooOptionsProvider(
        ticker_factory=lambda _symbol: _Ticker(calls=calls, puts=puts),
        rate_limiter=lambda: None,
        clock=lambda: NOW,
    )

    observation = provider.fetch_chain(
        "AAPL", date(2026, 9, 18), source_spot_price=100
    )

    assert observation.contracts[0].bid is None
    assert observation.contracts[0].volume == 0
    assert observation.contracts[1].open_interest is None


@pytest.mark.parametrize("missing", ["strike", "openInterest", "impliedVolatility"])
def test_missing_required_columns_raise_typed_schema_error(missing) -> None:
    calls, puts = _frames()
    calls = calls.drop(columns=[missing])
    provider = YahooOptionsProvider(
        ticker_factory=lambda _symbol: _Ticker(calls=calls, puts=puts),
        rate_limiter=lambda: None,
        clock=lambda: NOW,
    )

    with pytest.raises(OptionsSchemaError, match=missing):
        provider.fetch_chain("AAPL", date(2026, 9, 18), source_spot_price=100)


def test_empty_both_sides_raise_typed_schema_error() -> None:
    provider = YahooOptionsProvider(
        ticker_factory=lambda _symbol: _Ticker(calls=pd.DataFrame(), puts=pd.DataFrame()),
        rate_limiter=lambda: None,
        clock=lambda: NOW,
    )

    with pytest.raises(OptionsSchemaError, match="empty"):
        provider.fetch_chain("AAPL", date(2026, 9, 18), source_spot_price=100)


def test_provider_attempts_once_and_leaves_retry_budget_to_the_use_case() -> None:
    attempts = []

    def factory(_symbol):
        attempts.append("attempt")
        raise TimeoutError("temporary")

    waits = []
    provider = YahooOptionsProvider(
        ticker_factory=factory,
        rate_limiter=lambda: waits.append("wait"),
        clock=lambda: NOW,
    )

    with pytest.raises(TransientOptionsProviderError):
        provider.fetch_chain("AAPL", date(2026, 9, 18), source_spot_price=100)

    assert attempts == ["attempt"]
    assert waits == ["wait"]


def test_throttling_is_classified_after_one_provider_attempt() -> None:
    attempts = []

    def factory(_symbol):
        attempts.append("attempt")
        raise RuntimeError("429 Too Many Requests")

    provider = YahooOptionsProvider(
        ticker_factory=factory,
        rate_limiter=lambda: None,
        clock=lambda: NOW,
    )

    with pytest.raises(ThrottledOptionsProviderError):
        provider.list_expirations("AAPL")
    assert len(attempts) == 1


def test_irx_rate_uses_latest_close_on_or_before_pinned_date() -> None:
    history = pd.DataFrame(
        {"Close": [4.20, 4.35, 9.99]},
        index=pd.to_datetime(["2026-09-02", "2026-09-03", "2026-09-05"]),
    )
    provider = YahooOptionsProvider(
        ticker_factory=lambda symbol: _Ticker(history=history) if symbol == "^IRX" else None,
        rate_limiter=lambda: None,
        clock=lambda: NOW,
    )

    assert provider.risk_free_rate(date(2026, 9, 4)) == pytest.approx(0.0435)
