"""Synchronous Yahoo option-chain adapter behind the domain provider port."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from typing import Any, TypeVar

import pandas as pd
import yfinance as yf

from app.domain.options_analytics.models import (
    ChainObservation,
    NormalizedOptionContract,
    OptionSide,
)


class OptionsProviderError(RuntimeError):
    """Base error for truthful provider failures."""


class TransientOptionsProviderError(OptionsProviderError):
    pass


class ThrottledOptionsProviderError(TransientOptionsProviderError):
    pass


class OptionsSchemaError(OptionsProviderError):
    pass


class OptionsUnavailableError(OptionsProviderError):
    pass


_T = TypeVar("_T")
_REQUIRED_COLUMNS = {
    "strike",
    "bid",
    "ask",
    "lastPrice",
    "volume",
    "openInterest",
    "impliedVolatility",
    "lastTradeDate",
    "contractSize",
}


def _finite_float(value: Any) -> float | None:
    if value is None or value is pd.NA:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _non_negative_int(value: Any) -> int | None:
    numeric = _finite_float(value)
    if numeric is None or numeric < 0:
        return None
    return int(numeric)


def _utc_datetime(value: Any) -> datetime | None:
    if value is None or value is pd.NA:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


class YahooOptionsProvider:
    def __init__(
        self,
        *,
        ticker_factory: Callable[[str], Any] = yf.Ticker,
        rate_limiter: Callable[[], None] = lambda: None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        max_attempts: int = 3,
    ) -> None:
        self._ticker_factory = ticker_factory
        self._rate_limiter = rate_limiter
        self._clock = clock
        self._max_attempts = min(max(int(max_attempts), 1), 3)

    def list_expirations(self, symbol: str) -> tuple[date, ...]:
        def operation() -> tuple[date, ...]:
            ticker = self._ticker_factory(symbol.strip().upper())
            try:
                return tuple(date.fromisoformat(str(value)) for value in ticker.options)
            except (TypeError, ValueError) as exc:
                raise OptionsSchemaError("Yahoo returned an invalid expiration") from exc

        return self._attempt(operation)

    def fetch_chain(
        self,
        symbol: str,
        expiration: date,
        *,
        source_spot_price: float,
    ) -> ChainObservation:
        canonical_symbol = symbol.strip().upper()

        def operation() -> ChainObservation:
            ticker = self._ticker_factory(canonical_symbol)
            chain = ticker.option_chain(expiration.isoformat())
            calls = self._normalize_frame(chain.calls, OptionSide.CALL)
            puts = self._normalize_frame(chain.puts, OptionSide.PUT)
            if not calls and not puts:
                raise OptionsSchemaError("Yahoo returned an empty option chain")
            provider_spot = _finite_float(
                getattr(ticker, "fast_info", {}).get("last_price")
            )
            return ChainObservation(
                symbol=canonical_symbol,
                expiration=expiration,
                source_spot_price=float(source_spot_price),
                provider_spot_price=provider_spot,
                fetched_at=self._clock().astimezone(timezone.utc),
                contracts=tuple((*calls, *puts)),
            )

        return self._attempt(operation)

    def risk_free_rate(self, on_or_before: date) -> float:
        def operation() -> float:
            ticker = self._ticker_factory("^IRX")
            history = ticker.history(
                start=(on_or_before - timedelta(days=14)).isoformat(),
                end=(on_or_before + timedelta(days=1)).isoformat(),
                auto_adjust=False,
            )
            if history is None or history.empty or "Close" not in history.columns:
                raise OptionsUnavailableError("IRX close is unavailable")
            eligible = [
                (pd.Timestamp(index).date(), _finite_float(value))
                for index, value in history["Close"].items()
                if pd.Timestamp(index).date() <= on_or_before
            ]
            usable = [(day, value) for day, value in eligible if value is not None]
            if not usable:
                raise OptionsUnavailableError(
                    "IRX has no finite close on or before the pinned date"
                )
            _, close = max(usable, key=lambda row: row[0])
            return float(close) / 100.0

        return self._attempt(operation)

    def _attempt(self, operation: Callable[[], _T]) -> _T:
        last_error: Exception | None = None
        for _attempt_number in range(1, self._max_attempts + 1):
            self._rate_limiter()
            try:
                return operation()
            except (OptionsSchemaError, OptionsUnavailableError):
                raise
            except Exception as exc:  # provider clients expose heterogeneous errors
                last_error = exc
        message = str(last_error or "Yahoo options request failed")
        if "429" in message or "too many requests" in message.lower():
            raise ThrottledOptionsProviderError(message) from last_error
        raise TransientOptionsProviderError(message) from last_error

    @staticmethod
    def _normalize_frame(
        frame: pd.DataFrame | None, side: OptionSide
    ) -> tuple[NormalizedOptionContract, ...]:
        if frame is None or frame.empty:
            return ()
        missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise OptionsSchemaError(
                f"Yahoo option chain is missing columns: {', '.join(missing)}"
            )
        contracts: list[NormalizedOptionContract] = []
        for _, row in frame.iterrows():
            strike = _finite_float(row["strike"])
            if strike is None or strike <= 0:
                continue
            contract_size = (
                str(row["contractSize"]).strip().upper()
                if row["contractSize"] is not None
                else None
            )
            contracts.append(
                NormalizedOptionContract(
                    side=side,
                    strike=strike,
                    bid=_finite_float(row["bid"]),
                    ask=_finite_float(row["ask"]),
                    last_price=_finite_float(row["lastPrice"]),
                    volume=_non_negative_int(row["volume"]),
                    open_interest=_non_negative_int(row["openInterest"]),
                    implied_volatility=_finite_float(row["impliedVolatility"]),
                    last_trade_at=_utc_datetime(row["lastTradeDate"]),
                    contract_size=contract_size,
                    multiplier=100 if contract_size == "REGULAR" else None,
                )
            )
        return tuple(contracts)

