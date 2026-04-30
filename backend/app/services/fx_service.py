"""FX Service — deterministic USD normalisation for multi-market fundamentals.

Primary responsibilities
------------------------
- Resolve market -> currency (deterministic, no API call).
- Fetch/cache a daily USD conversion rate per currency.
- Produce reproducible ``fx_metadata`` snapshots for persistence on
  ``StockFundamental`` rows.

Design
------
Storage-time conversion: USD values (``market_cap_usd``, ``adv_usd``) are
computed when writing fundamentals and frozen into the row along with the
exact rate that was used. This is what makes cross-market scans
*auditable and replayable* (T3 acceptance): given any row you can answer
"what rate produced this value?" without needing the external source to
still be reachable.

The ``fx_rates`` table is the historical log (one row per
currency/date/source). The per-fundamentals-row ``fx_metadata`` JSONB is
the row-level snapshot. Together they cover both time-series analysis
and single-row replay without a join.

Rate source
-----------
The default rate fetcher uses yfinance's FX tickers (``HKDUSD=X``, etc.).
Callers can inject a different fetcher for tests or alternative sources.
The service treats USD as the identity currency (rate = 1.0) without
calling any fetcher.

Scope (T3)
----------
This module *provides* normalised values. Consumers (scanners, API, UI)
will adopt them in T4 / T6.3 / T8 — we don't modify scanner filters in
this change.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from ..models.fx_rate import FXRate
from .redis_pool import get_redis_client

logger = logging.getLogger(__name__)


USD = "USD"

# Market -> primary currency. Mirrors security_master_service._MARKET_DEFAULTS;
# a drift-prevention test (test_fx_service.py::test_agrees_with_security_master_defaults)
# fails CI if the two registries disagree.
MARKET_CURRENCY_MAP: Mapping[str, str] = {
    "US": "USD",
    "HK": "HKD",
    "IN": "INR",
    "JP": "JPY",
    "KR": "KRW",
    "TW": "TWD",
    "CN": "CNY",
}

SUPPORTED_CURRENCIES = frozenset(MARKET_CURRENCY_MAP.values())


def currency_for_market(market: str | None) -> str:
    """Return the primary currency for ``market``; USD for unknowns."""
    if not market:
        return USD
    return MARKET_CURRENCY_MAP.get(market.strip().upper(), USD)


RateFetcher = Callable[[str], Optional[float | Tuple[float, date] | Tuple[float, date, str]]]
"""``fetcher(from_currency) -> rate`` / ``(rate, as_of_date)`` / ``(rate, as_of_date, source)``.

Returning ``None`` signals "unavailable" so the caller can fall back to
NULL USD columns rather than silently assume a rate.
"""


def _yfinance_rate_fetcher(from_currency: str) -> Optional[Tuple[float, date]]:
    """Default rate fetcher — queries yfinance for ``{CUR}USD=X`` close."""
    try:
        import yfinance as yf  # local import to avoid hard dep at module load
    except ModuleNotFoundError:  # pragma: no cover
        logger.warning("yfinance not available; FX fetch skipped for %s", from_currency)
        return None

    try:
        ticker = yf.Ticker(f"{from_currency}USD=X")
        hist = ticker.history(period="5d")
        if hist is None or hist.empty:
            logger.warning("No FX history returned for %s->USD", from_currency)
            return None
        closes = hist["Close"].dropna()
        if closes.empty:
            logger.warning("No FX close values returned for %s->USD", from_currency)
            return None
        value = float(closes.iloc[-1])
        if value <= 0:
            return None
        market_ts = closes.index[-1]
        market_date = market_ts.date() if hasattr(market_ts, "date") else date.today()
        return value, market_date
    except Exception as exc:  # pragma: no cover - network variability
        logger.warning("FX fetch for %s failed: %s", from_currency, exc)
        return None


@dataclass(frozen=True)
class FXQuote:
    """A single currency's conversion quote."""
    from_currency: str
    to_currency: str
    rate: float
    as_of_date: date
    source: str

    def to_metadata(self) -> Dict[str, Any]:
        """Serialise as the JSONB blob stored on StockFundamental rows."""
        return {
            "from_currency": self.from_currency,
            "to_currency": self.to_currency,
            "rate": self.rate,
            "as_of_date": self.as_of_date.isoformat(),
            "source": self.source,
        }

    @staticmethod
    def unavailable_metadata(from_currency: str) -> Dict[str, Any]:
        """Blob to write when a rate could not be resolved.

        Shares the key schema of ``to_metadata`` so downstream consumers
        (T4 quality-aware fallback, T8 UI) can rely on a single shape.
        """
        return {
            "from_currency": from_currency,
            "to_currency": USD,
            "rate": None,
            "as_of_date": None,
            "source": "unavailable",
        }


class FXService:
    """Provides USD conversion quotes with Redis cache + DB persistence.

    Lookup order on ``get_usd_rate``:
    1. In-memory cache (per-service-instance, for the life of a batch run)
    2. Redis (24h TTL)
    3. DB (``fx_rates`` table) — most recent row for the currency
    4. Rate fetcher (default yfinance)

    On fetch success, the quote is written to DB and Redis so subsequent
    callers in the same process/pool hit the fast path.
    """

    REDIS_KEY_FORMAT = "fx:{source_currency}:USD"
    REDIS_TTL_SECONDS = 24 * 60 * 60  # 24h — FX rates are slow-moving

    SOURCE_IDENTITY = "identity"
    SOURCE_YFINANCE = "yfinance"
    SOURCE_DATABASE = "database"
    SOURCE_FETCHER = "fetcher"

    def __init__(
        self,
        *,
        rate_fetcher: Optional[RateFetcher] = None,
        rate_fetch_source: Optional[str] = None,
        session_factory: Optional[Callable] = None,
        redis_client: Optional[Any] = None,
    ):
        if rate_fetcher is None:
            self._rate_fetcher = _yfinance_rate_fetcher
            self._rate_fetch_source = self.SOURCE_YFINANCE
        else:
            self._rate_fetcher = rate_fetcher
            self._rate_fetch_source = rate_fetch_source or self.SOURCE_FETCHER
        if session_factory is None:
            from ..database import SessionLocal
            session_factory = SessionLocal
        self._session_factory = session_factory
        if redis_client is None:
            redis_client = get_redis_client()
        self._redis_client = redis_client
        # In-process cache: avoids re-hitting Redis for the same currency
        # within a single batch refresh.
        self._memo: Dict[str, FXQuote] = {}

    # --- Public API --------------------------------------------------------

    def get_usd_rate(self, from_currency: str | None) -> Optional[FXQuote]:
        """Return an ``FXQuote`` for ``from_currency`` -> USD, or ``None``.

        ``None`` indicates the rate is genuinely unavailable — callers
        should write NULL USD columns rather than fabricate a rate.
        """
        if from_currency is None:
            return None
        currency = from_currency.strip().upper()
        if not currency:
            return None
        if currency == USD:
            return FXQuote(
                from_currency=USD,
                to_currency=USD,
                rate=1.0,
                as_of_date=date.today(),
                source=self.SOURCE_IDENTITY,
            )
        today = date.today()
        latest_close = self._latest_trading_close(today)
        stale_quote: Optional[FXQuote] = None

        # 1. In-memory cache — stale if the quote is from a prior day. Without
        # this check, a long-lived Celery worker would keep yesterday's rate
        # forever since the memo has no TTL.
        memoed = self._memo.get(currency)
        if memoed is not None and self._is_fresh_quote_date(
            memoed.as_of_date,
            today=today,
            latest_close=latest_close,
        ):
            return memoed
        if memoed is not None:
            stale_quote = memoed

        # 2. Redis
        quote = self._read_redis(currency)
        if quote is not None:
            if self._is_fresh_quote_date(
                quote.as_of_date,
                today=today,
                latest_close=latest_close,
            ):
                self._memo[currency] = quote
                return quote
            stale_quote = quote

        # 3. DB (most recent row)
        quote = self._read_database(currency)
        if quote is not None:
            if self._is_fresh_quote_date(
                quote.as_of_date,
                today=today,
                latest_close=latest_close,
            ):
                self._memo[currency] = quote
                self._write_redis(quote)
                return quote
            if stale_quote is None or quote.as_of_date > stale_quote.as_of_date:
                stale_quote = quote

        # 4. Upstream fetch
        fetched = self._rate_fetcher(currency)
        quote = self._quote_from_fetch_result(currency, fetched)
        if quote is None:
            if stale_quote is not None:
                logger.warning(
                    "FX fetch unavailable for %s -> USD; using stale %s quote from %s",
                    currency,
                    stale_quote.source,
                    stale_quote.as_of_date,
                )
                self._memo[currency] = stale_quote
                self._write_redis(stale_quote)
                return stale_quote
            logger.warning(
                "FX rate unavailable for %s -> USD; USD fields will be NULL",
                currency,
            )
            return None
        self._persist_database(quote)
        self._write_redis(quote)
        self._memo[currency] = quote
        return quote

    @staticmethod
    def _latest_trading_close(reference_date: date) -> date:
        """Return the most recent weekday close date (weekends roll back)."""
        close_date = reference_date
        while close_date.weekday() >= 5:
            close_date -= timedelta(days=1)
        return close_date

    @staticmethod
    def _is_fresh_quote_date(
        quote_date: date,
        *,
        today: date,
        latest_close: date,
    ) -> bool:
        """Treat either today's quote or latest trading close as fresh."""
        return quote_date == today or quote_date == latest_close

    def convert_to_usd(
        self,
        amount: Optional[float],
        from_currency: str | None,
    ) -> tuple[Optional[float], Optional[FXQuote]]:
        """Return ``(usd_amount, quote)``.

        Returns ``(None, None)`` if ``amount`` is missing *or* the rate
        is unavailable. Callers that need to distinguish the two cases
        (e.g., to persist "attempted but failed" metadata) should call
        ``get_usd_rate`` directly and handle ``amount`` themselves.
        """
        if amount is None:
            return None, None
        quote = self.get_usd_rate(from_currency)
        if quote is None:
            return None, None
        return amount * quote.rate, quote

    def prefetch(self, currencies: list[str]) -> Dict[str, FXQuote]:
        """Warm the in-memory cache for a batch. Idempotent.

        Returns the dict of quotes successfully resolved, keyed by
        currency. Missing currencies are simply absent from the result.
        """
        quotes: Dict[str, FXQuote] = {}
        for currency in currencies:
            quote = self.get_usd_rate(currency)
            if quote is not None:
                quotes[currency] = quote
        return quotes

    def _quote_from_fetch_result(
        self,
        currency: str,
        fetched: Optional[float | Tuple[float, date] | Tuple[float, date, str]],
    ) -> Optional[FXQuote]:
        """Normalize fetcher output to a concrete ``FXQuote``."""
        if fetched is None:
            return None
        source = self._rate_fetch_source
        if isinstance(fetched, tuple):
            if len(fetched) == 3:
                rate, as_of_date, source = fetched
            elif len(fetched) == 2:
                rate, as_of_date = fetched
            else:
                return None
        else:
            rate, as_of_date = fetched, date.today()
        if rate is None:
            return None
        rate_value = float(rate)
        if rate_value <= 0:
            return None
        return FXQuote(
            from_currency=currency,
            to_currency=USD,
            rate=rate_value,
            as_of_date=as_of_date,
            source=source,
        )

    # --- Redis ------------------------------------------------------------

    def _read_redis(self, currency: str) -> Optional[FXQuote]:
        if self._redis_client is None:
            return None
        try:
            raw = self._redis_client.get(self.REDIS_KEY_FORMAT.format(source_currency=currency))
        except Exception as exc:  # pragma: no cover - transient Redis hiccup
            logger.debug("FX redis read failed for %s: %s", currency, exc)
            return None
        if not raw:
            return None
        try:
            # JSON (not pickle) because this Redis instance is shared with
            # the Celery broker/results and externally reachable tools like
            # redis-cli — keep payloads human-inspectable and free of
            # pickle deserialisation CVE surface.
            payload = json.loads(raw)
            return FXQuote(
                from_currency=payload["from_currency"],
                to_currency=payload["to_currency"],
                rate=payload["rate"],
                as_of_date=date.fromisoformat(payload["as_of_date"]),
                source=payload.get("source") or self.SOURCE_DATABASE,
            )
        except Exception as exc:  # pragma: no cover - corruption
            logger.debug("FX redis decode failed for %s: %s", currency, exc)
            return None

    def _write_redis(self, quote: FXQuote) -> None:
        if self._redis_client is None:
            return
        try:
            self._redis_client.setex(
                self.REDIS_KEY_FORMAT.format(source_currency=quote.from_currency),
                self.REDIS_TTL_SECONDS,
                json.dumps(quote.to_metadata()),
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("FX redis write failed for %s: %s", quote.from_currency, exc)

    # --- Database ---------------------------------------------------------

    def _read_database(self, currency: str) -> Optional[FXQuote]:
        db = self._session_factory()
        try:
            row = (
                db.query(FXRate)
                .filter(FXRate.from_currency == currency, FXRate.to_currency == USD)
                .order_by(FXRate.as_of_date.desc(), FXRate.id.desc())
                .first()
            )
            if row is None:
                return None
            return FXQuote(
                from_currency=row.from_currency,
                to_currency=row.to_currency,
                rate=row.rate,
                as_of_date=row.as_of_date,
                source=row.source or self.SOURCE_DATABASE,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("FX DB read failed for %s: %s", currency, exc)
            return None
        finally:
            db.close()

    def _persist_database(self, quote: FXQuote) -> None:
        db = self._session_factory()
        try:
            bind = db.get_bind()
            if bind is not None and bind.dialect.name == "postgresql":
                stmt = pg_insert(FXRate).values(
                    from_currency=quote.from_currency,
                    to_currency=quote.to_currency,
                    as_of_date=quote.as_of_date,
                    rate=quote.rate,
                    source=quote.source,
                )
                db.execute(
                    stmt.on_conflict_do_update(
                        constraint="uq_fx_rates_currency_date_source",
                        set_={"rate": stmt.excluded.rate},
                    )
                )
                db.commit()
                return

            # Non-Postgres fallback for tests/local tooling. Production uses
            # the atomic ON CONFLICT path above so duplicate writes do not emit
            # Postgres ERROR log entries.
            db.add(FXRate(
                from_currency=quote.from_currency,
                to_currency=quote.to_currency,
                as_of_date=quote.as_of_date,
                rate=quote.rate,
                source=quote.source,
            ))
            db.commit()
            return
        except IntegrityError:
            db.rollback()
        except Exception as exc:  # pragma: no cover - unexpected DB failure
            logger.warning("FX DB persist failed for %s: %s", quote.from_currency, exc)
            db.rollback()
            return
        finally:
            db.close()

        # UNIQUE conflict path: another writer won the race; update the rate.
        db = self._session_factory()
        try:
            (
                db.query(FXRate)
                .filter(
                    FXRate.from_currency == quote.from_currency,
                    FXRate.to_currency == quote.to_currency,
                    FXRate.as_of_date == quote.as_of_date,
                    FXRate.source == quote.source,
                )
                .update({"rate": quote.rate}, synchronize_session=False)
            )
            db.commit()
        except Exception as exc:  # pragma: no cover
            logger.warning("FX DB rate-update failed for %s: %s", quote.from_currency, exc)
            db.rollback()
        finally:
            db.close()


_default_service: Optional[FXService] = None


def get_fx_service() -> FXService:
    """Return a process-wide default ``FXService`` (lazy init)."""
    global _default_service
    if _default_service is None:
        _default_service = FXService()
    return _default_service
