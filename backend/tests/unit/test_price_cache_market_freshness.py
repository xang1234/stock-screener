"""Price-cache freshness follows each symbol's own market calendar (issue #413).

Real ``MarketCalendarService`` calendars with a frozen clock. 2026-07-03 is a
US holiday (Independence Day observed) and an HK trading day; HK trades
09:30-16:00 HKT (01:30-08:00 UTC), US 09:30-16:00 ET (13:30-20:00 UTC).
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest
from redis.exceptions import ResponseError as RedisResponseError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.cache.redis_codec import decode_frame, encode_frame
from app.database import Base
from app.models.stock import StockPrice
from app.models.stock_universe import UNIVERSE_STATUS_ACTIVE, StockUniverse
from app.services.market_calendar_service import (
    CalendarCoverageExpired,
    MarketCalendarService,
)
from app.services.price_cache_service import PriceCacheService


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


class _FrozenCalendar:
    """Real calendars, frozen 'now' for the methods that default to the clock."""

    def __init__(self, now: datetime) -> None:
        self._calendar = MarketCalendarService()
        self.now = now

    def __getattr__(self, name):
        return getattr(self._calendar, name)

    def last_completed_trading_day(self, market, now=None, **kwargs):
        return self._calendar.last_completed_trading_day(market, now or self.now, **kwargs)


class _BrokenCalendar:
    def __getattr__(self, name):
        def _raise(*args, **kwargs):
            raise CalendarCoverageExpired("coverage expired")

        return _raise


class _DictRedis:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.values[key] = value

    def scan(self, cursor, match=None, count=None):
        prefix, _, suffix = match.partition("*")
        suffix = suffix.split("*")[-1]
        keys = [
            key for key in self.values
            if key.startswith(prefix) and key.endswith(suffix)
            and key.count(":") == match.count(":")
        ]
        return 0, keys

    def pipeline(self):
        redis = self

        class _Pipeline:
            def __init__(self):
                self.calls = []

            def get(self, key):
                self.calls.append(("get", key, None))
                return self

            def setex(self, key, ttl, value):
                self.calls.append(("set", key, value))
                return self

            def execute(self, raise_on_error=True):
                results = []
                for op, key, value in self.calls:
                    if op == "set":
                        redis.values[key] = value
                        results.append(True)
                    else:
                        results.append(redis.values.get(key))
                return results

        return _Pipeline()


def _meta(fetched_at: datetime, *, legacy_flag: bool) -> str:
    """Metadata as the old US-clock writer stored it."""
    eastern = fetched_at.astimezone(pd.Timestamp.now(tz="America/New_York").tz)
    return json.dumps({
        "fetch_timestamp": eastern.isoformat(),
        "market_was_open": legacy_flag,
        "data_type": "intraday" if legacy_flag else "closing",
        "needs_refresh_after_close": legacy_flag,
    })


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = factory()
    for symbol, market in (("0700.HK", "HK"), ("9988.HK", "HK"), ("AAPL", "US"), ("MSFT", "US")):
        db.add(StockUniverse(
            symbol=symbol,
            market=market,
            exchange="XHKG" if market == "HK" else "XNAS",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            status_reason="active",
        ))
    db.commit()
    db.close()
    return factory


def _service(session_factory, calendar, redis=None) -> PriceCacheService:
    return PriceCacheService(
        redis_client=redis,
        session_factory=session_factory,
        market_calendar=calendar,
    )


# ── Expected session per market ────────────────────────────────────────


def test_expected_session_follows_each_market_calendar(session_factory):
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)))

    assert service._get_expected_data_date("HK") == date(2026, 7, 3)
    assert service._get_expected_data_date("US") == date(2026, 7, 2)  # US holiday


def test_calendar_failure_makes_non_us_data_stale(session_factory):
    service = _service(session_factory, _BrokenCalendar())

    assert service._is_data_fresh(date(2026, 7, 3), market="HK") is False


def test_calendar_falls_back_to_standalone_service_outside_runtime(monkeypatch):
    import app.wiring.bootstrap as bootstrap

    def _uninitialized():
        raise RuntimeError("RuntimeServices are not initialized for this context.")

    monkeypatch.setattr(bootstrap, "get_market_calendar_service", _uninitialized)
    service = PriceCacheService(redis_client=None, session_factory=lambda: None)

    assert isinstance(service._calendar(), MarketCalendarService)


# ── Intraday staleness judged from fetch_timestamp ─────────────────────


def test_hk_bar_fetched_mid_session_goes_stale_after_hk_close(session_factory):
    # Stamped "closing" by the old US-clock writer (US was shut at 11:00 HKT).
    meta = json.loads(_meta(_utc(2026, 7, 2, 3, 0), legacy_flag=False))
    during = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 2, 7, 0)))   # 15:00 HKT
    after = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 2, 9, 0)))    # 17:00 HKT

    assert during._is_fetch_metadata_stale(meta, market="HK") is False
    assert after._is_fetch_metadata_stale(meta, market="HK") is True


def test_hk_bar_fetched_after_hk_close_is_final_even_if_us_was_open(session_factory):
    # Old writer flagged it intraday because the US session was open (10:00 ET).
    meta = json.loads(_meta(_utc(2026, 7, 2, 14, 0), legacy_flag=True))
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 2, 21, 0)))

    assert service._is_fetch_metadata_stale(meta, market="HK") is False


def test_us_intraday_fetch_goes_stale_after_us_close(session_factory):
    meta = json.loads(_meta(_utc(2026, 7, 1, 18, 0), legacy_flag=True))  # 14:00 ET
    before = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 1, 19, 0)))  # 15:00 ET
    after = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 1, 21, 0)))   # 17:00 ET

    assert before._is_fetch_metadata_stale(meta, market="US") is False
    assert after._is_fetch_metadata_stale(meta, market="US") is True


def test_us_early_close_day_expects_same_day_after_half_day_close(session_factory):
    # 2026-11-27: NYSE closes at 13:00 ET; 14:00 ET is past the 30-minute buffer.
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 11, 27, 19, 0)))

    assert service._get_expected_data_date("US") == date(2026, 11, 27)


def test_us_fetch_inside_settlement_buffer_is_partial(session_factory):
    meta = json.loads(_meta(_utc(2026, 7, 1, 20, 15), legacy_flag=False))  # 16:15 ET
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 1, 21, 0)))  # 17:00 ET

    assert service._is_fetch_metadata_stale(meta, market="US") is True


def test_calendar_failure_makes_non_us_metadata_stale(session_factory):
    meta = json.loads(_meta(_utc(2026, 7, 2, 9, 0), legacy_flag=False))
    service = _service(session_factory, _BrokenCalendar())

    assert service._is_fetch_metadata_stale(meta, market="HK") is True


class _CountingCalendar(_FrozenCalendar):
    def __init__(self, now):
        super().__init__(now)
        self.calls = 0

    def session_close(self, *args, **kwargs):
        self.calls += 1
        return self._calendar.session_close(*args, **kwargs)

    def last_completed_trading_day(self, *args, **kwargs):
        self.calls += 1
        return super().last_completed_trading_day(*args, **kwargs)


def test_bulk_freshness_checks_reuse_calendar_answers(session_factory):
    """Per-symbol checks must not repeat calendar work (~2 ms/call) for every symbol."""
    calendar = _CountingCalendar(_utc(2026, 7, 2, 9, 0))
    service = _service(session_factory, calendar)
    metas = [json.loads(_meta(_utc(2026, 7, 2, 3, minute % 60), legacy_flag=False)) for minute in range(500)]

    for meta in metas:
        assert service._is_fetch_metadata_stale(meta, market="HK") is True
    for _ in range(500):
        service._get_expected_data_date("HK")

    assert calendar.calls <= 4


# ── Cache-only reads use the symbol's market ───────────────────────────


def _store_hk_prices(session_factory, symbol: str, last_day: date) -> None:
    days = pd.bdate_range(end=pd.Timestamp(last_day), periods=60)
    db = session_factory()
    for index, day in enumerate(days):
        close = 300.0 + index
        db.add(StockPrice(
            symbol=symbol, date=day.date(), open=close, high=close + 1,
            low=close - 1, close=close, adj_close=close, volume=1_000_000,
        ))
    db.commit()
    db.close()


def test_cached_only_fresh_uses_hk_session_on_us_holiday(session_factory):
    _store_hk_prices(session_factory, "0700.HK", date(2026, 7, 3))
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)))

    frame = service.get_cached_only_fresh("0700.HK", period="2y")

    assert frame is not None
    assert frame.index[-1].date() == date(2026, 7, 3)


def test_many_cached_only_fresh_rejects_hk_bar_fetched_mid_session(session_factory):
    _store_hk_prices(session_factory, "0700.HK", date(2026, 7, 3))
    _store_hk_prices(session_factory, "9988.HK", date(2026, 7, 3))
    redis = _DictRedis({
        # Metadata written by callers that omitted market lives under the US key.
        "price:US:0700.HK:fetch_meta": _meta(_utc(2026, 7, 3, 3, 0), legacy_flag=False),
        "price:US:9988.HK:fetch_meta": _meta(_utc(2026, 7, 3, 9, 0), legacy_flag=False),
    })
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)

    result = service.get_many_cached_only_fresh(["0700.HK", "9988.HK"], period="2y")

    assert result["0700.HK"] is None          # partial bar from 11:00 HKT
    assert result["9988.HK"] is not None      # fetched at 17:00 HKT, final


def test_many_cached_only_fresh_reads_market_scoped_metadata(session_factory):
    # Bulk-fallback fetches store metadata under the symbol's own market key.
    _store_hk_prices(session_factory, "0700.HK", date(2026, 7, 3))
    redis = _DictRedis({
        "price:HK:0700.HK:fetch_meta": _meta(_utc(2026, 7, 3, 3, 0), legacy_flag=False),
    })
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)

    assert service.get_many_cached_only_fresh(["0700.HK"], period="2y")["0700.HK"] is None


class _CountingRedis(_DictRedis):
    """Counts direct GETs and pipeline executions to pin Redis round-trips."""

    def __init__(self, values=None, *, fail_pipeline=False, error_keys=()):
        super().__init__(values)
        self.direct_gets = 0
        self.pipeline_executes = 0
        self.fail_pipeline = fail_pipeline
        self.error_keys = set(error_keys)

    def get(self, key):
        self.direct_gets += 1
        return super().get(key)

    def pipeline(self):
        redis = self

        class _CountingPipeline:
            def __init__(self):
                self.keys = []

            def get(self, key):
                self.keys.append(key)
                return self

            def execute(self, raise_on_error=True):
                """Mirrors redis-py: command errors raise unless raise_on_error=False."""
                redis.pipeline_executes += 1
                if redis.fail_pipeline:
                    raise ConnectionError("redis down")
                results = []
                for key in self.keys:
                    if key in redis.error_keys:
                        error = RedisResponseError(f"WRONGTYPE {key}")
                        if raise_on_error:
                            raise error
                        results.append(error)
                    else:
                        results.append(redis.values.get(key))
                return results

        return _CountingPipeline()


def _store_bulk_prices(session_factory):
    _store_hk_prices(session_factory, "0700.HK", date(2026, 7, 3))
    _store_hk_prices(session_factory, "9988.HK", date(2026, 7, 3))
    _store_hk_prices(session_factory, "AAPL", date(2026, 7, 2))  # US holiday on 07-03


def test_many_cached_only_fresh_reads_metadata_in_one_pipeline(session_factory):
    _store_bulk_prices(session_factory)
    redis = _CountingRedis({
        "price:HK:0700.HK:fetch_meta": _meta(_utc(2026, 7, 3, 3, 0), legacy_flag=False),  # partial
        "price:US:9988.HK:fetch_meta": _meta(_utc(2026, 7, 3, 9, 0), legacy_flag=False),  # final
    })
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)

    result = service.get_many_cached_only_fresh(["0700.HK", "9988.HK", "AAPL"], period="2y")

    assert result["0700.HK"] is None
    assert result["9988.HK"] is not None
    assert result["AAPL"] is not None
    assert redis.direct_gets == 0
    assert redis.pipeline_executes == 1


def test_many_cached_only_fresh_keeps_redis_failure_behaviour(session_factory):
    """A Redis error means no metadata, as the per-symbol reads did: rows stay fresh by date."""
    _store_bulk_prices(session_factory)
    redis = _CountingRedis({
        "price:HK:0700.HK:fetch_meta": _meta(_utc(2026, 7, 3, 3, 0), legacy_flag=False),
    }, fail_pipeline=True)
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)

    result = service.get_many_cached_only_fresh(["0700.HK", "9988.HK", "AAPL"], period="2y")

    assert all(result[symbol] is not None for symbol in ("0700.HK", "9988.HK", "AAPL"))


def test_many_cached_only_fresh_isolates_a_failing_metadata_key(session_factory):
    """One corrupted key (e.g. WRONGTYPE) must not discard every other symbol's metadata."""
    _store_bulk_prices(session_factory)
    redis = _CountingRedis(
        {
            "price:US:9988.HK:fetch_meta": _meta(_utc(2026, 7, 3, 3, 0), legacy_flag=False),  # partial
        },
        error_keys={"price:US:0700.HK:fetch_meta"},
    )
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)

    result = service.get_many_cached_only_fresh(["0700.HK", "9988.HK", "AAPL"], period="2y")

    assert result["9988.HK"] is None       # its partial marker is still honoured
    assert result["0700.HK"] is not None   # only the failing key is treated as missing
    assert result["AAPL"] is not None


def test_many_cached_only_fresh_uses_chunked_bulk_client(session_factory, monkeypatch):
    """Full-universe metadata reads go through the long-timeout bulk client, in chunks;
    a failing chunk loses only its own metadata."""
    import app.services.price_cache_service as module

    _store_bulk_prices(session_factory)
    metas = {
        "price:US:9988.HK:fetch_meta": _meta(_utc(2026, 7, 3, 3, 0), legacy_flag=False),  # partial
    }
    own_client = _CountingRedis(metas)
    bulk_client = _CountingRedis(metas)
    monkeypatch.setattr(module, "get_bulk_redis_client", lambda: bulk_client)
    monkeypatch.setattr(module.settings, "redis_pipeline_chunk_size", 1)
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), own_client)

    original_pipeline = bulk_client.pipeline
    calls = {"n": 0}

    def _first_chunk_times_out():
        pipeline = original_pipeline()
        calls["n"] += 1
        if calls["n"] == 1:
            def _timeout(raise_on_error=True):
                raise TimeoutError("socket timeout")
            pipeline.execute = _timeout
        return pipeline

    bulk_client.pipeline = _first_chunk_times_out

    result = service.get_many_cached_only_fresh(["0700.HK", "9988.HK", "AAPL"], period="2y")

    assert own_client.pipeline_executes == 0 and own_client.direct_gets == 0
    assert calls["n"] == 3                  # one pipeline per symbol with chunk size 1
    assert result["9988.HK"] is None        # a later chunk's partial marker still honoured


def test_bulk_get_many_isolates_a_failing_redis_key(session_factory, monkeypatch):
    """One corrupted key must not turn the whole get_many request into None."""
    import app.services.price_cache_service as module

    _store_bulk_prices(session_factory)
    redis = _CountingRedis(error_keys={"price:US:9988.HK:recent"})
    monkeypatch.setattr(module, "get_bulk_redis_client", lambda: None)
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)

    result = service.get_many(["0700.HK", "9988.HK"], period="2y")

    assert result["0700.HK"] is not None
    assert result["9988.HK"] is not None  # the failing key reads as a miss -> DB fallback


def test_stale_scan_isolates_a_failing_metadata_key(session_factory):
    redis = _CountingRedis(
        {
            "price:HK:0700.HK:fetch_meta": _meta(_utc(2026, 7, 2, 3, 0), legacy_flag=False),
            "price:US:AAPL:fetch_meta": "corrupt",
        },
        error_keys={"price:US:AAPL:fetch_meta"},
    )
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)

    assert service.get_stale_intraday_symbols() == ["0700.HK"]


def test_latest_fetch_metadata_wins_across_key_namespaces(session_factory):
    # Partial HK-key fetch at 11:00 HKT, then a final US-key refresh at 17:00 HKT.
    _store_hk_prices(session_factory, "0700.HK", date(2026, 7, 3))
    redis = _DictRedis({
        "price:HK:0700.HK:fetch_meta": _meta(_utc(2026, 7, 3, 3, 0), legacy_flag=False),
        "price:US:0700.HK:fetch_meta": _meta(_utc(2026, 7, 3, 9, 0), legacy_flag=False),
    })
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)

    assert service.get_many_cached_only_fresh(["0700.HK"], period="2y")["0700.HK"] is not None
    assert service.get_stale_intraday_symbols() == []


def test_unscoped_bulk_get_consults_symbol_market_metadata(session_factory, monkeypatch):
    """An unscoped get_many must not accept today's DB row fetched mid-session under the HK key."""
    import app.services.bulk_data_fetcher as bulk_module
    import app.services.price_cache_service as module

    _store_hk_prices(session_factory, "0700.HK", date(2026, 7, 3))
    redis = _DictRedis({
        "price:HK:0700.HK:fetch_meta": _meta(_utc(2026, 7, 3, 3, 0), legacy_flag=False),
    })
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)
    monkeypatch.setattr(module, "get_bulk_redis_client", lambda: None)
    monkeypatch.setattr(service, "_store_batch_in_cache_for_market", lambda *args, **kwargs: 0)
    fetched: list[str] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, **kwargs):
            fetched.extend(symbols)
            return {symbol: {"has_error": True, "error": "stub"} for symbol in symbols}

    monkeypatch.setattr(bulk_module, "BulkDataFetcher", _FakeFetcher)

    service.get_many(["0700.HK"], period="2y")

    assert fetched == ["0700.HK"]


def test_scoped_bulk_get_consults_unscoped_metadata(session_factory, monkeypatch):
    """The daily refresh writes US-key metadata; a scoped HK read must still see it."""
    import app.services.bulk_data_fetcher as bulk_module
    import app.services.price_cache_service as module

    _store_hk_prices(session_factory, "0700.HK", date(2026, 7, 3))
    redis = _DictRedis({
        "price:US:0700.HK:fetch_meta": _meta(_utc(2026, 7, 3, 3, 0), legacy_flag=False),  # 11:00 HKT
    })
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)
    monkeypatch.setattr(module, "get_bulk_redis_client", lambda: None)
    monkeypatch.setattr(service, "_store_batch_in_cache_for_market", lambda *args, **kwargs: 0)
    fetched: list[str] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, **kwargs):
            fetched.extend(symbols)
            return {symbol: {"has_error": True, "error": "stub"} for symbol in symbols}

    monkeypatch.setattr(bulk_module, "BulkDataFetcher", _FakeFetcher)

    service.get_many(["0700.HK"], period="2y", market_by_symbol={"0700.HK": "HK"})

    assert fetched == ["0700.HK"]
    assert service.get_cached_only_fresh("0700.HK", period="2y", market="HK") is None


def test_redis_payload_is_judged_by_its_own_namespace_metadata(session_factory, monkeypatch):
    """A partial US-key frame must not borrow freshness from newer HK-key metadata."""

    import app.services.price_cache_service as module

    _store_hk_prices(session_factory, "0700.HK", date(2026, 7, 3))  # final DB rows, last close 359
    days = pd.bdate_range(end=pd.Timestamp("2026-07-03"), periods=250)
    partial = pd.DataFrame(
        {"Open": 999.0, "High": 999.0, "Low": 999.0, "Close": 999.0, "Adj Close": 999.0, "Volume": 1},
        index=days,
    )
    redis = _DictRedis({
        "price:US:0700.HK:recent": encode_frame(partial),
        "price:US:0700.HK:fetch_meta": _meta(_utc(2026, 7, 3, 3, 0), legacy_flag=False),  # 11:00 HKT
        "price:HK:0700.HK:fetch_meta": _meta(_utc(2026, 7, 3, 9, 0), legacy_flag=False),  # 17:00 HKT
    })
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)
    monkeypatch.setattr(module, "get_bulk_redis_client", lambda: None)

    frame = service.get_many(["0700.HK"], period="2y")["0700.HK"]

    assert float(frame["Close"].iloc[-1]) == 359.0


def test_registered_non_universe_instrument_uses_its_own_market(session_factory):
    # ^HSI is fetched via the key-market registry, not stock_universe.
    redis = _DictRedis({
        "price:US:^HSI:fetch_meta": _meta(_utc(2026, 7, 2, 3, 0), legacy_flag=False),  # 11:00 HKT
    })
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 2, 9, 0)), redis)  # 17:00 HKT

    assert service._calendar_markets(["^HSI", "BTC-USD"]) == {"^HSI": "HK", "BTC-USD": "US"}
    assert service._is_intraday_data_stale("^HSI") is True


def test_bulk_fallback_refreshes_registered_non_universe_instruments(session_factory, monkeypatch):
    """^HSI is not in stock_universe but is a registered key-market instrument: fetch it as HK."""
    import app.services.bulk_data_fetcher as bulk_module

    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)))
    fetched: list[tuple[tuple[str, ...], str | None]] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, **kwargs):
            fetched.append((tuple(symbols), kwargs.get("market")))
            return {symbol: {"has_error": True, "error": "stub"} for symbol in symbols}

    monkeypatch.setattr(bulk_module, "BulkDataFetcher", _FakeFetcher)

    service.get_many(["^HSI"], period="2y")

    assert fetched == [(("^HSI",), "HK")]


def test_warming_redis_from_database_does_not_stamp_fetch_metadata(session_factory, monkeypatch):
    """A copy of DB rows is not a provider fetch; stamping it would vouch for the DB row later."""
    import app.services.price_cache_service as module

    _store_hk_prices(session_factory, "0700.HK", date(2026, 7, 3))
    redis = _DictRedis()
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)
    monkeypatch.setattr(module, "get_bulk_redis_client", lambda: None)

    frame = service.get_many(["0700.HK"], period="2y")["0700.HK"]
    service.get_historical_data("0700.HK", period="2y")

    assert frame is not None
    assert any(key.endswith(":recent") for key in redis.values)  # the warm still happens
    assert not [key for key in redis.values if key.endswith(":fetch_meta")]


def test_refreshed_batch_overwrites_both_key_namespaces(session_factory, monkeypatch):
    import app.services.price_cache_service as module

    monkeypatch.setattr(module, "get_eastern_now", lambda: _utc(2026, 7, 3, 9, 0).astimezone(module.EASTERN))
    redis = _DictRedis()
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 9, 0)), redis)
    db_writes = []
    monkeypatch.setattr(service, "_store_batch_in_database", lambda batch: db_writes.append(set(batch)))
    days = pd.bdate_range(end=pd.Timestamp("2026-07-03"), periods=5)
    frame = pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Adj Close": 1.0, "Volume": 1},
        index=days,
    )

    service.store_refreshed_batch({"0700.HK": frame, "AAPL": frame})

    assert {"price:US:0700.HK:recent", "price:HK:0700.HK:recent", "price:US:AAPL:recent"} <= set(redis.values)
    assert "price:HK:AAPL:recent" not in redis.values
    assert db_writes == [{"0700.HK", "AAPL"}]


# ── A short live top-up keeps the full cached history (#493) ───────────


def _bars(end, periods: int, close: float = 1.0) -> pd.DataFrame:
    days = pd.bdate_range(end=end, periods=periods)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Adj Close": close, "Volume": 1},
        index=days,
    )


def _store_bars(session_factory, symbol: str, frame: pd.DataFrame) -> None:
    db = session_factory()
    for day, row in frame.iterrows():
        db.add(StockPrice(
            symbol=symbol, date=day.date(), open=row["Open"], high=row["High"], low=row["Low"],
            close=row["Close"], adj_close=row["Adj Close"], volume=int(row["Volume"]),
        ))
    db.commit()
    db.close()


_HISTORY_BARS = 600  # more than 2y, so a 2y re-read would visibly truncate


def _top_up_after_history(session_factory, symbols):
    """~2.3y of stored bars ending a few days ago, then a top-up overlapping two of them."""
    today = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=1)[0]
    history_end = today - pd.offsets.BDay(3)
    for symbol in symbols:
        _store_bars(session_factory, symbol, _bars(history_end, _HISTORY_BARS))
    return _bars(today, 5, close=2.0)


# "5d" is not in PERIOD_DAYS: an unknown short period must not pass as 2y.
@pytest.mark.parametrize("period", ["7d", "5d"])
def test_short_top_up_caches_full_history_in_reader_namespaces(session_factory, period):
    redis = _DictRedis()
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)
    top_up = _top_up_after_history(session_factory, ("AAPL", "0700.HK"))

    service.store_refreshed_batch({"AAPL": top_up, "0700.HK": top_up}, period=period)

    for key in ("price:US:AAPL:recent", "price:US:0700.HK:recent", "price:HK:0700.HK:recent"):
        frame = decode_frame(redis.values[key])
        assert len(frame) == _HISTORY_BARS + 3, key  # stored + 5 fetched - 2 overlapping
        assert frame.index.is_unique and frame.index.is_monotonic_increasing
        assert frame.index[-1] == top_up.index[-1]
        assert service._covers_period(frame, "5y")
    for key in ("price:US:AAPL:fetch_meta", "price:HK:0700.HK:fetch_meta"):
        assert key in redis.values


def test_get_many_after_short_top_up_hits_redis_without_refill(session_factory, monkeypatch):
    import app.services.price_cache_service as module

    monkeypatch.setattr(module, "get_bulk_redis_client", lambda: None)
    redis = _DictRedis()
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)
    top_up = _top_up_after_history(session_factory, ("AAPL", "0700.HK"))
    service.store_refreshed_batch({"AAPL": top_up, "0700.HK": top_up}, period="7d")

    # Freshness has its own tests; this checks namespace, coverage and length.
    monkeypatch.setattr(service, "_get_expected_data_date", lambda market: top_up.index[-1].date())
    monkeypatch.setattr(service, "_is_fetch_metadata_stale", lambda *args, **kwargs: False)

    def no_refill(*args, **kwargs):
        raise AssertionError("served from Redis; no DB refill or provider fetch")

    monkeypatch.setattr(service, "_resolve_bulk_fallback", no_refill)

    # 5y readers (VolumeBreakthrough, 5y charts) get the whole frame; 2y
    # readers get it trimmed to their window.
    for period, min_bars in (("2y", 500), ("5y", _HISTORY_BARS + 3)):
        for symbols, market in ((["AAPL"], None), (["0700.HK"], "HK"), (["0700.HK"], None)):
            frame = service.get_many(symbols, period=period, market=market)[symbols[0]]
            assert frame is not None, (period, symbols, market)
            assert len(frame) >= min_bars and frame.index[-1] == top_up.index[-1]


def test_short_top_up_for_new_listing_caches_only_its_real_bars(session_factory):
    """The re-read returns just the committed bars; none are invented, so
    readers still find too little history for a full window."""
    redis = _DictRedis()
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)
    top_up = _bars(pd.Timestamp.today().normalize(), 5)

    service.store_refreshed_batch({"AAPL": top_up}, period="7d")

    frame = decode_frame(redis.values["price:US:AAPL:recent"])
    assert list(frame.index) == list(top_up.index)


def test_short_top_up_symbol_missing_from_reread_keeps_a_short_claim(session_factory, monkeypatch):
    redis = _DictRedis()
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)
    top_up = _top_up_after_history(session_factory, ("AAPL",))
    monkeypatch.setattr(
        service, "_get_many_from_database", lambda symbols, *a, **k: {s: (None, None) for s in symbols}
    )

    service.store_refreshed_batch({"AAPL": top_up}, period="7d")

    frame = decode_frame(redis.values["price:US:AAPL:recent"])
    assert len(frame) == 5
    assert not service._covers_period(frame, "2y")
    assert "price:US:AAPL:fetch_meta" in redis.values


def test_short_top_up_reread_failure_leaves_redis_untouched(session_factory, monkeypatch):
    """The rows are committed; a failed re-read must not replace good cached
    history with the short frame. Readers fall back to the database."""
    old = {"price:US:AAPL:recent": "old-frame", "price:US:AAPL:fetch_meta": "old-meta"}
    redis = _DictRedis(old)
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)
    monkeypatch.setattr(service, "_store_batch_in_database", lambda batch: None)

    class _FailingQuerySession:
        """Opens fine, fails mid-query: the case the helper used to swallow."""

        def query(self, *args, **kwargs):
            raise RuntimeError("database unavailable")

        def close(self):
            pass

    service._session_factory = _FailingQuerySession

    stored = service.store_refreshed_batch({"AAPL": _bars(pd.Timestamp.today(), 5)}, period="7d")

    assert stored == 0
    assert redis.values == old


def test_short_top_up_without_usable_rows_writes_nothing(session_factory):
    """A suspended stock returns NaN prices: nothing was stored, so nothing is vouched for."""
    redis = _DictRedis()
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)
    top_up = _top_up_after_history(session_factory, ("AAPL",))
    for column in ("Open", "High", "Low", "Close", "Adj Close"):
        top_up[column] = float("nan")
    top_up["Volume"] = 0

    service.store_refreshed_batch({"AAPL": top_up}, period="7d")

    assert redis.values == {}


# ── After-close stale scan covers every market ─────────────────────────


def test_stale_intraday_scan_finds_mid_session_bars_in_any_market(session_factory):
    redis = _DictRedis({
        "price:HK:0700.HK:fetch_meta": _meta(_utc(2026, 7, 2, 3, 0), legacy_flag=False),
        # 12:00 HKT: lunch break, market "closed" but the day's bar is still partial.
        "price:US:9988.HK:fetch_meta": _meta(_utc(2026, 7, 2, 4, 0), legacy_flag=False),
        "price:US:AAPL:fetch_meta": _meta(_utc(2026, 7, 2, 18, 0), legacy_flag=True),
        "price:US:MSFT:fetch_meta": _meta(_utc(2026, 7, 2, 21, 0), legacy_flag=False),
    })
    # 06:00 ET on the US holiday: the old scan returned nothing before 16:30 ET.
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)

    assert sorted(service.get_stale_intraday_symbols()) == ["0700.HK", "9988.HK", "AAPL"]


# ── Writers record the market-aware session state ──────────────────────


def test_fetch_metadata_marks_hk_fetch_during_hk_session_as_intraday(session_factory, monkeypatch):
    import app.services.price_cache_service as module

    fetched_at = _utc(2026, 7, 2, 3, 0)  # 11:00 HKT, US closed
    monkeypatch.setattr(module, "get_eastern_now", lambda: fetched_at.astimezone(module.EASTERN))
    redis = _DictRedis()
    service = _service(session_factory, _FrozenCalendar(fetched_at), redis)

    service._store_fetch_metadata("0700.HK", market="HK")

    meta = json.loads(redis.values["price:HK:0700.HK:fetch_meta"])
    assert meta["market"] == "HK"
    assert meta["needs_refresh_after_close"] is True
    assert meta["market_was_open"] is True


# ── Fetch metadata is stamped only after the DB write succeeds (#437) ──


def _commit_fails(session_factory):
    from sqlalchemy.exc import OperationalError

    def factory():
        db = session_factory()

        def commit():
            raise OperationalError("COMMIT", {}, Exception("lock timeout"))

        db.commit = commit
        return db

    return factory


def _recent_frame() -> pd.DataFrame:
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=5)
    return pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Adj Close": 1.0, "Volume": 1},
        index=days,
    )


def _full_fetch(service, frame):
    service._fetch_direct_historical_data = lambda symbol, period: frame
    service._fetch_full_and_cache("AAPL", "2y")


def _incremental_fetch(service, frame):
    # Two uncached bars, so a rejected latest bar still leaves one row to write.
    service._fetch_direct_historical_data = lambda symbol, period: frame
    service._fetch_incremental_and_merge(
        "AAPL", "2y", cached_data=frame.iloc[:-2], last_cached_date=frame.index[-3].date()
    )


_PRICE_WRITERS = {
    "batch": lambda service, frame: service.store_batch_in_cache({"AAPL": frame}),
    "refreshed_batch": lambda service, frame: service.store_refreshed_batch({"AAPL": frame}),
    "refreshed_batch_7d": lambda service, frame: service.store_refreshed_batch(
        {"AAPL": frame}, period="7d"
    ),
    "single": lambda service, frame: service.store_in_cache("AAPL", frame),
    "full_fetch": _full_fetch,
    "incremental_fetch": _incremental_fetch,
}


_BATCH_WRITERS = {"batch", "refreshed_batch", "refreshed_batch_7d"}


@pytest.mark.parametrize("name", _PRICE_WRITERS)
def test_failed_db_write_leaves_redis_frame_and_metadata_untouched(session_factory, name):
    # Writing the frame without fresh metadata would pair today's bar with the
    # older record, so a partial bar could pass as final; keep the old pair.
    from sqlalchemy.exc import OperationalError

    redis = _DictRedis()
    service = _service(_commit_fails(session_factory), _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)

    if name in _BATCH_WRITERS:
        # Refresh runners classify the error (retry transient ones) and mark the
        # batch failed; swallowing it would report a lost write as refreshed.
        with pytest.raises(OperationalError):
            _PRICE_WRITERS[name](service, _recent_frame())
    else:
        _PRICE_WRITERS[name](service, _recent_frame())

    assert redis.values == {}


def test_bulk_fallback_still_returns_fetched_frames_when_db_write_fails(session_factory, monkeypatch):
    import app.services.bulk_data_fetcher as bulk_module
    import app.services.price_cache_service as module

    monkeypatch.setattr(module, "get_bulk_redis_client", lambda: None)
    redis = _DictRedis()
    service = _service(_commit_fails(session_factory), _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)
    frame = _recent_frame()

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, **kwargs):
            return {symbol: {"has_error": False, "price_data": frame} for symbol in symbols}

    monkeypatch.setattr(bulk_module, "BulkDataFetcher", _FakeFetcher)

    result = service.get_many(["AAPL"], period="2y")

    assert result["AAPL"] is not None
    assert redis.values == {}


@pytest.mark.parametrize("write", _PRICE_WRITERS.values(), ids=_PRICE_WRITERS.keys())
def test_successful_db_write_stamps_fetch_metadata(session_factory, write):
    redis = _DictRedis()
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)

    write(service, _recent_frame())

    assert "price:US:AAPL:fetch_meta" in redis.values
    assert "price:US:AAPL:recent" in redis.values


# ── A dropped latest bar must not certify the row it failed to replace (#449) ──


def _frame_with_nan_latest_close() -> pd.DataFrame:
    frame = _recent_frame()
    frame.loc[frame.index[-1], "Close"] = float("nan")
    return frame


def _store_partial_row(session_factory, day) -> None:
    db = session_factory()
    db.add(StockPrice(symbol="AAPL", date=day, open=0.5, high=0.5, low=0.5, close=0.5, volume=1))
    db.commit()
    db.close()


@pytest.mark.parametrize("write", _PRICE_WRITERS.values(), ids=_PRICE_WRITERS.keys())
def test_dropped_latest_bar_does_not_vouch_for_the_stored_partial_row(session_factory, write):
    frame = _frame_with_nan_latest_close()
    _store_partial_row(session_factory, frame.index[-1].date())
    redis = _DictRedis()
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)

    write(service, frame)

    assert redis.values == {}


@pytest.mark.parametrize("write", _PRICE_WRITERS.values(), ids=_PRICE_WRITERS.keys())
def test_dropped_latest_bar_without_a_stored_row_is_still_stamped(session_factory, write):
    # Illiquid tickers get NaN rows on no-trade days; they must keep caching.
    redis = _DictRedis()
    service = _service(session_factory, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), redis)

    write(service, _frame_with_nan_latest_close())

    assert "price:US:AAPL:fetch_meta" in redis.values
    assert "price:US:AAPL:recent" in redis.values


def test_unavailable_stored_row_check_does_not_vouch_for_the_dropped_bar():
    def no_session():
        raise RuntimeError("database unavailable")

    service = _service(no_session, _FrozenCalendar(_utc(2026, 7, 3, 10, 0)), _DictRedis())
    raw = _frame_with_nan_latest_close()

    assert service._unreplaced_rejected_rows({"AAPL": raw}, {"AAPL": raw.iloc[:-1]}) == {"AAPL"}
