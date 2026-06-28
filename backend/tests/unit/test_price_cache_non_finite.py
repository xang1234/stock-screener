from __future__ import annotations

from datetime import date

import pandas as pd

from app.services.price_cache_service import PriceCacheService


def _price_frame(closes: list[float], days: list[date]) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Adj Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=pd.to_datetime(days),
    )
    data.index.name = "Date"
    return data


def test_store_batch_in_cache_skips_non_finite_close_rows():
    captured_rows = []

    class FakeQuery:
        def filter(self, *_args):
            return self

        def all(self):
            return []

    class FakeSession:
        def query(self, *_args):
            return FakeQuery()

        def bulk_insert_mappings(self, _model, rows):
            captured_rows.extend(rows)

        def bulk_update_mappings(self, _model, _rows):
            raise AssertionError("No existing rows should be updated")

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    service = PriceCacheService(redis_client=None, session_factory=FakeSession)
    payload = _price_frame(
        [101.0, float("nan")],
        [date(2026, 6, 25), date(2026, 6, 26)],
    )

    service.store_batch_in_cache({"SPY": payload}, also_store_db=True)

    assert [(row["symbol"], row["date"], row["close"]) for row in captured_rows] == [
        ("SPY", date(2026, 6, 25), 101.0)
    ]


def test_fetch_full_and_cache_uses_cleaned_price_frame_for_redis_db_and_return():
    service = PriceCacheService(redis_client=None, session_factory=lambda: None)
    raw = _price_frame(
        [101.0, float("nan")],
        [date(2026, 6, 25), date(2026, 6, 26)],
    )
    captured = {}

    service._fetch_direct_historical_data = lambda symbol, period: raw  # type: ignore[assignment]
    service._store_recent_in_redis = lambda symbol, data, market=None: captured.setdefault("redis", data)  # type: ignore[assignment]
    service._store_in_database = lambda symbol, data: captured.setdefault("db", data)  # type: ignore[assignment]

    result = service._fetch_full_and_cache("SPY", "2y")

    assert result is not None
    assert result["Close"].tolist() == [101.0]
    assert captured["redis"]["Close"].tolist() == [101.0]
    assert captured["db"]["Close"].tolist() == [101.0]


def test_incremental_merge_uses_cleaned_price_frame_for_redis_db_and_return():
    service = PriceCacheService(redis_client=None, session_factory=lambda: None)
    cached = _price_frame([100.0], [date(2026, 6, 24)])
    raw_incremental = _price_frame(
        [101.0, float("nan")],
        [date(2026, 6, 25), date(2026, 6, 26)],
    )
    captured = {}

    service._fetch_direct_historical_data = lambda symbol, period: raw_incremental  # type: ignore[assignment]
    service._store_recent_in_redis = lambda symbol, data, market=None: captured.setdefault("redis", data)  # type: ignore[assignment]
    service._store_in_database = lambda symbol, data: captured.setdefault("db", data)  # type: ignore[assignment]

    result = service._fetch_incremental_and_merge(
        "SPY",
        "2y",
        cached,
        date(2026, 6, 24),
    )

    assert result is not None
    assert result["Close"].tolist() == [100.0, 101.0]
    assert captured["redis"]["Close"].tolist() == [100.0, 101.0]
    assert captured["db"]["Close"].tolist() == [101.0]
