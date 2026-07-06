from __future__ import annotations

from contextlib import contextmanager
from datetime import date

import app.scripts.export_static_site as export_script
from app.services.benchmark_cache_service import BenchmarkFallbackPolicy


def test_compute_static_market_exposure_uses_primary_only_policy_for_us(monkeypatch):
    calls: list[dict[str, object]] = []
    db = object()

    @contextmanager
    def fake_session():
        yield db

    def fake_refresh(db_arg, market, as_of_date, *, benchmark_fallback_policy):
        calls.append(
            {
                "db": db_arg,
                "market": market,
                "as_of_date": as_of_date,
                "benchmark_fallback_policy": benchmark_fallback_policy,
            }
        )
        return {
            "market": market,
            "date": as_of_date,
            "exposure_score": 70.0,
            "benchmark_symbol": "SPY",
        }

    monkeypatch.setattr(export_script, "SessionLocal", fake_session)
    monkeypatch.setattr(
        "app.services.market_exposure_service.refresh_market_exposure_for_date",
        fake_refresh,
    )

    result = export_script._compute_static_market_exposure(  # noqa: SLF001
        as_of_date=date(2026, 7, 1),
        market="us",
    )

    assert result["benchmark_symbol"] == "SPY"
    assert calls == [
        {
            "db": db,
            "market": "US",
            "as_of_date": date(2026, 7, 1),
            "benchmark_fallback_policy": BenchmarkFallbackPolicy.PRIMARY_ONLY,
        }
    ]


def test_compute_static_market_exposure_allows_fallback_policy_for_non_us(monkeypatch):
    calls: list[dict[str, object]] = []
    db = object()

    @contextmanager
    def fake_session():
        yield db

    def fake_refresh(db_arg, market, as_of_date, *, benchmark_fallback_policy):
        calls.append(
            {
                "db": db_arg,
                "market": market,
                "as_of_date": as_of_date,
                "benchmark_fallback_policy": benchmark_fallback_policy,
            }
        )
        return {
            "market": market,
            "date": as_of_date,
            "exposure_score": 70.0,
            "benchmark_symbol": "^HSI",
        }

    monkeypatch.setattr(export_script, "SessionLocal", fake_session)
    monkeypatch.setattr(
        "app.services.market_exposure_service.refresh_market_exposure_for_date",
        fake_refresh,
    )

    result = export_script._compute_static_market_exposure(  # noqa: SLF001
        as_of_date=date(2026, 7, 1),
        market="hk",
    )

    assert result["benchmark_symbol"] == "^HSI"
    assert calls == [
        {
            "db": db,
            "market": "HK",
            "as_of_date": date(2026, 7, 1),
            "benchmark_fallback_policy": BenchmarkFallbackPolicy.ALLOW,
        }
    ]
