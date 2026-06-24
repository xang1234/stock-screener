from __future__ import annotations

from datetime import date
from types import SimpleNamespace


def test_build_price_refresh_planning_input_logs_diagnostic_stages(monkeypatch):
    import app.services.price_refresh_plan_builder as module

    stages = []

    class _Stage:
        def __init__(self, _logger, name, **extra):
            self.name = name
            self.extra = extra

        def __enter__(self):
            stages.append(("start", self.name, self.extra))

        def __exit__(self, exc_type, exc, tb):
            stages.append(("finish", self.name, self.extra))

    class _StockUniverse:
        symbol = SimpleNamespace(in_=lambda _symbols: True)
        market = "US"
        is_active = True
        market_cap = SimpleNamespace(
            desc=lambda: SimpleNamespace(nullslast=lambda: "market_cap_desc")
        )

    class _Query:
        def filter(self, *_args):
            return self

        def order_by(self, *_args):
            return self

        def all(self):
            return [SimpleNamespace(symbol="AAPL", market="US")]

    class _Db:
        def query(self, *_args):
            return _Query()

    monkeypatch.setattr(module, "log_runtime_stage", _Stage)
    monkeypatch.setattr(module, "_key_market_refresh_symbols", lambda *_args: {})
    monkeypatch.setattr("app.models.stock_universe.StockUniverse", _StockUniverse)
    monkeypatch.setattr(
        module,
        "classify_price_history",
        lambda *_args, **_kwargs: SimpleNamespace(
            fresh=("AAPL",),
            stale=(),
            no_history=(),
        ),
    )

    module.build_price_refresh_planning_input(
        _Db(),
        mode="delta",
        market="US",
        effective_market="US",
        normalize_market=lambda market: str(market).upper(),
        market_calendar_service=SimpleNamespace(
            last_completed_trading_day=lambda _market: date(2026, 6, 18)
        ),
        sync_github_seed=lambda *_args, **_kwargs: {"status": "missing"},
    )

    assert stages == [
        (
            "start",
            "price_refresh.load_universe",
            {"market": "US", "mode": "delta"},
        ),
        (
            "finish",
            "price_refresh.load_universe",
            {"market": "US", "mode": "delta"},
        ),
        (
            "start",
            "price_refresh.sync_github_seed",
            {"market": "US", "mode": "delta", "symbol_count": 1},
        ),
        (
            "finish",
            "price_refresh.sync_github_seed",
            {"market": "US", "mode": "delta", "symbol_count": 1},
        ),
        (
            "start",
            "price_refresh.classify_coverage",
            {
                "market": "US",
                "mode": "delta",
                "symbol_count": 1,
                "target_as_of": "2026-06-18",
            },
        ),
        (
            "finish",
            "price_refresh.classify_coverage",
            {
                "market": "US",
                "mode": "delta",
                "symbol_count": 1,
                "target_as_of": "2026-06-18",
            },
        ),
    ]
