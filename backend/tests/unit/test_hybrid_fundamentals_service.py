from __future__ import annotations

from unittest.mock import MagicMock


def test_constructor_preserves_injected_finviz_limiter(
    monkeypatch,
):
    import app.services.hybrid_fundamentals_service as module

    injected_limiter = object()
    finviz_service = MagicMock()
    finviz_service._rate_limiter = injected_limiter
    fake_price_cache = MagicMock()

    service = module.HybridFundamentalsService(
        price_cache=fake_price_cache,
        finviz_service=finviz_service,
    )

    assert service.price_cache is fake_price_cache
    assert service._finviz_rate_limiter is injected_limiter
    assert service.bulk_fetcher._rate_limiter is injected_limiter


def test_store_all_caches_uses_injected_session_factory(monkeypatch):
    import app.services.hybrid_fundamentals_service as module

    ownership_service = MagicMock()
    ownership_service.bulk_update.return_value = 2
    ownership_ctor = MagicMock(return_value=ownership_service)
    monkeypatch.setattr(module, "InstitutionalOwnershipService", ownership_ctor)
    finviz_service = MagicMock()
    finviz_service._rate_limiter = MagicMock()

    service = module.HybridFundamentalsService(
        price_cache=MagicMock(),
        finviz_service=finviz_service,
    )
    fundamentals_cache = MagicMock()
    fake_db = MagicMock()
    session_factory = MagicMock(return_value=fake_db)

    stats = service.store_all_caches(
        {
            "AAPL": {"symbol": "AAPL", "market_cap": 1_000},
            "BAD": {"has_error": True},
        },
        fundamentals_cache,
        session_factory=session_factory,
    )

    fundamentals_cache.store.assert_called_once_with(
        "AAPL",
        {"symbol": "AAPL", "market_cap": 1_000},
        data_source="hybrid",
        market=None,
    )
    session_factory.assert_called_once_with()
    ownership_ctor.assert_called_once_with(fake_db)
    ownership_service.bulk_update.assert_called_once()
    fake_db.close.assert_called_once_with()
    assert stats["fundamentals_stored"] == 1
    assert stats["ownership_updated"] == 2
    assert stats["failed"] == 1
