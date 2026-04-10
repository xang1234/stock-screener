from __future__ import annotations

from unittest.mock import MagicMock


def test_store_all_caches_uses_injected_session_factory(monkeypatch):
    import app.services.hybrid_fundamentals_service as module

    ownership_service = MagicMock()
    ownership_service.bulk_update.return_value = 2
    ownership_ctor = MagicMock(return_value=ownership_service)
    monkeypatch.setattr(module, "InstitutionalOwnershipService", ownership_ctor)

    service = module.HybridFundamentalsService(price_cache=MagicMock())
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
    )
    session_factory.assert_called_once_with()
    ownership_ctor.assert_called_once_with(fake_db)
    ownership_service.bulk_update.assert_called_once()
    fake_db.close.assert_called_once_with()
    assert stats["fundamentals_stored"] == 1
    assert stats["ownership_updated"] == 2
    assert stats["failed"] == 1
