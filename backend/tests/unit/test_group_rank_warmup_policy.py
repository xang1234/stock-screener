from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from app.services.group_rank_warmup_policy import evaluate_same_day_group_rank_warmup


def test_group_rank_warmup_policy_allows_partial_cache_above_market_threshold() -> None:
    price_cache = MagicMock()
    price_cache.get_warmup_metadata.return_value = {
        "status": "partial",
        "count": 1081,
        "total": 1969,
        "completed_at": datetime.now().isoformat(),
    }

    decision = evaluate_same_day_group_rank_warmup(price_cache, market="TW")

    assert decision.error is None
    assert decision.require_complete_cache is False
    assert decision.min_cache_coverage == 0.50


def test_group_rank_warmup_policy_requires_complete_cache_when_partial_is_too_low() -> None:
    price_cache = MagicMock()
    price_cache.get_warmup_metadata.return_value = {
        "status": "partial",
        "count": 1,
        "total": 10,
        "completed_at": datetime.now().isoformat(),
    }

    decision = evaluate_same_day_group_rank_warmup(price_cache, market="US")

    assert decision.error is not None
    assert decision.require_complete_cache is True
    assert decision.min_cache_coverage is None
