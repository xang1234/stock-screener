from __future__ import annotations

import warnings

import pandas as pd

from app.services.theme_correlation_service import ThemeCorrelationService


def test_calculate_returns_tolerates_sparse_price_history_without_futurewarning():
    service = ThemeCorrelationService(db=None)
    price_df = pd.DataFrame(
        {
            "AAPL": [100.0, None, 101.0, 103.0, 104.0],
            "NVDA": [200.0, 202.0, None, 205.0, 207.0],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        returns_df = service._calculate_returns(price_df)

    assert not returns_df.empty
    assert list(returns_df.columns) == ["AAPL", "NVDA"]
