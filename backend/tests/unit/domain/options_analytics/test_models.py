from __future__ import annotations

import pytest
from app.domain.options_analytics.models import (
    DividendSource,
    OptionCandidateInput,
)


def test_dividend_source_is_normalized_to_a_closed_enum() -> None:
    candidate = OptionCandidateInput(
        symbol="AAPL",
        composite_score=99,
        daily_dollar_volume=200_000_000,
        spot_price=100,
        dividend_yield=0.01,
        dividend_source="pinned_feature_run",
    )

    assert candidate.dividend_source is DividendSource.PINNED_FEATURE_RUN


def test_pinned_dividend_source_requires_a_finite_non_negative_value() -> None:
    with pytest.raises(ValueError, match="Pinned dividend source"):
        OptionCandidateInput(
            symbol="AAPL",
            composite_score=99,
            daily_dollar_volume=200_000_000,
            spot_price=100,
            dividend_yield=None,
            dividend_source="pinned_feature_run",
        )


def test_zero_assumption_normalizes_the_dividend_value() -> None:
    candidate = OptionCandidateInput(
        symbol="AAPL",
        composite_score=99,
        daily_dollar_volume=200_000_000,
        spot_price=100,
        dividend_yield=None,
        dividend_source="zero_assumption",
    )

    assert candidate.dividend_yield == 0.0
    assert candidate.dividend_source is DividendSource.ZERO_ASSUMPTION
