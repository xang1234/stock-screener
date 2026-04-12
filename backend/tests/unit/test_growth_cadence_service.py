from __future__ import annotations

import pandas as pd

from app.services.growth_cadence_service import (
    BASIS_COMPARABLE_YOY,
    BASIS_QOQ,
    BASIS_UNAVAILABLE,
    CADENCE_INSUFFICIENT,
    CADENCE_QUARTERLY,
    CADENCE_SEMIANNUAL,
    compute_cadence_aware_growth,
)


def _income_frame(columns: list[pd.Timestamp], eps: list[float], revenue: list[float]) -> pd.DataFrame:
    data = {col: [eps[idx], revenue[idx]] for idx, col in enumerate(columns)}
    return pd.DataFrame(data, index=["Diluted EPS", "Total Revenue"])


def test_quarterly_cadence_uses_qoq_basis():
    cols = [
        pd.Timestamp("2025-12-31"),
        pd.Timestamp("2025-09-30"),
        pd.Timestamp("2025-06-30"),
        pd.Timestamp("2025-03-31"),
        pd.Timestamp("2024-12-31"),
    ]
    df = _income_frame(cols, eps=[1.2, 1.0, 0.9, 0.8, 0.6], revenue=[120, 110, 108, 100, 90])

    result = compute_cadence_aware_growth(df, market="US")

    assert result["growth_reporting_cadence"] == CADENCE_QUARTERLY
    assert result["growth_metric_basis"] == BASIS_QOQ
    assert result["eps_growth_qq"] == 20.0
    assert result["sales_growth_qq"] == 9.09
    assert result["eps_growth_yy"] == 100.0
    assert result["sales_growth_yy"] == 33.33
    assert result["growth_comparable_period_date"] == "2024-12-31 00:00:00"


def test_hk_semiannual_cadence_uses_comparable_period_yoy_for_primary_metrics():
    cols = [
        pd.Timestamp("2025-12-31"),
        pd.Timestamp("2025-06-30"),
        pd.Timestamp("2024-12-31"),
    ]
    df = _income_frame(cols, eps=[1.2, 1.0, 0.8], revenue=[120, 110, 100])

    result = compute_cadence_aware_growth(df, market="HK")

    assert result["growth_reporting_cadence"] == CADENCE_SEMIANNUAL
    assert result["growth_metric_basis"] == BASIS_COMPARABLE_YOY
    # No fabricated 6-month comparison for HK: primary growth mirrors comparable YoY.
    assert result["eps_growth_qq"] == 50.0
    assert result["sales_growth_qq"] == 20.0
    assert result["eps_growth_yy"] == 50.0
    assert result["sales_growth_yy"] == 20.0


def test_us_non_quarterly_does_not_emit_fabricated_qoq():
    cols = [
        pd.Timestamp("2025-12-31"),
        pd.Timestamp("2025-06-30"),
        pd.Timestamp("2024-12-31"),
    ]
    df = _income_frame(cols, eps=[1.2, 1.0, 0.8], revenue=[120, 110, 100])

    result = compute_cadence_aware_growth(df, market="US")

    assert result["growth_reporting_cadence"] == CADENCE_SEMIANNUAL
    assert result["growth_metric_basis"] == BASIS_UNAVAILABLE
    assert result["eps_growth_qq"] is None
    assert result["sales_growth_qq"] is None
    assert result["eps_growth_yy"] == 50.0
    assert result["sales_growth_yy"] == 20.0


def test_insufficient_history_marks_unavailable_basis():
    cols = [pd.Timestamp("2025-12-31")]
    df = _income_frame(cols, eps=[1.2], revenue=[120])

    result = compute_cadence_aware_growth(df, market="JP")

    assert result["growth_reporting_cadence"] == CADENCE_INSUFFICIENT
    assert result["growth_metric_basis"] == "unavailable"
    assert result["eps_growth_qq"] is None
    assert result["sales_growth_qq"] is None
