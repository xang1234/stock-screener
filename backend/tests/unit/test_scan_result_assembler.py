"""Characterization tests for scan-result assembly."""

import pandas as pd

from app.scanners.base_screener import ScreenerResult, StockData
from app.scanners.scan_result_assembler import (
    ScanResultAssembler,
    ScanResultAssemblyRequest,
)


def test_assembler_preserves_result_contract_and_projects_opportunity_state():
    prices = pd.DataFrame(
        {"Close": [120.0], "Volume": [1_000_000]},
        index=pd.date_range("2026-08-21", periods=1),
    )
    stock_data = StockData(
        symbol="NVDA",
        price_data=prices,
        benchmark_data=None,
        fundamentals={
            "market_cap": 4_000_000_000_000,
            "avg_volume": 2_000_000,
            "dividend_yield": 0.6,
        },
    )
    screener_results = {
        "minervini": ScreenerResult(
            score=88.0,
            passes=True,
            rating="Buy",
            breakdown={"trend": 40.0},
            details={"rs_rating": 95.0, "stage": 2},
            screener_name="minervini",
        )
    }
    projection_calls = []

    def project(result, projected_stock_data, _parameters):
        projection_calls.append((dict(result), projected_stock_data))
        return {
            "correction_survivor": True,
            "resilience_score": 91.0,
            "action_state": "setup_ready",
            "opportunity_state": {"schema_version": 1},
        }

    result = ScanResultAssembler(opportunity_projector=project).assemble(
        ScanResultAssemblyRequest(
            symbol="NVDA",
            stock_data=stock_data,
            screener_results=screener_results,
            composite_score=88.0,
            overall_rating="Buy",
            composite_method="weighted_average",
            applicable_screeners=("minervini",),
            unavailable_screeners=(),
            history_bars=260,
            scan_mode="full",
            data_status="complete",
            is_scannable=True,
            field_completeness_score=94,
        )
    )

    assert result["symbol"] == "NVDA"
    assert result["composite_score"] == 88.0
    assert result["rs_rating"] == 95.0
    assert result["market_cap"] == 4_000_000_000_000
    assert result["avg_dollar_volume"] == 240_000_000
    assert result["dividend_yield"] == 0.6
    assert result["opportunity_state"]["schema_version"] == 1
    assert projection_calls[0][1] is stock_data
