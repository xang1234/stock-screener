from app.domain.scanning.models import ScanResultItemDomain
from app.services.group_detail_payloads import (
    constituent_stock_payloads_from_group_rows,
    scan_result_item_to_group_row,
)


def _scan_item(symbol: str, *, rs_rating: float, composite_score: float) -> ScanResultItemDomain:
    return ScanResultItemDomain(
        symbol=symbol,
        composite_score=composite_score,
        rating="Buy",
        current_price=123.45,
        screener_outputs={},
        screeners_run=[],
        composite_method="weighted_average",
        screeners_passed=0,
        screeners_total=0,
        extended_fields={
            "company_name": f"{symbol} Corp",
            "ibd_industry_group": "Software",
            "rs_rating": rs_rating,
            "rs_rating_1m": rs_rating - 1,
            "rs_rating_3m": rs_rating - 3,
            "price_sparkline_data": [1.0, 1.1],
            "price_trend": 1,
            "price_change_1d": 2.5,
            "rs_sparkline_data": [1.0, 1.2],
            "rs_trend": 1,
            "stage": 2,
        },
    )


def test_scan_result_item_to_group_row_keeps_group_detail_fields():
    row = scan_result_item_to_group_row(
        _scan_item("HIGH", rs_rating=97.0, composite_score=88.0)
    )

    assert row == {
        "symbol": "HIGH",
        "company_name": "HIGH Corp",
        "composite_score": 88.0,
        "current_price": 123.45,
        "rs_rating": 97.0,
        "rs_rating_1m": 96.0,
        "rs_rating_3m": 94.0,
        "rs_rating_12m": None,
        "eps_growth_qq": None,
        "eps_growth_yy": None,
        "sales_growth_qq": None,
        "sales_growth_yy": None,
        "stage": 2,
        "market_cap": None,
        "market_cap_usd": None,
        "ibd_industry_group": "Software",
        "price_sparkline_data": [1.0, 1.1],
        "price_trend": 1,
        "price_change_1d": 2.5,
        "rs_sparkline_data": [1.0, 1.2],
        "rs_trend": 1,
    }


def test_constituent_payloads_sort_once_and_use_response_schema_sanitization():
    rows = [
        {
            "symbol": "LOW",
            "company_name": "Low Corp",
            "current_price": 10.0,
            "rs_rating": 50.0,
            "composite_score": 99.0,
            "price_sparkline_data": [1.0, None],
        },
        {
            "symbol": "HIGH",
            "company_name": "High Corp",
            "current_price": 20.0,
            "rs_rating": 95.0,
            "composite_score": 20.0,
            "price_sparkline_data": [1.0, 1.2],
        },
    ]

    payloads = constituent_stock_payloads_from_group_rows(rows)

    assert [payload["symbol"] for payload in payloads] == ["HIGH", "LOW"]
    assert payloads[0]["price"] == 20.0
    assert payloads[0]["price_sparkline_data"] == [1.0, 1.2]
    assert payloads[1]["price_sparkline_data"] is None
