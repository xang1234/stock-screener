from __future__ import annotations

from datetime import date

from app.infra.db.models.feature_store import StockFeatureDaily
from app.infra.db.repositories.feature_store_repo import _map_feature_to_scan_result


def test_map_feature_to_scan_result_coerces_scalar_market_themes():
    row = StockFeatureDaily(
        run_id=1,
        symbol="0700.HK",
        as_of_date=date(2026, 4, 2),
        composite_score=95.0,
        overall_rating=5,
        passes_count=4,
        details_json={
            "rating": "Buy",
            "current_price": 410.0,
            "screeners_run": ["minervini"],
            "market_themes": "AI Infrastructure",
        },
    )

    item = _map_feature_to_scan_result(
        row,
        joined={
            "company_name": "Tencent",
            "market": "HK",
            "exchange": "XHKG",
            "currency": "HKD",
        },
        include_sparklines=False,
    )

    assert item.extended_fields["market_themes"] == ["AI Infrastructure"]


def test_map_feature_to_scan_result_preserves_insufficient_data_rating_from_details():
    row = StockFeatureDaily(
        run_id=1,
        symbol="0100.HK",
        as_of_date=date(2026, 4, 2),
        composite_score=None,
        overall_rating=None,
        passes_count=0,
        details_json={
            "rating": "Insufficient Data",
            "data_status": "insufficient_history",
            "scan_mode": "listing_only",
            "history_bars": 20,
        },
    )

    item = _map_feature_to_scan_result(
        row,
        joined={
            "company_name": "MINIMAX-W",
            "market": "HK",
            "exchange": "XHKG",
            "currency": "HKD",
        },
        include_sparklines=False,
    )

    assert item.rating == "Insufficient Data"
