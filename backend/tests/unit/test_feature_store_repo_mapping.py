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


def test_map_feature_to_scan_result_projects_the_persisted_screener_ratings():
    """The per-screener ratings are stored facts and reach ``extended_fields``."""
    row = StockFeatureDaily(
        run_id=12,
        symbol="XRPN",
        as_of_date=date(2026, 10, 2),
        composite_score=88.0,
        overall_rating=5,
        passes_count=3,
        details_json={
            "rating": "Strong Buy",
            "minervini_rating": "Pass",
            "canslim_rating": "Pass",
            "volume_breakthrough_rating": "Strong Buy",
        },
    )

    item = _map_feature_to_scan_result(row, joined={}, include_sparklines=False)

    assert item.extended_fields["minervini_rating"] == "Pass"
    assert item.extended_fields["canslim_rating"] == "Pass"
    assert item.extended_fields["volume_breakthrough_rating"] == "Strong Buy"


def _vcp_row(details_block: dict) -> StockFeatureDaily:
    """Build a feature-store row carrying one persisted ``details`` block."""
    return StockFeatureDaily(
        run_id=12,
        symbol="ALAI",
        as_of_date=date(2026, 10, 2),
        composite_score=90.0,
        overall_rating=4,
        passes_count=2,
        details_json={
            "rating": "Buy",
            "vcp_detected": True,
            "details": details_block,
        },
    )


def test_map_feature_to_scan_result_reads_the_persisted_vcp_base_count():
    """``num_bases`` comes from the stored Minervini block, verbatim."""
    row = _vcp_row(
        {
            "screeners": {
                "minervini": {
                    "details": {"full_analysis": {"vcp": {"num_bases": 5}}}
                }
            }
        }
    )

    item = _map_feature_to_scan_result(row, joined={}, include_sparklines=False)

    assert item.extended_fields["vcp_detected"] is True
    assert item.extended_fields["vcp_num_bases"] == 5


def test_map_feature_to_scan_result_reports_no_base_count_when_it_was_not_stored():
    """A detected VCP without a stored block yields ``None``, never a guess."""
    row = _vcp_row({})

    item = _map_feature_to_scan_result(row, joined={}, include_sparklines=False)

    assert item.extended_fields["vcp_detected"] is True
    assert item.extended_fields["vcp_num_bases"] is None


def test_map_feature_to_scan_result_projects_the_persisted_row_state():
    """Coverage and opportunity facts pass through unchanged."""
    row = StockFeatureDaily(
        run_id=12,
        symbol="ALAI",
        as_of_date=date(2026, 10, 2),
        composite_score=88.0,
        overall_rating=4,
        passes_count=3,
        details_json={
            "rating": "Buy",
            "data_status": "ok",
            "is_scannable": True,
            "action_state": "WATCH",
            "opportunity_state": {"policy_version": "correction-survivors-v1"},
            "volume_surge": True,
        },
    )

    item = _map_feature_to_scan_result(row, joined={}, include_sparklines=False)

    assert item.extended_fields["data_status"] == "ok"
    assert item.extended_fields["is_scannable"] is True
    assert item.extended_fields["action_state"] == "WATCH"
    assert item.extended_fields["opportunity_state"] == {
        "policy_version": "correction-survivors-v1"
    }
    assert item.extended_fields["volume_surge"] is True


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
