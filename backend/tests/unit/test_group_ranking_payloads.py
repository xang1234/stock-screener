from datetime import date

from app.services.group_ranking_payloads import (
    compute_group_rankings_from_serialized_rows,
)


def test_compute_group_rankings_from_serialized_rows_ranks_groups_once():
    rankings = compute_group_rankings_from_serialized_rows(
        [
            {
                "symbol": "AAA",
                "company_name": "A Corp",
                "ibd_industry_group": "Software",
                "rs_rating": 90.0,
                "composite_score": 80.0,
                "market_cap_usd": 100.0,
            },
            {
                "symbol": "BBB",
                "company_name": "B Corp",
                "ibd_industry_group": "Software",
                "rs_rating": 70.0,
                "composite_score": 99.0,
                "market_cap_usd": 300.0,
            },
            {
                "symbol": "CCC",
                "company_name": "C Corp",
                "ibd_industry_group": "Semiconductors",
                "rs_rating": 95.0,
                "composite_score": 60.0,
                "market_cap_usd": 50.0,
            },
            {
                "symbol": "NO_GROUP",
                "rs_rating": 100.0,
            },
        ],
        ranking_date=date(2026, 6, 26),
    )

    assert [row["industry_group"] for row in rankings] == [
        "Semiconductors",
        "Software",
    ]
    assert rankings[0] == {
        "industry_group": "Semiconductors",
        "date": "2026-06-26",
        "rank": 1,
        "avg_rs_rating": 95.0,
        "median_rs_rating": 95.0,
        "weighted_avg_rs_rating": 95.0,
        "rs_std_dev": 0.0,
        "num_stocks": 1,
        "num_stocks_rs_above_80": 1,
        "pct_rs_above_80": 100.0,
        "top_symbol": "CCC",
        "top_symbol_name": "C Corp",
        "top_rs_rating": 95.0,
        "rank_change_1w": None,
        "rank_change_1m": None,
        "rank_change_3m": None,
        "rank_change_6m": None,
    }
    assert rankings[1]["avg_rs_rating"] == 80.0
    assert rankings[1]["median_rs_rating"] == 80.0
    assert rankings[1]["weighted_avg_rs_rating"] == 75.0
    assert rankings[1]["rs_std_dev"] == 10.0
    assert rankings[1]["top_symbol"] == "AAA"
