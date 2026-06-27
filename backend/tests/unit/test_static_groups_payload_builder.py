from __future__ import annotations

from importlib import import_module

import pytest


def test_build_static_groups_payload_emits_the_response_envelope_once():
    try:
        module = import_module("app.services.static_groups_payload_builder")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing static groups payload builder: {exc}")

    StaticGroupsSnapshot = module.StaticGroupsSnapshot
    build_static_groups_payload = module.build_static_groups_payload

    snapshot = StaticGroupsSnapshot(
        date="2026-04-04",
        rankings=[
            {
                "industry_group": "Semiconductors",
                "date": "2026-04-04",
                "rank": 1,
                "avg_rs_rating": 91.0,
                "median_rs_rating": 90.0,
                "weighted_avg_rs_rating": 92.0,
                "rs_std_dev": 2.0,
                "num_stocks": 4,
                "num_stocks_rs_above_80": 3,
                "pct_rs_above_80": 75.0,
                "top_symbol": "AAA",
                "top_symbol_name": "AAA Corp",
                "top_rs_rating": 98,
                "rank_change_1w": 2,
                "rank_change_1m": None,
                "rank_change_3m": None,
                "rank_change_6m": None,
            }
        ],
        movers={
            "period": "1m",
            "gainers": [],
            "losers": [],
        },
        group_details={"Semiconductors": {"industry_group": "Semiconductors"}},
        market="HK",
    )

    payload = build_static_groups_payload(
        snapshot,
        generated_at="2026-04-04T22:00:00Z",
        schema_version="static-site-v2",
    )

    assert payload["schema_version"] == "static-site-v2"
    assert payload["generated_at"] == "2026-04-04T22:00:00Z"
    assert payload["market"] == "HK"
    assert payload["payload"]["rankings"]["date"] == "2026-04-04"
    assert payload["payload"]["rankings"]["total_groups"] == 1
    assert payload["payload"]["movers_period"] == "1m"
    assert payload["payload"]["movers"]["period"] == "1m"
    assert payload["payload"]["group_details"] == {
        "Semiconductors": {"industry_group": "Semiconductors"}
    }


def test_static_groups_snapshot_requires_market():
    module = import_module("app.services.static_groups_payload_builder")
    StaticGroupsSnapshot = module.StaticGroupsSnapshot

    with pytest.raises(TypeError):
        StaticGroupsSnapshot(
            date="2026-04-04",
            rankings=[],
            movers={"period": "1w", "gainers": [], "losers": []},
            group_details={},
        )
