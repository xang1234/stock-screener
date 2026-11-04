"""Tools exercised end to end through the SDK's in-memory client."""

import copy
from datetime import datetime, timezone

import pytest
from mcp import Client

from stockscreen_mcp import server

from conftest import condition, preset

pytestmark = [pytest.mark.anyio, pytest.mark.usefixtures("use_fake_data")]


async def call(tool, **arguments):
    async with Client(server.mcp) as client:
        result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    return result.structured_content


async def call_error(tool, **arguments):
    async with Client(server.mcp) as client:
        result = await client.call_tool(tool, arguments)
    assert result.is_error
    return result.content[0].text


async def test_every_tool_is_listed_as_read_only():
    async with Client(server.mcp) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == {
        "list_markets", "list_screens", "run_screen", "get_stock", "audit_stocks", "market_overview",
        "group_rankings",
    }
    for tool in tools.values():
        assert tool.annotations.read_only_hint is True
        assert tool.description
        undocumented = [p for p, s in tool.input_schema.get("properties", {}).items()
                        if not str(s.get("description", "")).strip()]
        assert not undocumented, f"{tool.name}: {undocumented}"


async def test_list_markets_reports_dates_status_and_features():
    result = await call("list_markets")

    us = next(m for m in result["markets"] if m["code"] == "US")
    assert us == {
        "code": "US", "name": "United States", "as_of_date": "2026-10-09", "status": "current",
        "features": ["breadth", "groups", "scan"],
    }
    assert result["unavailable_markets"] == ["IN"]


async def test_list_screens_hides_opportunity_presets_where_the_site_does():
    us = {s["id"] for s in (await call("list_screens", market="US"))["screens"]}
    au = {s["id"] for s in (await call("list_screens", market="AU"))["screens"]}

    assert us == {"minervini", "correction_survivors", "se_cup_handle"}
    assert au == {"minervini", "se_cup_handle"}


async def test_preset_screen_matches_and_sorts_like_the_site():
    result = await call("run_screen", market="US", screen="minervini")

    assert result["screen"] == "Minervini Trend Template"
    assert result["total_matches"] == 2
    assert [row["symbol"] for row in result["rows"]] == ["AAA", "BBB"]
    assert result["sorted_by"] == "minervini_score desc"
    first = result["rows"][0]
    assert first["rs_rating"] == 95 and first["minervini_score"] == 90
    assert "price_sparkline_data" not in first


async def test_custom_filters_combine_with_a_preset_and_show_their_fields():
    result = await call(
        "run_screen", market="US", screen="minervini",
        filters=[{"field": "ibd_group_rank", "max": 10}],
    )

    assert [row["symbol"] for row in result["rows"]] == ["AAA"]
    assert result["rows"][0]["ibd_group_rank"] == 3


async def test_filters_alone_use_the_market_default_sort_and_respect_limit():
    result = await call("run_screen", market="US", filters=[{"field": "stage", "equals": 2}], limit=2)

    assert result["total_matches"] == 4
    assert [row["symbol"] for row in result["rows"]] == ["AAA", "DDD"]
    assert result["sorted_by"] == "composite_score desc"


async def test_unknown_screen_and_unknown_field_are_explained():
    assert "minervini" in await call_error("run_screen", market="US", screen="minerva")
    message = await call_error("run_screen", market="US", filters=[{"field": "rs_ratin", "min": 90}])
    assert "rs_ratin" in message and "rs_rating" in message
    assert "Pass a screen" in await call_error("run_screen", market="US")


async def test_get_stock_returns_the_row_and_the_screens_it_passes():
    result = await call("get_stock", symbol="aaa", market="US")

    assert result["symbol"] == "AAA"
    assert result["rs_rating"] == 95
    assert result["passes_screens"] == ["minervini", "correction_survivors", "se_cup_handle"]
    assert "field_availability" not in result
    assert "not found" in await call_error("get_stock", symbol="ZZZ", market="US")


async def test_market_overview_summarises_health_breadth_and_groups():
    result = await call("market_overview", market="US")

    assert result["exposure"]["exposure_score"] == 45.0
    assert result["exposure"]["stance"] == "Downtrend/Caution"
    assert "history" not in result["exposure"]
    assert result["breadth"]["latest"]["stocks_up_4pct"] == 29
    assert "eligibility_signature" not in result["breadth"]["latest"]
    assert [day["date"] for day in result["breadth"]["recent"]] == [
        "2026-10-09", "2026-10-08", "2026-10-07", "2026-10-06", "2026-10-05",
    ]
    assert result["key_markets"][0] == {
        "symbol": "SPY", "name": "S&P 500 ETF", "close": 778.57, "change_1d_pct": 0.6, "date": "2026-10-09",
    }
    assert result["top_groups"][0]["group"] == "Semiconductors"


async def test_market_overview_says_when_breadth_is_not_published():
    result = await call("market_overview", market="AU")

    assert result["breadth"] == {"available": False, "reason": "Breadth is not published for AU."}


async def test_group_rankings_list_and_one_group_in_detail():
    ranked = await call("group_rankings", market="US")
    assert [g["group"] for g in ranked["groups"]] == ["Semiconductors", "Software"]
    assert ranked["movers_1w"]["losers"][0] == {"group": "Software", "rank": 2, "rank_change_1w": -30}

    detail = await call("group_rankings", market="US", group="semiconductors", limit=1)
    assert detail["group"]["group"] == "Semiconductors"
    assert [s["symbol"] for s in detail["group"]["stocks"]] == ["CCC"]  # highest RS first
    assert "price_sparkline_data" not in detail["group"]["stocks"][0]
    assert detail["group"]["rank_history"][0] == {"date": "2026-10-09", "rank": 9, "avg_rs_rating": 89.0}
    assert len(detail["group"]["rank_history"]) == 8
    assert "Semiconductors" in await call_error("group_rankings", market="US", group="Semis Typo")


@pytest.mark.parametrize(
    ("publication", "status"),
    [
        ({"state": "current", "session_date": "2026-10-09", "late_after": "2099-01-01T00:00:00Z"}, "current"),
        ({"state": "current", "session_date": "2026-10-09", "late_after": "2000-01-01T00:00:00Z"}, "late"),
        ({"state": "stale", "session_date": "2026-10-08", "late_after": None}, "late"),
        ({"state": "pending", "session_date": "2026-10-09", "late_after": "2099-01-01T00:00:00Z"}, "current"),
        ({"state": "unknown", "session_date": "2026-10-09", "late_after": "2099-01-01T00:00:00Z"}, "unknown"),
        ({"state": "current", "session_date": None, "late_after": "2099-01-01T00:00:00Z"}, "unknown"),
        ({"state": "current", "session_date": "2026-10-09", "late_after": "not a date"}, "unknown"),
        ({"state": "current", "session_date": "2026-10-09", "late_after": "2099-01-01T00:00:00"}, "current"),
    ],
)
def test_publication_status_follows_the_site(publication, status):
    now = datetime(2026, 10, 12, tzinfo=timezone.utc)
    assert server._status({"publication": publication}, now) == status


def test_picked_rows_never_contain_non_json_numbers():
    row = {"symbol": "X", "scan_mode": "listing_only", "volume": 5.0, "x": float("nan")}
    assert server._pick(row, ["symbol", "listing_aware_volume", "x", "volume"]) == {"symbol": "X", "volume": 5.0}


async def test_presets_without_an_expression_are_skipped(use_fake_data):
    scan, _ = use_fake_data.scan("US")
    scan["preset_screens"].append({"id": "legacy", "name": "Legacy", "filters": {"minRs": 80}})

    screens = await call("list_screens", market="US")
    assert "legacy" not in {s["id"] for s in screens["screens"]}
    assert (await call("get_stock", symbol="AAA", market="US"))["symbol"] == "AAA"


async def test_preset_order_breaks_ties_by_symbol_like_the_site(use_fake_data):
    _, rows = use_fake_data.scan("US")
    rows.reverse()  # insertion order now puts BBB before AAA
    for row in rows:
        if row["symbol"] in {"AAA", "BBB"}:
            row["minervini_score"] = 80  # tie

    result = await call("run_screen", market="US", screen="minervini")
    assert [row["symbol"] for row in result["rows"]] == ["AAA", "BBB"]


async def test_group_constituents_without_an_rs_rating_come_last(use_fake_data):
    # 80 of 191 live US groups include unrated stocks; they must not crowd out the leaders.
    stocks = use_fake_data.page("US", "groups")["payload"]["group_details"]["Semiconductors"]["stocks"]
    stocks.insert(0, {"symbol": "NEW", "company_name": "New Listing", "rs_rating": None})

    detail = await call("group_rankings", market="US", group="Semiconductors")
    assert [s["symbol"] for s in detail["group"]["stocks"]] == ["CCC", "AAA", "NEW"]


async def test_group_constituents_with_equal_rs_keep_the_higher_composite_first(use_fake_data):
    # group_detail_payloads.py publishes constituents by (rs_rating, composite_score) desc;
    # live groups have thousands of equal-RS neighbours, so ties decide who makes the limit.
    stocks = use_fake_data.page("US", "groups")["payload"]["group_details"]["Semiconductors"]["stocks"]
    stocks[:] = [
        {"symbol": "AAB", "rs_rating": 95.0, "composite_score": 40.0},
        {"symbol": "ZZZ", "rs_rating": 95.0, "composite_score": 90.0},
        {"symbol": "MMM", "rs_rating": 95.0, "composite_score": None},
    ]

    detail = await call("group_rankings", market="US", group="Semiconductors", limit=2)
    assert [s["symbol"] for s in detail["group"]["stocks"]] == ["ZZZ", "AAB"]


async def test_run_screen_honours_preset_groups_like_get_stock(use_fake_data):
    scan, _ = use_fake_data.scan("US")
    grouped = copy.deepcopy(scan["preset_screens"][0])  # minervini: AAA and BBB pass
    grouped["id"] = "minervini_semis"
    grouped["filter_expression"]["groups"] = [
        {"id": "semis", "name": "Semis", "enabled": True, "match": "all",
         "conditions": [{"field": "ibd_industry_group", "kind": "categorical", "values": ["Semiconductors"]}]},
    ]
    scan["preset_screens"].append(grouped)

    result = await call("run_screen", market="US", screen="minervini_semis")
    assert [row["symbol"] for row in result["rows"]] == ["AAA"]
    assert "minervini_semis" not in (await call("get_stock", symbol="BBB", market="US"))["passes_screens"]


async def test_group_lookup_accepts_a_unique_partial_name():
    # IBD names are abbreviated ("Elec-Semiconductor Equip"); a unique substring should do.
    detail = await call("group_rankings", market="US", group="semiconduct")
    assert detail["group"]["group"] == "Semiconductors"


CCC_FAILS_STAGE = {"field": "stage", "value": 3, "needs": {"min": 2, "max": 2}, "short_by": 1.0}
AAA_SNAPSHOT = {"company_name": "AAA Corp", "current_price": 10.0, "rs_rating": 95, "stage": 2, "composite_score": 81,
                "ibd_industry_group": "Semiconductors", "ibd_group_rank": 3}
CCC_SNAPSHOT = {"company_name": "CCC Corp", "current_price": 10.0, "rs_rating": 99, "stage": 3, "composite_score": 90,
                "ibd_industry_group": "Semiconductors", "ibd_group_rank": 3}


async def test_audit_stocks_summarises_a_passing_and_a_failing_symbol():
    result = await call("audit_stocks", symbols=["AAA", "CCC"], screens=["minervini"])

    assert result == {
        "market": "US",
        "as_of_date": "2026-10-09",
        "screens": ["minervini"],
        "stocks": [
            {"symbol": "AAA", "snapshot": AAA_SNAPSHOT, "passes": ["minervini"], "fails": [], "near_misses": {}},
            {"symbol": "CCC", "snapshot": CCC_SNAPSHOT, "passes": [], "fails": ["minervini"],
             "near_misses": {"minervini": CCC_FAILS_STAGE}},
        ],
        "not_found": [],
    }


async def test_audit_stocks_detail_adds_every_failed_rule_on_every_failing_screen():
    result = await call("audit_stocks", symbols=["AAA", "CCC"], detail=True)

    assert result["stocks"] == [
        {"symbol": "AAA", "snapshot": AAA_SNAPSHOT, "passes": ["minervini", "correction_survivors", "se_cup_handle"],
         "fails": [], "near_misses": {}, "failed_rules": {}},
        {"symbol": "CCC", "snapshot": CCC_SNAPSHOT, "passes": [],
         "fails": ["minervini", "correction_survivors", "se_cup_handle"],
         "near_misses": {"minervini": CCC_FAILS_STAGE},
         "failed_rules": {
             "minervini": [CCC_FAILS_STAGE],
             "correction_survivors": [{"field": "correction_survivor", "value": None, "needs": {"is": True}}],
             "se_cup_handle": [{"field": "se_pattern_primary", "value": None, "needs": {"in": ["cup_handle"]}}],
         }},
    ]


async def test_audit_stocks_near_misses_fail_exactly_one_rule(use_fake_data):
    _, rows = use_fake_data.scan("US")
    next(r for r in rows if r["symbol"] == "BBB").update(rs_rating=60, stage=3)

    result = await call("audit_stocks", symbols=["AAA", "BBB", "CCC"], screens=["minervini"], detail=True)

    assert {s["symbol"]: s["near_misses"] for s in result["stocks"]} == {
        "AAA": {}, "BBB": {}, "CCC": {"minervini": CCC_FAILS_STAGE},
    }
    assert result["stocks"][1]["failed_rules"] == {"minervini": [
        {"field": "rs_rating", "value": 60, "needs": {"min": 70}, "short_by": 10.0},
        CCC_FAILS_STAGE,
    ]}


async def test_audit_stocks_near_misses_need_at_least_two_rules():
    ccc = (await call("audit_stocks", symbols=["CCC"]))["stocks"][0]

    assert ccc["fails"] == ["minervini", "correction_survivors", "se_cup_handle"]
    assert ccc["near_misses"] == {"minervini": CCC_FAILS_STAGE}


async def test_audit_stocks_short_by_stays_positive_when_the_gap_rounds_to_zero(use_fake_data):
    scan, rows = use_fake_data.scan("US")
    next(r for r in rows if r["symbol"] == "AAA")["atr"] = 1.4999999
    scan["preset_screens"].append(preset("atr", "ATR", [condition("atr", "range", min=1.5, max=None)]))

    aaa = (await call("audit_stocks", symbols=["AAA"], screens=["atr"], detail=True))["stocks"][0]

    assert aaa["failed_rules"] == {
        "atr": [{"field": "atr", "value": 1.4999999, "needs": {"min": 1.5}, "short_by": 1.0000000005838672e-07}],
    }


async def test_audit_stocks_matches_any_case_dedupes_and_reports_not_found():
    result = await call("audit_stocks", symbols=["ccc", "zZz", " Aaa ", "CCC", "zzz"], screens=["minervini"])

    assert [s["symbol"] for s in result["stocks"]] == ["CCC", "AAA"]
    assert result["not_found"] == ["ZZZ"]

    nothing = await call("audit_stocks", symbols=["zzz"], screens=["minervini"])
    assert (nothing["stocks"], nothing["not_found"]) == ([], ["ZZZ"])


async def test_audit_stocks_unknown_screen_lists_the_available_ids():
    message = await call_error("audit_stocks", symbols=["AAA"], screens=["minervini", "minerva"])
    assert "Unknown screen 'minerva'. Available: minervini, correction_survivors, se_cup_handle." in message
    # Hidden where the market has no opportunity state, as in list_screens.
    message = await call_error("audit_stocks", symbols=["AAA"], market="AU", screens=["correction_survivors"])
    assert "Unknown screen 'correction_survivors'. Available: minervini, se_cup_handle." in message


async def test_audit_stocks_defaults_to_every_visible_screen_in_list_screens_order():
    assert (await call("audit_stocks", symbols=["AAA"]))["screens"] == [
        "minervini", "correction_survivors", "se_cup_handle",
    ]
    assert (await call("audit_stocks", symbols=["AAA"], screens=None))["screens"] == [
        "minervini", "correction_survivors", "se_cup_handle",
    ]
    assert (await call("audit_stocks", symbols=["AAA"], market="AU"))["screens"] == ["minervini", "se_cup_handle"]
    assert (await call("audit_stocks", symbols=["AAA"], screens=["se_cup_handle", "MINERVINI"]))["screens"] == [
        "minervini", "se_cup_handle",
    ]


async def test_audit_stocks_rejects_an_empty_screens_list():
    assert "screens" in await call_error("audit_stocks", symbols=["AAA"], screens=[])


async def test_audit_stocks_default_screens_keep_a_mixed_case_preset_id(use_fake_data):
    scan, _ = use_fake_data.scan("US")
    scan["preset_screens"].append(preset("MixedCase", "Mixed", [condition("stage", "range", min=2, max=2)]))

    aaa = (await call("audit_stocks", symbols=["AAA"]))["stocks"][0]

    assert aaa["passes"] == ["minervini", "correction_survivors", "se_cup_handle", "MixedCase"]


async def test_audit_stocks_passes_agree_with_get_stock():
    result = await call("audit_stocks", symbols=["AAA", "BBB", "CCC", "DDD", "EEE"])

    passing = {s["symbol"]: s["passes"] for s in result["stocks"]}
    assert passing == {
        "AAA": ["minervini", "correction_survivors", "se_cup_handle"], "BBB": ["minervini"],
        "CCC": [], "DDD": [], "EEE": [],
    }
    for symbol, screens in passing.items():
        assert screens == (await call("get_stock", symbol=symbol))["passes_screens"]
    assert result["stocks"][1]["fails"] == ["correction_survivors", "se_cup_handle"]


async def test_audit_stocks_missing_data_fails_with_no_shortfall():
    result = await call("audit_stocks", symbols=["DDD"], screens=["minervini"])

    assert result["stocks"][0]["near_misses"] == {
        "minervini": {"field": "rs_rating", "value": None, "needs": {"min": 70}},
    }


async def test_audit_stocks_notes_a_failure_from_an_optional_group(use_fake_data):
    scan, _ = use_fake_data.scan("US")
    grouped = copy.deepcopy(scan["preset_screens"][0])  # minervini: AAA and BBB pass its required rules
    grouped["id"] = "minervini_semis"
    grouped["filter_expression"]["groups"] = [
        {"id": "semis", "name": "Semis", "enabled": True, "match": "all",
         "conditions": [{"field": "ibd_industry_group", "kind": "categorical", "values": ["Semiconductors"]}]},
    ]
    scan["preset_screens"].append(grouped)

    aaa, bbb = (await call("audit_stocks", symbols=["AAA", "BBB"], screens=["minervini_semis"], detail=True))["stocks"]

    assert (aaa["passes"], aaa["failed_rules"]) == (["minervini_semis"], {})
    assert "notes" not in aaa
    assert (bbb["fails"], bbb["near_misses"], bbb["failed_rules"]) == (["minervini_semis"], {}, {"minervini_semis": []})
    assert bbb["notes"] == {"minervini_semis": "fails an optional rule group"}


async def test_audit_stocks_limits_the_symbol_list():
    listed = [f"S{i}" for i in range(50)]
    assert (await call("audit_stocks", symbols=listed, screens=["minervini"]))["not_found"] == listed
    assert "symbols" in await call_error("audit_stocks", symbols=[*listed, "S50"])
    assert "symbols" in await call_error("audit_stocks", symbols=[])


FIVE_FOUND = ["AAA", "BBB", "CCC", "DDD", "EEE"]  # 5 symbols x 6 rules over the three fake screens is 30 rule cells


async def test_audit_stocks_detail_refuses_more_rule_cells_than_the_limit(monkeypatch):
    monkeypatch.setattr(server, "AUDIT_DETAIL_MAX_RULES", 29)

    message = await call_error("audit_stocks", symbols=FIVE_FOUND, detail=True)
    assert message.endswith(
        "5 symbols x 6 rules = 30 rule cells; the detail limit is 29. "
        "Up to 4 symbols fit with these screens, or pass a shorter `screens` list."
    )

    # Both recoveries it names work, and the summary is never capped.
    assert len((await call("audit_stocks", symbols=FIVE_FOUND[:4], detail=True))["stocks"]) == 4
    assert len((await call("audit_stocks", symbols=FIVE_FOUND, screens=["minervini"], detail=True))["stocks"]) == 5
    assert len((await call("audit_stocks", symbols=FIVE_FOUND))["stocks"]) == 5


async def test_audit_stocks_detail_says_none_fit_when_one_symbol_is_over_the_limit(monkeypatch):
    monkeypatch.setattr(server, "AUDIT_DETAIL_MAX_RULES", 5)

    message = await call_error("audit_stocks", symbols=["AAA"], detail=True)
    assert message.endswith(
        "1 symbols x 6 rules = 6 rule cells; the detail limit is 5. "
        "None fit with these screens, so pass a shorter `screens` list."
    )


async def test_audit_stocks_detail_accepts_exactly_the_limit(monkeypatch):
    monkeypatch.setattr(server, "AUDIT_DETAIL_MAX_RULES", 30)

    assert len((await call("audit_stocks", symbols=FIVE_FOUND, detail=True))["stocks"]) == 5


async def test_audit_stocks_detail_limit_counts_found_symbols_only(monkeypatch):
    monkeypatch.setattr(server, "AUDIT_DETAIL_MAX_RULES", 6)
    unknown = [f"S{i}" for i in range(48)]

    result = await call("audit_stocks", symbols=["AAA", "aaa", *unknown], detail=True)

    assert [s["symbol"] for s in result["stocks"]] == ["AAA"]
    assert result["not_found"] == unknown


async def test_audit_stocks_docstring_states_the_detail_limit():
    async with Client(server.mcp) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert f"may not exceed {server.AUDIT_DETAIL_MAX_RULES}" in " ".join(tools["audit_stocks"].description.split())
