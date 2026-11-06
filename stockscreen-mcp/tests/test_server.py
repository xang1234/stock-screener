"""Tools exercised end to end through the SDK's in-memory client."""

import copy
import json
import re
from datetime import datetime, timezone

import pytest
from mcp import Client

from stockscreen_mcp import server

from conftest import condition, horizon_block, preset

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
        "group_rankings", "screen_track_record", "describe_fields",
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
        "features": ["breadth", "groups", "scan", "track_record"],
    }
    assert result["unavailable_markets"] == ["IN"]
    assert next(m for m in result["markets"] if m["code"] == "AU")["features"] == ["groups", "scan"]
    assert next(m for m in result["markets"] if m["code"] == "HK")["features"] == ["breadth", "groups", "scan"]


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


async def test_run_screen_returns_exactly_this_projection():
    result = await call("run_screen", market="us", screen="MINERVINI", limit=1)

    assert result == {
        "market": "US",
        "as_of_date": "2026-10-09",
        "screen": "Minervini Trend Template",
        "total_matches": 2,
        "sorted_by": "minervini_score desc",
        "rows": [{
            "symbol": "AAA", "company_name": "AAA Corp", "current_price": 10.0, "rs_rating": 95, "stage": 2,
            "composite_score": 81, "minervini_score": 90, "ibd_industry_group": "Semiconductors",
            "ibd_group_rank": 3, "eps_growth_qq": 40.0, "ma_alignment": True,
        }],
    }


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
    assert await call_error("run_screen", market="US", screen="minerva") == (
        "Error executing tool run_screen: Unknown screen 'minerva'. Did you mean: minervini? Available: minervini, correction_survivors, se_cup_handle."
    )
    assert await call_error("run_screen", market="US", screen="zzz") == (
        "Error executing tool run_screen: Unknown screen 'zzz'. Available: minervini, correction_survivors, se_cup_handle."
    )
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
            {"symbol": "AAA", "snapshot": AAA_SNAPSHOT, "passes": ["minervini"], "near_misses": {}},
            {"symbol": "CCC", "snapshot": CCC_SNAPSHOT, "passes": [],
             "near_misses": {"minervini": "stage short_by 1.0"}},
        ],
        "not_found": [],
    }


async def test_audit_stocks_detail_adds_every_failed_rule_on_every_failing_screen():
    result = await call("audit_stocks", symbols=["AAA", "CCC"], detail=True)

    assert result["stocks"] == [
        {"symbol": "AAA", "snapshot": AAA_SNAPSHOT, "passes": ["minervini", "correction_survivors", "se_cup_handle"],
         "near_misses": {}, "failed_rules": {}},
        {"symbol": "CCC", "snapshot": CCC_SNAPSHOT, "passes": [],
         "near_misses": {"minervini": "stage short_by 1.0"},
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
        "AAA": {}, "BBB": {}, "CCC": {"minervini": "stage short_by 1.0"},
    }
    assert result["stocks"][1]["failed_rules"] == {"minervini": [
        {"field": "rs_rating", "value": 60, "needs": {"min": 70}, "short_by": 10.0},
        CCC_FAILS_STAGE,
    ]}


async def test_audit_stocks_near_misses_need_at_least_two_rules():
    ccc = (await call("audit_stocks", symbols=["CCC"]))["stocks"][0]

    assert ccc["passes"] == []
    assert ccc["near_misses"] == {"minervini": "stage short_by 1.0"}


async def test_audit_stocks_short_by_stays_positive_when_the_gap_is_tiny(use_fake_data):
    scan, rows = use_fake_data.scan("US")
    next(r for r in rows if r["symbol"] == "AAA")["atr"] = 1.4999999
    scan["preset_screens"].append(preset("atr", "ATR", [condition("atr", "range", min=1.5, max=None)]))

    aaa = (await call("audit_stocks", symbols=["AAA"], screens=["atr"], detail=True))["stocks"][0]

    assert aaa["failed_rules"] == {
        "atr": [{"field": "atr", "value": 1.4999999, "needs": {"min": 1.5}, "short_by": 1.00000000058e-07}],
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


async def test_audit_stocks_passes_agree_with_get_stock():
    result = await call("audit_stocks", symbols=["AAA", "BBB", "CCC", "DDD", "EEE"])

    passing = {s["symbol"]: s["passes"] for s in result["stocks"]}
    assert passing == {
        "AAA": ["minervini", "correction_survivors", "se_cup_handle"], "BBB": ["minervini"],
        "CCC": [], "DDD": [], "EEE": [],
    }
    for symbol, screens in passing.items():
        assert screens == (await call("get_stock", symbol=symbol))["passes_screens"]


async def test_audit_stocks_missing_data_fails_with_no_shortfall():
    result = await call("audit_stocks", symbols=["DDD"], screens=["minervini"])

    assert result["stocks"][0]["near_misses"] == {"minervini": "rs_rating"}


async def test_audit_stocks_non_numeric_near_miss_is_the_bare_field(use_fake_data):
    scan, rows = use_fake_data.scan("US")
    next(r for r in rows if r["symbol"] == "AAA")["ma_alignment"] = False
    scan["preset_screens"].append(preset("aligned", "Aligned", [
        condition("stage", "range", min=2, max=2), condition("ma_alignment", "boolean", value=True),
    ]))

    aaa = (await call("audit_stocks", symbols=["AAA"], screens=["aligned"]))["stocks"][0]

    assert aaa["near_misses"] == {"aligned": "ma_alignment"}


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
    assert (bbb["passes"], bbb["near_misses"], bbb["failed_rules"]) == ([], {}, {"minervini_semis": []})
    assert bbb["notes"] == {"minervini_semis": "fails an optional rule group"}


async def test_screen_ids_from_list_screens_work_in_any_case(use_fake_data):
    scan, _ = use_fake_data.scan("US")
    mixed = copy.deepcopy(scan["preset_screens"][0])
    mixed["id"] = "MixedCase"
    scan["preset_screens"].append(mixed)

    assert "MixedCase" in {s["id"] for s in (await call("list_screens"))["screens"]}
    audited = await call("audit_stocks", symbols=["AAA"], screens=["MixedCase"])
    assert (audited["screens"], audited["stocks"][0]["passes"]) == (["MixedCase"], ["MixedCase"])
    assert (await call("run_screen", screen="MixedCase"))["total_matches"] == 2


async def test_audit_stocks_limits_the_symbol_list():
    listed = [f"S{i}" for i in range(50)]
    assert (await call("audit_stocks", symbols=listed, screens=["minervini"]))["not_found"] == listed
    assert "symbols" in await call_error("audit_stocks", symbols=[*listed, "S50"])
    assert "symbols" in await call_error("audit_stocks", symbols=[])


FIVE_FOUND = ["AAA", "BBB", "CCC", "DDD", "EEE"]


async def test_audit_stocks_limit_measures_the_text_the_client_gets(use_fake_data, monkeypatch):
    _, rows = use_fake_data.scan("US")
    next(r for r in rows if r["symbol"] == "AAA")["company_name"] = "日本電信電話"
    symbols = [*FIVE_FOUND, "ZZZ"]
    async with Client(server.mcp) as client:
        text = (await client.call_tool("audit_stocks", {"symbols": symbols, "detail": True})).content[0].text

    monkeypatch.setattr(server, "MAX_RESPONSE_CHARS", len(text))
    await call("audit_stocks", symbols=symbols, detail=True)

    monkeypatch.setattr(server, "MAX_RESPONSE_CHARS", len(text) - 1)
    message = await call_error("audit_stocks", symbols=symbols, detail=True)
    assert f"5 symbols over 3 screens would return {len(text):,} characters" in message
    assert "About 4 symbols fit" in message
    assert message.endswith("Pass fewer symbols, a shorter `screens` list, or drop `detail`.")


async def test_audit_stocks_summary_size_error_does_not_suggest_dropping_detail(monkeypatch):
    symbols = [*FIVE_FOUND, "ZZZ"]
    async with Client(server.mcp) as client:
        summary_text = (await client.call_tool("audit_stocks", {"symbols": symbols})).content[0].text

    monkeypatch.setattr(server, "MAX_RESPONSE_CHARS", len(summary_text) - 1)
    message = await call_error("audit_stocks", symbols=symbols)
    assert message.endswith("Pass fewer symbols or a shorter `screens` list.")


async def test_track_record_returns_one_horizon_for_every_cohort_in_file_order():
    result = await call("screen_track_record")

    how_to_read = result.pop("how_to_read")
    assert "not a forecast or investment advice" in how_to_read
    assert result == {
        "market": "US",
        "as_of_date": "2026-10-09",
        "benchmark": "SPY",
        "recording_since": "2026-06-01",
        "scored_since": "2026-06-02",
        "ledger_through": "2026-10-09",
        "missing_sessions": 1,
        "horizon_sessions": 20,
        "cohorts": [
            {
                "screen": "minervini", "name": "Minervini", "retired": False, "definition": "def-old",
                "origin": "artifact", "first_date": "2026-06-01", "last_date": "2026-09-30",
                "sample_size": 80, "signal_dates": 22, "effective_dates": 18.2, "pending_windows": 40,
                "oldest_open_sessions": 19, "thin": False, "positive_rate": 0.5, "median_return_pct": -1.5,
                "excess_sample_size": 0, "skipped_missing_history": 0, "suppressed_overlapping": 0,
            },
            {
                "screen": "minervini", "name": "Minervini", "retired": False, "definition": "def-new",
                "origin": "live", "first_date": "2026-10-01", "last_date": "2026-10-09",
                "sample_size": 0, "signal_dates": 0, "effective_dates": 0.0, "pending_windows": 12,
                "oldest_open_sessions": 6, "thin": True,
                "excess_sample_size": 0, "skipped_missing_history": 0, "suppressed_overlapping": 0,
            },
            {
                "screen": "se_cup_handle", "name": "se_cup_handle", "retired": True, "definition": "def-cup",
                "origin": "artifact_pre_stage_fix", "first_date": "2026-06-01", "last_date": "2026-06-30",
                "sample_size": 3, "signal_dates": 2, "effective_dates": 1.8, "pending_windows": 0, "thin": True,
                "positive_rate": 1.0, "beat_benchmark_rate": 0.0, "median_return_pct": 4.0,
                "median_excess_return_pct": -0.2, "excess_sample_size": 3,
                "skipped_missing_history": 0, "suppressed_overlapping": 0,
            },
        ],
    }


async def test_track_record_horizon_picks_that_block_and_screen_ignores_case():
    result = await call("screen_track_record", market="us", screen=" MiNeRvInI ", horizon=5)

    assert result["horizon_sessions"] == 5
    assert [(c["screen"], c["definition"], c["sample_size"], c.get("median_return_pct"))
            for c in result["cohorts"]] == [
        ("minervini", "def-old", 120, 2.1),
        ("minervini", "def-new", 0, None),
    ]


async def test_track_record_pending_horizon_has_counts_and_no_statistics():
    result = await call("screen_track_record", screen="minervini", horizon=60)

    assert result["cohorts"][0] == {
        "screen": "minervini", "name": "Minervini", "retired": False, "definition": "def-old",
        "origin": "artifact", "first_date": "2026-06-01", "last_date": "2026-09-30",
        "sample_size": 0, "signal_dates": 0, "effective_dates": 0.0, "pending_windows": 120,
        "oldest_open_sessions": 59, "thin": True, "excess_sample_size": 0, "skipped_missing_history": 0,
        "suppressed_overlapping": 0,
    }


async def test_track_record_unknown_screen_lists_recorded_ids_once():
    message = await call_error("screen_track_record", screen="canslim")

    assert message == (
        "Error executing tool screen_track_record: "
        "No track record for screen 'canslim'. Recorded: minervini, se_cup_handle."
    )


async def test_track_record_market_without_the_asset_names_the_markets_with_one():
    message = await call_error("screen_track_record", market="au")

    assert message == (
        "Error executing tool screen_track_record: "
        "The current publish has no track record for AU, so this says nothing about how its screens "
        "performed. Markets with a track record: US."
    )


async def test_track_record_says_when_no_market_has_one(fake_publish):
    del fake_publish.manifest["markets"]["US"]["assets"]

    message = await call_error("screen_track_record")

    assert message == (
        "Error executing tool screen_track_record: "
        "The current publish has no track record for US, so this says nothing about how its screens "
        "performed. No market has one in this publish."
    )


async def test_track_record_refuses_a_schema_it_does_not_know(fake_publish):
    fake_publish.files["g/1/markets/us/scoreboard.json"]["schema_version"] = "static-scoreboard-v2"

    message = await call_error("screen_track_record")

    assert message == (
        "Error executing tool screen_track_record: "
        "This track record uses schema 'static-scoreboard-v2', which this version of stockscreen-mcp "
        "cannot read. Update the client's command to `uvx stockscreen-mcp@latest` and restart it, or install the latest bundle if you use one."
    )


async def test_track_record_rejects_a_horizon_that_is_not_published():
    message = await call_error("screen_track_record", horizon=7)

    assert message == "Error executing tool screen_track_record: horizon must be 5, 20 or 60, not 7."
    assert (await call("screen_track_record", horizon=60))["horizon_sessions"] == 60


async def test_track_record_input_schema_has_no_enum():
    async with Client(server.mcp) as client:
        tool = next(t for t in (await client.list_tools()).tools if t.name == "screen_track_record")

    assert tool.input_schema["properties"]["horizon"]["type"] == "integer"
    assert "enum" not in json.dumps(tool.input_schema)


async def test_track_record_reports_a_republish_that_dropped_the_asset_on_the_first_call(
    use_fake_data, fake_publish
):
    use_fake_data.manifest()  # caches g/1/, which advertises a scoreboard
    fake_publish.republish("g/2/")
    del fake_publish.manifest["markets"]["US"]["assets"]
    del fake_publish.files["g/2/markets/us/scoreboard.json"]

    message = await call_error("screen_track_record")

    assert message == (
        "Error executing tool screen_track_record: "
        "The current publish has no track record for US, so this says nothing about how its screens "
        "performed. No market has one in this publish."
    )


async def test_track_record_limit_measures_the_text_the_client_gets(monkeypatch):
    async with Client(server.mcp) as client:
        text = (await client.call_tool("screen_track_record", {})).content[0].text

    monkeypatch.setattr(server, "MAX_RESPONSE_CHARS", len(text))
    assert len((await call("screen_track_record"))["cohorts"]) == 3

    monkeypatch.setattr(server, "MAX_RESPONSE_CHARS", len(text) - 1)
    message = await call_error("screen_track_record")
    assert message == (
        "Error executing tool screen_track_record: "
        f"3 rows would return {len(text):,} characters; the limit is {len(text) - 1:,}. "
        "Pass `screen` for one screen's rows. Recorded: minervini, se_cup_handle."
    )


async def test_track_record_row_omits_block_fields_it_does_not_define(fake_publish):
    block = fake_publish.files["g/1/markets/us/scoreboard.json"]["cohorts"][0]["horizons"][1]
    block["avg_return_pct"] = 9.9

    row = (await call("screen_track_record"))["cohorts"][0]

    assert "avg_return_pct" not in row
    assert "horizon_sessions" not in row
    assert row["sample_size"] == 80


async def test_track_record_note_mentions_every_field_it_returns(fake_publish):
    filled = horizon_block(20, **dict.fromkeys(server.TRACK_RECORD_STATS, 1))
    fake_publish.files["g/1/markets/us/scoreboard.json"]["cohorts"][0]["horizons"][1] = filled

    result = await call("screen_track_record")

    row = result["cohorts"][0]
    assert set(server.TRACK_RECORD_STATS) <= row.keys()
    words = set(re.findall(r"\w+", result["how_to_read"]))
    assert (result.keys() - {"market", "how_to_read"}) | row.keys() <= words
