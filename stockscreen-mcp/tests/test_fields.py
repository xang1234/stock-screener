import json
import re
from importlib.resources import files

import pytest
from mcp import Client

from stockscreen_mcp import server
from stockscreen_mcp.screens import expression_fields, row_value

from conftest import PRESETS, ROWS_CHUNK, ROWS_INITIAL

pytestmark = [pytest.mark.anyio, pytest.mark.usefixtures("use_fake_data")]

DICTIONARY = json.loads(files("stockscreen_mcp").joinpath("fields.json").read_text("utf-8"))
FIELDS = DICTIONARY["fields"]
ROWS = [*ROWS_INITIAL, *ROWS_CHUNK]


def test_every_fixture_key_and_preset_rule_is_defined():
    keys = {key for row in ROWS for key in row}
    rules = {field for p in PRESETS for field in expression_fields(p["filter_expression"])}
    assert sorted((keys | rules) - FIELDS.keys()) == []


def test_every_field_a_tool_returns_by_name_is_defined():
    returned = {*server.ROW_FIELDS, *server.SNAPSHOT_FIELDS, *server.GROUP_STOCK_FIELDS}
    assert sorted(returned - FIELDS.keys()) == []


def test_entries_use_known_groups_types_and_units():
    assert sorted({e["group"] for e in FIELDS.values()} - DICTIONARY["groups"].keys()) == []
    assert {e["type"] for e in FIELDS.values()} <= {"number", "integer", "boolean", "text", "date", "list", "object"}
    assert {e["unit"] for e in FIELDS.values()} <= {
        "percent", "percentile", "score", "ratio", "rank", "local_currency", "usd", "count", "weeks", "date",
        "label", "flag", "none",
    }


def test_the_units_the_live_checks_prove_stay_in_the_dictionary():
    assert {name: FIELDS[name]["unit"] for name in ("volume", "pct_day", "pct_week", "pct_month")} == {
        "volume": "local_currency", "pct_day": "percentile", "pct_week": "percentile", "pct_month": "percentile",
    }


def test_every_alias_points_at_a_real_field():
    aliases = {name: e["alias_of"] for name, e in FIELDS.items() if "alias_of" in e}
    assert aliases == {"price": "current_price"}
    assert sorted(set(aliases.values()) - FIELDS.keys()) == []


def test_rule_names_resolve_through_the_evaluator():
    row = {"symbol": "AAA", "company_name": "AAA Corp", "current_price": 10.0, "volume": 5.0, "scan_mode": "full"}
    assert row_value(row, "price") == 10.0
    assert row_value(row, "listing_search") == "AAA AAA Corp"
    assert row_value(row, "listing_aware_volume") == 5.0


def test_fixture_rows_respect_every_stated_range():
    ranged = {name: e["range"] for name, e in FIELDS.items() if "range" in e}
    outside = [
        (row["symbol"], name, row[name])
        for row in ROWS
        for name, (low, high) in ranged.items()
        if row.get(name) is not None and not low <= row[name] <= high
    ]
    assert outside == []


async def describe(**arguments):
    async with Client(server.mcp) as client:
        result = await client.call_tool("describe_fields", arguments)
    assert not result.is_error, result.content
    return result.structured_content


async def describe_error(**arguments):
    async with Client(server.mcp) as client:
        result = await client.call_tool("describe_fields", arguments)
    assert result.is_error
    return result.content[0].text


async def test_describe_fields_returns_the_entry_with_this_markets_counts():
    result = await describe(market="hk", fields=["ibd_industry_group"])

    assert result["market"] == "HK"
    assert result["as_of_date"] == "2026-10-09"
    assert result["rows"] == 5
    assert result["fields"] == {
        "ibd_industry_group": {
            "group": "identity", "type": "text", "unit": "label", "meaning": "IBD-style industry group name.",
            "rows_with_value": 3, "values": {"Semiconductors": 2, "Software": 1},
        },
    }
    assert result["groups"]["vcp"] == "Volatility contraction pattern detector output."


def fake_scan(monkeypatch, rows):
    monkeypatch.setattr(server.data, "scan", lambda market: ({"as_of_date": "2026-10-09"}, rows))


async def test_describe_fields_reports_no_rows_for_a_field_no_row_has():
    assert (await describe(fields=["eps_rating"]))["fields"]["eps_rating"]["rows_with_value"] == 0


async def test_describe_fields_measures_aliases_through_the_evaluator():
    described = (await describe(fields=["price", "listing_aware_volume", "volume"]))["fields"]
    assert {name: entry["rows_with_value"] for name, entry in described.items()} == {
        "price": 5, "listing_aware_volume": 3, "volume": 3,
    }
    assert described["price"]["alias_of"] == "current_price"


async def test_describe_fields_has_no_row_count_for_fields_the_tools_drop():
    described = (await describe(fields=["field_availability", "rs_sparkline_data"]))["fields"]
    assert described["field_availability"]["rows_with_value"] is None
    assert described["rs_sparkline_data"]["rows_with_value"] is None


async def test_describe_fields_does_not_count_empty_lists_or_objects(monkeypatch):
    fake_scan(monkeypatch, [
        {"market_themes": [], "opportunity_state": {}, "company_name": ""},
        {"market_themes": ["AI"], "opportunity_state": {"as_of_date": "2026-10-09"}, "company_name": "AAA Corp"},
        {"market_themes": None, "opportunity_state": None, "company_name": None},
    ])
    described = (await describe(fields=["market_themes", "opportunity_state", "company_name"]))["fields"]
    assert {name: entry["rows_with_value"] for name, entry in described.items()} == {
        "market_themes": 1, "opportunity_state": 1, "company_name": 1,
    }


async def test_describe_fields_counts_true_rows_only_on_booleans(monkeypatch):
    fake_scan(monkeypatch, [{"se_setup_ready": False, "correction_survivor": True, "vcp_score": 1.0},
                            {"se_setup_ready": False, "correction_survivor": False, "vcp_score": 0.0},
                            {"se_setup_ready": None, "correction_survivor": True, "vcp_score": 0.0}])
    described = (await describe(fields=["se_setup_ready", "correction_survivor", "vcp_score"]))["fields"]

    assert (described["se_setup_ready"]["rows_with_value"], described["se_setup_ready"]["true_rows"]) == (2, 0)
    assert (described["correction_survivor"]["rows_with_value"], described["correction_survivor"]["true_rows"]) == (3, 2)
    assert "true_rows" not in described["vcp_score"]


async def test_describe_fields_lists_labels_most_common_first(monkeypatch):
    labels = ["watch", "exit_risk", "exit_risk", "extended", "exit_risk", "watch", None]
    fake_scan(monkeypatch, [{"action_state": label, "stage": 2, "market_themes": ["AI", "Chips"]} for label in labels])
    described = (await describe(fields=["action_state", "stage", "market_themes", "vcp_score"]))["fields"]

    assert list(described["action_state"]["values"].items()) == [("exit_risk", 3), ("watch", 2), ("extended", 1)]
    assert "values" not in described["stage"]
    assert described["market_themes"]["values"] == {"AI": 7, "Chips": 7}
    assert "values" not in described["vcp_score"]


async def test_describe_fields_counts_a_repeated_label_once_per_row(monkeypatch):
    fake_scan(monkeypatch, [{"market_themes": ["AI", "AI", "Chips"]}, {"market_themes": ["AI"]}])
    described = (await describe(fields=["market_themes"]))["fields"]

    assert described["market_themes"]["values"] == {"AI": 2, "Chips": 1}


async def test_describe_fields_omits_values_above_fifteen_labels(monkeypatch):
    fake_scan(monkeypatch, [{"gics_sector": f"sector {i}"} for i in range(15)])
    assert len((await describe(fields=["gics_sector"]))["fields"]["gics_sector"]["values"]) == 15

    fake_scan(monkeypatch, [{"gics_sector": f"sector {i}"} for i in range(16)])
    entry = (await describe(fields=["gics_sector"]))["fields"]["gics_sector"]
    assert entry["rows_with_value"] == 16
    assert "values" not in entry


async def test_describe_fields_filters_by_group_and_by_name():
    assert sorted((await describe(group="trend"))["fields"]) == [
        "ma_alignment", "pocket_pivot", "power_trend", "stage", "stage_name",
    ]
    assert sorted((await describe(fields=["stage", "volume"]))["fields"]) == ["stage", "volume"]
    assert sorted((await describe(fields=["stage", "volume"], group="trend"))["fields"]) == ["stage"]


async def test_describe_fields_hides_internal_fields_unless_named():
    assert "scan_mode" not in (await describe())["fields"]
    assert "scan_mode" not in (await describe(group="data_quality"))["fields"]
    assert (await describe(fields=["scan_mode"]))["fields"]["scan_mode"]["internal"] is True


def test_hidden_stock_fields_are_the_bookkeeping_fields_and_rule_names():
    assert server.HIDDEN_STOCK_FIELDS == {
        "market_rs_run_id", "rs_formula_version", "rs_universe_size", "scan_mode", "screeners_run",
        "applicable_screeners", "unavailable_screeners", "matched_groups", "data_status",
        "listing_search", "listing_aware_volume",
    }


async def test_run_screen_filters_reject_the_rule_names_marked_internal():
    async with Client(server.mcp) as client:
        for name in ("listing_search", "listing_aware_volume"):
            result = await client.call_tool("run_screen", {"filters": [{"field": name, "min": 1}]})
            assert result.is_error
            assert f"Unknown field {name!r}" in result.content[0].text


async def test_get_stock_hides_the_fields_the_dictionary_marks_internal():
    async with Client(server.mcp) as client:
        stock = (await client.call_tool("get_stock", {"symbol": "AAA"})).structured_content
    assert stock["exchange"] == "XNAS"
    assert "scan_mode" not in stock
    assert "data_status" not in stock
    assert (await describe(fields=["scan_mode"]))["fields"]["scan_mode"]["rows_with_value"] == 5


async def test_describe_fields_suggests_close_names_for_an_unknown_field():
    assert await describe_error(fields=["rs_ratng"]) == (
        "Error executing tool describe_fields: Unknown field 'rs_ratng'. "
        "Did you mean: rs_rating, rs_rating_3m, rs_rating_1m, eps_rating, rs_rating_12m?"
    )
    assert await describe_error(fields=["zzzzzz"]) == (
        "Error executing tool describe_fields: Unknown field 'zzzzzz'."
    )


async def test_describe_fields_lists_the_groups_for_an_unknown_group():
    assert await describe_error(group="momentum") == (
        "Error executing tool describe_fields: Unknown group 'momentum'. Groups: identity, price_volume, performance, "
        "relative_strength, trend, fundamentals, scores, vcp, setup_engine, opportunity, data_quality."
    )


async def test_describe_fields_rejects_an_empty_field_list():
    assert "at least 1 item" in await describe_error(fields=[])


async def test_describe_fields_reports_an_unknown_market():
    assert await describe_error(market="ZZ") == (
        "Error executing tool describe_fields: Unknown market 'ZZ'. Available markets: AU, HK, US."
    )


async def test_the_unfiltered_response_fits_the_shared_limit():
    async with Client(server.mcp) as client:
        result = await client.call_tool("describe_fields", {})
    assert not result.is_error
    assert len(result.content[0].text) <= server.MAX_RESPONSE_CHARS


def _backticked_names(text):
    return set(re.findall(r"`([a-z][a-z0-9_]*)`", text))


async def _tool_names(client):
    return {tool.name for tool in (await client.list_tools()).tools}


async def test_the_skill_teaches_every_tool_and_names_only_real_fields():
    skill = files("stockscreen_mcp").joinpath("SKILL.md").read_text("utf-8")
    assert skill.startswith("---\nname: stockscreen-mcp\ndescription: ")
    async with Client(server.mcp) as client:
        tools = await _tool_names(client)
    named = _backticked_names(skill)
    assert sorted(named - tools - FIELDS.keys()) == []
    assert sorted(tools - named) == []


async def test_prompts_name_only_real_tools_and_point_at_the_skill():
    async with Client(server.mcp) as client:
        tools = await _tool_names(client)
        watchlist = (await client.get_prompt("review_watchlist", {"symbols": "NVDA, AAPL", "market": "jp"})).messages
        candidates = (await client.get_prompt("find_candidates", {})).messages
    watchlist_text = watchlist[0].content.text
    candidates_text = candidates[0].content.text

    assert "Review these JP stocks: NVDA, AAPL." in watchlist_text
    assert "candidates in the US market" in candidates_text
    assert _backticked_names(watchlist_text) == {
        "market_overview", "audit_stocks", "group_rankings", "get_stock", "describe_fields",
    }
    assert _backticked_names(candidates_text) == {
        "market_overview", "list_screens", "screen_track_record", "run_screen", "group_rankings", "get_stock",
        "describe_fields",
    }
    assert sorted((_backticked_names(watchlist_text) | _backticked_names(candidates_text)) - tools) == []
    assert "stockscreen://skill" in watchlist_text
    assert "stockscreen://skill" in candidates_text


async def test_prompts_document_their_arguments():
    async with Client(server.mcp) as client:
        prompts = {p.name: p for p in (await client.list_prompts()).prompts}
    assert sorted(prompts) == ["find_candidates", "review_watchlist"]
    arguments = {(p.name, a.name, a.required) for p in prompts.values() for a in p.arguments}
    assert arguments == {
        ("review_watchlist", "symbols", True),
        ("review_watchlist", "market", False),
        ("find_candidates", "market", False),
    }
    assert all(a.description for p in prompts.values() for a in p.arguments)


async def test_resources_serve_the_packaged_files():
    async with Client(server.mcp) as client:
        listed = {str(r.uri): r.mime_type for r in (await client.list_resources()).resources}
        fields = (await client.read_resource("stockscreen://fields")).contents[0].text
        skill = (await client.read_resource("stockscreen://skill")).contents[0].text
    assert listed == {"stockscreen://fields": "application/json", "stockscreen://skill": "text/markdown"}
    assert json.loads(fields)["fields"]["price"]["alias_of"] == "current_price"
    assert skill.startswith("---\nname: stockscreen-mcp\n")
