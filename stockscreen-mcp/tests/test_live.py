"""Every tool against the real publish on GitHub Pages. Opt in: STOCKSCREEN_LIVE=1 pytest -m live."""

import bisect
import json
import math
import os
import re
import statistics
from importlib.resources import files

import pytest
from mcp import Client

from stockscreen_mcp import data as data_module
from stockscreen_mcp import server
from stockscreen_mcp.data import HEAVY_FIELDS, StaticData

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("STOCKSCREEN_LIVE") != "1", reason="set STOCKSCREEN_LIVE=1"),
]

FIELDS = json.loads(files("stockscreen_mcp").joinpath("fields.json").read_text("utf-8"))["fields"]
# The evaluator resolves these, so no scan row has them as keys.
EVALUATOR_NAMES = {"price", "listing_search", "listing_aware_volume"}
MOVE_BY_RANK = {"pct_day": "price_change_1d", "pct_week": "perf_week", "pct_month": "perf_month"}
IS_TYPE = {
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "text": lambda v: isinstance(v, str),
    "date": lambda v: isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) is not None,
    "list": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


@pytest.fixture(autouse=True)
def real_data(monkeypatch):
    monkeypatch.setattr(server, "data", StaticData())


@pytest.fixture(scope="module")
def market_rows():
    data = StaticData()
    published = [code for code in data.manifest()["markets"] if data.page_entry(code, "scan") is not None]
    return {code: data.scan(code)[1] for code in published}


async def call(client, tool, **arguments):
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    return result.structured_content


async def test_all_tools_against_the_live_publish():
    async with Client(server.mcp) as client:
        markets = await call(client, "list_markets")
        assert {m["code"] for m in markets["markets"]} >= {"US", "HK", "JP"}

        screens = await call(client, "list_screens", market="US")
        assert {"minervini", "canslim", "vcp"} <= {s["id"] for s in screens["screens"]}

        minervini = await call(client, "run_screen", market="US", screen="minervini", limit=5)
        assert minervini["total_matches"] >= len(minervini["rows"])
        for row in minervini["rows"]:
            assert row["rs_rating"] >= 70 and row["stage"] == 2 and row["minervini_score"] >= 70

        if minervini["rows"]:
            symbol = minervini["rows"][0]["symbol"]
            stock = await call(client, "get_stock", symbol=symbol, market="US")
            assert "minervini" in stock["passes_screens"]
            audit = await call(client, "audit_stocks", symbols=[symbol], market="US", detail=True)
            verdict = audit["stocks"][0]
            assert verdict["passes"] == stock["passes_screens"]
            assert set(verdict["failed_rules"]) == set(audit["screens"]) - set(verdict["passes"])

        overview = await call(client, "market_overview", market="US")
        assert overview["exposure"]["exposure_score"] is not None
        assert overview["breadth"]["latest"]["date"] == overview["as_of_date"]

        groups = await call(client, "group_rankings", market="US", limit=3)
        assert len(groups["groups"]) == 3
        top = groups["groups"][0]["group"]
        detail = await call(client, "group_rankings", market="US", group=top, limit=3)
        assert detail["group"]["rank"] == 1

        track_record = await client.call_tool("screen_track_record", {"market": "US", "horizon": 5})
        if track_record.is_error:
            assert "The current publish has no track record for US" in track_record.content[0].text
        else:
            assert track_record.structured_content["horizon_sessions"] == 5


async def test_describe_fields_against_the_live_publish():
    async with Client(server.mcp) as client:
        described = await call(client, "describe_fields", market="US")
        assert described["rows"] > 1000
        assert 0 < described["fields"]["rs_rating"]["rows_with_value"] < described["rows"]
        assert described["fields"]["volume"]["unit"] == "local_currency"
        assert "scan_mode" not in described["fields"]
        assert described["fields"]["field_availability"]["rows_with_value"] is None


async def test_the_unfiltered_response_fits_the_shared_limit_on_real_rows():
    async with Client(server.mcp) as client:
        for code in server.data.manifest()["markets"]:
            if server.data.page_entry(code, "scan") is None:
                continue
            result = await client.call_tool("describe_fields", {"market": code})
            assert len(result.content[0].text) <= server.MAX_RESPONSE_CHARS, code


def test_the_field_dictionary_covers_real_rows_and_their_ranges(market_rows):
    ranged = {name: e["range"] for name, e in FIELDS.items() if "range" in e}
    for code, rows in market_rows.items():
        assert sorted({key for row in rows for key in row} - FIELDS.keys()) == [], code
        outside = [
            (row["symbol"], name, row[name])
            for row in rows
            for name, (low, high) in ranged.items()
            if row.get(name) is not None and not low <= row[name] <= high
        ]
        assert outside == [], code


def test_every_dictionary_entry_exists_on_real_rows(market_rows):
    keys = {key for rows in market_rows.values() for row in rows for key in row}
    assert sorted(FIELDS.keys() - keys - HEAVY_FIELDS - EVALUATOR_NAMES) == []
    assert sorted(EVALUATOR_NAMES & keys) == []


def test_the_fields_the_tools_drop_exist_in_the_published_json(monkeypatch, market_rows):
    monkeypatch.setattr(data_module, "prune", lambda value: value)
    _, rows = StaticData().scan(min(market_rows, key=lambda code: len(market_rows[code])))
    assert sorted(HEAVY_FIELDS - {key for row in rows for key in row}) == []


def test_every_field_holds_values_of_its_stated_type(market_rows):
    wrong = [
        (code, row["symbol"], name, row[name])
        for code, rows in market_rows.items()
        for row in rows
        for name, entry in FIELDS.items()
        if row.get(name) is not None and not IS_TYPE[entry["type"]](row[name])
    ]
    assert wrong == []


def test_volume_is_traded_value_in_the_listing_currency(market_rows):
    if "US" in market_rows:
        us = {row["symbol"]: row for row in market_rows["US"]}
        # These trade under 200M shares a day, so 1e9 is above any share count and below their traded value.
        assert {symbol: us[symbol]["volume"] > 1e9 for symbol in ("AAPL", "MSFT", "NVDA")} == {
            "AAPL": True, "MSFT": True, "NVDA": True,
        }
    for code, rows in market_rows.items():
        # adv_usd is shares x price x FX, so volume over adv_usd sits near the FX rate for money in the listing
        # currency, near 1 for USD, and off by the price for a share count.
        priced = [
            (row["volume"] / row["adv_usd"], row["market_cap"] / row["market_cap_usd"], row["currency"])
            for row in rows
            if row.get("volume") and row.get("adv_usd") and row.get("market_cap") and row.get("market_cap_usd")
        ]
        assert 0.5 < statistics.median(ratio / fx for ratio, fx, _ in priced) < 2, code
        # ponytail: the two readings split at the square root of the FX rate, so a rate near 1 leaves little margin.
        foreign = [(ratio, fx) for ratio, fx, currency in priced if currency != "USD"]
        nearer_fx = sum(abs(math.log(ratio / fx)) < abs(math.log(ratio)) for ratio, fx in foreign)
        assert not foreign or nearer_fx > len(foreign) / 2, (code, nearer_fx, len(foreign))


def test_percentile_ranks_are_the_published_moves_ranked(market_rows):
    for code, rows in market_rows.items():
        for rank_field, move_field in MOVE_BY_RANK.items():
            moves = sorted(row[move_field] for row in rows if row.get(move_field) is not None)
            wrong = []
            for row in rows:
                move = row.get(move_field)
                # Tied moves all take the rank of the last of them, as StaticScanBundleExporter does.
                expected = None if move is None else round(bisect.bisect_right(moves, move) / len(moves) * 100, 2)
                if row.get(rank_field) != pytest.approx(expected, abs=0.01):
                    wrong.append((row["symbol"], move, row.get(rank_field), expected))
            assert wrong == [], (code, rank_field)
