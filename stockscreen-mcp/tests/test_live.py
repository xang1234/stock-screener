"""Every tool against the real publish on GitHub Pages. Opt in: STOCKSCREEN_LIVE=1 pytest -m live."""

import os

import pytest
from mcp import Client

from stockscreen_mcp import server
from stockscreen_mcp.data import StaticData

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("STOCKSCREEN_LIVE") != "1", reason="set STOCKSCREEN_LIVE=1"),
]


@pytest.fixture(autouse=True)
def real_data(monkeypatch):
    monkeypatch.setattr(server, "data", StaticData())


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
