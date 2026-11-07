"""what_changed through the SDK's in-memory client, over archive files the real writer built."""

import copy
import gzip
import http.client
import json
import re
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from mcp import Client

from stockscreen_mcp import server
from stockscreen_mcp.data import MANIFEST_TTL_SECONDS
from stockscreen_mcp.history import TRACKED_FIELDS, History

from conftest import HISTORY, PRESETS, REPO, SINCE_ROWS, UNTIL_ROWS, archived, condition

pytestmark = [pytest.mark.anyio, pytest.mark.usefixtures("use_fake_data", "use_fake_history")]

NOTHING = {"count": 0, "symbols": [], "truncated": False}
MOVES = ("entered", "entered_with_new_data", "left", "left_with_lost_data")
RS_FORMULA = "balanced-horizon-percentile-v2"


def listed(*symbols, count=None):
    count = len(symbols) if count is None else count
    return {"count": count, "symbols": list(symbols), "truncated": count > len(symbols)}


def side(as_of_date, origin="live", rs_universe_size=5000, rs_formula_version=RS_FORMULA):
    return {
        "as_of_date": as_of_date, "published_at": f"{as_of_date}T22:00:00Z", "origin": origin,
        "rs_universe_size": rs_universe_size, "rs_formula_version": rs_formula_version,
    }


async def call(**arguments):
    async with Client(server.mcp) as client:
        result = await client.call_tool("what_changed", arguments)
    assert not result.is_error, result.content
    return result.structured_content


async def call_error(**arguments):
    async with Client(server.mcp) as client:
        result = await client.call_tool("what_changed", arguments)
    assert result.is_error
    return result.content[0].text.removeprefix("Error executing tool what_changed: ")


async def test_defaults_compare_the_two_newest_sessions_screen_by_screen():
    result = await call()

    assert result.pop("how_to_read") == server.WHAT_CHANGED_NOTES
    assert result == {
        "market": "US",
        "since": side("2026-10-08"),
        "until": side("2026-10-09"),
        "sessions_between": 0,
        "warnings": [],
        "screens": {
            "correction_survivors": {
                "name": "Correction Survivors", "definition_changed": False, "entered": NOTHING, "entered_with_new_data": NOTHING,
                "left": NOTHING, "left_with_lost_data": NOTHING, "new_to_publish": NOTHING,
                "dropped_from_publish": NOTHING,
            },
            "minervini": {
                "name": "Minervini Trend Template", "definition_changed": False, "entered": listed("CCC", "DDD"),
                "entered_with_new_data": NOTHING, "left": listed("BBB"), "left_with_lost_data": NOTHING,
                "new_to_publish": listed("NEW"), "dropped_from_publish": listed("ZZZ"),
            },
            "se_cup_handle": {
                "name": "Cup with Handle", "definition_changed": False, "entered": NOTHING, "entered_with_new_data": NOTHING,
                "left": NOTHING, "left_with_lost_data": NOTHING, "new_to_publish": NOTHING,
                "dropped_from_publish": NOTHING,
            },
        },
    }


@pytest.mark.parametrize(("arguments", "since", "until", "between"), [
    ({"since": "2026-10-06", "until": "2026-10-08"}, "2026-10-05", "2026-10-08", 1),
    ({"until": "2026-10-08"}, "2026-10-07", "2026-10-08", 0),
    ({"since": "2026-10-06"}, "2026-10-05", "2026-10-09", 2),
    ({"since": "2026-10-09"}, "2026-10-08", "2026-10-09", 0),
    ({"since": "2026-10-05", "until": "2026-12-31"}, "2026-10-05", "2026-10-09", 2),
])
async def test_dates_pick_the_newest_archived_session_on_or_before_each(arguments, since, until, between):
    result = await call(**arguments)

    assert (result["since"]["as_of_date"], result["until"]["as_of_date"], result["sessions_between"]) == (
        since, until, between,
    )


@pytest.mark.parametrize(("arguments", "message"), [
    ({"until": "2026-10-01"}, "No archived US session on or before 2026-10-01. The archive starts at 2026-10-05."),
    ({"since": "2026-10-04"}, (
        "No archived US session before 2026-10-09 and on or before 2026-10-04. The archive starts at 2026-10-05."
    )),
    ({"until": "2026-10-05"}, "No archived US session before 2026-10-05. The archive starts at 2026-10-05."),
])
async def test_a_date_before_the_archive_names_its_first_session(arguments, message):
    assert await call_error(**arguments) == message


async def test_since_after_until_is_refused():
    assert await call_error(since="2026-10-09", until="2026-10-08") == (
        "since 2026-10-09 is after until 2026-10-08. Pass the older date as since."
    )


async def test_one_archived_session_says_when_the_archive_started(fake_archive):
    fake_archive.add("HK", "2026-10-09", UNTIL_ROWS)

    assert await call_error(market="hk") == (
        "The HK archive started at 2026-10-09 and holds one session, so there is nothing to compare until the next "
        "publish is recorded."
    )


async def test_a_rule_field_gaining_or_losing_its_value_is_reported_apart_from_a_move(fake_archive):
    fake_archive.add("US", "2026-10-08", [
        *SINCE_ROWS, archived("GAIN", rs_rating=None, minervini_score=80),
        archived("LOSS", rs_rating=80, minervini_score=80), archived("OTH", rs_rating=65, minervini_score=80),
    ])
    fake_archive.add("US", "2026-10-09", [
        *UNTIL_ROWS, archived("GAIN", rs_rating=80, minervini_score=80),
        archived("LOSS", rs_rating=None, minervini_score=80),
        archived("OTH", rs_rating=75, minervini_score=80, eps_growth_qq=5.0),
    ])

    result = await call(screen="minervini", symbols=["GAIN", "LOSS", "OTH"])

    assert result["screens"]["minervini"] == {
        "name": "Minervini Trend Template", "definition_changed": False,
        "entered": listed("CCC", "DDD", "OTH"), "entered_with_new_data": listed("GAIN"),
        "left": listed("BBB"), "left_with_lost_data": listed("LOSS"),
        "new_to_publish": listed("NEW"), "dropped_from_publish": listed("ZZZ"),
    }
    assert {symbol: {move: change[move] for move in MOVES} for symbol, change in result["symbols"].items()} == {
        "GAIN": {"entered": [], "entered_with_new_data": ["minervini"], "left": [], "left_with_lost_data": []},
        "LOSS": {"entered": [], "entered_with_new_data": [], "left": [], "left_with_lost_data": ["minervini"]},
        "OTH": {"entered": ["minervini"], "entered_with_new_data": [], "left": [], "left_with_lost_data": []},
    }


async def test_limit_cuts_each_list_and_says_so():
    minervini = (await call(screen="minervini", limit=1))["screens"]["minervini"]

    assert minervini["entered"] == {"count": 2, "symbols": ["CCC"], "truncated": True}
    assert minervini["left"] == {"count": 1, "symbols": ["BBB"], "truncated": False}


# The tracked fields the fixture rows never set, so null in both sessions.
NEVER_SET = [
    "rs_rating_1m", "rs_rating_3m", "rs_rating_12m", "canslim_score", "ibd_industry_group", "ibd_group_rank",
    "eps_growth_qq", "sales_growth_qq", "growth_metric_basis", "week_52_high_distance", "market_cap_usd",
]


def absent(in_since, in_until):
    return {
        "in_since": in_since, "in_until": in_until, "entered": [], "entered_with_new_data": [], "left": [],
        "left_with_lost_data": [], "changed": {}, "missing": [], "warnings": [],
    }


async def test_symbols_report_presence_moves_and_only_the_fields_that_changed():
    result = await call(symbols=["aaa", " BBB ", "NEW", "ZZZ", "NOPE", "AAA"])

    assert result["symbols"] == {
        "AAA": {
            "in_since": True, "in_until": True, "entered": [], "entered_with_new_data": [], "left": [],
            "left_with_lost_data": [], "changed": {"current_price": {"from": 10.0, "to": 10.5}, "rs_rating": {"from": 95, "to": 96}},
            "missing": NEVER_SET, "warnings": [],
        },
        "BBB": {
            "in_since": True, "in_until": True, "entered": [], "entered_with_new_data": [], "left": ["minervini"],
            "left_with_lost_data": [], "changed": {"rs_rating": {"from": 75, "to": 60}}, "missing": NEVER_SET, "warnings": [],
        },
        "NEW": absent(False, True),
        "ZZZ": absent(True, False),
        "NOPE": absent(False, False),
    }


async def test_a_field_null_in_both_sessions_is_missing_not_unchanged(fake_archive):
    fake_archive.add("US", "2026-10-08", [archived("AAA", rs_rating=None, minervini_score=90), *SINCE_ROWS[1:]])
    fake_archive.add("US", "2026-10-09", [archived("AAA", rs_rating=None, minervini_score=91), *UNTIL_ROWS[1:]])

    aaa = (await call(symbols=["AAA"]))["symbols"]["AAA"]

    assert aaa["changed"] == {"minervini_score": {"from": 90, "to": 91}}
    assert aaa["missing"] == [
        "rs_rating", "rs_rating_1m", "rs_rating_3m", "rs_rating_12m", "composite_score", "canslim_score",
        "ibd_industry_group", "ibd_group_rank", "eps_growth_qq", "sales_growth_qq", "growth_metric_basis",
        "week_52_high_distance", "market_cap_usd",
    ]


@pytest.mark.parametrize(("since_basis", "until_basis", "warnings"), [
    ("comparable_period_yoy", "quarterly_qoq", [
        "growth_metric_basis went from comparable_period_yoy to quarterly_qoq, so eps_growth_qq and sales_growth_qq "
        "measure different comparisons in the two sessions, so their change can come from the comparison rather than "
        "the stock.",
    ]),
    ("quarterly_qoq", "quarterly_qoq", []),
])
async def test_a_growth_change_shows_each_sides_basis(fake_archive, since_basis, until_basis, warnings):
    fake_archive.add("US", "2026-10-08", [
        archived("GRO", eps_growth_qq=10.0, sales_growth_qq=5.0, growth_metric_basis=since_basis), *SINCE_ROWS,
    ])
    fake_archive.add("US", "2026-10-09", [
        archived("GRO", eps_growth_qq=20.0, sales_growth_qq=5.0, growth_metric_basis=until_basis), *UNTIL_ROWS,
    ])

    gro = (await call(symbols=["GRO"]))["symbols"]["GRO"]

    assert gro["changed"] == {
        "eps_growth_qq": {"from": 10.0, "to": 20.0, "from_basis": since_basis, "to_basis": until_basis},
        **({} if since_basis == until_basis else {"growth_metric_basis": {"from": since_basis, "to": until_basis}}),
    }
    assert gro["warnings"] == warnings


async def test_a_value_that_went_missing_is_a_change_to_null(fake_archive):
    fake_archive.add("US", "2026-10-09", [archived("AAA", rs_rating=None), *UNTIL_ROWS[1:]])

    changed = (await call(symbols=["AAA"]))["symbols"]["AAA"]["changed"]

    assert changed == {"rs_rating": {"from": 95, "to": None}, "minervini_score": {"from": 90, "to": None},
                       "composite_score": {"from": 80, "to": None}}


async def test_symbols_are_limited_to_50():
    listed_symbols = [f"S{i}" for i in range(50)]
    assert len((await call(symbols=listed_symbols))["symbols"]) == 50

    message = await call_error(symbols=[*listed_symbols, "S50"])
    assert "symbols" in message and "at most 50 items" in message


async def test_screen_narrows_the_screens_and_each_symbols_moves(fake_archive):
    since = [archived("CCC", rs_rating=99, minervini_score=95, stage=3, correction_survivor=False)]
    until = [archived("CCC", rs_rating=99, minervini_score=95, correction_survivor=True)]
    fake_archive.add("US", "2026-10-08", [*SINCE_ROWS[:2], *since, *SINCE_ROWS[3:]])
    fake_archive.add("US", "2026-10-09", [*UNTIL_ROWS[:2], *until, *UNTIL_ROWS[3:]])
    assert (await call(symbols=["CCC"]))["symbols"]["CCC"]["entered"] == ["correction_survivors", "minervini"]

    result = await call(screen=" MINERVINI ", symbols=["CCC"])

    assert list(result["screens"]) == ["minervini"]
    assert result["symbols"]["CCC"]["entered"] == ["minervini"]


async def test_an_unknown_screen_names_the_screens_in_both_sessions(fake_archive):
    fake_archive.add("US", "2026-10-09", UNTIL_ROWS, presets=[*PRESETS, {**PRESETS[0], "id": "canslim"}])

    assert await call_error(screen="vcp") == (
        "Unknown screen 'vcp'. Screens in these sessions: canslim, correction_survivors, minervini, se_cup_handle."
    )


def stricter_minervini():
    presets = copy.deepcopy(PRESETS)
    presets[0]["filter_expression"]["required"]["conditions"][1] = condition("rs_rating", "range", min=80, max=None)
    return presets


async def test_definition_changed_marks_a_screen_whose_rules_differ(fake_archive):
    fake_archive.add("US", "2026-10-09", UNTIL_ROWS, presets=stricter_minervini())

    screens = (await call())["screens"]

    assert {preset: s["definition_changed"] for preset, s in screens.items()} == {
        "correction_survivors": False, "minervini": True, "se_cup_handle": False,
    }


def rows_with(rows, **fields):
    return [{**row, **fields} for row in rows]


ORIGIN_WARNING = (
    "The {side} session, {date}, came from a retained build (origin {origin}), so its published_at only approximates "
    "publication and the site may never have shown that build."
)
STAGE_FIX_WARNING = (
    "One session was built before the 2026-10-21 stage fix and the other after it, so stage changes between them can "
    "come from the fix rather than the stocks."
)


@pytest.mark.parametrize(("change", "warnings"), [
    ({"until_rows": rows_with(UNTIL_ROWS, rs_universe_size=5501)}, [
        "The RS universe went from 5,000 to 5,501 stocks, a change of more than 10 percent, so RS ratings and the "
        "screens that use them moved partly because the universe changed.",
    ]),
    ({"until_rows": rows_with(UNTIL_ROWS, rs_universe_size=4500)}, []),
    ({"until_rows": rows_with(UNTIL_ROWS, rs_universe_size=5500)}, []),
    ({"until_rows": rows_with(UNTIL_ROWS, rs_formula_version="balanced-horizon-percentile-v3")}, [
        "The RS formula changed from balanced-horizon-percentile-v2 to balanced-horizon-percentile-v3, so RS ratings "
        "do not compare across these sessions.",
    ]),
    ({"until_rows": rows_with(UNTIL_ROWS, rs_formula_version=None)}, [
        "The RS formula is unknown in the until session, 2026-10-09, so its RS ratings may not compare with the "
        "since session's.",
    ]),
    ({"since_rows": rows_with(SINCE_ROWS, rs_formula_version=None)}, [
        "The RS formula is unknown in the since session, 2026-10-08, so its RS ratings may not compare with the "
        "until session's.",
    ]),
    ({"since_rows": rows_with(SINCE_ROWS, rs_formula_version=None),
      "until_rows": rows_with(UNTIL_ROWS, rs_formula_version=None)}, []),
    ({"since_origin": "artifact"}, [ORIGIN_WARNING.format(side="since", date="2026-10-08", origin="artifact")]),
    ({"until_origin": "artifact"}, [ORIGIN_WARNING.format(side="until", date="2026-10-09", origin="artifact")]),
    ({"since_origin": "artifact_pre_stage_fix"}, [
        ORIGIN_WARNING.format(side="since", date="2026-10-08", origin="artifact_pre_stage_fix"), STAGE_FIX_WARNING,
    ]),
    ({"since_origin": "artifact_pre_stage_fix", "until_origin": "artifact_pre_stage_fix"}, [
        ORIGIN_WARNING.format(side="since", date="2026-10-08", origin="artifact_pre_stage_fix"),
        ORIGIN_WARNING.format(side="until", date="2026-10-09", origin="artifact_pre_stage_fix"),
    ]),
    ({"until_presets": stricter_minervini()}, [
        "The rules of minervini differ between the sessions, so their entered and left lists can come from the rules "
        "rather than the stocks.",
    ]),
    ({"newest_publish": "2026-10-12"}, [
        "The site now shows 2026-10-12, which the archive has not recorded yet. Its newest session is 2026-10-09.",
    ]),
    ({"newest_publish": "2026-10-08"}, []),
])
async def test_each_warning_fires_only_when_its_condition_holds(fake_archive, fake_publish, change, warnings):
    fake_archive.add("US", "2026-10-08", change.get("since_rows", SINCE_ROWS), origin=change.get("since_origin", "live"))
    fake_archive.add(
        "US", "2026-10-09", change.get("until_rows", UNTIL_ROWS), presets=change.get("until_presets", PRESETS),
        origin=change.get("until_origin", "live"),
    )
    fake_publish.manifest["markets"]["US"]["as_of_date"] = change.get("newest_publish", "2026-10-09")

    assert (await call())["warnings"] == warnings


async def test_a_screen_on_one_side_only_counts_as_redefined(fake_archive):
    fake_archive.add("US", "2026-10-09", UNTIL_ROWS, presets=[*PRESETS, {**PRESETS[0], "id": "canslim"}])

    result = await call(screen="canslim")

    assert result["screens"]["canslim"]["definition_changed"] is True
    assert result["screens"]["canslim"]["entered"] == listed("AAA", "CCC", "DDD")
    assert result["warnings"] == [
        "The rules of canslim differ between the sessions, so their entered and left lists can come from the rules "
        "rather than the stocks.",
    ]


async def test_a_missing_index_in_an_existing_release_can_last_until_the_next_publish(fake_archive):
    del fake_archive.files[fake_archive.url("US", "index.json")]

    assert await call_error() == (
        "The US history index is missing from its Release. A publish rewriting it is back within a minute, but if a "
        "publish failed after creating the Release, the index stays missing until the next publish. Try again in a "
        "few minutes, not in a tight loop."
    )


@pytest.mark.parametrize(("failing", "message"), [
    ("https://example.test/releases/download/market-history-us-v1/index.json",
     "The US history index is unavailable (500); try again in a minute."),
    ("https://example.test/releases/tag/market-history-us-v1",
     "The US history index and its Release page are unavailable (500); try again in a minute."),
])
async def test_other_http_failures_report_their_status(fake_archive, failing, message):
    del fake_archive.files[fake_archive.url("US", "index.json")]
    fake_archive.errors[failing] = 500

    assert await call_error() == message


INDEX_URL = "https://example.test/releases/download/market-history-us-v1/index.json"
SESSION_URL = "https://example.test/releases/download/market-history-us-v1/us-2026-10-09.json.gz"


@pytest.mark.parametrize(("error", "shown"), [
    (urllib.error.URLError("connection refused"), "URLError("),
    (TimeoutError("timed out"), "TimeoutError("),
    (http.client.IncompleteRead(b""), "IncompleteRead("),
])
async def test_a_network_failure_says_it_could_not_reach_the_archive(fake_archive, error, shown):
    fake_archive.errors[INDEX_URL] = error

    assert (await call_error()).startswith(f"Could not reach the market history at {INDEX_URL}: {shown}")


async def test_a_malformed_history_url_says_it_could_not_reach_the_archive(monkeypatch):
    monkeypatch.setattr(server, "history", History(base_url="example.test/releases/download"))

    assert (await call_error()).startswith(
        "Could not reach the market history at example.test/releases/download/market-history-us-v1/index.json: "
        "ValueError("
    )


def gzip_body(text):
    return gzip.compress(text.encode(), mtime=0)


@pytest.mark.parametrize(("url", "body", "shown"), [
    (INDEX_URL, b"\x1f\x8bnot gzip", "BadGzipFile("),
    (INDEX_URL, gzip_body('{"schema_version": "market-history-index-v1"}')[:-8], "EOFError("),
    (INDEX_URL, gzip_body("{}")[:10] + b"\xff\xff\xff\xff", "error("),
    (INDEX_URL, b"not json", "JSONDecodeError("),
    (INDEX_URL, b'{"schema_version": "market-history-index-v1"}', "KeyError("),
    (INDEX_URL, b'{"schema_version": "market-history-index-v1", "sessions": 5}', "TypeError("),
    (INDEX_URL, b"[]", "AttributeError("),
    (SESSION_URL, json.dumps({
        "schema_version": "market-history-v1", "market": "US", "as_of_date": "2026-10-09", "published_at": "x",
        "origin": "live", "presets": {}, "columns": ["rs_universe_size", "symbol"], "rows": [[]],
    }).encode(), "IndexError("),
])
async def test_a_corrupt_file_says_it_could_not_be_read(fake_archive, url, body, shown):
    fake_archive.files[url] = body

    assert (await call_error()).startswith(f"Could not read the market history file {url}: {shown}")


async def test_a_market_without_a_release_has_not_started():
    assert await call_error(market="HK") == (
        "The market history archive has not started for HK, so there are no sessions to compare yet. It records "
        "every publish from its first run on; try again after HK's next publish."
    )


async def test_an_unknown_market_names_the_published_ones():
    assert await call_error(market=" xx ") == (
        "Unknown market 'XX'. Available markets: AU, HK, US. Also accepted, for their archives, though missing from "
        "this publish: IN."
    )


async def test_a_market_missing_from_this_publish_still_reads_its_archive(fake_archive):
    fake_archive.add("IN", "2026-10-08", SINCE_ROWS)
    fake_archive.add("IN", "2026-10-09", UNTIL_ROWS)

    result = await call(market="IN")

    assert (result["market"], result["warnings"]) == ("IN", [])


@pytest.mark.parametrize(("asset", "schema", "message"), [
    ("index.json", "market-history-index-v2", "The US history index uses schema 'market-history-index-v2'"),
    ("us-2026-10-09.json.gz", "market-history-v2", "The US 2026-10-09 session uses schema 'market-history-v2'"),
])
async def test_a_newer_schema_asks_for_an_upgrade(fake_archive, asset, schema, message):
    url = fake_archive.url("US", asset)
    body = fake_archive.files[url]
    document = json.loads(gzip.decompress(body) if asset.endswith(".gz") else body)
    fake_archive.files[url] = json.dumps({**document, "schema_version": schema}).encode()

    assert await call_error() == (
        f"{message}, which this version of stockscreen-mcp cannot read. Update the client's command to "
        "`uvx stockscreen-mcp@latest` and restart it, or install the latest bundle if you use one."
    )


def session_fetches(fake_archive):
    return {
        name: sum(url.endswith(f"/{name}") for url in fake_archive.calls)
        for name in sorted({url.rsplit("/", 1)[1] for url in fake_archive.calls if url.endswith(".json.gz")})
    }


async def test_sessions_are_cached_by_market_and_date_four_at_most(fake_archive):
    fake_archive.add("HK", "2026-10-08", SINCE_ROWS)
    fake_archive.add("HK", "2026-10-09", UNTIL_ROWS)
    await call()
    await call(since="2026-10-05", until="2026-10-07")
    await call()  # all four US sessions cached, 10-05 and 10-07 now the least recently used
    await call(market="HK")  # same dates as US, fetched rather than reused, evicting US 10-05 and 10-07
    await call(since="2026-10-07", until="2026-10-08")  # refetching 10-07 evicts 10-08 before it is read

    assert session_fetches(fake_archive) == {
        "hk-2026-10-08.json.gz": 1, "hk-2026-10-09.json.gz": 1, "us-2026-10-05.json.gz": 1,
        "us-2026-10-07.json.gz": 2, "us-2026-10-08.json.gz": 2, "us-2026-10-09.json.gz": 1,
    }


async def test_the_index_is_refetched_after_the_manifest_ttl(fake_archive, clock):
    def index_fetches():
        return sum(url.endswith("/index.json") for url in fake_archive.calls)

    assert (await call())["until"]["as_of_date"] == "2026-10-09"
    fake_archive.add("US", "2026-10-12", UNTIL_ROWS)
    assert (await call())["until"]["as_of_date"] == "2026-10-09"
    assert index_fetches() == 1

    clock.now += MANIFEST_TTL_SECONDS + 1
    result = await call()

    assert (result["since"]["as_of_date"], result["until"]["as_of_date"]) == ("2026-10-09", "2026-10-12")
    assert index_fetches() == 2


async def test_the_limit_measures_the_text_the_client_gets(monkeypatch):
    async with Client(server.mcp) as client:
        text = (await client.call_tool("what_changed", {})).content[0].text

    monkeypatch.setattr(server, "MAX_RESPONSE_CHARS", len(text))
    assert len((await call())["screens"]) == 3

    monkeypatch.setattr(server, "MAX_RESPONSE_CHARS", len(text) - 1)
    assert await call_error() == (
        f"3 screens would return {len(text):,} characters; the limit is {len(text) - 1:,}. "
        "Pass `screen` for one screen, a lower `limit`, or fewer `symbols`."
    )


async def test_the_note_names_every_field_it_returns(fake_archive):
    fake_archive.add("US", "2026-10-09", [archived("AAA", rs_rating=96, eps_growth_qq=1.0), *UNTIL_ROWS[1:]])
    result = await call(symbols=["AAA"])

    screen = result["screens"]["minervini"]
    aaa = result["symbols"]["AAA"]
    returned = {
        *result, *result["since"], *screen, *screen["entered"], *aaa, *aaa["changed"]["rs_rating"],
        *aaa["changed"]["eps_growth_qq"],
    }
    assert sorted(returned - {"market", "how_to_read"} - set(re.findall(r"\w+", result["how_to_read"]))) == []


def test_every_tracked_field_is_in_the_field_dictionary():
    dictionary = REPO / "contracts" / "scan_field_dictionary.json"
    if not dictionary.exists():
        pytest.skip("not inside the repository checkout")
    fields = json.loads(dictionary.read_text())["fields"]

    assert sorted(set(TRACKED_FIELDS) - fields.keys()) == []


@pytest.fixture
def github_like_server(fake_archive):
    """Release downloads as GitHub serves them: a 302 to a blob store that sends the .gz bytes as
    application/octet-stream with no Content-Encoding."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/releases/download/"):
                self.send_response(302)
                blob = self.path.replace("/releases/download/", "/blob/", 1)
                self.send_header("Location", f"http://{self.headers['Host']}{blob}")
                self.end_headers()
                return
            body = fake_archive.files.get(HISTORY + self.path.removeprefix("/blob/"))
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/releases/download/"
    httpd.shutdown()
    httpd.server_close()


async def test_gzip_sessions_served_as_plain_bytes_after_a_redirect_are_read(monkeypatch, github_like_server):
    monkeypatch.setattr(server, "history", History(base_url=github_like_server))

    minervini = (await call(screen="minervini"))["screens"]["minervini"]

    assert (minervini["entered"], minervini["left"]) == (listed("CCC", "DDD"), listed("BBB"))
