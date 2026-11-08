"""MCP tools over the stock-screener project's published daily data."""

from __future__ import annotations

import difflib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from importlib.resources import files
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .data import HEAVY_FIELDS, StaticData
from .history import TRACKED_FIELDS, History, diff, pick_sessions
from .screens import (
    evaluate_expression,
    explain_expression,
    expression_fields,
    filters_to_conditions,
    preset_requires_opportunity_state,
    row_value,
    sort_rows,
)

SITE_URL = "https://xang1234.github.io/stock-screener/"

INSTRUCTIONS = f"""\
Daily stock-screening data from the open-source stock-screener project
(https://github.com/xang1234/stock-screener), read from its published snapshot
({SITE_URL}). It covers 12 markets: US, CA, DE, HK, CN, JP, KR, TW, IN, SG, MY, AU.

- Data is an end-of-day snapshot published after each market's close; every
  result has an as_of_date. There is no intraday data.
- rs_rating, rs_rating_1m/3m/12m are 1-99 relative-strength percentiles within
  the stock's own market (99 = strongest). Group ranks are IBD-style industry
  groups ranked by their stocks' RS (1 = strongest).
- Start with list_screens / run_screen for candidates, get_stock for one ticker,
  audit_stocks to check a watchlist or holdings against the screens, market_overview
  for market health and breadth, group_rankings for leading groups.
- screen_track_record reports how stocks did after newly entering a screen, one holding
  period per call. Follow its how_to_read note when quoting the numbers.
- what_changed compares two archived sessions, listing the stocks that entered and left each
  screen and the field changes of named stocks. Read its warnings before attributing a move to a stock.
- Call describe_fields before quoting or filtering on an unfamiliar field. Names mislead, and what
  rows hold varies by market.
- The stockscreen://skill resource teaches the full workflow and reading rules.
- This is educational screening data, not investment advice.
"""

mcp = MCPServer("stockscreen", instructions=INSTRUCTIONS)
data = StaticData()
history = History()


def _packaged_text(name: str) -> str:
    return files("stockscreen_mcp").joinpath(name).read_text("utf-8")


READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True)
FIELD_DICTIONARY = json.loads(_packaged_text("fields.json"))
Market = Annotated[str, Field(description="Market code, e.g. US, HK, JP, IN, DE. See list_markets.")]

# Columns shown for every screen row, before the fields used to filter and sort.
ROW_FIELDS = (
    "symbol", "company_name", "current_price", "price_change_1d", "rs_rating", "rs_rating_1m",
    "rs_rating_3m", "rs_rating_12m", "stage", "composite_score", "minervini_score", "canslim_score",
    "ibd_industry_group", "ibd_group_rank", "eps_growth_qq", "sales_growth_qq", "week_52_high_distance",
    "market_cap_usd",
)
HIDDEN_STOCK_FIELDS = frozenset(name for name, e in FIELD_DICTIONARY["fields"].items() if e.get("internal"))
BREADTH_FIELDS = (
    "date", "stocks_up_4pct", "stocks_down_4pct", "ratio_5day", "ratio_10day", "advancing_count",
    "declining_count", "new_high_52week_count", "new_low_52week_count", "t2108_pct",
    "stocks_up_25pct_quarter", "stocks_down_25pct_quarter", "stocks_up_25pct_month",
    "stocks_down_25pct_month", "stocks_up_50pct_month", "stocks_down_50pct_month",
    "stocks_up_13pct_34days", "stocks_down_13pct_34days",
)
GROUP_STOCK_FIELDS = (
    "symbol", "company_name", "rs_rating", "rs_rating_1m", "rs_rating_3m", "rs_rating_12m", "stage",
    "price", "price_change_1d", "eps_growth_qq", "sales_growth_qq", "composite_score",
)
SNAPSHOT_FIELDS = (
    "company_name", "current_price", "price_change_1d", "rs_rating", "rs_rating_3m", "stage", "composite_score",
    "ibd_industry_group", "ibd_group_rank",
)
# Claude Code truncates tool results past 25,000 tokens by default, which is about 77,500 to 82,000 characters of
# numeric indent-2 JSON at 3.1 to 3.28 characters per token (cl100k estimates). The worst 50-symbol audit summary
# measured 67 KB and a normal publish's 84 track record rows about 60 KB. A re-hash of every preset reaches about 78 KB.
MAX_RESPONSE_CHARS = 80_000
MAX_LISTED_LABELS = 15
# uvx clients stay on old versions, so a client that cannot read a newer file must say so.
SCOREBOARD_SCHEMA = "static-scoreboard-v1"
TRACK_RECORD_HORIZONS = (5, 20, 60)
TRACK_RECORD_STATS = (
    "sample_size", "signal_dates", "effective_dates", "pending_windows", "oldest_open_sessions", "thin",
    "positive_rate", "beat_benchmark_rate", "median_return_pct", "median_excess_return_pct", "excess_sample_size",
    "skipped_missing_history", "suppressed_overlapping",
)
TRACK_RECORD_NOTES = (
    "A pick is a stock entering a screen it was not in on the previous recorded day, so re-entries count. "
    "A screen's first recorded day, a rule change and crossing the stage fix are baseline days, which produce no "
    "picks. "
    "Entry is the first open after publication. "
    "Exit is the close of the Nth session, where N is horizon_sessions and the entry day is session 1. "
    "Returns are in local currency, price only, with no costs. "
    "Statistics are medians and rates, with no averages. "
    "positive_rate is the share of closed windows with a return above 0, and beat_benchmark_rate the share of closed "
    "windows with benchmark coverage that beat it. Both are fractions from 0 to 1. "
    "median_return_pct and median_excess_return_pct are percent. "
    "A statistic with nothing behind it is absent. "
    "Prices run through as_of_date. "
    "scored_since is the first recorded day inside the last 365 days, and only days from then are scored, so a "
    "recently started record is short. "
    "recording_since is the first recorded day and ledger_through the last. "
    "missing_sessions counts completed sessions between recording_since and ledger_through with no record for the "
    "whole market, including sessions older than scored_since. "
    "Each row in cohorts is one screen definition and origin, so a screen can have several. Do not pool them. "
    "screen is the preset id to pass back to this tool. name is the screen's display name, or its id once retired. "
    "definition is a hash of the screen's rules, and retired marks a screen list_screens no longer has. "
    "first_date and last_date are the first and last scored days a row spans, baseline days included. "
    "Origin live was recorded at publication, artifact was read from a retained build whose time only approximates "
    "publication, and artifact_pre_stage_fix is a retained build from before a stage fix. "
    "sample_size counts closed windows and signal_dates the dates they came from. "
    "effective_dates counts a busy day as about one date. "
    "thin means sample_size below 20 or effective_dates below 5, so do not rank a thin row. "
    "benchmark is what excess is measured against. "
    "beat_benchmark_rate and median_excess_return_pct rest on excess_sample_size, which can be far smaller than "
    "sample_size, and thin does not check it, so read it before quoting them. "
    "They are absent with no coverage. "
    "pending_windows counts windows still open, and oldest_open_sessions how far along the oldest is. "
    "sample_size 0 with pending_windows above 0 is not a bad result. "
    "All counts zero means the row has no picks. "
    "suppressed_overlapping counts re-entries while that stock's earlier window is open, possibly in another row. "
    "skipped_missing_history counts picks with no usable price, including stocks that later delist, which can "
    "flatter results. "
    "This is a record of past screens, not a forecast or investment advice."
)
WHAT_CHANGED_NOTES = (
    "since and until are two archived sessions, each the site's publish for its as_of_date as first published. "
    "sessions_between counts the archived sessions strictly between them. "
    "Moves can come from data coverage, not only from the stocks, such as a value that became available or missing, "
    "a symbol the publish gained or lost, a rule change, or a change of RS universe. "
    "Read warnings before attributing a move to a stock. "
    "Each entry in screens is one preset id. name is its display name and definition_changed is true when its rules "
    "differ between the sessions, so its moves can come from the rules. "
    "Six lists hold the symbols with a row in both sessions that moved into or out of the screen, each symbol in one. "
    "entered_with_new_data and left_with_lost_data hold the moves where a field the screen's rules read went from "
    "null to a value, or from a value to null, such as a stock joining or leaving the RS ranking. Data coverage may "
    "have moved those, not the stock. entered_with_basis_change and left_with_basis_change hold the other moves where "
    "a growth field the rules read held a value in both sessions and changed while growth_metric_basis changed, such "
    "as from comparable_period_yoy to quarterly_qoq or from null to a basis. The comparison may have moved those, not "
    "the stock. entered and left hold the remaining moves. "
    "new_to_publish holds symbols matching it in until with no row in since, and dropped_from_publish symbols matching "
    "it in since with no row in until. Those changed the publish, not the screen. "
    "count is the full number, symbols holds up to the limit in symbol order, and truncated says some were cut. "
    "Each entry in symbols has in_since and in_until, which say whether the symbol had a row, the same six move "
    "lists naming the screens it moved into and out of, and changed, which holds each tracked field whose value "
    "differs as from and to. A growth field in changed also has from_basis and to_basis, its growth_metric_basis in each "
    "session. missing lists the tracked fields null in both sessions, which are unknown, not unchanged. A tracked "
    "field in neither changed nor missing held the same value in both. The symbol's own warnings flag a growth field "
    "that held a value in both sessions and changed while growth_metric_basis changed. When both bases are known, the "
    "two values measure different comparisons. When one side's basis is null or unavailable, its comparison is "
    "unknown, so the change may not compare. null and unavailable both mean unknown, so a change between them is not "
    "flagged. For a symbol missing from either "
    "session, every list and changed are empty. A null is a missing value, not a zero. "
    f"Tracked fields: {', '.join(TRACKED_FIELDS)}. "
    "Prices are not split-adjusted across sessions, so a split shows as a price move. "
    "rs_universe_size is how many stocks the market's RS ranking held, and rs_formula_version the RS formula. "
    "published_at is when the session was first published. "
    "origin live was recorded at publication, artifact was read from a retained build whose time only approximates "
    "publication, and artifact_pre_stage_fix is a retained build from before a stage fix. "
    "This is a record of past screens, not a forecast or investment advice."
)


class Filter(BaseModel):
    """One condition on a scan field. Use min/max for numbers, equals for booleans or text."""

    field: str = Field(description=(
        "Scan field, e.g. rs_rating, rs_rating_3m, stage (1-4), ibd_group_rank, eps_growth_qq, "
        "sales_growth_qq, current_price, volume (average daily traded value in the listing currency, not shares), "
        "market_cap_usd, week_52_high_distance, minervini_score, canslim_score, vcp_score, ma_alignment, "
        "passes_template, se_pattern_primary. describe_fields defines them all."
    ))
    min: float | None = Field(default=None, description="Inclusive lower bound.")
    max: float | None = Field(default=None, description="Inclusive upper bound.")
    equals: bool | str | float | None = Field(default=None, description="Exact value; overrides min/max.")


@dataclass(frozen=True)
class ScreenResult:
    """Every row a screen matched, sorted, before the MCP tool or the CLI narrows them."""

    market: str
    as_of_date: str | None
    preset: dict[str, Any] | None
    sort_by: str
    descending: bool
    columns: list[str]
    rows: list[dict[str, Any]]

    @property
    def total(self) -> int:
        return len(self.rows)


def _pick(row: dict[str, Any], fields: tuple[str, ...] | list[str]) -> dict[str, Any]:
    picked = {}
    for field in fields:
        value = row_value(row, field)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            continue  # inf/NaN (e.g. listing_aware_volume) are not valid JSON
        picked[field] = value
    return picked


def _response_chars(result: dict[str, Any]) -> int:
    return len(json.dumps(result, indent=2, ensure_ascii=False))


def _parse_deadline(value: Any) -> datetime | None:
    try:
        deadline = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return deadline if deadline.tzinfo else deadline.replace(tzinfo=timezone.utc)


def _status(entry: dict[str, Any], now: datetime) -> str:
    """StaticLayout.publicationStatus: late once past its deadline, unknown without one."""
    publication = entry.get("publication")
    if not publication or publication.get("state") == "unknown" or not publication.get("session_date"):
        return "unknown"
    if publication.get("state") == "stale":
        return "late"
    deadline = _parse_deadline(publication.get("late_after")) if publication.get("late_after") else None
    if deadline is None:
        return "unknown"
    return "late" if now > deadline else "current"


def market_status(market: str) -> str:
    """A market's publication status: "current", "late" or "unknown"."""
    return _status(data.market(market), datetime.now(timezone.utc))


def _scoreboard_asset(entry: dict[str, Any]) -> dict[str, Any] | None:
    return entry.get("assets", {}).get("scoreboard")


def _visible_presets(code: str, scan: dict[str, Any]) -> list[dict[str, Any]]:
    # Presets without a filter_expression use the site's legacy filter format; all current
    # publishes use expressions, so those are skipped rather than translated.
    presets = [p for p in scan.get("preset_screens", []) if "filter_expression" in p]
    if data.market(code).get("features", {}).get("opportunity_state"):
        return presets
    return [p for p in presets if not preset_requires_opportunity_state(p)]


@mcp.tool(annotations=READ_ONLY)
def list_markets() -> dict[str, Any]:
    """List the markets that have published data, so you know which market codes the other tools accept
    and how fresh each one is.

    Call this first when the user names a country or exchange, or asks how current the data is.
    For one market's health, breadth and leading groups, use market_overview instead.

    Each market returns its code (pass it as `market` to the other tools), name, as_of_date (the
    trading day the data covers), status ("current"; "late" if it is behind its latest completed
    session or its next update is over a day overdue; or "unknown"), and features: which of "scan"
    (run_screen, get_stock, list_screens, audit_stocks), "breadth" (market_overview breadth), "groups"
    (group_rankings) and "track_record" (screen_track_record) are published for it. unavailable_markets
    lists markets that failed this publish. Takes no arguments. Results are cached for up to 10 minutes;
    markets are published after their close by two scheduled runs, one for Asia and one for US/Europe.
    """
    manifest = data.manifest()
    now = datetime.now(timezone.utc)
    markets = [
        {
            "code": code,
            "name": entry.get("display_name", code),
            "as_of_date": entry.get("as_of_date"),
            "status": _status(entry, now),
            "features": sorted(
                [k for k, on in entry.get("features", {}).items() if on and k in {"scan", "breadth", "groups"}]
                + (["track_record"] if _scoreboard_asset(entry) else [])
            ),
        }
        for code, entry in sorted(manifest["markets"].items())
    ]
    return {
        "markets": markets,
        "unavailable_markets": manifest.get("unavailable_markets", []),
        "published_at": manifest.get("generated_at"),
        "source": SITE_URL,
    }


@mcp.tool(annotations=READ_ONLY)
def list_screens(market: Market = "US") -> dict[str, Any]:
    """Preset screens (Minervini, CANSLIM, VCP, chart patterns, momentum...) with their exact rules.

    Pass a screen's id to run_screen or audit_stocks.
    """
    scan, _ = data.scan(market)
    screens = [
        {
            "id": p["id"],
            "name": p["name"],
            "description": p.get("description"),
            "rules": p["filter_expression"]["required"]["conditions"],
            "sorted_by": f"{p.get('sort_by')} {p.get('sort_order', 'desc')}",
        }
        for p in _visible_presets(market, scan)
    ]
    return {"market": market.upper(), "as_of_date": scan.get("as_of_date"), "screens": screens}


def _row_counts(rows: list[dict[str, Any]], name: str, entry: dict[str, Any]) -> dict[str, Any]:
    if name in HEAVY_FIELDS:
        return {"rows_with_value": None}
    present = [v for v in (row_value(row, name) for row in rows) if v not in (None, "", [], {})]
    counts: dict[str, Any] = {"rows_with_value": len(present)}
    if entry["type"] == "boolean":
        counts["true_rows"] = sum(v is True for v in present)
    # JSON keys are strings, and a filter on the string "4" does not match the integer 4.
    if entry["unit"] == "label" and entry["type"] != "integer":
        labels = Counter(label for v in present for label in (set(v) if isinstance(v, list) else [v]))
        if len(labels) <= MAX_LISTED_LABELS:
            counts["values"] = {str(label): count for label, count in labels.most_common()}
    return counts


@mcp.tool(annotations=READ_ONLY)
def describe_fields(
    market: Market = "US",
    fields: Annotated[list[str] | None, Field(
        min_length=1, description="Exact field names to define, internal ones included; defaults to every public field.",
    )] = None,
    group: Annotated[str | None, Field(
        description="Only the fields in this group, e.g. fundamentals. The response lists every group.",
    )] = None,
) -> dict[str, Any]:
    """Define scan fields and count what a market's rows hold for them.

    Call it before quoting or filtering on a field you have not used, because names mislead: volume is
    traded value rather than shares, and pct_day is a percentile rank rather than a percent change.

    Each field returns its group, type, unit and meaning. note says what a reader would otherwise get
    wrong, range appears only where the code guarantees a hard bound, and alias_of names the row field
    this name reads.

    rows_with_value counts this market's rows with a non-null, non-empty value. At 0 the field supports
    no claim here. It is null for fields the tools drop, which exist
    only in the published JSON. true_rows appears on boolean fields and counts the rows that are true.
    values appears on text and list label fields as {label: rows}, most common first, unless the market
    has more than 15 distinct labels. Filters can test only the text ones. These counts say what this publish holds and not why. They cannot separate a
    field that cannot be computed from a result where nothing qualified.

    Internal fields, which are bookkeeping fields and rule names, appear only when named
    in `fields`. `fields` and `group` together keep the named fields that are in the group.
    """
    known = FIELD_DICTIONARY["fields"]
    groups = FIELD_DICTIONARY["groups"]
    if group is not None and group not in groups:
        raise ToolError(f"Unknown group {group!r}. Groups: {', '.join(groups)}.")
    for name in fields or []:
        if name not in known:
            close = difflib.get_close_matches(name, list(known), n=5)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise ToolError(f"Unknown field {name!r}.{hint}")
    scan, rows = data.scan(market)
    names = fields or [name for name in known if name not in HIDDEN_STOCK_FIELDS]
    return {
        "market": market.upper(),
        "as_of_date": scan.get("as_of_date"),
        "rows": len(rows),
        "groups": groups,
        "fields": {
            name: {**known[name], **_row_counts(rows, name, known[name])}
            for name in names
            if group is None or known[name]["group"] == group
        },
    }


def match_screen(
    market: str,
    screen: str | None = None,
    filters: list[Filter] | None = None,
    sort_by: str | None = None,
    descending: bool = True,
) -> ScreenResult:
    if screen is None and not filters:
        raise ToolError("Pass a screen (see list_screens) or filters, or both.")
    scan, rows = data.scan(market)

    preset = None
    if screen is not None:
        presets = {p["id"].lower(): p for p in _visible_presets(market, scan)}
        wanted = screen.strip().lower()
        preset = presets.get(wanted)
        if preset is None:
            close = difflib.get_close_matches(wanted, list(presets), n=5)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise ToolError(f"Unknown screen {screen!r}.{hint} Available: {', '.join(presets)}.")

    filter_dicts = [f.model_dump() for f in filters or []]
    row_fields = set().union(*(row.keys() for row in rows))
    # Filters accept the evaluator's aliases; sorting reads raw row fields, as on the site.
    filter_fields = row_fields | {"price", "perf_week", "perf_month", "market_cap_usd"}
    checks = [(f["field"], filter_fields) for f in filter_dicts] + ([(sort_by, row_fields)] if sort_by else [])
    for field, known in checks:
        if field not in known:
            close = difflib.get_close_matches(field, sorted(known), n=5)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise ToolError(f"Unknown field {field!r}.{hint}")
    # The whole preset expression (groups included), AND every custom filter.
    custom = {"required": {"match": "all", "conditions": filters_to_conditions(filter_dicts)}, "groups": []}
    expressions = [preset["filter_expression"], custom] if preset else [custom]
    matches = [row for row in rows if all(evaluate_expression(row, e) for e in expressions)]

    site_order = sort_by is None
    if site_order:
        default_sort = scan.get("sort") or {}
        sort_by = (preset or {}).get("sort_by") or default_sort.get("field", "composite_score")
        descending = ((preset or {}).get("sort_order") or default_sort.get("order", "desc")) == "desc"
    matches = sort_rows(matches, sort_by, descending=descending, site_order=site_order)

    used = [field for e in expressions for field in expression_fields(e)]
    extra = [f for f in (*used, sort_by) if f not in ROW_FIELDS]
    return ScreenResult(
        market=market.upper(),
        as_of_date=scan.get("as_of_date"),
        preset=preset,
        sort_by=sort_by,
        descending=descending,
        columns=[*ROW_FIELDS, *dict.fromkeys(extra)],
        rows=matches,
    )


@mcp.tool(annotations=READ_ONLY)
def run_screen(
    market: Market = "US",
    screen: Annotated[str | None, Field(description="Preset id from list_screens, e.g. minervini, canslim, vcp.")] = None,
    filters: Annotated[list[Filter] | None, Field(description="Extra conditions, all of which must hold.")] = None,
    sort_by: Annotated[str | None, Field(description="Field to sort by; defaults to the screen's own sort.")] = None,
    descending: Annotated[bool, Field(description="Direction for sort_by; ignored without sort_by.")] = True,
    limit: Annotated[int, Field(ge=1, le=100, description="Rows to return; total_matches counts all.")] = 25,
) -> dict[str, Any]:
    """Run a preset screen and/or custom filters over a market's full daily scan (every listed stock).

    A preset matches exactly the stocks the project's website shows for it. Filters are
    combined with AND, including with the preset.
    """
    result = match_screen(market, screen, filters, sort_by, descending)
    return {
        "market": result.market,
        "as_of_date": result.as_of_date,
        "screen": result.preset["name"] if result.preset else None,
        "total_matches": result.total,
        "sorted_by": f"{result.sort_by} {'desc' if result.descending else 'asc'}",
        "rows": [_pick(row, result.columns) for row in result.rows[:limit]],
    }


@mcp.tool(annotations=READ_ONLY)
def get_stock(
    symbol: Annotated[str, Field(description="Ticker as listed in that market, e.g. NVDA, 0700.HK, 7203.T.")],
    market: Market = "US",
) -> dict[str, Any]:
    """Everything the daily scan knows about one stock: RS ratings, stage, scores, growth, group rank,
    setup/pattern fields, and which preset screens it currently passes."""
    scan, rows = data.scan(market)
    wanted = symbol.strip().upper()
    row = next((r for r in rows if str(r.get("symbol", "")).upper() == wanted), None)
    if row is None:
        raise ToolError(f"{wanted} not found in the {market.upper()} scan. Check the symbol and market.")
    passes = [p["id"] for p in _visible_presets(market, scan) if evaluate_expression(row, p["filter_expression"])]
    stock = {k: v for k, v in row.items() if k not in HIDDEN_STOCK_FIELDS and v is not None}
    return {**stock, "as_of_date": scan.get("as_of_date"), "passes_screens": passes}


def _near_miss(rule: dict[str, Any]) -> str:
    return f"{rule['field']} short_by {rule['short_by']}" if "short_by" in rule else rule["field"]


@mcp.tool(annotations=READ_ONLY)
def audit_stocks(
    symbols: Annotated[list[str], Field(
        min_length=1, max_length=50, description="Tickers to check in this market, e.g. a watchlist or holdings.",
    )],
    market: Market = "US",
    screens: Annotated[list[str] | None, Field(
        min_length=1, description="Preset ids from list_screens to check; defaults to every screen in the market.",
    )] = None,
    detail: Annotated[bool, Field(
        description=(
            "Also return every failed rule per screen under failed_rules. "
            "Responses over 80,000 characters are refused with a hint of how many symbols fit."
        ),
    )] = False,
) -> dict[str, Any]:
    """Check a watchlist or holdings against the preset screens in one call: which screens each stock
    passes, and which it nearly passes.

    Use it to review a list of stocks, find near misses, or see what a position fails today. Use
    get_stock for every field on one symbol, and run_screen to discover new candidates. A long list
    over every screen returns a lot of text, so pass a `screens` list when you only need some.

    Each stock returns a snapshot, `passes` (screen ids in list_screens order, matching the website
    and get_stock's passes_screens), near_misses, and notes when there are any. The top-level `screens`
    list minus `passes` gives the failing ids.
    near_misses maps each screen the stock fails on exactly one of at least two rules to
    "field short_by N", or to the field alone when short_by is absent. It ignores how far short the
    stock is, so check short_by before calling one close. short_by is the positive distance between a
    failed numeric rule's bound and the stock's value (rs_rating 60 against a min of 70 is short_by 10);
    on a max rule it is how far the value is above the bound (ibd_group_rank 154 against a max of 40 is
    short_by 114); it is absent when the value is missing and on non-numeric rules. notes maps a screen
    to the reason when its rules alone don't explain the result: a failing optional rule group, or a
    disabled required group.

    detail=True adds failed_rules: every failed rule for each failing screen. Any response over 80,000
    characters of JSON is refused with an error that says how many symbols fit, so for detail pass fewer
    symbols or a shorter `screens` list.

    Symbols absent from the market's scan are listed in not_found rather than raising an error.
    """
    scan, rows = data.scan(market)
    presets = {p["id"].lower(): p for p in _visible_presets(market, scan)}
    wanted = set(presets) if screens is None else {s.strip().lower() for s in screens}
    unknown = sorted(wanted - presets.keys())
    if unknown:
        raise ToolError(f"Unknown screen {', '.join(map(repr, unknown))}. Available: {', '.join(presets)}.")
    audited = [p for key, p in presets.items() if key in wanted]

    by_symbol = {str(r.get("symbol", "")).upper(): r for r in rows}
    requested = list(dict.fromkeys(s.strip().upper() for s in symbols))
    found = [by_symbol[s] for s in requested if s in by_symbol]
    stocks = []
    for row in found:
        explained = {p["id"]: explain_expression(row, p["filter_expression"]) for p in audited}
        stock = {
            "symbol": row["symbol"],
            "snapshot": _pick(row, SNAPSHOT_FIELDS),
            "passes": [i for i, e in explained.items() if e["passes"]],
            "near_misses": {i: _near_miss(e["failed"][0]) for i, e in explained.items() if e["near_miss"]},
        }
        notes = {i: e["note"] for i, e in explained.items() if "note" in e}
        if notes:
            stock["notes"] = notes
        if detail:
            stock["failed_rules"] = {i: e["failed"] for i, e in explained.items() if not e["passes"]}
        stocks.append(stock)
    result = {
        "market": market.upper(),
        "as_of_date": scan.get("as_of_date"),
        "screens": [p["id"] for p in audited],
        "stocks": stocks,
        "not_found": [s for s in requested if s not in by_symbol],
    }
    size = _response_chars(result)
    if size > MAX_RESPONSE_CHARS:
        fit = len(found) * MAX_RESPONSE_CHARS // size
        recovery = (
            "Pass fewer symbols, a shorter `screens` list, or drop `detail`."
            if detail
            else "Pass fewer symbols or a shorter `screens` list."
        )
        raise ToolError(
            f"{len(found)} symbols over {len(audited)} screens would return {size:,} characters; "
            f"the limit is {MAX_RESPONSE_CHARS:,}. About {fit} symbols fit with these screens. {recovery}"
        )
    return result


def _track_record_row(cohort: dict[str, Any], horizon: int) -> dict[str, Any]:
    block = {b["horizon_sessions"]: b for b in cohort["horizons"]}[horizon]
    return {
        "screen": cohort["preset"],
        "name": cohort["name"],
        "retired": cohort["retired"],
        "definition": cohort["definition"],
        "origin": cohort["origin"],
        "first_date": cohort["first_date"],
        "last_date": cohort["last_date"],
        **_pick(block, TRACK_RECORD_STATS),
    }


@mcp.tool(annotations=READ_ONLY)
def screen_track_record(
    market: Market = "US",
    screen: Annotated[str | None, Field(
        description=(
            "Preset id, e.g. minervini. Retired screens that list_screens no longer has are still accepted; "
            "defaults to every recorded screen."
        ),
    )] = None,
    horizon: Annotated[int, Field(description="Sessions each pick is held: 5, 20 or 60.")] = 20,
) -> dict[str, Any]:
    """What happened after stocks newly entered each preset screen, at one holding period.

    Use it to judge how much weight a screen's output deserves, or to see whether a screen has a
    record yet. Use run_screen for today's matches. Each call reports one horizon; call again with
    horizon=5 or horizon=60 for the others.

    Rows in `cohorts` keep the published order, which is not a ranking. how_to_read in each result
    defines the fields; read it before quoting a number.

    Any response over 80,000 characters of JSON is refused with an error; pass `screen` for one
    screen's rows.

    A market with no published record returns an error naming the markets that have one. In list_markets
    those markets show the track_record feature.
    """
    if horizon not in TRACK_RECORD_HORIZONS:
        raise ToolError(f"horizon must be 5, 20 or 60, not {horizon}.")
    board = data.asset(market, "scoreboard")
    if board is None:
        having = [code for code, entry in sorted(data.manifest()["markets"].items()) if _scoreboard_asset(entry)]
        others = (
            f"Markets with a track record: {', '.join(having)}." if having else "No market has one in this publish."
        )
        raise ToolError(
            f"The current publish has no track record for {market.upper()}, so this says nothing about how its "
            f"screens performed. {others}"
        )
    if board.get("schema_version") != SCOREBOARD_SCHEMA:
        raise ToolError(
            f"This track record uses schema {board.get('schema_version')!r}, which this version of "
            "stockscreen-mcp cannot read. Update the client's command to `uvx stockscreen-mcp@latest` and restart it, "
            "or install the latest bundle if you use one."
        )
    cohorts = board["cohorts"]
    recorded = ", ".join(dict.fromkeys(c["preset"] for c in cohorts)) or "none"
    if screen is not None:
        wanted = screen.strip().lower()
        matching = [c for c in cohorts if c["preset"].lower() == wanted]
        if not matching:
            raise ToolError(f"No track record for screen {screen!r}. Recorded: {recorded}.")
        cohorts = matching
    result = {
        "market": market.upper(),
        "as_of_date": board["as_of_date"],
        "benchmark": board["benchmark"],
        "recording_since": board["recording_since"],
        "scored_since": board["scored_since"],
        "ledger_through": board["ledger_through"],
        "missing_sessions": board["missing_sessions"],
        "horizon_sessions": horizon,
        "how_to_read": TRACK_RECORD_NOTES,
        "cohorts": [_track_record_row(c, horizon) for c in cohorts],
    }
    size = _response_chars(result)
    if size > MAX_RESPONSE_CHARS:
        raise ToolError(
            f"{len(cohorts)} rows would return {size:,} characters; the limit is {MAX_RESPONSE_CHARS:,}. "
            f"Pass `screen` for one screen's rows. Recorded: {recorded}."
        )
    return result


@mcp.tool(annotations=READ_ONLY)
def what_changed(
    market: Market = "US",
    since: Annotated[date | None, Field(
        description=(
            "Older date, YYYY-MM-DD. Picks the newest archived session on or before it and before until; defaults to "
            "the session just before until."
        ),
    )] = None,
    until: Annotated[date | None, Field(
        description="Newer date, YYYY-MM-DD. Picks the newest archived session on or before it; defaults to the newest.",
    )] = None,
    symbols: Annotated[list[str] | None, Field(
        min_length=1, max_length=50,
        description="Tickers to report one by one, with the screens each entered and left and its changed fields.",
    )] = None,
    screen: Annotated[str | None, Field(description="Preset id, e.g. minervini, to report only that screen.")] = None,
    limit: Annotated[int, Field(
        ge=1, le=100, description="Symbols listed per list in each screen, such as entered or new_to_publish.",
    )] = 25,
) -> dict[str, Any]:
    """What changed in a market between two archived sessions. It lists the stocks that entered and left each
    preset screen, and for named symbols the screens they moved into and out of, the tracked fields that changed,
    and the tracked fields null in both sessions.

    Use it for "what changed since yesterday" or since a date. Use run_screen and get_stock for the newest
    publish, and screen_track_record for how entries did afterwards. With no dates it compares the two newest
    archived sessions.

    Moves where a field the screen's rules read went from null to a value, or back, are listed apart as
    entered_with_new_data and left_with_lost_data, since data coverage may have moved them. Moves where a growth
    field the rules read changed across a change of growth_metric_basis are listed apart as
    entered_with_basis_change and left_with_basis_change, since the comparison may have moved them.

    Read `warnings` before attributing a move to a stock, because a rule change, an RS universe or formula change,
    the stage fix or a session read from a retained build can move a stock that did nothing. how_to_read in each
    result defines the fields.

    The archive holds each session as first published, from when recording started. A date before the first
    archived session is an error that names it. Any response over 80,000 characters of JSON is refused; pass
    `screen`, a lower `limit` or fewer `symbols`.
    """
    code = market.strip().upper()
    try:
        manifest = data.manifest()
    except ToolError:
        # The archive is on Releases, not Pages, so it can still answer; only the market check and freshness go.
        manifest = None
    # A market that failed this publish keeps its archive.
    unavailable = [] if manifest is None else manifest.get("unavailable_markets", [])
    if manifest is not None and code not in manifest["markets"] and code not in unavailable:
        also = (
            f" Also accepted, for their archives, though missing from this publish: {', '.join(sorted(unavailable))}."
            if unavailable else ""
        )
        raise ToolError(f"Unknown market {code!r}. Available markets: {', '.join(sorted(manifest['markets']))}.{also}")
    archived = history.sessions(code)
    older, newer = pick_sessions(code, archived, since, until)
    newest_publish = None if manifest is None else manifest["markets"].get(code, {}).get("as_of_date")
    result = {
        "market": code,
        **diff(
            history.session(code, older), history.session(code, newer), archived, newest_publish,
            screen=screen, symbols=symbols, limit=limit,
        ),
        "how_to_read": WHAT_CHANGED_NOTES,
    }
    if manifest is None:
        result["warnings"].append(
            "The site's current manifest could not be read, so whether the site holds a newer session than the "
            "archive was not checked."
        )
    size = _response_chars(result)
    if size > MAX_RESPONSE_CHARS:
        raise ToolError(
            f"{len(result['screens'])} screens would return {size:,} characters; the limit is {MAX_RESPONSE_CHARS:,}. "
            "Pass `screen` for one screen, a lower `limit`, or fewer `symbols`."
        )
    return result


@mcp.tool(annotations=READ_ONLY)
def market_overview(market: Market = "US") -> dict[str, Any]:
    """Market health in one call: exposure score and stance (distribution days, follow-through day,
    trend), StockBee-style breadth (4% movers, 5/10-day ratios, T2108, 52-week highs/lows),
    key index/asset moves, and the strongest industry groups."""
    home = data.page(market, "home")
    if home is None:
        raise ToolError(f"No daily snapshot is published for {market.upper()}.")
    exposure = home.get("market_health_exposure")
    breadth_page = data.page(market, "breadth")
    if breadth_page and breadth_page.get("available"):
        payload = breadth_page["payload"]
        history = sorted(payload.get("history_90d", []), key=lambda d: d["date"], reverse=True)
        breadth = {
            "latest": _pick(payload["current"], BREADTH_FIELDS),
            "recent": [_pick(d, BREADTH_FIELDS) for d in history[:5]],
        }
    else:
        breadth = {"available": False, "reason": f"Breadth is not published for {market.upper()}."}
    return {
        "market": market.upper(),
        "as_of_date": home.get("as_of_date"),
        "exposure": {k: v for k, v in exposure.items() if k != "history"} if exposure else None,
        "breadth": breadth,
        "key_markets": [
            {"symbol": k["symbol"], "name": k.get("display_name"), "close": k.get("latest_close"),
             "change_1d_pct": k.get("change_1d"), "date": k.get("latest_date")}
            for k in home.get("key_markets", [])
        ],
        "top_groups": [_group_summary(g) for g in home.get("top_groups", [])[:5]],
        "scan_counts": {k: v for k, v in (home.get("scan_summary") or {}).items() if k != "top_results"},
    }


def _constituent_key(stock: dict[str, Any]) -> tuple[float, float]:
    """backend/app/services/group_detail_payloads.py constituent_sort_key: (rs, composite),
    missing values as -inf, so unrated stocks go last and RS ties keep the stronger composite."""
    rs, composite = stock.get("rs_rating"), stock.get("composite_score")
    return (-math.inf if rs is None else rs, -math.inf if composite is None else composite)


def _group_summary(g: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": g.get("rank"),
        "group": g.get("industry_group"),
        "avg_rs_rating": g.get("avg_rs_rating"),
        "median_rs_rating": g.get("median_rs_rating"),
        "pct_rs_above_80": g.get("pct_rs_above_80"),
        "num_stocks": g.get("num_stocks"),
        "rank_change_1w": g.get("rank_change_1w"),
        "rank_change_1m": g.get("rank_change_1m"),
        "rank_change_3m": g.get("rank_change_3m"),
        "rank_change_6m": g.get("rank_change_6m"),
        "top_stock": g.get("top_symbol"),
    }


@mcp.tool(annotations=READ_ONLY)
def group_rankings(
    market: Market = "US",
    group: Annotated[str | None, Field(description="Industry group name for its detail and constituents.")] = None,
    limit: Annotated[int, Field(
        ge=1, le=200, description="Groups to return, or with `group`, constituent stocks to return.",
    )] = 20,
) -> dict[str, Any]:
    """Rank a market's IBD-style industry groups by their stocks' relative strength (1 = strongest),
    with 1W/1M/3M/6M rank changes and this week's biggest movers.

    Use it to find leading or improving industries. Pass `group` (full name, or any part of it
    that matches only one group) for that group's rank history and its stocks, strongest RS
    first. For a quick top-5 alongside breadth, market_overview is enough; to filter stocks by
    group rank, use run_screen with ibd_group_rank. Rankings are end-of-day and update once a day.
    """
    page = data.page(market, "groups")
    if page is None or not page.get("available"):
        raise ToolError(f"Group rankings are not published for {market.upper()}.")
    payload = page["payload"]
    result: dict[str, Any] = {"market": market.upper(), "as_of_date": page.get("rs_as_of_date")}

    if group is None:
        rankings = payload["rankings"]["rankings"]
        movers = payload.get("movers") or {}
        result["total_groups"] = len(rankings)
        result["groups"] = [_group_summary(g) for g in rankings[:limit]]
        result["movers_1w"] = {
            side: [{"group": g["industry_group"], "rank": g["rank"], "rank_change_1w": g.get("rank_change_1w")}
                   for g in movers.get(side, [])[:5]]
            for side in ("gainers", "losers")
        }
        return result

    details = payload.get("group_details", {})
    wanted = group.strip().lower()
    by_name = {name.lower(): name for name in details}
    # IBD names are abbreviated ("Elec-Semiconductor Equip"), so a unique substring also works.
    partial = [name for name in details if wanted in name.lower()]
    name = by_name.get(wanted) or (partial[0] if len(partial) == 1 else None)
    if name is None:
        close = difflib.get_close_matches(group, list(details), n=5, cutoff=0.3)
        raise ToolError(f"Unknown group {group!r}. Close matches: {', '.join(close) or 'none'}.")
    detail = details[name]
    history = sorted(detail.get("history", []), key=lambda h: h["date"], reverse=True)
    stocks = sorted(detail.get("stocks", []), key=_constituent_key, reverse=True)
    result["group"] = {
        "group": name,
        "rank": detail.get("current_rank"),
        "avg_rs_rating": detail.get("current_avg_rs"),
        "median_rs_rating": detail.get("current_median_rs"),
        "pct_rs_above_80": detail.get("pct_rs_above_80"),
        "num_stocks": detail.get("num_stocks"),
        "rank_change_1w": detail.get("rank_change_1w"),
        "rank_change_1m": detail.get("rank_change_1m"),
        "rank_change_3m": detail.get("rank_change_3m"),
        "rank_change_6m": detail.get("rank_change_6m"),
        "rank_history": [{"date": h["date"], "rank": h.get("rank"), "avg_rs_rating": h.get("avg_rs_rating")}
                         for h in history[:8]],
        "stocks": [_pick(s, GROUP_STOCK_FIELDS) for s in stocks[:limit]],
    }
    return result


@mcp.resource(
    "stockscreen://fields",
    name="fields",
    description="Meaning, unit and hard range of every scan field, as JSON. describe_fields adds per-market counts.",
    mime_type="application/json",
)
def fields_resource() -> str:
    return _packaged_text("fields.json")


@mcp.resource(
    "stockscreen://skill",
    name="skill",
    description="Workflow and reading rules for answering stock questions with these tools.",
    mime_type="text/markdown",
)
def skill_resource() -> str:
    return _packaged_text("SKILL.md")


CLOSING_INSTRUCTIONS = (
    "Call `describe_fields` before quoting an unfamiliar field. State the as_of_date and any data gaps, and say this "
    "is screening data, not investment advice."
)


@mcp.prompt()
def review_watchlist(
    symbols: Annotated[str, Field(description="Tickers to review, separated by commas, e.g. NVDA, AAPL.")],
    market: Market = "US",
) -> str:
    """Review a watchlist or holdings against the market regime and the preset screens."""
    return (
        f"Review these {market.upper()} stocks: {symbols}. Read the stockscreen://skill resource first. "
        "Call `market_overview` for the regime, `audit_stocks` for the screens each stock passes and nearly passes, "
        "`group_rankings` to see whether its group leads, and `get_stock` for any name that needs more detail. "
        f"{CLOSING_INSTRUCTIONS}"
    )


@mcp.prompt()
def find_candidates(market: Market = "US") -> str:
    """Find screening candidates in a market, from the regime down to individual names."""
    return (
        f"Find screening candidates in the {market.upper()} market. Read the stockscreen://skill resource first. "
        "Call `market_overview` for the regime, `list_screens` and `screen_track_record` to choose screens, "
        "`run_screen` for the matches, `group_rankings` to confirm leadership, and `get_stock` for the finalists. "
        f"{CLOSING_INSTRUCTIONS}"
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
