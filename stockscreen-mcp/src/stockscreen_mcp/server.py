"""MCP tools over the stock-screener project's published daily data."""

from __future__ import annotations

import difflib
import math
from datetime import datetime, timezone
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .data import StaticData
from .screens import (
    evaluate_expression,
    explain_condition,
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
  market_overview for market health and breadth, group_rankings for leading groups.
- This is educational screening data, not investment advice.
"""

mcp = MCPServer("stockscreen", instructions=INSTRUCTIONS)
data = StaticData()

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True)
Market = Annotated[str, Field(description="Market code, e.g. US, HK, JP, IN, DE. See list_markets.")]

# Columns shown for every screen row, before the fields used to filter and sort.
ROW_FIELDS = (
    "symbol", "company_name", "current_price", "price_change_1d", "rs_rating", "rs_rating_1m",
    "rs_rating_3m", "rs_rating_12m", "stage", "composite_score", "minervini_score", "canslim_score",
    "ibd_industry_group", "ibd_group_rank", "eps_growth_qq", "sales_growth_qq", "week_52_high_distance",
    "market_cap_usd",
)
# Internal bookkeeping fields nobody asks about.
HIDDEN_STOCK_FIELDS = frozenset({
    "market_rs_run_id", "rs_formula_version", "rs_universe_size", "scan_mode", "screeners_run",
    "applicable_screeners", "unavailable_screeners", "matched_groups", "data_status",
})
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


class Filter(BaseModel):
    """One condition on a scan field. Use min/max for numbers, equals for booleans or text."""

    field: str = Field(description=(
        "Scan field, e.g. rs_rating, rs_rating_3m, stage (1-4), ibd_group_rank, eps_growth_qq, "
        "sales_growth_qq, current_price, volume, market_cap_usd, week_52_high_distance, "
        "minervini_score, canslim_score, vcp_score, ma_alignment, passes_template, se_pattern_primary."
    ))
    min: float | None = Field(default=None, description="Inclusive lower bound.")
    max: float | None = Field(default=None, description="Inclusive upper bound.")
    equals: bool | str | float | None = Field(default=None, description="Exact value; overrides min/max.")


def _pick(row: dict[str, Any], fields: tuple[str, ...] | list[str]) -> dict[str, Any]:
    picked = {}
    for field in fields:
        value = row_value(row, field)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            continue  # inf/NaN (e.g. listing_aware_volume) are not valid JSON
        picked[field] = value
    return picked


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
    (run_screen, get_stock, list_screens), "breadth" (market_overview breadth) and "groups"
    (group_rankings) are published for it. unavailable_markets lists markets that failed this
    publish. Takes no arguments. Results are cached for up to 10 minutes; markets are published
    after their close by two scheduled runs, one for Asia and one for US/Europe.
    """
    manifest = data.manifest()
    now = datetime.now(timezone.utc)
    markets = [
        {
            "code": code,
            "name": entry.get("display_name", code),
            "as_of_date": entry.get("as_of_date"),
            "status": _status(entry, now),
            "features": sorted(k for k, on in entry.get("features", {}).items()
                               if on and k in {"scan", "breadth", "groups"}),
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

    Pass a screen's id to run_screen.
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
    if screen is None and not filters:
        raise ToolError("Pass a screen (see list_screens) or filters, or both.")
    scan, rows = data.scan(market)

    preset = None
    if screen is not None:
        presets = {p["id"]: p for p in _visible_presets(market, scan)}
        preset = presets.get(screen.strip().lower())
        if preset is None:
            raise ToolError(f"Unknown screen {screen!r}. Available: {', '.join(presets)}.")

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
    columns = [*ROW_FIELDS, *dict.fromkeys(extra)]
    return {
        "market": market.upper(),
        "as_of_date": scan.get("as_of_date"),
        "screen": preset["name"] if preset else None,
        "total_matches": len(matches),
        "sorted_by": f"{sort_by} {'desc' if descending else 'asc'}",
        "rows": [_pick(row, columns) for row in matches[:limit]],
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


@mcp.tool(annotations=READ_ONLY)
def audit_stocks(
    symbols: Annotated[list[str], Field(
        min_length=1, max_length=50, description="Tickers to check in this market, e.g. a watchlist or holdings.",
    )],
    market: Market = "US",
    screens: Annotated[list[str] | None, Field(
        description="Preset ids from list_screens to check; defaults to every screen in the market.",
    )] = None,
    include_passing: Annotated[bool, Field(description="Also list the rules each stock meets, under `passed`.")] = False,
) -> dict[str, Any]:
    """Check a list of stocks against the preset screens in one call: for each symbol and screen,
    pass/fail plus every rule it fails and by how much.

    Use it to review a watchlist or holdings, find near misses, or see what has deteriorated in a
    position, meaning which rules it fails today. Use get_stock instead for every field on one
    symbol, and run_screen to discover new candidates.

    short_by is the positive distance between a failed numeric rule's bound and the stock's value
    (rs_rating 60 against a min of 70 is short_by 10); it is absent when the value is missing and
    on non-numeric rules. `passes` matches the website and get_stock's passes_screens. failed and
    passed cover the rules list_screens shows, and a `note` marks a failure that comes from an
    optional rule group instead. near_misses lists the screens a stock fails on exactly one rule.
    Symbols absent from the market's scan are listed in not_found rather than raising an error.
    Output grows with symbols times screens, so pass `screens` for more than a few symbols.
    """
    scan, rows = data.scan(market)
    presets = {p["id"]: p for p in _visible_presets(market, scan)}
    wanted = {s.strip().lower() for s in screens or presets}
    unknown = sorted(wanted - presets.keys())
    if unknown:
        raise ToolError(f"Unknown screen {', '.join(map(repr, unknown))}. Available: {', '.join(presets)}.")
    audited = [p for p in presets.values() if p["id"] in wanted]

    by_symbol = {str(r.get("symbol", "")).upper(): r for r in rows}
    requested = list(dict.fromkeys(s.strip().upper() for s in symbols))
    stocks = []
    for symbol in requested:
        row = by_symbol.get(symbol)
        if row is None:
            continue
        results = {}
        for preset in audited:
            expression = preset["filter_expression"]
            explained = [explain_condition(row, c) for c in expression["required"]["conditions"]]
            result = {
                "passes": evaluate_expression(row, expression),
                "failed": [e for e in explained if not e["passes"]],
            }
            if include_passing:
                result["passed"] = [e for e in explained if e["passes"]]
            has_groups = any(g.get("enabled") is not False for g in expression.get("groups") or [])
            if has_groups and not result["passes"] and not result["failed"]:
                result["note"] = "fails an optional rule group"
            results[preset["id"]] = result
        stocks.append({
            "symbol": row["symbol"],
            "snapshot": _pick(row, SNAPSHOT_FIELDS),
            "screens": results,
            "near_misses": [i for i, r in results.items() if not r["passes"] and len(r["failed"]) == 1],
        })
    return {
        "market": market.upper(),
        "as_of_date": scan.get("as_of_date"),
        "screens": [p["id"] for p in audited],
        "stocks": stocks,
        "not_found": [s for s in requested if s not in by_symbol],
    }


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


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
