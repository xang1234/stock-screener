#!/usr/bin/env python3
"""Render crawlable HTML pages from a built static site's own static-data.

The demo is a hash-routed SPA, so search engines see one empty page. This writes
plain HTML at stable URLs (stocks, curated screens, group rankings, breadth, the
scoreboard, a hub and a sitemap) into the built site, from the same published JSON the app reads.

    python3 scripts/render_seo_pages.py --dist frontend/dist \
        --site-url https://xang1234.github.io/stock-screener/

Stdlib only. Screen matching and ordering reuse stockscreen-mcp's port of the
site's evaluator and sort (imported by path; that module is pure stdlib).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "stockscreen-mcp" / "src"))
from stockscreen_mcp.screens import evaluate_expression, sort_rows  # noqa: E402

REPO_URL = "https://github.com/xang1234/stock-screener"
CURATED_SCREENS = ("minervini", "canslim", "vcp", "se_cup_handle", "se_high_tight_flag", "new_highs")
SCREEN_PAGE_LIMIT = 100
FIELD_LABELS = {
    "minervini_score": "Minervini score", "canslim_score": "CANSLIM score", "rs_rating": "RS rating",
    "rs_rating_3m": "3-month RS rating", "ma_alignment": "moving averages aligned (50 > 150 > 200)",
    "stage": "Weinstein stage", "eps_growth_qq": "quarterly EPS growth %", "vcp_detected": "VCP detected",
    "week_52_high_distance": "distance from 52-week high %", "volume_surge": "volume vs 50-day average",
    "se_pattern_primary": "detected base pattern", "composite_score": "composite score",
    "vcp_score": "VCP score", "se_setup_score": "setup score",
}
SCOREBOARD_HORIZONS = (5, 20, 60)
SCOREBOARD_SECTIONS = (
    ("live", "Recorded live"),
    ("artifact", "Recovered from retained build artifacts, not recorded live"),
    ("artifact_pre_stage_fix",
     "Recovered from retained build artifacts, not recorded live, from before the Stage calculation fix (#671)"),
)
CSS = """\
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:960px;margin:0 auto;padding:16px;color:#1b1f24;line-height:1.5}
a{color:#1f5fbf}nav{font-size:14px;margin-bottom:8px}h1{font-size:26px;margin:8px 0}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}th,td{border-bottom:1px solid #e3e6ea;padding:6px 8px;text-align:left}
th{background:#f5f7f9}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:#5c6670;font-size:14px}.cta{display:inline-block;margin:8px 0;padding:8px 14px;background:#1f5fbf;color:#fff;border-radius:6px;text-decoration:none}
footer{margin-top:32px;border-top:1px solid #e3e6ea;padding-top:12px;font-size:13px;color:#5c6670}
ul.cols{columns:4 140px;padding-left:18px}svg{vertical-align:middle}
"""


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def slug(symbol: str) -> str:
    """Path-safe and unambiguous on any server: unusual characters become _XX hex."""
    safe = re.sub(r"[^A-Za-z0-9.\-]", lambda m: f"_{ord(m.group()):02X}", symbol)
    return "_2E" + safe[1:] if safe.startswith(".") else safe  # never "." or ".."


def num(value: Any, digits: int = 1, suffix: str = "") -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "–"
    return f"{value:,.{digits}f}{suffix}"


def short(value: Any) -> str:
    """Whole numbers without decimals (90.0 -> 90), others to one decimal."""
    if isinstance(value, float) and value.is_integer():
        return f"{value:,.0f}"
    return num(value, 1)


def signed(value: Any, suffix: str = "%") -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "–"
    return f"{value:+d}{suffix}" if isinstance(value, int) else f"{value:+.1f}{suffix}"


def money(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "–"
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(value) >= size:
            return f"${value / size:.1f}{unit}"
    return f"${value:,.0f}"


def sparkline(points: Any) -> str:
    values = [v for v in points or [] if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(values) < 2:
        return ""
    low, high = min(values), max(values)
    span = (high - low) or 1
    step = 160 / (len(values) - 1)
    path = " ".join(f"{i * step:.1f},{38 - (v - low) / span * 36:.1f}" for i, v in enumerate(values))
    color = "#1a7f37" if values[-1] >= values[0] else "#cf222e"
    return (f'<svg width="160" height="40" viewBox="0 0 160 40" role="img" aria-label="Recent price trend">'
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{path}"/></svg>')


def rule_text(condition: dict[str, Any]) -> str:
    label = FIELD_LABELS.get(condition["field"], condition["field"].replace("_", " "))
    kind = condition.get("kind")
    if kind == "boolean":
        return label if condition.get("value") else f"not {label}"
    if kind == "categorical":
        values = ", ".join(str(v).replace("_", " ") for v in condition.get("values", []))
        return f"{label}: {'not ' if condition.get('mode') == 'exclude' else ''}{values}"
    low, high = condition.get("min"), condition.get("max")
    if low is not None and high is not None:
        return f"{label} = {low:g}" if low == high else f"{label} between {low:g} and {high:g}"
    if low is not None:
        return f"{label} ≥ {low:g}"
    return f"{label} ≤ {high:g}" if high is not None else label


def plural(count: int, noun: str) -> str:
    return f"{count:,} {noun}{'' if count == 1 else 's'}"


def scoreboard_horizon(cohort: dict[str, Any], sessions: int) -> dict[str, Any]:
    return {h["horizon_sessions"]: h for h in cohort["horizons"]}[sessions]


def scoreboard_cell(cohort: dict[str, Any], sessions: int) -> str:
    """A snippet carries no grey, so thin and empty cells say so in words instead of showing a number."""
    if cohort["events"] == 0:
        return "<td>no picks</td>"
    h = scoreboard_horizon(cohort, sessions)
    if h["sample_size"] == 0:
        verdict = "pending" if h["pending_windows"] else "no closed windows"
    elif h["thin"]:
        verdict = "too few"
    else:
        verdict = f"{signed(h['median_return_pct'])} median"
    dates = plural(h["signal_dates"], "date")
    counts = " · ".join(filter(None, (
        f"{h['sample_size']:,} closed",
        f"{dates}, effective {h['effective_dates']:.1f}" if h["sample_size"] else dates,
        f"{h['pending_windows']:,} pending",
        h["skipped_missing_history"] and f"{h['skipped_missing_history']:,} no price",
        h["suppressed_overlapping"] and f"{h['suppressed_overlapping']:,} overlapping",
    )))
    return f"<td>{esc(verdict)}<br><span class=muted>{esc(counts)}</span></td>"


def scoreboard_section(title: str, cohorts: list[dict[str, Any]]) -> str:
    note = ("" if any(h["sample_size"] for c in cohorts for h in c["horizons"])
            else "<p>No window has closed yet. Nothing here is a result.</p>")
    rows = "".join(
        f"<tr><td>{esc(c['name'])}<br><span class=muted>{esc(c['first_date'])} to {esc(c['last_date'])}"
        + (" · Retired" if c["retired"] else "") + "</span></td>"
        + "".join(scoreboard_cell(c, sessions) for sessions in SCOREBOARD_HORIZONS) + "</tr>"
        for c in cohorts)
    head = "".join(f"<th>{sessions} sessions</th>" for sessions in SCOREBOARD_HORIZONS)
    table = f"<table><tr><th>Screen</th>{head}</tr>{rows}</table>" if cohorts else ""
    return f"<h2>{esc(title)}</h2>{note}{table}"


class Site:
    def __init__(self, dist: Path, site_url: str) -> None:
        self.dist = dist
        self.site_url = site_url if site_url.endswith("/") else site_url + "/"
        self.base = urlparse(self.site_url).path or "/"
        self.sitemap: list[tuple[str, str]] = []

    def href(self, *parts: str) -> str:
        return self.base + "".join(f"{p}/" for p in parts)

    def write(self, parts: tuple[str, ...], lastmod: str, title: str, description: str, body: str,
              crumbs: list[tuple[str, str]]) -> None:
        canonical = self.site_url + "".join(f"{p}/" for p in parts)
        nav = " › ".join(f'<a href="{esc(url)}">{esc(label)}</a>' for label, url in crumbs)
        page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><meta name="description" content="{esc(description)}">
<link rel="canonical" href="{esc(canonical)}"><link rel="stylesheet" href="{esc(self.base)}seo.css">
<link rel="icon" href="{esc(self.base)}fire.svg" type="image/svg+xml">
<meta property="og:type" content="website"><meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description)}"><meta property="og:url" content="{esc(canonical)}">
</head><body><nav>{nav}</nav><main>
{body}
</main><footer>Daily end-of-day data from the open-source <a href="{REPO_URL}">stock-screener</a> project,
also available to AI assistants via <a href="{REPO_URL}/tree/main/stockscreen-mcp">stockscreen-mcp</a>.
Educational screening data, not investment advice.</footer></body></html>
"""
        target = self.dist.joinpath(*parts)
        target.mkdir(parents=True, exist_ok=True)
        (target / "index.html").write_text(page, encoding="utf-8")
        self.sitemap.append((canonical, lastmod))


def load_market(data_root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    def read_file(rel: str | None) -> Any:
        path = data_root / rel if rel else None
        return json.loads(path.read_text()) if path and path.exists() else None

    def read(page: str) -> Any:
        return read_file(entry.get("pages", {}).get(page, {}).get("path"))

    def read_scoreboard() -> Any:
        rel = entry.get("assets", {}).get("scoreboard", {}).get("path")
        try:
            return read_file(rel)
        except ValueError as exc:
            print(f"::warning::{rel} cannot be parsed, skipping its page: {exc!r}")
            return None

    scan = read("scan") or {}
    rows = {r["symbol"]: r for r in scan.get("initial_rows", [])}
    for chunk in scan.get("chunks", []):
        for r in json.loads((data_root / chunk["path"]).read_text())["rows"]:
            rows[r["symbol"]] = r
    return {"scan": scan, "rows": list(rows.values()), "home": read("home") or {},
            "groups": read("groups"), "breadth": read("breadth"),
            "scoreboard": read_scoreboard()}


def render_scoreboard(site: Site, code: str, name: str, scoreboard: dict[str, Any],
                      crumbs: list[tuple[str, str]]) -> tuple[str, str, str]:
    cohorts = scoreboard["cohorts"]
    board_as_of = scoreboard["as_of_date"]
    at_20 = [scoreboard_horizon(c, 20) for c in cohorts]
    closed = sum(h["sample_size"] for h in at_20)
    pending = sum(h["pending_windows"] for h in at_20)
    picks = plural(closed + pending, "pick")
    sections = "".join(
        scoreboard_section(title, members)
        for origin, title in SCOREBOARD_SECTIONS
        if (members := [c for c in cohorts if c["origin"] == origin]) or origin == "live")
    benchmark = scoreboard["benchmark"]
    rules = ("Entry is the first open after publication. Exit is the close of the Nth session.",
             "A pick is a stock newly entering the screen. Stocks on the list on the first recorded day, or on a day the screen's definition or the Stage calculation changed, are not counted.",
             "Returns are price only, in local currency, with no costs." + (f" Excess is against {benchmark}." if benchmark else ""),
             "Prices are as the provider reports them, so an unadjusted split can appear as a false move. Medians limit the effect.",
             "Funds that match a screen are included.",
             "A stock counts once per screen until its window ends. A repeat that enters earlier is left out.",
             "Effective dates count each date by how many picks it holds, so one busy day counts as about one.",
             "Stocks that later delist drop out as no price, which can flatter results.")
    scored_since = scoreboard["scored_since"]
    scored = (f"Scored since {esc(scored_since)} · " if scored_since and scored_since != scoreboard["recording_since"] else "")
    site.write(("scoreboard", code.lower()), board_as_of, f"{name} scoreboard – how preset screen picks moved afterward ({board_as_of})",
               f"{name} preset screens tracked after publication: {plural(len({c['preset'] for c in cohorts}), 'screen')}, "
               f"{picks}, {plural(closed, 'window')} closed and {pending:,} pending at 20 sessions, as of {board_as_of}.",
               f"<h1>{esc(name)} scoreboard</h1><p class=muted>As of {esc(board_as_of)} · "
               f"Recording since {esc(scoreboard['recording_since'])} · {scored}Ledger through {esc(scoreboard['ledger_through'])} · "
               f"{plural(scoreboard['missing_sessions'], 'missing session')}</p>"
               "<ul>" + "".join(f"<li>{esc(rule)}</li>" for rule in rules) + f"</ul>{sections}"
               f'<a class=cta href="{esc(f"{site.base}#/scoreboard?market={code}")}">Open the interactive scoreboard</a>',
               crumbs)
    return "Scoreboard", site.href("scoreboard", code.lower()), f"{picks} tracked"


def render_market(site: Site, code: str, entry: dict[str, Any], data: dict[str, Any]) -> list[str]:
    m = code.lower()
    name = entry.get("display_name", code)
    as_of = entry.get("as_of_date") or data["scan"].get("as_of_date") or ""
    app = f"{site.base}#/scan?market={code}"
    home_crumbs = [("Stock Screener", site.href("explore")), (name, site.href("markets", m))]
    # IPO-style scans can mark short-history rows scannable without an RS rating.
    stocks = [r for r in data["rows"] if r.get("is_scannable") and r.get("symbol")
              and r.get("data_status") == "complete" and r.get("rs_rating") is not None]
    has_page = {r["symbol"] for r in stocks}
    presets = {p["id"]: p for p in data["scan"].get("preset_screens", []) if "filter_expression" in p}
    curated = [presets[i] for i in CURATED_SCREENS if i in presets]

    def stock_link(row: dict[str, Any]) -> str:
        label = esc(row["symbol"])
        return f'<a href="{esc(site.href("stocks", m, slug(row["symbol"])))}">{label}</a>' if row["symbol"] in has_page else label

    # Screens: membership and order as the site shows them.
    screen_matches = {}
    for preset in curated:
        matched = [r for r in data["rows"] if evaluate_expression(r, preset["filter_expression"])]
        sort_by = preset.get("sort_by") or "composite_score"
        screen_matches[preset["id"]] = sort_rows(
            matched, sort_by, descending=preset.get("sort_order", "desc") == "desc")
        label = FIELD_LABELS.get(sort_by, sort_by.replace('_', ' '))
        heading = label[:1].upper() + label[1:]
        # Show the column the list is sorted by (RS already has its own column).
        sort_header = "" if sort_by == "rs_rating" else f"<th class=n>{esc(heading)}</th>"
        rows_html = "".join(
            f"<tr><td>{stock_link(r)}</td><td>{esc(r.get('company_name'))}</td>"
            + ("" if sort_by == "rs_rating" else f"<td class=n>{short(r.get(sort_by))}</td>")
            + f"<td class=n>{num(r.get('rs_rating'), 0)}</td>"
            f"<td class=n>{num(r.get('current_price'), 2)}</td><td>{esc(r.get('ibd_industry_group'))}</td></tr>"
            for r in screen_matches[preset["id"]][:SCREEN_PAGE_LIMIT])
        count = len(screen_matches[preset["id"]])
        rules = "".join(f"<li>{esc(rule_text(c))}</li>" for c in preset["filter_expression"]["required"]["conditions"])
        site.write(
            ("screens", m, preset["id"]), as_of,
            f"{preset['name']} stocks today – {name} ({as_of})",
            f"{count} {name} stocks pass the {preset['name']} screen as of {as_of}. {preset.get('description') or ''}",
            f"<h1>{esc(preset['name'])} – {esc(name)}</h1><p class=muted>As of {esc(as_of)} · {count} stocks</p>"
            f"<p>{esc(preset.get('description'))}</p><h2>Rules</h2><ul>{rules}</ul>"
            f"<table><tr><th>Symbol</th><th>Company</th>{sort_header}<th class=n>RS</th><th class=n>Price</th><th>Group</th></tr>{rows_html}</table>"
            + (f"<p class=muted>Showing the first {SCREEN_PAGE_LIMIT}.</p>" if count > SCREEN_PAGE_LIMIT else "")
            + f'<a class=cta href="{esc(app)}">Open the interactive screener</a>',
            home_crumbs)

    # Stock pages.
    for row in stocks:
        passes = [p for p in curated if any(r is row for r in screen_matches[p["id"]])]
        facts = [
            ("Price", f"{num(row.get('current_price'), 2)} {esc(row.get('currency') or '')}"),
            ("1-day change", signed(row.get("price_change_1d"))),
            ("RS rating (overall / 1M / 3M / 12M)", " / ".join(num(row.get(k), 0) for k in
                                                               ("rs_rating", "rs_rating_1m", "rs_rating_3m", "rs_rating_12m"))),
            ("Stage", esc(row.get("stage_name") or row.get("stage") or "–")),
            ("Composite / Minervini / CANSLIM score", " / ".join(num(row.get(k), 0) for k in
                                                                ("composite_score", "minervini_score", "canslim_score"))),
            ("Industry group (rank)", f"{esc(row.get('ibd_industry_group') or '–')} ({num(row.get('ibd_group_rank'), 0)})"),
            ("EPS growth Q/Q", signed(row.get("eps_growth_qq"))),
            ("Sales growth Q/Q", signed(row.get("sales_growth_qq"))),
            ("Distance from 52-week high", signed(row.get("week_52_high_distance"))),
            ("Market cap (USD)", money(row.get("market_cap_usd"))),
        ]
        table = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in facts)
        screens = "".join(f'<li><a href="{esc(site.href("screens", m, p["id"]))}">{esc(p["name"])}</a></li>' for p in passes)
        rs = num(row.get("rs_rating"), 0)
        title = f"{row['symbol']} {row.get('company_name') or ''} – RS rating {rs}, {row.get('stage_name') or 'stage n/a'}"
        site.write(
            ("stocks", m, slug(row["symbol"])), as_of, title,
            f"{row['symbol']} ({row.get('company_name') or name}): RS rating {rs}, "
            f"{row.get('stage_name') or 'stage n/a'}, group {row.get('ibd_industry_group') or 'n/a'}, as of {as_of}.",
            f"<h1>{esc(row['symbol'])} · {esc(row.get('company_name'))}</h1>"
            f"<p class=muted>{esc(name)} · RS rating {rs} · As of {esc(as_of)}</p>{sparkline(row.get('price_sparkline_data'))}"
            f"<table>{table}</table>"
            + (f"<h2>Screens it passes today</h2><ul>{screens}</ul>" if screens else "")
            + f'<a class=cta href="{esc(app)}">Open {esc(name)} in the interactive screener</a>',
            home_crumbs + [("Stocks", site.href("stocks", m))])

    # Stock directory.
    by_letter: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(stocks, key=lambda r: r["symbol"]):
        by_letter.setdefault(row["symbol"][0].upper(), []).append(row)
    directory = "".join(
        f"<h2>{esc(letter)}</h2><ul class=cols>" + "".join(f"<li>{stock_link(r)}</li>" for r in rows) + "</ul>"
        for letter, rows in by_letter.items())
    site.write(("stocks", m), as_of, f"All {name} stocks – RS ratings and screens ({as_of})",
               f"Directory of {len(stocks)} {name} stocks with daily RS ratings, stage and screen results.",
               f"<h1>{esc(name)} stocks</h1><p class=muted>{len(stocks)} stocks · As of {esc(as_of)}</p>{directory}",
               home_crumbs)

    links = [("Stock directory", site.href("stocks", m), f"{len(stocks)} stocks")]
    links += [(p["name"], site.href("screens", m, p["id"]), f"{len(screen_matches[p['id']])} stocks") for p in curated]

    groups = data["groups"]
    if groups and groups.get("available"):
        ranked = groups["payload"]["rankings"]["rankings"]
        groups_as_of = groups.get("rs_as_of_date") or as_of  # last-good sections can trail the scan
        rows_html = "".join(
            f"<tr><td class=n>{esc(g.get('rank'))}</td><td>{esc(g.get('industry_group'))}</td><td class=n>{num(g.get('avg_rs_rating'))}</td>"
            f"<td class=n>{signed(g.get('rank_change_1w'), '')}</td><td class=n>{signed(g.get('rank_change_1m'), '')}</td>"
            f"<td class=n>{signed(g.get('rank_change_3m'), '')}</td><td>{esc(g.get('top_symbol'))}</td></tr>" for g in ranked)
        site.write(("groups", m), groups_as_of, f"{name} industry group rankings by relative strength ({groups_as_of})",
                   f"{len(ranked)} {name} industry groups ranked by relative strength, with 1-week, 1-month and 3-month rank changes.",
                   f"<h1>{esc(name)} industry group rankings</h1><p class=muted>As of {esc(groups_as_of)}</p>"
                   f"<table><tr><th class=n>Rank</th><th>Group</th><th class=n>Avg RS</th><th class=n>1W</th><th class=n>1M</th>"
                   f"<th class=n>3M</th><th>Top stock</th></tr>{rows_html}</table>",
                   home_crumbs)
        links.append(("Industry group rankings", site.href("groups", m), f"{len(ranked)} groups"))

    breadth = data["breadth"]
    if breadth and breadth.get("available"):
        current = breadth["payload"]["current"]
        breadth_as_of = current.get("date") or as_of
        recent = sorted(breadth["payload"].get("history_90d", []), key=lambda d: d["date"], reverse=True)[:10]
        latest = [("Stocks up 4%+ today", current.get("stocks_up_4pct")), ("Stocks down 4%+ today", current.get("stocks_down_4pct")),
                  ("5-day ratio", num(current.get("ratio_5day"), 2)), ("10-day ratio", num(current.get("ratio_10day"), 2)),
                  ("T2108 (% above 40-day MA)", num(current.get("t2108_pct"), 1)),
                  ("New 52-week highs", current.get("new_high_52week_count")), ("New 52-week lows", current.get("new_low_52week_count"))]
        rows_html = "".join(f"<tr><td>{esc(d['date'])}</td><td class=n>{esc(d.get('stocks_up_4pct'))}</td>"
                            f"<td class=n>{esc(d.get('stocks_down_4pct'))}</td></tr>" for d in recent)
        site.write(("breadth", m), breadth_as_of, f"{name} market breadth – 4% movers, T2108 ({breadth_as_of})",
                   f"StockBee-style market breadth for {name}: stocks up and down 4%, 5- and 10-day ratios, T2108, new highs and lows.",
                   f"<h1>{esc(name)} market breadth</h1><p class=muted>As of {esc(breadth_as_of)}</p>"
                   "<table>" + "".join(f"<tr><th>{esc(k)}</th><td class=n>{esc(v)}</td></tr>" for k, v in latest) + "</table>"
                   f"<h2>Last 10 sessions</h2><table><tr><th>Date</th><th class=n>Up 4%+</th><th class=n>Down 4%+</th></tr>{rows_html}</table>",
                   home_crumbs)
        links.append(("Market breadth", site.href("breadth", m), "4% movers, T2108"))

    if data["scoreboard"] is not None:
        try:
            links.append(render_scoreboard(site, code, name, data["scoreboard"], home_crumbs))
        except (KeyError, TypeError, ValueError) as exc:
            print(f"::warning::{code} scoreboard.json has the wrong shape, skipping its page: {exc!r}")

    exposure = data["home"].get("market_health_exposure") or {}
    health = (f"<p>Market exposure {num(exposure.get('exposure_score'), 0)}/100 · {esc(exposure.get('stance'))} · "
              f"{esc(exposure.get('distribution_day_count'))} distribution days</p>" if exposure else "")
    site.write(("markets", m), as_of, f"{name} stock screener – daily screens, RS ratings, group rankings ({as_of})",
               f"Daily {name} stock screens (Minervini, CANSLIM, VCP), RS ratings, industry group rankings and breadth.",
               f"<h1>{esc(name)}</h1><p class=muted>As of {esc(as_of)}</p>{health}<ul>"
               + "".join(f'<li><a href="{esc(u)}">{esc(t)}</a> <span class=muted>{esc(d)}</span></li>' for t, u, d in links)
               + f'</ul><a class=cta href="{esc(app)}">Open the interactive screener</a>',
               [("Stock Screener", site.href("explore"))])
    return [name, as_of]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", required=True, type=Path, help="Built site directory containing static-data/")
    parser.add_argument("--site-url", required=True, help="Public base URL of the site, e.g. https://x.github.io/repo/")
    args = parser.parse_args(argv)

    data_dir = args.dist / "static-data"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    data_root = data_dir / (manifest.get("data_root") or "")  # absent when STATIC_DATA_GENERATIONS=false
    site = Site(args.dist, args.site_url)
    unavailable = set(manifest.get("unavailable_markets") or [])

    markets = []
    for code, entry in sorted(manifest["markets"].items()):
        if code in unavailable or "scan" not in entry.get("pages", {}):
            continue
        name, as_of = render_market(site, code, entry, load_market(data_root, entry))
        markets.append((code, name, as_of))

    newest = max((as_of for _, _, as_of in markets), default="")
    site.write(("explore",), newest, "Free daily stock screener for 12 markets – RS ratings, screens, group rankings",
               "Daily Minervini, CANSLIM and VCP screens, relative-strength ratings, industry group rankings and market breadth.",
               "<h1>Stock Screener: markets</h1><ul>"
               + "".join(f'<li><a href="{esc(site.href("markets", c.lower()))}">{esc(n)}</a> <span class=muted>as of {esc(d)}</span></li>'
                         for c, n, d in markets)
               + f'</ul><a class=cta href="{esc(site.base)}">Open the interactive app</a>',
               [("Stock Screener", site.href("explore"))])

    (args.dist / "seo.css").write_text(CSS, encoding="utf-8")
    urls = "".join(f"<url><loc>{esc(loc)}</loc><lastmod>{esc(mod)}</lastmod></url>" for loc, mod in site.sitemap if mod)
    (args.dist / "sitemap.xml").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>\n',
        encoding="utf-8")
    if len(site.sitemap) > 50_000:  # ponytail: one sitemap; split into a sitemap index when this fires
        print(f"::warning::sitemap.xml has {len(site.sitemap)} URLs; search engines read only the first 50,000")
    print(f"Rendered {len(site.sitemap)} pages for {len(markets)} markets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
