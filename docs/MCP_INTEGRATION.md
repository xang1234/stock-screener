# MCP Integration

> **No-install option:** [`stockscreen-mcp`](../stockscreen-mcp/) (`uvx stockscreen-mcp`) serves the published daily snapshot (screens, stock lookup, market health and breadth, group rankings) to any MCP client. It needs no running app. The server described below is the live app's own MCP server, which needs the full stack. It adds feature-run comparisons, watchlists, themes and task status.

StockScreenClaude exposes a [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server for AI copilot workflows. When connected to an MCP-capable agent (like [Hermes](https://hermes-agent.nousresearch.com/)), the agent can query market state, compare feature runs, inspect watchlists and themes, and prepare automated market briefs.

## Available Tools

The MCP server exposes 15 tools:

| Tool | Description |
|------|-------------|
| `market_overview` | Current market state snapshot (breadth, sentiment, key indices) |
| `compare_feature_runs` | Diff two published feature runs to identify biggest movers |
| `find_candidates` | Query stocks with opinionated filters; a `market` argument pins that market's published run (see [Selecting a candidate market](#selecting-a-candidate-market)) |
| `explain_symbol` | Explain a stock's rating (brief or full depth) |
| `watchlist_snapshot` | Summary of a named watchlist with current data |
| `theme_state` | Inspect theme rankings, momentum, and lifecycle |
| `task_status` | Check background job health and last execution times |
| `watchlist_add` | Add symbols to a watchlist (requires opt-in) |
| `group_rankings` | IBD industry group rankings and rank movers |
| `stock_lookup` | Universe, fundamentals and optional technicals for one symbol |
| `stock_snapshot` | Combined snapshot for one symbol |
| `breadth_snapshot` | Recent breadth snapshots for the last N trading days |
| `daily_digest` | Full daily digest with stance, leaders, themes and risks |
| `market_calendar_status` | Canonical market-calendar facts for one market (read-only) |
| `earnings_lookup` | Stored earnings-calendar observation with canonical freshness (read-only) |

Every tool returns a structured envelope with `summary`, `facts`, `citations`, `freshness`, and `next_actions`.

## Calendar and Earnings Facts

Both tools are read-only and exist so that consumers (the Hermes watchlist scanner among them)
do not need the temporary `xang-mcp-gateway` for these two facts. Neither tool writes, fetches a
provider or derives its own rules.

### `market_calendar_status`

Input: `{"market": "US"}` (default `US`; an unsupported market is `invalid_arguments`).

The answer carries the canonical facts plus the keys the gateway used, so a consumer can read the
same names:

```json
{
  "market": "US",
  "market_timezone": "America/New_York",
  "market_now": "2026-10-15T11:00:00-04:00",
  "market_date": "2026-10-15",
  "is_trading_day": true,
  "last_completed_trading_day": "2026-10-14",
  "expected_trading_date": "2026-10-15",
  "calendar_source": "xang_market_calendar_service",
  "summary": "...",
  "facts": [],
  "citations": [],
  "freshness": {"generated_at": "...", "as_of_date": "2026-10-15", "sources": {}},
  "next_actions": []
}
```

`expected_trading_date` keeps the gateway's meaning: the market-calendar day while the market
trades — **mid-session and pre-open included** — otherwise the last completed session. It is deliberately not
`last_completed_trading_day` during a running session; that value only moves after the close and
would make today's run look stale.

The values come from `app.services.market_calendar_service.MarketCalendarService`. If the service
cannot answer (missing coverage, provider drift), the call is an `isError` response with
`code: "calendar_unavailable"` and **no** `expected_trading_date` — there is no half-filled success
payload.

### `earnings_lookup`

Input: `{"symbol": "AAPL"}`. The ticker shape is validated first (`A-Z`, digits, `.`, `-`); a
malformed symbol is `invalid_arguments` before any query runs.

The answer keeps the gateway contract per row of `public.stock_fundamentals`:

```json
{
  "symbol": "AAPL",
  "next_earnings_date": "2026-11-05",
  "event_calendar_as_of_date": "2026-10-10",
  "source": "xang_yahoo_calendar",
  "freshness_status": "ok",
  "date_confirmation": "unknown",
  "available": true,
  "reasons": ["date_confirmation_unrecorded"],
  "observation_age_days": 5,
  "event_calendar_max_age_days": 14,
  "event_calendar_producer_refresh_after_days": 7,
  "event_calendar_future_tolerance_days": 3,
  "generated_at": "..."
}
```

Rules, taken from `app.services.yahoo_earnings_calendar` (no second implementation):

- `freshness_status`: `ok` inside the observation window, `stale` past it, `invalid` when the
  observation lies in the future beyond tolerance or the stored date is unusable, `unknown` when
  there is no observation to judge.
- `available` answers "is there a usable observation": a current observation that reports **no**
  upcoming date is `available: true` with
  `reasons: ["no_upcoming_earnings_date", "date_confirmation_unrecorded"]`.
- `date_confirmation` is always `unknown`. The table stores no confirmation proof, so neither
  `confirmed` nor `estimated` can be claimed; a current date is not a confirmed date.
- Missing or stale facts are named in `reasons` instead of being filled in.

Deviations from the gateway, on purpose: a malformed symbol and a failing database are `isError`
responses (`invalid_arguments`, `database_unavailable`) rather than a success payload with an
`invalid`/`unknown` status, because a consumer must not read an envelope off a failed lookup. The
error message never contains connection strings or driver text.

## Example Workflows

### After-Close Market Brief
Query `market_overview`, compare the latest two feature runs, snapshot your watchlists, and check theme state. The agent synthesizes a brief covering market health, notable movers, theme momentum, and risks.

### Feature Run Comparison
Use `compare_feature_runs` to see which stocks gained or lost the most score between two daily runs. Useful for identifying breakouts, breakdowns, and emerging setups.

### Watchlist Stewardship
Combine `watchlist_snapshot` with `explain_symbol` to audit your watchlist. The agent identifies which positions still meet screening criteria and which have deteriorated.

### Run Freshness and Event Risk
Call `market_calendar_status` for the market, then `earnings_lookup` for each held symbol. Treat
`expected_trading_date` as the day the run should carry, and `freshness_status != "ok"` or
`available: false` as "no evidence" — never as a date.

### Theme Scouting
Use `theme_state` to review trending and emerging themes, then `find_candidates` to discover stocks associated with high-momentum themes.

### Selecting a candidate market
`find_candidates` takes an optional top-level `market` argument:

```json
{"market":"DE","filters":{"min_rs_rating":80,"stage":2},"limit":100}
```

The server normalizes the code, validates it against the supported-market registry and resolves `latest_published_market:<MARKET>`, then verifies that the referenced run is published and that its stored market metadata agrees with the request. It fails closed: there is no global-pointer or other-market fallback. Errors are `market_run_unavailable` (missing pointer, unpublished run, custom universe, missing or conflicting market evidence) and `invalid_arguments` (bad market code); a stored row with an unknown status or run type returns `invalid_stored_data`, because the request was valid and the row is not.

A successful response keeps the candidate projection and adds the verified `market`, `universe_type: "market"` and `pointer_key` to `run`, plus `scope.market`. A valid empty result still carries that provenance, which distinguishes it from an unavailable publication. The one exception is an unknown watchlist name: it returns the existing not-found envelope with empty `symbols` and no `run`/`scope`, because it reports a bad scope rather than a missing publication.

Omitting `market` preserves the previous global latest-run behaviour and response shape. `universe` keeps its own meaning (`all`, `latest_published`, or a watchlist name) and is not a market selector, although it may be combined with an explicit market. The limit remains 100, so clients collecting a full universe must detect truncation and pin every query for one market to the same run identity; market-specific freshness must use that market's own calendar.

This argument changes nothing else: scoring, strategy rules, earnings collection, calendar calculation, authentication and write permissions are untouched, and the lookup creates or changes no pointer.

## Configuration

In `backend/.env`:

```dotenv
MCP_SERVER_NAME=stockscreen-market-copilot
MCP_WATCHLIST_WRITES_ENABLED=false
```

`MCP_WATCHLIST_WRITES_ENABLED=false` is the safe default. Enable it only if you want MCP clients to modify watchlists via `watchlist_add`.

## Setup

For full setup instructions including Hermes configuration, smoke testing, and weekday automation, see [integrations/hermes/README.md](../integrations/hermes/README.md).

Quick smoke test:

```bash
cd backend
PYTHONPATH="$PWD" ./venv/bin/python -m app.interfaces.mcp.server
# The process waits on stdio for an MCP client. Ctrl+C to stop.
```
