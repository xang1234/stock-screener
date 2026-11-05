# MCP Integration

> **No-install option:** [`stockscreen-mcp`](../stockscreen-mcp/) (`uvx stockscreen-mcp`) serves the published daily snapshot (screens, stock lookup, market health and breadth, group rankings) to any MCP client. It needs no running app. The server described below is the live app's own MCP server, which needs the full stack. It adds feature-run comparisons, watchlists, themes and task status.

StockScreenClaude exposes a [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server for AI copilot workflows. When connected to an MCP-capable agent (like [Hermes](https://hermes-agent.nousresearch.com/)), the agent can query market state, compare feature runs, inspect watchlists and themes, and prepare automated market briefs.

## Available Tools

The MCP server exposes 15 tools:

| Tool | Description |
|------|-------------|
| `market_overview` | Current market state snapshot (breadth, sentiment, key indices) |
| `compare_feature_runs` | Diff two published feature runs to identify biggest movers |
| `find_candidates` | Query stocks with opinionated filters |
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

Both tools are read-only and replace the two facts consumers (the Hermes watchlist scanner among
them) still take from the temporary `xang-mcp-gateway`. Neither writes, fetches a provider or
derives its own rules.

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
  "calendar_source": "xang_market_calendar_service"
}
```

`expected_trading_date` names the session whose bars can exist right now. While the session is
**open** that is the market-calendar day; before the open, and on a day without a session, it is
the last completed one — pointing at a session that has not traded yet would make an up-to-date
cache look stale for the rest of the day. After the close the two dates are the same again.
Values come from
`app.services.market_calendar_service.MarketCalendarService`; if the service cannot answer, the
call is `isError` with `code: "calendar_unavailable"` and **no** `expected_trading_date`.

### `earnings_lookup`

Input: `{"symbol": "AAPL"}`. The ticker shape is validated first (a letter or digit, then `A-Z`,
digits, `.` and `-`, up to 20 characters), so the stored provider forms `0700.HK`, `7203.T` and
`2330.TW` are accepted as well as `BRK.B`; a malformed symbol is `invalid_arguments` before any
query runs. Freshness is measured in the symbol's own market date, taken from the suffix registry
(`.HK`, `.T`, `.TW`, ...); a plain ticker keeps the Eastern date the scanners use. When a suffixed
symbol's calendar cannot answer, the Eastern date is used instead and the payload says so in
`freshness_reference_basis: "eastern_fallback"` - a successful answer is then not market-local
evidence. The field is `"market"` for a suffixed symbol served by its own calendar and
`"eastern"` for a plain ticker; `freshness_reference_date` names the date actually used. The answer
keeps the gateway contract per row of `public.stock_fundamentals`:

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
  "freshness_reference_date": "2026-10-15",
  "freshness_reference_basis": "eastern",
  "event_calendar_max_age_days": 14,
  "event_calendar_producer_refresh_after_days": 7,
  "event_calendar_future_tolerance_days": 3,
  "generated_at": "..."
}
```

Rules, taken from `app.services.yahoo_earnings_calendar` (no second implementation):

- `freshness_status`: `ok` inside the observation window, `stale` past it or when the stored
  earnings date has already passed, `invalid` when the observation itself lies in the future
  beyond tolerance, `unknown` when there is no observation to judge. A past earnings date is a
  normal state of the weekly producer cadence, not damaged storage, and is named
  `earnings_date_in_past`.
- `available` answers "is there a usable observation": a current observation that reports **no**
  upcoming date is `available: true` with
  `reasons: ["no_upcoming_earnings_date", "date_confirmation_unrecorded"]`.
- `date_confirmation` is always `unknown`. The table stores no confirmation proof, so neither
  `confirmed` nor `estimated` can be claimed; a current date is not a confirmed date.
- Missing or stale facts are named in `reasons` instead of being filled in.

Deviations from the gateway, on purpose: a malformed symbol and a failing database are `isError`
(`invalid_arguments`, `database_unavailable`) rather than a success payload, because a consumer
must not read an envelope off a failed lookup. Neither message carries connection strings or
driver text.

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
