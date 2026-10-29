# StockScreenClaude MCP Tool Map

This skill assumes the Hermes MCP server is configured under:

```yaml
mcp_servers:
  stockscreen_market:
    ...
```

Hermes prefixes MCP tools as `mcp_<server>_<tool>`, so the expected tool names are:

- `mcp_stockscreen_market_market_overview`
- `mcp_stockscreen_market_compare_feature_runs`
- `mcp_stockscreen_market_find_candidates`
- `mcp_stockscreen_market_explain_symbol`
- `mcp_stockscreen_market_watchlist_snapshot`
- `mcp_stockscreen_market_theme_state`
- `mcp_stockscreen_market_task_status`
- `mcp_stockscreen_market_watchlist_add`
- `mcp_stockscreen_market_group_rankings`
- `mcp_stockscreen_market_stock_lookup`
- `mcp_stockscreen_market_stock_snapshot`
- `mcp_stockscreen_market_breadth_snapshot`
- `mcp_stockscreen_market_daily_digest`
- `mcp_stockscreen_market_market_calendar_status`
- `mcp_stockscreen_market_earnings_lookup`

If you rename the Hermes MCP server key, these prefixed names will change too.

## Shared response contract

Every tool returns:

- `summary`: concise natural-language state
- `facts`: machine-readable key facts
- `citations`: stable internal references
- `freshness`: generated-at timestamp plus source freshness
- `next_actions`: concrete follow-up steps when data is stale, missing, or actionable

## Calendar and earnings facts

Both tools are read-only and replace the two facts that used to come from the
temporary `xang-mcp-gateway`.

- `market_calendar_status({"market": "US"})` returns `market_date`,
  `is_trading_day`, `last_completed_trading_day`, `expected_trading_date`,
  `market_timezone` and `market_now`. Use `expected_trading_date` as the day a
  published run should carry: while the market trades it is **today's session**,
  not `last_completed_trading_day`. A failed calendar answers `isError` with
  `calendar_unavailable`; there is no date to read then.
- `earnings_lookup({"symbol": "AAPL"})` returns the stored observation with
  `freshness_status` (`ok`/`stale`/`invalid`/`unknown`), `available`,
  `date_confirmation` (always `unknown`) and `reasons`. A current observation
  that reports no upcoming date is `available: true` with
  `no_upcoming_earnings_date`; a stale or missing one is not evidence.

## Tool selection guide

### After-close brief

1. `mcp_stockscreen_market_market_overview`
2. `mcp_stockscreen_market_compare_feature_runs`
3. `mcp_stockscreen_market_theme_state`
4. `mcp_stockscreen_market_task_status`
5. `mcp_stockscreen_market_watchlist_snapshot` for each requested watchlist
6. Optional: `mcp_stockscreen_market_explain_symbol` for the most important mover

Output shape:

- Market State
- Run Diff
- Themes
- Watchlists
- Risks
- Next Actions

### Run-diff analyst

Start with `mcp_stockscreen_market_compare_feature_runs`. Highlight:

- `summary_stats.upgraded_count`
- `summary_stats.downgraded_count`
- `summary_stats.avg_score_change`
- top `movers`
- any `added` or `removed` names that change the character of the run

### Watchlist steward

1. `mcp_stockscreen_market_watchlist_snapshot`
2. `mcp_stockscreen_market_find_candidates` with either broad filters or the watchlist name as the `universe`
3. `mcp_stockscreen_market_explain_symbol` for the top proposed add
4. `mcp_stockscreen_market_watchlist_add` only after explicit user approval and only if writes are enabled

### Theme scout

1. `mcp_stockscreen_market_theme_state(theme_name=?, limit=?)`
2. `mcp_stockscreen_market_find_candidates` filtered to the relevant industry group or sector if you need a candidate list
3. `mcp_stockscreen_market_explain_symbol` for the strongest theme constituents

## Guardrails

- Prefer read-only analysis unless the user clearly asks for a write.
- Treat missing breadth, failed tasks, or missing runs as first-class outputs, not hidden footnotes.
- Keep conclusions anchored to the latest published run, not generic market priors.
