---
name: market-copilot
description: Use when Hermes is connected to StockScreenClaude through the `stockscreen_market` MCP server and needs to produce market briefs, compare feature runs, inspect watchlists and themes, explain symbols, or propose watchlist additions.
version: 1.0.0
metadata:
  hermes:
    category: finance
    tags: [stocks, investing, mcp, watchlists, themes]
    requires_tools:
      - mcp_stockscreen_market_market_overview
      - mcp_stockscreen_market_compare_feature_runs
      - mcp_stockscreen_market_watchlist_snapshot
---
# Market Copilot

## When to Use

Use this skill when the task is about StockScreenClaude market state rather than generic market commentary.

Typical triggers:

- After-close market briefs
- Latest feature-run diff analysis
- Watchlist review or watchlist candidate proposals
- Theme inspection and alert triage
- Symbol explanation from the latest published run

Read `references/mcp-tools.md` when you need the exact MCP tool names or the workflow order for a specific job.

## Procedure

1. Start with `mcp_stockscreen_market_market_overview` unless the user asked for one narrow object only.
2. Pull the next tool that matches the task:
   - run diffs: `mcp_stockscreen_market_compare_feature_runs`
   - candidate search: `mcp_stockscreen_market_find_candidates`
   - one-symbol analysis: `mcp_stockscreen_market_explain_symbol`
   - watchlist review: `mcp_stockscreen_market_watchlist_snapshot`
   - theme review: `mcp_stockscreen_market_theme_state`
   - pipeline health: `mcp_stockscreen_market_task_status`
3. Build the answer from structured fields first: `summary`, `facts`, `freshness`, `citations`, and `next_actions`.
4. Treat stale freshness and failed scheduled tasks as gating signals. Lead with those caveats before making recommendations.
5. Keep the workflow deterministic at the tool layer and do the synthesis in prose only after the needed tool calls are done.
6. Use `mcp_stockscreen_market_watchlist_add` only when the user explicitly wants a watchlist changed and the tool response shows writes are enabled.

## Workflow Rules

- After-close brief: lead with market state, then run diff, then themes, then watchlists, then risks and next actions.
- Run-diff analysis: quantify adds, removals, movers, and upgraded or downgraded names before interpreting them.
- Watchlist stewardship: snapshot the watchlist first, then use `find_candidates` or `explain_symbol` to justify adds.
- Theme scouting: prefer `theme_state` before symbol-level explanation so the theme narrative comes from current metrics and alerts.

## Pitfalls

- Do not present the data as live unless the `freshness` block supports that claim.
- Do not assume `watchlist_add` will work. The server defaults to read-only.
- Do not skip `task_status` when the request depends on pipeline freshness or scheduled jobs.
- Do not replace tool facts with unsupported market opinions.

## Verification

- The answer should mention the relevant as-of date or freshness signal.
- Any recommendation to add or monitor a symbol should be traceable to at least one MCP tool result.
- If data is missing or stale, the response should surface the tool-provided `next_actions` instead of guessing.
