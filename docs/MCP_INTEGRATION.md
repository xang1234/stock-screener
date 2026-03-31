# MCP Integration

StockScreenClaude exposes a [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server for AI copilot workflows. When connected to an MCP-capable agent (like [Hermes](https://hermes-agent.nousresearch.com/)), the agent can query market state, compare feature runs, inspect watchlists and themes, and prepare automated market briefs.

## Available Tools

The MCP server exposes 8 tools:

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

Every tool returns a structured envelope with `summary`, `facts`, `citations`, `freshness`, and `next_actions`.

## Example Workflows

### After-Close Market Brief
Query `market_overview`, compare the latest two feature runs, snapshot your watchlists, and check theme state. The agent synthesizes a brief covering market health, notable movers, theme momentum, and risks.

### Feature Run Comparison
Use `compare_feature_runs` to see which stocks gained or lost the most score between two daily runs. Useful for identifying breakouts, breakdowns, and emerging setups.

### Watchlist Stewardship
Combine `watchlist_snapshot` with `explain_symbol` to audit your watchlist. The agent identifies which positions still meet screening criteria and which have deteriorated.

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
