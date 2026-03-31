# Hermes Market Copilot

This integration exposes StockScreenClaude market state to [Hermes Agent MCP](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp/) and ships a Hermes skill pack that turns those tools into repeatable market workflows.

It is intentionally split in two:

- MCP server: deterministic tool access to market state, feature-run diffs, watchlists, themes, and task health.
- Hermes skill: workflow guidance for after-close briefs, run-diff analysis, watchlist stewardship, and theme scouting.

## What ships here

- Local stdio MCP server entrypoint: `backend/app/interfaces/mcp/server.py`
- Tool adapters: `backend/app/interfaces/mcp/market_copilot.py`
- Hermes skill directory: `integrations/hermes/skills/market-copilot/`

## Prerequisites

- Hermes Agent installed with MCP support. Hermes' MCP docs cover the expected `mcp_servers` config shape and local stdio setup.
- A working StockScreenClaude backend virtualenv at `backend/venv/`
- A reachable StockScreenClaude database via `DATABASE_URL`

## Backend configuration

In `backend/.env`, keep these settings explicit:

```dotenv
DATABASE_URL=sqlite:////ABS/PATH/TO/StockScreenClaude/data/stockscanner.db
MCP_SERVER_NAME=stockscreen-market-copilot
MCP_WATCHLIST_WRITES_ENABLED=false
```

`MCP_WATCHLIST_WRITES_ENABLED=false` is the safe default. Leave it off unless you want Hermes to be able to call `watchlist_add`.

## Manual smoke test

From the repo root:

```bash
cd backend
PYTHONPATH="$PWD" ./venv/bin/python -m app.interfaces.mcp.server
```

The process will wait on stdio for an MCP client. Stop it with `Ctrl+C` after confirming it starts cleanly.

## Hermes configuration

Add the MCP server and external skill directory to `~/.hermes/config.yaml`.

Keep the server key as `stockscreen_market`. Hermes prefixes MCP tools as `mcp_<server>_<tool>`, and the bundled skill assumes that exact prefix.

```yaml
mcp_servers:
  stockscreen_market:
    command: "/ABS/PATH/TO/StockScreenClaude/backend/venv/bin/python"
    args: ["-m", "app.interfaces.mcp.server"]
    env:
      PYTHONPATH: "/ABS/PATH/TO/StockScreenClaude/backend"
      DATABASE_URL: "sqlite:////ABS/PATH/TO/StockScreenClaude/data/stockscanner.db"
      MCP_SERVER_NAME: "stockscreen-market-copilot"
      MCP_WATCHLIST_WRITES_ENABLED: "false"
    tools:
      include:
        - market_overview
        - compare_feature_runs
        - find_candidates
        - explain_symbol
        - watchlist_snapshot
        - theme_state
        - task_status
        - watchlist_add
      prompts: false
      resources: false

skills:
  external_dirs:
    - /ABS/PATH/TO/StockScreenClaude/integrations/hermes/skills
```

If you change the server key from `stockscreen_market`, Hermes will register different prefixed tool names and you will need to update the skill instructions accordingly.

After saving the config:

```bash
hermes chat
```

If Hermes is already running, use `/reload-mcp` after updating the config.

## Included MCP tools

The server exposes these v1 tools:

- `market_overview(as_of_date?)`
- `compare_feature_runs(run_a?, run_b?, limit=25)`
- `find_candidates(filters, universe?, limit=25)`
- `explain_symbol(symbol, depth="brief"|"full")`
- `watchlist_snapshot(watchlist)`
- `theme_state(theme_name?, limit=10)`
- `task_status(task_name?)`
- `watchlist_add(watchlist, symbols, reason?)`

Every tool returns the same top-level envelope:

- `summary`
- `facts`
- `citations`
- `freshness`
- `next_actions`

## Using the Hermes skill

Once the external skill directory is registered, Hermes can load the included skill as:

```text
/market-copilot
```

Example prompts:

- `/market-copilot Write an after-close brief for my Leaders and Breakouts watchlists.`
- `/market-copilot Compare the latest two feature runs and explain the biggest movers.`
- `/market-copilot Review the AI Infrastructure theme and tell me which symbols deserve deeper work.`

## Weekday after-close automation

Hermes cron jobs run in fresh sessions, so the prompt needs to carry its own workflow. After setting a home channel in Hermes, a good recurring prompt is:

```text
Every weekday at 6:15pm America/New_York, use the market-copilot skill with the stockscreen_market MCP tools to prepare an after-close brief. Start with market_overview, compare the latest two published feature runs with limit 10, inspect theme_state with limit 5, check task_status, and snapshot the Leaders and Breakouts watchlists. Write five short sections: Market State, Run Diff, Themes, Watchlists, Risks, Next Actions. If freshness is stale or any scheduled task failed, lead with that caveat.
```

That prompt follows Hermes' scheduled-task model: fresh session, explicit instructions, and delivery to the configured home channel.
