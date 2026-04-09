# Hermes Assistant Integration

StockScreenClaude now uses Hermes as a **private assistant runtime** behind the backend. The browser never calls Hermes directly. The backend owns auth, transcript persistence, SSE normalization, and watchlist confirmation flows.

The integration has three parts:

- Backend proxy: `/api/v1/assistant/*`
- Authenticated MCP data plane: `/mcp`
- Hermes skills: `integrations/hermes/skills/market-copilot/`

## Deployment posture

- Supported database: PostgreSQL
- Supported app posture: single-tenant server deployment
- Supported assistant posture: Hermes runs on the same host or Docker network as the backend
- Not supported: browser-direct Hermes access, desktop-mode Hermes wiring, repo/file/shell access in the product assistant path

## Backend configuration

In `backend/.env` or your Docker env source, keep these explicit:

```dotenv
DATABASE_URL=postgresql://user:password@localhost/stockscanner
SERVER_AUTH_PASSWORD=replace-me
HERMES_API_BASE=http://127.0.0.1:8642/v1
HERMES_API_KEY=replace-me
HERMES_MODEL=hermes-agent
HERMES_REQUEST_TIMEOUT_SECONDS=120
MCP_SERVER_NAME=stockscreen-market-copilot
MCP_WATCHLIST_WRITES_ENABLED=false
MCP_HTTP_ENABLED=true
```

`MCP_WATCHLIST_WRITES_ENABLED=false` remains the safe default. The product assistant does not mutate watchlists directly; confirmed watchlist writes still go through the app backend.

## Hermes configuration

Hermes should call the backend’s authenticated HTTP MCP transport, not the old stdio-only path. A sample config template is included at [config.yaml.example](config.yaml.example).

Important points:

- Keep Hermes private on the Docker network or localhost.
- Configure Hermes to call `http://backend:8000/mcp` in Docker or `http://127.0.0.1:8000/mcp` locally.
- Send the same `SERVER_AUTH_PASSWORD` as an `x-server-auth` header so Hermes can reach the protected MCP transport.
- Register the external skills directory so Hermes can load `market-copilot`.

For local development, prefer the repo helper instead of hand-editing `~/.hermes`:

```bash
bash scripts/run_local_hermes_gateway.sh
```

That script:

- sources the repo `.env` and `backend/.env`
- runs Hermes with `HERMES_HOME=data/hermes`
- generates `data/hermes/config.yaml` with the local MCP URL `http://127.0.0.1:8000/mcp`
- starts the OpenAI-compatible Hermes API on `http://127.0.0.1:8642/v1`

## Docker Compose

`docker-compose.yml` now includes an opt-in Hermes sidecar under the `assistant` profile:

```bash
docker compose --profile assistant up -d
```

The service is internal-only:

- no browser-facing port is published
- backend uses `HERMES_API_BASE=http://hermes:8642/v1`
- Hermes state lives under `data/hermes/`

Before enabling the profile, place a real Hermes config file in `data/hermes/config.yaml` using the example template and your actual secrets.

## Assistant runtime contract

The backend proxy sends Hermes a system prompt that tells it to:

- use StockScreenClaude MCP tools first for scans, themes, breadth, watchlists, and symbol posture
- use broader web research only when internal data is missing or stale
- separate internal signals from external web/news context
- mention freshness caveats explicitly
- stay in research-assistant mode and avoid certainty language

The frontend assistant lives at `/chatbot` for v1, but the UI is labeled **Assistant** and also appears as a global popup drawer.

## Included MCP tools

The current deterministic MCP surface is:

- `market_overview`
- `compare_feature_runs`
- `find_candidates`
- `explain_symbol`
- `watchlist_snapshot`
- `theme_state`
- `task_status`
- `watchlist_add`
- `group_rankings`
- `stock_lookup`
- `stock_snapshot`
- `breadth_snapshot`
- `daily_digest`

Every tool returns the same top-level envelope:

- `summary`
- `facts`
- `citations`
- `freshness`
- `next_actions`

## Manual smoke checks

Local backend + Hermes:

```bash
cd backend
PYTHONPATH="$PWD" ./venv/bin/python -m uvicorn app.main:app --reload
bash ../scripts/run_local_hermes_gateway.sh
curl -s -H "x-server-auth: $SERVER_AUTH_PASSWORD" http://127.0.0.1:8000/api/v1/assistant/health
```

MCP transport:

```bash
curl -s \
  -H "Content-Type: application/json" \
  -H "x-server-auth: $SERVER_AUTH_PASSWORD" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
  http://127.0.0.1:8000/mcp/
```

Full app:

- sign in through the normal server login
- open `/chatbot`
- ask for a scan/theme/breadth summary
- confirm the answer shows citations and tool activity
- open the global assistant drawer from a non-chat route
- test the watchlist preview flow from the latest assistant answer
