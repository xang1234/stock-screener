# Environment Variables Reference

StockScreenClaude uses two environment files depending on deployment mode:
- **Local development:** `backend/.env` (see `backend/.env.example`)
- **Docker deployment:** `.env` in the project root (see `.env.docker.example`)

## LLM API Keys

At least one LLM provider key is required for assistant and theme extraction workflows. Scanning and other features work without API keys.

| Provider | Env Var | Get Key | Notes |
|----------|---------|---------|-------|
| Groq | `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Fast inference, free tier (recommended to start) |
| Z.AI | `ZAI_API_KEY` | [platform.z.ai](https://platform.z.ai) | GLM models |
| Minimax | `MINIMAX_API_KEY` | [platform.minimax.io](https://platform.minimax.io) | Default for theme extraction |
| OpenCode Go | `OPENCODE_GO_API_KEY` | [opencode.ai](https://opencode.ai/docs/go/) | Default `deepseek-v4-flash` provider for Social Signal extraction |
| Ollama | `OLLAMA_API_KEY` | [ollama.com](https://ollama.com) | Open-weight models; set `OLLAMA_API_BASE` for a local daemon (no key needed) |

Multiple keys for load balancing: `GROQ_API_KEYS=key1,key2,key3` (comma-separated).

### Ollama

`OLLAMA_API_BASE` is a bare host, not a `/v1` URL — LiteLLM appends `/api/chat` itself:

| Target | `OLLAMA_API_BASE` | `OLLAMA_API_KEY` |
|---|---|---|
| Ollama Cloud | `https://ollama.com` (default) | required |
| Local daemon | `http://ollama:11434` | not needed |

Select a model in the UI as `ollama/<model>`, for example `ollama/deepseek-v4.1-flash`.

## Web Search Keys (Optional)

Enables assistant web research fallback.

| Provider | Env Var | Get Key |
|----------|---------|---------|
| Tavily | `TAVILY_API_KEY` | [tavily.com](https://tavily.com) |
| Serper | `SERPER_API_KEY` | [serper.dev](https://serper.dev) |

## Data Source Keys

| Source | Env Var | Get Key | Notes |
|--------|---------|---------|-------|
| Alpha Vantage | `ALPHA_VANTAGE_API_KEY` | [alphavantage.co](https://www.alphavantage.co/support/#api-key) | Free tier: 25 req/day |

## Database

### PostgreSQL

| Variable | Local Default | Docker Default | Description |
|----------|---------------|----------------|-------------|
| `POSTGRES_DB` | `stockscanner` | `stockscanner` | Database name |
| `POSTGRES_USER` | `stockscanner` | `stockscanner` | Database user |
| `POSTGRES_PASSWORD` | `stockscanner` | `stockscanner` | Database password |
| `DATABASE_URL` | `postgresql://stockscanner:stockscanner@localhost:5432/stockscanner` | `postgresql://stockscanner:stockscanner@postgres:5432/stockscanner` | Full connection string |

## Redis / Celery

| Variable | Local Default | Docker Default | Description |
|----------|---------------|----------------|-------------|
| `REDIS_HOST` | `localhost` | `redis` | Redis hostname |
| `REDIS_PORT` | `6379` | `6379` | Redis port |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | `redis://redis:6379/0` | Celery broker |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/1` | `redis://redis:6379/1` | Celery result backend |
| `CELERY_TIMEZONE` | `America/New_York` | `America/New_York` | Timezone for scheduled tasks |

## Server

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port |
| `CORS_ORIGINS` | `http://localhost:5173` (local) | Comma-separated allowed origins |
| `SERVER_AUTH_PASSWORD` | (empty) | Required for browser login in server/Docker deployments |
| `SERVER_AUTH_SESSION_SECRET` | (empty) | Optional cookie-signing secret; defaults to `SERVER_AUTH_PASSWORD` |
| `SERVER_AUTH_SECURE_COOKIE` | `false` | Force Secure auth cookies; set `true` when TLS terminates at a trusted HTTPS proxy |
| `SERVER_EXPOSE_API_DOCS` | `false` | Keep `/docs`, `/redoc`, and `/openapi.json` disabled unless you explicitly need them |
| `ADMIN_API_KEY` | (empty) | Required for `/api/v1/config/*` endpoints |
| `ADMIN_PRINCIPAL_ID` | (empty) | Stable audit identity bound to `ADMIN_API_KEY`, and the only identity granted Economic Taxonomy review/publication authority. Without it, admin endpoints keep working but are audited as `admin:unbound-api-key` (a warning is logged once) and taxonomy publication fails closed |

## Docker Deployment

| Variable | Example | Description |
|----------|---------|-------------|
| `DOMAIN` | `stocks.yourdomain.com` | For HTTPS/Caddy scenario only |
| `CORS_ORIGINS` | `https://stocks.yourdomain.com` | Must match your access URL |
| `SERVER_AUTH_PASSWORD` | `choose-a-long-random-password` | Required shared password for server login |
| `BACKEND_IMAGE` | `ghcr.io/you/stockscreenclaude-backend` | GHCR image (release overlay) |
| `FRONTEND_IMAGE` | `ghcr.io/you/stockscreenclaude-frontend` | GHCR image (release overlay) |
| `APP_IMAGE_TAG` | `v1.2.3` | Release tag to deploy |

### Container orchestration

The backend runs its Alembic migrations inside the application lifespan, blocking, before
uvicorn accepts HTTP. While `backend.healthcheck.start_period` runs, a failing probe does not
count towards `retries`. The whole Celery tier declares `condition: service_healthy` on the
backend, so a backend that stays unhealthy past that grace blocks the **start** of the worker
tier; it does not stop workers that are already running.

`start_period` is set well above the longest expected migration (`900s` in `docker-compose.yml`).
**That value alone is not sufficient.** An orchestrator applies its own, independent wait:

| Setting | Where | Value |
|---------|-------|-------|
| `backend.healthcheck.start_period` | `docker-compose.yml` | `900s` |
| `deployWaitTimeout` | Arcane, project settings | `1200` |

Two deadlines can end the wait, and **the health check is usually the earlier one**:

```
start_period + interval * retries   900 + 30 * 3 = 990 s   (never a successful probe)
deployWaitTimeout                                        1200 s
```

So `deployWaitTimeout` must cover the expected time to healthy — it does not have to be
reached, and it is not automatically the first limit. Setting it below the health-check
deadline would cap the grace for no reason, which is why the two are ordered this way.

Measured on a QNAP TS-473A, container start to first successful `/readyz`:

```
15:34:17   container created, uvicorn parent started 15:34:19
15:34:43   migrations 20260925_0058 .. 20260926_0060, ~6 s total
15:34:53   first /readyz 200          -> 36 s
```

Four probes failed before that, all inside the grace. The 36 s is not representative of a
schema-changing revision: revision `20260926_0058` alone (an index over a 216 MB table) took
**519 s** of migration time on this host, which is what `900s` is sized against. That figure
is migration time from the Alembic log, not startup time — size the limits against the full
interval from container start to the first successful `/readyz`.

Before this change the grace was `30 s` and the wait `600`, so the deploy aborted after 93 s
and could not recover.

Docker Compose itself needs no such pairing **when `docker compose up` is run without
`--wait`**. That option (`up --wait --wait-timeout N`) adds its own deadline for services to
become running or healthy — configure it the same way when it is used. The table above matters
whenever the stack is driven by an orchestrator such as Arcane, Portainer, or a CI deploy step;
set the equivalent "time to healthy" limit there.

If a migration is expected to outlive both limits, the durable fix is to run migrations as a
dedicated step before the API starts, rather than extending the grace further.

## Twitter/X Ingestion

| Variable | Default | Description |
|----------|---------|-------------|
| `X_INGEST_PROVIDER` | `official` | Twitter/X ingestion provider: `official` or `xui` (external private package) |
| `TWITTER_BEARER_TOKEN` | empty | Official X API bearer token |
| `X_API_MAX_PAGES_PER_SOURCE` | `10` | Official X API pagination cap per source fetch; hitting the cap ingests the fetched partial window and advances the checkpoint to the highest observed post id |
| `X_API_MAX_RESULTS_PER_PAGE` | `50` | Official X API page size, clamped to X API bounds of 5-100 |
| `XUI_LIMIT_PER_SOURCE` | `50` | Max items per source fetch (consumed by the external private `xui` provider when `X_INGEST_PROVIDER=xui`) |
| `TWITTER_REQUEST_DELAY` | `5.0` | Delay between fetches (seconds) |

For the private override, install the private package locally with
`pip install git+ssh://git@github.com/xang1234/xui.git` and set
`X_INGEST_PROVIDER=xui`.

## MCP Server

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SERVER_NAME` | `stockscreen-market-copilot` | MCP server identity |
| `MCP_WATCHLIST_WRITES_ENABLED` | `false` | Allow MCP clients to modify watchlists |

## Advanced

| Variable | Default | Description |
|----------|---------|-------------|
| `PRICE_CACHE_TTL` | `604800` | Price cache TTL in seconds (7 days) |
| `FUNDAMENTAL_CACHE_TTL` | `604800` | Fundamentals cache TTL (7 days) |
| `QUARTERLY_CACHE_TTL` | `2592000` | Quarterly data cache TTL (30 days) |
| `DATA_FETCH_LOCK_WAIT_SECONDS` | `7200` | Max wait for data fetch lock |
| `SETUP_ENGINE_ENABLED` | `true` | Feature flag for Setup Engine scanner |

## Scanning

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_UNIVERSE` | `all` | Default scan universe |
| `SCAN_BATCH_SIZE` | `20` | Batch size for scan processing |
| `SCAN_COMPUTE_PROCESSES` | `0` | Processes that compute scan and daily feature-snapshot results. `0` = automatic: the CPUs the worker container may use (cgroup CPU quota aware), capped at 4, and in-process on macOS/Windows. `1` = in-process. `N` = up to `N`. Each process adds memory, and a worker capped below 2 CPUs (such as the `docker-compose.prod.yml` scan workers at `cpus: '0.5'`) computes in-process; raise that worker's `cpus` and `memory` limits to benefit. |
| `STATIC_SNAPSHOT_PARALLEL_WORKERS` | `8` | Upper bound on compute processes for static-site and bootstrap cache-only snapshots (also capped at available CPUs) |
| `YFINANCE_RATE_LIMIT` | `1` | yfinance requests per second |
| `ALPHAVANTAGE_RATE_LIMIT` | `25` | Alpha Vantage requests per day |
