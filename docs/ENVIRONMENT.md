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

Multiple keys for load balancing: `GROQ_API_KEYS=key1,key2,key3` (comma-separated).

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

## Docker Deployment

| Variable | Example | Description |
|----------|---------|-------------|
| `DOMAIN` | `stocks.yourdomain.com` | For HTTPS/Caddy scenario only |
| `CORS_ORIGINS` | `https://stocks.yourdomain.com` | Must match your access URL |
| `SERVER_AUTH_PASSWORD` | `choose-a-long-random-password` | Required shared password for server login |
| `BACKEND_IMAGE` | `ghcr.io/you/stockscreenclaude-backend` | GHCR image (release overlay) |
| `FRONTEND_IMAGE` | `ghcr.io/you/stockscreenclaude-frontend` | GHCR image (release overlay) |
| `APP_IMAGE_TAG` | `v1.2.3` | Release tag to deploy |

## Twitter/X Ingestion

| Variable | Default | Description |
|----------|---------|-------------|
| `X_INGEST_PROVIDER` | `official` | Twitter/X ingestion provider: `official` or `xui` |
| `TWITTER_BEARER_TOKEN` | empty | Official X API bearer token |
| `X_API_MAX_PAGES_PER_SOURCE` | `10` | Official X API pagination cap per source fetch; hitting the cap fails the source without advancing its checkpoint |
| `XUI_ENABLED` | `false` | Legacy repo-local xui-reader flag; private xui override is selected by `X_INGEST_PROVIDER=xui` |
| `XUI_CONFIG_PATH` | Platform app-data dir locally; `/app/data/xui-reader/config.toml` in Docker | Path to xui session state for bridge/local tooling |
| `XUI_PROFILE` | `default` | xui session profile |
| `XUI_LIMIT_PER_SOURCE` | `50` | Max items per source fetch |
| `XUI_NEW_ONLY` | `true` | Private xui new-item mode |
| `XUI_CHECKPOINT_MODE` | `auto` | Private xui checkpoint behavior |
| `XUI_BRIDGE_ENABLED` | `false` | Enable browser session bridge in xui mode |
| `XUI_BRIDGE_ALLOWED_ORIGINS` | See `.env.example` | Allowed CORS origins for xui bridge |
| `XUI_BRIDGE_CHALLENGE_TTL_SECONDS` | `120` | xui bridge challenge TTL |
| `XUI_BRIDGE_MAX_COOKIES` | `300` | Max cookies to accept for xui bridge |
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
| `YFINANCE_RATE_LIMIT` | `1` | yfinance requests per second |
| `ALPHAVANTAGE_RATE_LIMIT` | `25` | Alpha Vantage requests per day |
