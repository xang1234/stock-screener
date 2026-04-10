# Architecture

Technical overview of StockScreenClaude's system design. For API reference and database schema details, see the [Backend README](../backend/README.md).

## System Overview

```
┌─────────────┐     ┌──────────────────────────────┐     ┌─────────────┐
│   Frontend   │────>│       Backend (FastAPI)       │────>│  PostgreSQL  │
│  React/Vite  │<────│  /api/v1/*                    │<────│              │
│   nginx      │     │                               │     └─────────────┘
└─────────────┘     │  ┌─────────────────────────┐  │     ┌─────────────┐
                     │  │    Celery Workers        │  │────>│    Redis     │
                     │  │  celery | data_fetch     │  │<────│  DB0: broker │
                     │  │  user_scans | beat       │  │     │  DB1: results│
                     │  └─────────────────────────┘  │     │  DB2: cache  │
                     └──────────────────────────────┘     └─────────────┘
```

## Layered Backend Architecture

Clean separation of concerns following Domain-Driven Design:

| Layer | Directory | Responsibility |
|-------|-----------|----------------|
| **Domain** | `domain/` | Business rules, ports (interfaces), value objects |
| **Use Cases** | `use_cases/` | Application services, orchestration |
| **Infrastructure** | `infra/` | SQLAlchemy repositories, Celery tasks, external APIs |
| **API** | `api/` | FastAPI routes, Pydantic schemas, HTTP concerns |

Dependency injection wired in `wiring/bootstrap.py`. Services layer (`services/`) contains 70+ business logic modules.

## Multi-Screener Orchestrator

The scan orchestrator (`scanners/scan_orchestrator.py`) coordinates all screener types:

- All screeners extend `BaseStockScreener` abstract class
- The `DataPreparationLayer` fetches data once and distributes it to all active screeners
- Composite scoring via configurable aggregation: `weighted_average`, `maximum`, or `minimum`
- Screener registry enables plugin-style scanner registration

### Available Screeners

| Screener | Key Criteria |
|----------|-------------|
| **Minervini Template** | RS > 70-80, Stage 2 uptrend, MA alignment (50>150>200), price 30%+ above 52-week low |
| **CANSLIM** | Quarterly EPS > 25%, Annual EPS growth > 25% (3yr), volume patterns, RS > 70 |
| **IPO Scanner** | Recent IPO status, momentum, volume/price action |
| **Volume Breakthrough** | Unusual volume spikes with confirming price action |
| **Setup Engine** | Base detection, breakout confirmation, Bollinger squeeze, RS line strength |
| **Custom Scanner** | 80+ configurable filters (price, volume, technicals, fundamentals, scores) |

## Celery Task Queues

Three-queue architecture to separate concerns and prevent rate limit violations:

| Queue | Workers | Purpose |
|-------|---------|---------|
| `celery` | 4 (local), 2 (Docker) | General compute tasks |
| `data_fetch` | 1 | External API calls (serialized to respect rate limits) |
| `user_scans` | 2 | User-triggered scan tasks |

All external API tasks route to `data_fetch` to prevent rate limit violations. Celery Beat handles scheduled tasks (daily refresh, breadth calculation, theme discovery).

## Redis Caching Strategy

Three-tier cache: Redis > PostgreSQL > External API.

| DB | Purpose | TTL |
|----|---------|-----|
| DB 0 | Celery broker | N/A |
| DB 1 | Celery results | 24h, auto-cleanup |
| DB 2 | Application cache | Varies |

Application cache (DB 2):
- **Price data:** 7-day TTL, stores 5 years of OHLCV (required for Volume Breakthrough Scanner)
- **Fundamentals:** 7-day TTL with database fallback on Redis miss
- **SPY benchmark:** 24-hour TTL with distributed locking to prevent thundering herd

Shared connection pool managed via `services/redis_pool.py`.

## Feature Store

Pre-computed daily stock snapshots stored in `stock_feature_daily`. A scheduled feature run scores every stock in the universe, then atomically publishes via pointer swap in `feature_run_pointers`.

Lifecycle: `RUNNING` > `COMPLETED` > quality checks > `PUBLISHED` (or `QUARANTINED`)

Scan API endpoints read from the latest published run for fast query responses.

## Database

| Deployment | Database | Notes |
|-----------|----------|-------|
| Local dev / Docker | PostgreSQL | Configured via `DATABASE_URL` / `POSTGRES_*` env vars |

### Key Tables by Category

**Core Stock Data:**
`stock_prices`, `stock_fundamentals`, `stock_universe`, `stock_technicals`

**Feature Store:**
`feature_runs`, `stock_feature_daily`, `feature_run_pointers`

**Scanning:**
`scans`, `scan_results`

**Market Analysis:**
`market_breadth`, `ibd_industry_groups`, `ibd_group_ranks`

**Themes:**
`theme_clusters`, `theme_constituents`, `theme_metrics`, `content_sources`, `content_items`

**User Data:**
`user_watchlists`, `watchlist_items`, `user_themes`, `user_theme_stocks`, `filter_presets`

**Assistant:**
`chatbot_conversations`, `chatbot_messages`

## Application Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Routine | Market dashboard with Key Markets, Themes, Watchlists, Stockbee tabs |
| `/scan` | Bulk Scanner | Multi-screener scanning with 80+ filters and CSV export |
| `/breadth` | Market Breadth | StockBee-style breadth indicators and trends |
| `/groups` | Group Rankings | IBD industry group rankings with movers |
| `/themes` | Themes | AI-powered theme discovery with trending/emerging detection |
| `/chatbot` | Assistant | Hermes-backed assistant with internal market tools and web context |
| `/stock/:symbol` | Stock Detail | Individual stock analysis with charts and fundamentals |

## API Endpoint Groups

Interactive Swagger docs available at `http://localhost:8000/docs`.

**Core:**
- `/api/v1/scans` — Scan management and results
- `/api/v1/stocks` — Stock data, fundamentals, chart data
- `/api/v1/features` — Feature store management

**Market Analysis:**
- `/api/v1/breadth` — Market breadth indicators
- `/api/v1/groups` — IBD group rankings
- `/api/v1/themes` — Theme discovery and analysis
- `/api/v1/technical` — Technical indicators

**AI & Research:**
- `/api/v1/assistant` — Assistant sessions, streaming, health, and watchlist preview

**User Data:**
- `/api/v1/user-watchlists` — Watchlist management
- `/api/v1/user-themes` — User theme management
- `/api/v1/market-scan` — Dashboard market scan lists
- `/api/v1/filter-presets` — Saved scan filter configurations

**System:**
- `/api/v1/universe` — Stock universe management
- `/api/v1/cache` — Cache management
- `/api/v1/tasks` — Background task status
- `/api/v1/config` — Admin configuration
- `/api/v1/data-fetch-status` — Data fetch monitoring
- `/api/v1/ticker-validation` — Ticker symbol validation
- `/api/v1/app-runtime` — App capabilities and runtime info

**Health Endpoints (root-level):**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/livez` | GET | Liveness probe (zero dependencies) |
| `/readyz` | GET | Readiness probe (checks DB + Redis) |
| `/health` | GET | Deprecated alias for `/readyz` |
