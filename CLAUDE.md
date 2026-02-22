# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stock screening platform implementing CANSLIM (William O'Neil) and Minervini methodologies, with theme discovery, AI chatbot, and market analysis. Full-stack application with FastAPI backend, React frontend, SQLite database, Redis caching, and Celery for background tasks.

## Development Commands

### Backend
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm run dev      # Vite dev server on :5173
npm run build    # Production build
npm run lint     # ESLint
```

### Celery Workers (required for scans)
```bash
cd backend
./start_celery.sh    # Starts both queues

# Or manually:
./venv/bin/celery -A app.celery_app worker --pool=solo -Q celery -n general@%h
./venv/bin/celery -A app.celery_app worker --pool=solo -Q data_fetch -n datafetch@%h
./venv/bin/celery -A app.celery_app beat --loglevel=info  # Scheduler
```

### Docker Deployment

Layered Docker Compose architecture with three scenarios:

```bash
# Local development
cp .env.docker.example .env   # Add API keys for chatbot
docker-compose up

# Homelab (behind reverse proxy like Traefik/nginx proxy manager)
cp .env.docker.example .env.docker
# Edit: CORS_ORIGINS=https://stocks.home.lan
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# VPS with auto-HTTPS (Hostinger, DigitalOcean, etc.)
cp .env.docker.example .env.docker
# Edit: DOMAIN=stocks.yourdomain.com, CORS_ORIGINS=https://stocks.yourdomain.com
docker-compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.https.yml up -d
```

**Docker files:**
- `docker-compose.yml` - Base config (local dev)
- `docker-compose.prod.yml` - Production overlay (resource limits, health checks, logging)
- `docker-compose.https.yml` - HTTPS overlay (Caddy with Let's Encrypt)
- `.env.docker.example` - Docker environment template
- `Caddyfile` - Caddy TLS configuration

**Note:** Backend runs as non-root user (uid 1000). After upgrade: `sudo chown -R 1000:1000 ./data`

### Running Tests

#### Backend (pytest)
```bash
cd backend
source venv/bin/activate

# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests (requires running server at localhost:8000)
pytest tests/integration/ -m integration

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_canslim_scanner.py
```

#### Frontend (Vitest + React Testing Library)
```bash
cd frontend

# Run all tests once (CI mode)
npm run test:run

# Run tests in watch mode (development)
npm run test

# Run a specific test file
npx vitest run src/components/Scan/ResultsTable.test.jsx

# Lint test files
npm run lint
```

**Note:** Vitest 4.x requires Node 18+. On this machine, the system Node is v14 — use NVM to activate Node 22:
```bash
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
```

**Test file locations:**
- `frontend/src/components/Scan/` — component tests (ResultsTable, FilterPanel, SetupEngineDrawer)
- `frontend/src/components/Scan/filters/` — filter sub-component tests (CompactRangeInput, CompactCheckbox)
- `frontend/src/utils/` — utility tests (formatUtils)
- `frontend/src/test/fixtures/` — shared test fixtures
- `frontend/src/test/renderWithProviders.jsx` — MUI ThemeProvider test wrapper

### Diagnostic Scripts
Utility scripts are in `backend/scripts/`:
```bash
cd backend
source venv/bin/activate

python scripts/inspect_redis.py            # Inspect Redis cache keys
python scripts/cache_diagnostic.py         # Trace cache flow (DB → Redis)
python scripts/check_cache_status.py       # Check price cache status
python scripts/clear_redis_price_cache.py  # Clear Redis cache after config change
python scripts/force_full_cache_refresh.py # Force full cache refresh
python scripts/cleanup_orphaned_scans.py   # Clean up stale scans, VACUUM DB
```

## Architecture

### Backend Structure
- **FastAPI** application in `backend/app/main.py`
- **API routes** in `backend/app/api/v1/` - RESTful endpoints for stocks, scans, themes, chatbot, signals
- **Services** in `backend/app/services/` - Business logic layer (70+ service files)
- **Scanners** in `backend/app/scanners/` - Stock screening implementations
- **Tasks** in `backend/app/tasks/` - Celery background tasks
- **Models** in `backend/app/models/` - SQLAlchemy ORM models
- **Schemas** in `backend/app/schemas/` - Pydantic request/response schemas

### Key Architectural Patterns

**Multi-Screener Orchestrator** (`scanners/scan_orchestrator.py`):
- Coordinates Minervini, CANSLIM, IPO, Volume Breakthrough, and Custom screeners
- All screeners extend `BaseStockScreener` abstract class
- Data fetched once and shared across screeners
- Composite scoring via configurable aggregation (weighted_average, maximum, minimum)

**Two-Queue Celery Architecture**:
- `celery` queue: General compute tasks (4 workers local, 1 in Docker for SQLite safety)
- `data_fetch` queue: API calls (1 worker, serialized to respect rate limits)
- All external API tasks route to `data_fetch` to prevent rate limit violations

**Redis Caching Strategy** (three-tier: Redis → SQLite → API):
- DB 0: Celery broker
- DB 1: Celery results (24h TTL, auto-cleanup)
- DB 2: Application cache via shared connection pool (`services/redis_pool.py`)
  - Price cache: 7d TTL, stores 5 years of OHLCV (required for Volume Breakthrough Scanner)
  - Fundamentals cache: 7d TTL with DB fallback on Redis miss
  - Benchmark (SPY): 24h TTL with distributed locking to prevent thundering herd

**LLM Integration** (`services/chatbot/`):
- Multi-provider support: Groq, DeepSeek, Together AI, OpenRouter, Gemini
- Agent orchestrator with tool executor pattern
- Research mode with web search (Tavily, Serper, DuckDuckGo)

### Frontend Structure
- **React 18** with Vite
- **Material-UI** for components
- **React Query** (TanStack) for data fetching
- **TanStack Table** for results display
- Pages in `frontend/src/pages/`: ScanPage, ChatbotPage, ThemesPage, GroupRankingsPage, BreadthPage, SignalsPage

### Frontend API Client Convention

**CRITICAL: API paths must NOT include `/api` prefix**

The axios client in `frontend/src/api/client.js` handles the `/api` prefix via `baseURL`:
- **Local dev**: `baseURL = 'http://localhost:8000/api'`
- **Docker**: `baseURL = '/api'` (set via `VITE_API_URL` build arg)

When adding new API endpoints, use paths starting with `/v1/`:
```javascript
// ✅ CORRECT - path without /api prefix
const response = await apiClient.get('/v1/themes/rankings');

// ❌ WRONG - will cause double prefix in Docker (/api/api/v1/...)
const response = await apiClient.get('/api/v1/themes/rankings');
```

For API modules with a `BASE_PATH` constant:
```javascript
// ✅ CORRECT
const BASE_PATH = '/v1/user-themes';

// ❌ WRONG
const BASE_PATH = '/api/v1/user-themes';
```

**Why this matters**: In Docker, nginx proxies `/api/*` to the backend. If paths include `/api`, you get `/api/api/v1/...` which returns 404.

## Data Sources & Rate Limits
- **yfinance**: 1 req/sec (self-imposed)
- **Finviz**: Rate-limited via wrapper
- **Alpha Vantage**: 25 req/day free tier
- **SEC EDGAR**: 10 req/sec (150ms between requests)

## Environment Variables

**Local development**: `backend/.env` (see `backend/.env.example`)
**Docker deployment**: `.env.docker` in project root (see `.env.docker.example`)

**Required for chatbot** (at least one LLM provider):
- `GROQ_API_KEY`, `GROQ_API_KEYS` - Groq (fast inference, free tier)
- `GEMINI_API_KEY` / `GOOGLE_API_KEY` - Google Gemini (theme extraction)
- `DEEPSEEK_API_KEY` - DeepSeek (cost-effective fallback)
- `TOGETHER_API_KEY` - Together AI (wide model selection)
- `OPENROUTER_API_KEY` - OpenRouter (100+ models)

**Web search** (enables research mode):
- `TAVILY_API_KEY`, `SERPER_API_KEY`

**Data sources**:
- `ALPHA_VANTAGE_API_KEY` - Fundamental data (25 req/day free tier)

**Infrastructure**:
- `DATABASE_URL` - **Must use absolute path** (e.g., `sqlite:////Users/admin/StockScreenClaude/data/stockscanner.db`)
- `REDIS_HOST`, `CELERY_BROKER_URL` - Redis/Celery configuration
- `CORS_ORIGINS` - Comma-separated allowed origins (for production)

**LLM routing** (optional):
- `LLM_DEFAULT_PROVIDER` - Primary provider: groq, deepseek, together_ai, openrouter, gemini
- `LLM_CHATBOT_MODEL`, `LLM_RESEARCH_MODEL` - Model overrides (LiteLLM format)
- `LLM_FALLBACK_ENABLED` - Enable automatic provider fallback

## Database

**CRITICAL: Database Location**
- **Production database**: `data/stockscanner.db` (project root) - 2.7GB with all stock data, chat history, themes
- **DO NOT** create or use databases in `backend/data/`, `frontend/data/`, or any other location
- The `.env` file uses an **absolute path** to prevent working-directory issues:
  ```
  DATABASE_URL=sqlite:////Users/admin/StockScreenClaude/data/stockscanner.db
  ```
- If you see an empty database or missing data, verify `DATABASE_URL` points to the correct absolute path

**Valid database files:**
- `data/stockscanner.db` - Main application database (KEEP)
- `backend/celerybeat-schedule.db` - Celery Beat scheduler state (KEEP)

**Key tables:**
- `stock_prices`, `stock_fundamentals`, `stock_universe` - Core stock data
- `scans`, `scan_results` - Scan metadata and results with multi-screener scores
- `ibd_groups`, `ibd_group_ranks` - Industry group rankings
- `theme_clusters`, `theme_constituents` - Theme discovery
- `signals` - Technical signal detections
- `chat_sessions`, `chat_messages` - Chatbot conversation history

Migrations in `backend/migrations/`. SQLite for both development and Docker deployment.

## Screening Methodologies

**Minervini Template**: RS Rating > 70-80, Stage 2 uptrend, MA alignment (50 > 150 > 200), price 30%+ above 52-week low

**CANSLIM**: Current quarterly EPS > 25%, Annual EPS growth > 25% 3yr, new highs, volume patterns, RS > 70, institutional ownership 40-70%

## macOS Development Note
For Celery on macOS, use `--pool=solo` (set via `start_celery.sh`) to avoid fork() crashes with curl_cffi. Also set:
```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export TOKENIZERS_PARALLELISM=false
```

## Git Conventions

This project uses **Conventional Commits** for all commit messages. Format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring (no feature or fix)
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Build system or dependency changes
- `ci`: CI/CD configuration changes
- `chore`: Maintenance tasks (deps, configs)

**Examples:**
```
feat(scanner): add volume breakthrough screener
fix(chatbot): handle empty response from LLM provider
docs: update API endpoint documentation
refactor(api): consolidate stock data fetching logic
test(canslim): add unit tests for EPS calculation
```

**Scopes** (optional): `api`, `scanner`, `chatbot`, `frontend`, `celery`, `db`, `cache`, `themes`, `signals`
