# Development Guide

Full from-source setup for contributors and developers on any platform.

## Prerequisites

- Python 3.11+
- Node.js 18+ (use [NVM](https://github.com/nvm-sh/nvm) if your system Node is older)
- Redis server (local or via Docker)

## Backend Setup

```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e ../xui-reader
python -m playwright install chromium

# Configure environment
cp .env.example .env
# Edit .env — at minimum set DATABASE_URL (absolute path) and at least one LLM API key
```

### Start Redis

```bash
redis-server
# or: brew services start redis (macOS)
# or: docker run -d -p 6379:6379 redis:7-alpine
```

### Start Celery Workers (Required for Scans)

```bash
cd backend
./start_celery.sh

# Or manually:
./venv/bin/celery -A app.celery_app worker --pool=solo -Q celery -n general@%h
./venv/bin/celery -A app.celery_app worker --pool=solo -Q data_fetch -n datafetch@%h
./venv/bin/celery -A app.celery_app worker --pool=solo -Q user_scans -n userscans@%h
./venv/bin/celery -A app.celery_app beat --loglevel=info  # Scheduler
```

On macOS and other local non-Docker workflows, keep `--pool=solo`. Docker uses PostgreSQL and Linux `prefork` workers; those settings don't apply to local runs.

### Start the API Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API available at **http://localhost:8000**. Interactive docs at **http://localhost:8000/docs**.

## Frontend Setup

```bash
cd frontend
npm install
npm run dev      # Development server on :5173
npm run build    # Production build
npm run lint     # ESLint
```

Requires backend API running on port 8000.

## Twitter/X Ingestion Setup (Optional)

Bootstrap shared xui profile/session state for theme discovery:

```bash
xui config init --path ../data/xui-reader/config.toml
xui profiles create default --path ../data/xui-reader/config.toml
xui auth login --profile default --path ../data/xui-reader/config.toml
```

Alternative for Google-linked X accounts: use **Themes > Manage Sources > "Connect From Current Browser"** after loading the unpacked extension from `browser-extension/xui-session-bridge`.

## macOS Celery Notes

Celery on macOS requires `--pool=solo` to avoid fork() crashes from Objective-C runtime safety checks. The `start_celery.sh` script handles this automatically along with:

```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export TOKENIZERS_PARALLELISM=false
```

## Running Tests

### Backend (pytest)

```bash
cd backend
source venv/bin/activate

pytest                                     # All tests
pytest tests/unit/                         # Unit tests only
pytest tests/integration/ -m integration   # Integration tests (requires running server)
pytest -v                                  # Verbose output
pytest tests/unit/test_canslim_scanner.py  # Specific test file
```

### Frontend (Vitest + React Testing Library)

```bash
cd frontend

npm run test:run    # All tests once (CI mode)
npm run test        # Watch mode (development)
npx vitest run src/components/Scan/ResultsTable.test.jsx  # Specific file
```

> **Note:** Vitest requires Node 18+. On machines with older system Node, activate NVM first:
> ```bash
> export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
> ```

## Quality Gates

The project has 5 quality gate levels plus a theme identity gate. Run them locally before pushing:

```bash
make help          # List all available targets
make gates         # Run all 5 SE quality gates
make gate-1        # Detector correctness
make gate-2        # Temporal integrity
make gate-3        # Integration coverage
make gate-4        # Performance baselines (advisory)
make gate-5        # Golden regression
make gate-check    # Verify all SE test files are in a gate
make all           # Full CI (backend gates + frontend)
make golden-update # Regenerate golden snapshots
```

## Rate Limits

External data sources have rate limits that the backend respects:

| Source | Limit | Notes |
|--------|-------|-------|
| yfinance | 1 req/sec | Self-imposed |
| Finviz | Rate-limited | Via wrapper |
| Alpha Vantage | 25 req/day | Free tier |
| SEC EDGAR | 10 req/sec | 150ms between requests |

## Diagnostic Scripts

Utility scripts in `backend/scripts/`:

```bash
cd backend && source venv/bin/activate

python scripts/inspect_redis.py            # Inspect Redis cache keys
python scripts/cache_diagnostic.py         # Trace cache flow (DB -> Redis)
python scripts/check_cache_status.py       # Check price cache status
python scripts/clear_redis_price_cache.py  # Clear Redis cache after config change
python scripts/force_full_cache_refresh.py # Force full cache refresh
python scripts/cleanup_orphaned_scans.py   # Clean up stale scans, VACUUM DB
```

## Project Structure

- **Backend architecture, API reference, database schema:** see [Backend README](../backend/README.md)
- **Frontend components, patterns, conventions:** see [Frontend README](../frontend/README.md)
- **Environment variable reference:** see [Environment Variables](ENVIRONMENT.md)
- **System architecture overview:** see [Architecture](ARCHITECTURE.md)
