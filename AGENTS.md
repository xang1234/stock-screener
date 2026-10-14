## Project Overview

Stock screening platform implementing CANSLIM (William O'Neil) and Minervini methodologies, with theme discovery, AI chatbot, and market analysis. Full-stack application with FastAPI backend, React frontend, PostgreSQL, Redis caching, and Celery for background tasks.

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
./start_celery.sh    # Starts all required queues for enabled markets

# Or manually:
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CELERY_POOL="${CELERY_POOL:-solo}"

# Same markets as the API and Beat: backend/.env, else US.
export ENABLED_MARKETS="${ENABLED_MARKETS:-$(./venv/bin/python -c 'from app.config import settings; print(settings.enabled_markets)')}"

DATA_FETCH_QUEUES="$(./venv/bin/python - <<'PY'
from app.tasks.market_queues import all_data_fetch_queues
print(",".join(all_data_fetch_queues()))
PY
)"

./venv/bin/celery -A app.celery_app worker --pool="$CELERY_POOL" -Q celery -n general@%h &
./venv/bin/celery -A app.celery_app worker --pool="$CELERY_POOL" --concurrency=1 -Q "$DATA_FETCH_QUEUES" -n datafetch-global@%h &
./venv/bin/celery -A app.celery_app worker --pool="$CELERY_POOL" -Q user_scans_shared -n userscans-shared@%h &

IFS=',' read -ra MARKET_ARRAY <<< "$ENABLED_MARKETS"
for RAW_MARKET in "${MARKET_ARRAY[@]}"; do
  MARKET_LOWER="$(echo "$RAW_MARKET" | tr '[:upper:]' '[:lower:]' | xargs)"
  ./venv/bin/celery -A app.celery_app worker --pool="$CELERY_POOL" -Q "market_jobs_${MARKET_LOWER}" -n "marketjobs-${MARKET_LOWER}@%h" &
  ./venv/bin/celery -A app.celery_app worker --pool="$CELERY_POOL" -Q "user_scans_${MARKET_LOWER}" -n "userscans-${MARKET_LOWER}@%h" &
done

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
