# Docker Deployment

Docker is the recommended deployment method for servers, homelabs, and VPS hosting. The project uses a layered Docker Compose architecture with composable overlays for different scenarios.

Docker deployments use **PostgreSQL** as the application database. The shared `./data` mount handles non-database state (xui-reader config, Celery beat schedule, caches).

## Prerequisites

- Docker Engine 20.10+
- Docker Compose v2 (`docker compose` or `docker-compose`)

## Quick Start (Local Development)

Zero-config local deployment:

```bash
# 1. Set up environment (required for chatbot/LLM features)
cp .env.docker.example .env
# Edit .env: Set SERVER_AUTH_PASSWORD and add your API keys (GROQ_API_KEY, GEMINI_API_KEY, etc.)

# 2. Start all services
docker-compose up
```

This starts PostgreSQL, Redis, Backend API, Celery workers, backup service, and Frontend. Access at **http://localhost**.

> **Note:** Docker Compose reads environment variables from `.env` in the project root (not `.env.docker`). `SERVER_AUTH_PASSWORD` is required for server access, and LLM API keys are required for chatbot features.

## Homelab (Behind Reverse Proxy)

For deployment behind Traefik, nginx proxy manager, or similar:

```bash
# 1. Configure environment
cp .env.docker.example .env.docker
# Edit .env.docker: Set SERVER_AUTH_PASSWORD and CORS_ORIGINS=https://stocks.home.lan

# 2. Start with production settings
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 3. Configure your reverse proxy to forward to port 80
```

The production overlay adds resource limits, health checks, and JSON logging with rotation.

## Production with GHCR Images (Recommended)

Use pre-built images from GitHub Container Registry instead of building from source on the server:

```bash
# 1. Configure environment
cp .env.docker.example .env.docker
# Edit .env.docker:
#   BACKEND_IMAGE=ghcr.io/<owner>/stockscreenclaude-backend
#   FRONTEND_IMAGE=ghcr.io/<owner>/stockscreenclaude-frontend
#   APP_IMAGE_TAG=v1.2.3
#   SERVER_AUTH_PASSWORD=choose-a-long-random-password
#   CORS_ORIGINS=https://stocks.yourdomain.com

# 2. Pull the tagged release images
docker-compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.release.yml pull

# 3. Deploy without rebuilding locally
docker-compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.release.yml up -d --no-build
```

For HTTPS on a standalone VPS, add the Caddy overlay:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.release.yml -f docker-compose.https.yml pull
docker-compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.release.yml -f docker-compose.https.yml up -d --no-build
```

### Release and Rollback

- Push to `main` to publish rolling `main`, `sha-*`, and `latest` image tags to GHCR
- Push a git tag like `v1.2.3` to publish immutable release tags
- **Deploy:** Set `APP_IMAGE_TAG=v1.2.3` in `.env.docker`, run `pull` + `up -d --no-build`
- **Roll back:** Change `APP_IMAGE_TAG` to the previous tag and redeploy

If the repository or package is private, authenticate first:
```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-username> --password-stdin
```

## VPS with Auto-HTTPS (Hostinger, DigitalOcean, etc.)

Includes Caddy for automatic Let's Encrypt certificates:

```bash
# 1. Configure environment
cp .env.docker.example .env.docker
# Edit .env.docker: Set DOMAIN=stocks.yourdomain.com
# Edit .env.docker: Set SERVER_AUTH_PASSWORD=choose-a-long-random-password
# Edit .env.docker: Set CORS_ORIGINS=https://stocks.yourdomain.com

# 2. Ensure DNS A record points to your server IP

# 3. Start with HTTPS
docker-compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.https.yml up -d
```

Requirements:
- DNS A record pointing to your server
- Ports 80 and 443 open
- `DOMAIN` environment variable set

## Services Architecture

| Service | Purpose |
|---------|---------|
| `redis` | Celery broker (DB 0) and result backend (DB 1) |
| `postgres` | Application database |
| `backend` | FastAPI API server |
| `celery-worker` | General compute queue (2 workers) |
| `celery-datafetch` | Data fetch queue (1 worker, serialized for rate limits) |
| `celery-userscans` | User scan queue (2 workers) |
| `celery-beat` | Celery Beat scheduler |
| `db-backup` | Automated PostgreSQL backups to `./data/backups` |
| `frontend` | React app served via nginx |
| `caddy` | (HTTPS overlay only) TLS termination with Let's Encrypt |

## Docker Compose File Reference

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Base configuration for local development |
| `docker-compose.prod.yml` | Production overlay: resource limits, health checks, logging |
| `docker-compose.release.yml` | Release overlay: deploy tagged GHCR images instead of local builds |
| `docker-compose.https.yml` | HTTPS overlay: Caddy with automatic Let's Encrypt |
| `.env.docker.example` | Template for Docker environment variables |
| `Caddyfile` | Caddy configuration for TLS termination |

## PostgreSQL Notes

### Upgrade from Older Versions

The backend runs as non-root user (uid 1000). If upgrading from an older version:
```bash
sudo chown -R 1000:1000 ./data
```

### Backups and Restore

PostgreSQL backups are automatically written by the `db-backup` service via `pg_dump`. Restore with:
```bash
pg_restore -d <database> <dump-file>
```

### SQLite-to-PostgreSQL Cutover

One-time migration for existing Docker deployments:

```bash
# 1. Stop backend, workers, and beat so SQLite is quiescent
docker-compose stop backend celery-worker celery-datafetch celery-userscans celery-beat

# 2. Take a final SQLite backup
sqlite3 data/stockscanner.db ".backup 'data/backups/stockscanner_final_pre_pg.db'"

# 3. Start PostgreSQL only
docker-compose up -d postgres

# 4. Prepare schema against PostgreSQL
docker-compose run --rm backend python -c "from app.database import init_db; init_db()"

# 5. Import data into PostgreSQL
docker-compose run --rm \
  -e SQLITE_DB_PATH=/app/data/stockscanner.db \
  backend \
  python scripts/migrate_sqlite_to_postgres.py --source /app/data/stockscanner.db

# 6. Start backend first and smoke test the API
docker-compose up -d backend

# 7. Start workers and beat after API validation passes
docker-compose up -d celery-worker celery-datafetch celery-userscans celery-beat frontend db-backup
```

Fresh installs now auto-seed `ibd_industry_groups` from the bundled canonical CSV on backend startup when the table is empty.
If you already have an existing database with an empty `ibd_industry_groups` table, repair it with:
```bash
docker-compose run --rm backend python scripts/seed_ibd_industry_groups.py
```

## Troubleshooting

### Chatbot not responding
Docker Compose reads API keys from `.env` in the project root. If `.env` is missing or keys are empty, scanning still works but the chatbot won't. Check:
```bash
docker-compose exec backend env | grep -i API_KEY
```

### CORS errors in browser
Set `CORS_ORIGINS` in your environment file to match your access URL (e.g., `https://stocks.home.lan`). Restart the backend after changes.

### Permission denied on ./data
The backend container runs as uid 1000. Fix with:
```bash
sudo chown -R 1000:1000 ./data
```

### Container health checks failing
```bash
docker-compose ps          # Check service status
docker-compose logs backend # Check backend logs
curl http://localhost:8000/readyz  # Direct health check
```
