# Local Backend Development Guide

## Purpose

Run the Stock Screener backend locally while continuing to use Docker for infrastructure services.

### Architecture

#### Docker Services

* PostgreSQL
* Redis

#### Local Services

* FastAPI
* Celery

### Benefits

* Fast code reload
* Local debugging
* No backend container rebuilds
* No Celery container rebuilds
* Faster development iteration cycle

---

## Quick Start

### First-Time Setup

Follow the backend installation instructions in the project README.

Once the backend virtual environment has been created and dependencies installed, continue with the local backend workflow below.

```bash
cp docker-compose.override.example.yml docker-compose.override.yml

docker compose up -d postgres redis

cd backend

source ./scripts/activate_backend.sh

python -c "from app.config.settings import settings; print(settings.database_url)"
```

### Daily Startup

Start infrastructure:

```bash
docker compose up -d postgres redis
```

Terminal 1:

```bash
cd backend

source ./scripts/activate_backend.sh

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2:

```bash
cd backend

source ./scripts/activate_backend.sh

./start_celery.sh
```

### Verify

Login:

```bash
curl -i -c cookies.txt \
  -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"password":"<PASSWORD>"}'
```

Query scans:

```bash
curl -b cookies.txt \
  http://localhost:8000/api/v1/scans
```

If scan results are returned, the local backend stack is operational.

---

## Local Infrastructure Setup

When running FastAPI and Celery locally, PostgreSQL and Redis must be reachable from `localhost`.

Do **not** modify the main `docker-compose.yml`.

Instead, create a local override file from the provided example:

```bash
cp docker-compose.override.example.yml docker-compose.override.yml
```

Review the contents if needed:

```bash
cat docker-compose.override.yml
```

Expected contents:

```yaml
services:
  redis:
    ports:
      - "6379:6379"

  postgres:
    ports:
      - "5432:5432"
```

Docker Compose automatically loads `docker-compose.override.yml` when present.

### Start Infrastructure

```bash
docker compose up -d postgres redis
```

### Verify

```bash
docker ps
```

Expected:

```text
stock-screener-postgres-1
0.0.0.0:5432->5432/tcp

stock-screener-redis-1
0.0.0.0:6379->6379/tcp
```

Without this override file, local FastAPI and Celery processes cannot connect to PostgreSQL or Redis through `localhost`.

---

## Python Environment

The backend uses its own virtual environment.

```bash
cd backend
source venv/bin/activate
```

---

## Environment Gotcha

The repository root `.env` is intended for Docker deployments.

```env
DATABASE_URL=postgresql://stockscanner:stockscanner@postgres:5432/stockscanner

CELERY_BROKER_URL=redis://redis:6379/0

CELERY_RESULT_BACKEND=redis://redis:6379/1
```

The backend `.env` is intended for local development.

```env
DATABASE_URL=postgresql://stockscanner:stockscanner@localhost:5432/stockscanner

CELERY_BROKER_URL=redis://localhost:6379/0

CELERY_RESULT_BACKEND=redis://localhost:6379/1
```

### VS Code Environment Injection

VS Code may automatically load the repository root `.env` into terminals when:

```json
{
  "python.terminal.useEnvFile": true
}
```

This causes local FastAPI and Celery processes to attempt connections to:

```text
postgres
redis
```

These hostnames only exist inside the Docker network.

---

## Backend Local Activation

The activation script exists because VS Code or shell environments may inject Docker-oriented environment variables from the repository root `.env`, which override the backend local-development configuration.

Use:

```bash
source ./scripts/activate_backend.sh
```

The script:

* Clears Docker-specific environment variables
* Activates `backend/venv`
* Verifies backend configuration

Expected output:

```text
Backend local development environment activated.

DATABASE_URL=postgresql://stockscanner:stockscanner@localhost:5432/stockscanner

CELERY_BROKER_URL=redis://localhost:6379/0
```

Verify the active Python interpreter:

```bash
which python
```

Expected:

```text
.../backend/venv/bin/python
```

---

## Start FastAPI

Open a terminal:

```bash
cd backend

source ./scripts/activate_backend.sh

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Expected:

```text
Started server process
Waiting for application startup
```

---

## Start Celery

Open a second terminal:

```bash
cd backend

source ./scripts/activate_backend.sh

./start_celery.sh
```

Expected:

```text
Connected to redis://localhost:6379/0
```

Workers:

```text
general
datafetch-global
marketjobs-us
userscans-shared
userscans-us
```

Verify workers:

```bash
celery -A app.celery_app inspect ping
```

Expected:

```text
-> celery@hostname: OK
    pong
```

---

## Authentication Testing

Login:

```bash
curl -i -c cookies.txt \
  -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"password":"<PASSWORD>"}'
```

Expected:

```json
{
  "authenticated": true
}
```

Session cookie:

```text
stockscanner_session
```

---

## Authenticated Requests

Example:

```bash
curl -b cookies.txt \
  http://localhost:8000/api/v1/scans
```

---

## Shutdown

Stop infrastructure:

```bash
docker compose stop postgres redis
```

Remove infrastructure:

```bash
docker compose down
```

---

## Verification Checklist

Use this checklist when validating a fresh local setup:

* [ ] Docker PostgreSQL container running
* [ ] Docker Redis container running
* [ ] PostgreSQL exposed on port 5432
* [ ] Redis exposed on port 6379
* [ ] `source ./scripts/activate_backend.sh`
* [ ] `which python` points to `backend/venv/bin/python`
* [ ] `DATABASE_URL` resolves to `localhost`
* [ ] `CELERY_BROKER_URL` resolves to `localhost`
* [ ] FastAPI starts successfully
* [ ] Celery workers connect successfully
* [ ] Login endpoint returns `authenticated=true`
* [ ] `/api/v1/scans` returns results

---

## Lessons Learned

1. Use the backend virtual environment, not the repository root virtual environment.

2. PostgreSQL and Redis must be exposed to localhost when running FastAPI locally.

3. Environment variables override `.env` values.

4. The repository root `.env` is Docker-oriented.

5. The backend `.env` is local-development oriented.

6. Use:

   ```bash
   source ./scripts/activate_backend.sh
   ```

   not:

   ```bash
   ./scripts/activate_backend.sh
   ```

7. Authentication uses session cookies; login is required only once per cookie lifetime.

8. Local backend development is significantly faster than rebuilding backend and Celery containers during active development.

9. Environment variables take precedence over values loaded from `.env` files.
