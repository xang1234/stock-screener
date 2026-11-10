# Windows Installation

Docker Desktop is the recommended way to run StockScreenClaude on Windows. It provides the same PostgreSQL, Redis, API, worker, scheduler, and frontend topology used by supported server deployments.

The former Windows desktop installer and portable application are archived and are no longer built from the main branch.

## Recommended: Docker Desktop

### Prerequisites

- Windows 10 or 11 with WSL2 enabled
- [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) using Linux containers
- Git for Windows, including Git Bash
- Python 3.11 or newer on `PATH` for the market-aware Compose wrapper

### Install

Clone the repository and open PowerShell in the repository root:

```powershell
Copy-Item .env.docker.example .env
notepad .env
```

At minimum, set these values in `.env`:

```dotenv
SERVER_AUTH_PASSWORD=choose-a-long-random-password
ENABLED_MARKETS=US
```

API keys are optional unless you enable the corresponding chatbot or data-provider features.

Start the complete stack through the market-aware Compose wrapper:

```powershell
$env:STOCKSCREEN_PYTHON = "python"
bash ./scripts/docker-compose-enabled-markets.sh up -d
```

If Python is installed under a different command or path, set `STOCKSCREEN_PYTHON` to that executable instead.

Open [http://localhost](http://localhost) and sign in with `SERVER_AUTH_PASSWORD`.

The wrapper starts only the market workers selected by `ENABLED_MARKETS`. For example, changing the value to `US,HK` and running the command again creates the US and Hong Kong market workers while retaining the shared services.

### Common Docker Commands

Run these commands from the repository root:

```powershell
# Show service status
$env:STOCKSCREEN_PYTHON = "python"
bash ./scripts/docker-compose-enabled-markets.sh ps

# Follow logs
bash ./scripts/docker-compose-enabled-markets.sh logs -f

# Apply configuration or image changes
bash ./scripts/docker-compose-enabled-markets.sh up -d

# Stop and remove the stack
bash ./scripts/docker-compose-enabled-markets.sh down
```

Application state is stored under `./data` and in Docker volumes. Removing containers with `down` does not delete named volumes; do not add `--volumes` unless you intend to delete the PostgreSQL database.

For reverse proxies, release images, HTTPS, backups, and upgrades, continue with the [Docker deployment guide](INSTALL_DOCKER.md).

## Advanced: Native PowerShell Deployment

Use this path only when Docker Desktop is unavailable or when you need to operate each service directly. Native deployment requires you to maintain PostgreSQL, Redis, the API, Celery workers, Celery Beat, and a frontend web server independently.

### Prerequisites

- Python 3.11
- Node.js 18 or newer
- PostgreSQL reachable from Windows
- Redis reachable from Windows
- PowerShell 5.1 or newer

Redis can run in WSL2, as a Windows service, or in a standalone Docker container. The backend and every Celery process must use the same PostgreSQL and Redis configuration.

### Backend Setup

From the repository root:

```powershell
py -3.11 -m venv .\backend\venv
.\backend\venv\Scripts\Activate.ps1
pip install -r .\backend\requirements-runtime.lock
Copy-Item .\backend\.env.example .\backend\.env
notepad .\backend\.env
```

`requirements-runtime.lock` pins every package with hashes, the same set the Docker images install, so an install today matches one made last month. Theme embeddings are optional. To enable them, also run `pip install -r .\backend\requirements-theme-ml.txt`.

Configure at least:

```dotenv
DATABASE_URL=postgresql://user:password@localhost/stockscanner
REDIS_HOST=localhost
REDIS_PORT=6379
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
SERVER_AUTH_PASSWORD=choose-a-long-random-password
ENABLED_MARKETS=US
```

### Start the API

In one PowerShell window:

```powershell
Set-Location .\backend
.\venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API startup applies the current database migrations.

### Start Celery Workers

In a second PowerShell window:

```powershell
Set-Location .\backend
.\start_celery.ps1
```

The launcher reads `ENABLED_MARKETS`, validates it against the backend market catalog, and starts the current queue topology:

- one general worker for `celery`
- one concurrency-one global worker for the enabled `data_fetch_*` queues
- one safety-net worker for `user_scans_shared`
- one `market_jobs_<market>` worker per enabled market
- one `user_scans_<market>` worker per enabled market

To override the configured markets for one run:

```powershell
.\start_celery.ps1 -EnabledMarkets "US,HK"
```

Keep the `datafetch-global@%h` worker name. The backend uses that prefix during startup to inspect stale locks across all market scopes.

### Start Celery Beat

In a third PowerShell window:

```powershell
Set-Location .\backend
.\venv\Scripts\python -m celery -A app.celery_app beat --loglevel=info
```

Run only one Beat scheduler for a deployment, or scheduled work will be submitted more than once.

### Build and Serve the Frontend

```powershell
Set-Location .\frontend
npm ci
npm run build
```

Serve `frontend\dist` with IIS, Caddy, or another static web server. Route `/api` to `http://127.0.0.1:8000`. For local frontend development, use `npm run dev` instead.

## Troubleshooting

### PowerShell blocks virtual environment or worker scripts

Allow local scripts for the current PowerShell process only:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### A worker exits immediately

Confirm that:

- PostgreSQL and Redis are running and reachable.
- `backend\.env` contains the connection settings shown above.
- `backend\venv` exists and contains the installed requirements.
- every value in `ENABLED_MARKETS` is a supported market code.
- no other worker on the host uses the same Celery node name.

The PowerShell launcher stops the other workers it started if any worker exits unexpectedly. Correct the reported error and restart the launcher.

### A data-fetch lock appears stuck

Locks normally release in task cleanup and also expire after the configured TTL. Before forcing a release, verify that no data-fetch task is still running.

After signing in, inspect:

```text
GET /api/v1/data-fetch/status
```

If the reported task is definitely no longer running, an authenticated administrator can use:

```text
POST /api/v1/data-fetch/force-release-lock
```

Force release is unsafe while a task is active because it permits overlapping provider work. The `datafetch-global@%h` worker also checks heartbeat-backed stale locks when it starts.

### Docker Desktop cannot start the stack

Check that Docker Desktop is using Linux containers and that WSL2 integration is enabled. Then inspect resolved services and logs:

```powershell
bash ./scripts/docker-compose-enabled-markets.sh config --services
bash ./scripts/docker-compose-enabled-markets.sh logs --tail 200
```
