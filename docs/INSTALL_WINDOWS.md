# Windows Installation

> Archived deployment path: Windows desktop packaging is no longer part of the supported mainline deployment model. Keep this guide only for historical reference while the repository converges on the server-only path.

Three options for running StockScreenClaude on Windows, from simplest to most flexible.

## Option 1: Download the Installer (Recommended)

The easiest way to get started. No development tools required.

1. Go to [GitHub Releases](../../releases)
2. Download `StockScanner-Setup.exe` from the latest release
3. Run the installer and follow the prompts
4. Launch **StockScanner** from the Start Menu or desktop shortcut

The app opens in your default browser at `http://127.0.0.1:8765`. Everything runs locally on your machine.

**What gets installed:**
- App files under `%LOCALAPPDATA%\StockScanner`
- Local application state under `%LOCALAPPDATA%\StockScanner`
- No Redis or Celery required (desktop mode handles everything in-process)

**First launch:** The app seeds a starter universe from bundled CSV data so you can start scanning immediately. Regular refresh runs replace the starter baseline with live data over time.

> A portable zip (`StockScanner-Portable.zip`) is also available if you prefer not to install.

## Option 2: Build the Desktop Bundle

Use this if you want to build the installer yourself from source.

### Prerequisites
- Python 3.11
- Node.js 18+
- PowerShell
- Inno Setup 6 (optional, for the `.exe` installer)

### Build Steps

```powershell
# Build the frontend
cd .\frontend
npm ci
$env:VITE_API_URL = "/api"
npm run build
Remove-Item Env:VITE_API_URL

# Build the PyInstaller bundle
cd ..
py -3.11 -m venv .\backend\venv
.\backend\venv\Scripts\Activate.ps1
pip install -r .\backend\requirements-desktop.txt
pyinstaller .\backend\desktop\StockScanner.spec --noconfirm --clean
```

### Artifacts
- One-folder bundle: `dist\StockScanner\StockScanner.exe`
- Optional installer: `dist\installer\StockScanner-Setup.exe` (see below)

### Verify the Build

```powershell
Start-Process -FilePath .\dist\StockScanner\StockScanner.exe -ArgumentList "--no-browser","--port","8765"
python .\backend\desktop\smoke_test.py --base-url http://127.0.0.1:8765
.\dist\StockScanner\StockScanner.exe --stop
```

### Create the Installer (Optional)

```powershell
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" .\backend\desktop\windows_installer.iss
```

## Option 3: Source Deployment (Full Stack)

Use this when you want the full backend + frontend + worker stack on a Windows host, with Redis, Celery, and all background tasks.

### Prerequisites
- Python 3.11
- Node.js 18+
- Redis reachable from Windows (via Docker Desktop, WSL2, or a local service)

### Backend Setup

```powershell
py -3.11 -m venv .\backend\venv
.\backend\venv\Scripts\Activate.ps1
pip install -r .\backend\requirements.txt
pip install -e .\xui-reader
python -m playwright install chromium
Copy-Item .\backend\.env.example .\backend\.env
```

Edit `backend\.env` with your API keys and a PostgreSQL `DATABASE_URL`.

### Start the Backend

```powershell
cd .\backend
.\venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Start Celery Workers (Separate PowerShell Windows)

```powershell
cd .\backend
.\venv\Scripts\celery -A app.celery_app worker --pool=solo -Q celery -n general@$env:COMPUTERNAME
.\venv\Scripts\celery -A app.celery_app worker --pool=solo -Q data_fetch -n datafetch@$env:COMPUTERNAME
.\venv\Scripts\celery -A app.celery_app worker --pool=solo -Q user_scans -n userscans@$env:COMPUTERNAME
.\venv\Scripts\celery -A app.celery_app beat --loglevel=info
```

### Build and Serve the Frontend

```powershell
cd .\frontend
npm ci
npm run build
```

Serve `frontend\dist` with IIS, Caddy, or another static file server and reverse proxy `/api` to `http://127.0.0.1:8000`.

## Desktop Runtime Notes

### Bootstrap Options

By default, desktop mode skips heavy full-universe refresh for fast first-run startup. To enable the heavier bootstrap:

```powershell
$env:DESKTOP_BOOTSTRAP_REFRESH_UNIVERSE = "true"
$env:DESKTOP_BOOTSTRAP_FUNDAMENTALS_LIMIT = "25"
.\dist\StockScanner\StockScanner.exe
```

### Launcher Arguments

| Argument | Description |
|----------|-------------|
| `--no-browser` | Start without opening a browser window |
| `--port 8765` | Use a custom port |
| `--stop` | Stop a running desktop instance |

### Key Files

| File | Purpose |
|------|---------|
| `backend/desktop/launcher.py` | Desktop app launcher |
| `backend/desktop/stop.py` | Stop helper |
| `backend/desktop/smoke_test.py` | Post-build verification |
| `backend/desktop/windows_installer.iss` | Inno Setup installer script |
| `backend/desktop/StockScanner.spec` | PyInstaller spec for Windows |

## Troubleshooting

### PowerShell blocks virtualenv activation
```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\backend\venv\Scripts\Activate.ps1
```

### Redis not available (Source Deployment)
Install Redis via one of:
- **Docker Desktop:** `docker run -d -p 6379:6379 redis:7-alpine`
- **WSL2:** `sudo apt install redis-server && redis-server`
- **Windows Redis:** Download from [tporadowski/redis](https://github.com/tporadowski/redis/releases)

### App shows empty data on first launch
This is normal. The starter payload provides a minimal dataset. Run a scan or wait for the background refresh to populate data from live sources.
