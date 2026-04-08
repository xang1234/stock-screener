# macOS Installation

> Archived deployment path: macOS desktop packaging is no longer part of the supported mainline deployment model. Keep this guide only for historical reference while the repository converges on the server-only path.

Two options for running StockScreenClaude on macOS: download the pre-built DMG or build from source.

## Option 1: Download the DMG (Recommended)

The easiest way to get started. No development tools required.

1. Go to [GitHub Releases](../../releases)
2. Download `StockScanner.dmg` from the latest release
3. Open the DMG and drag **StockScanner** to your Applications folder
4. Launch StockScanner from Applications

The app opens a native window running the full platform locally. Everything runs on your machine with a bundled SQLite database.

**First launch:** The app seeds a starter universe from bundled data so you can start scanning immediately. A background refresh process gradually replaces starter data with live market data.

> **Gatekeeper:** If macOS shows "StockScanner can't be opened because Apple cannot check it for malicious software," right-click the app and select **Open**, then click **Open** in the dialog.

## Option 2: Build from Source

Use this if you want to build the macOS app bundle yourself.

### Prerequisites
- Python 3.11+
- Node.js 18+
- Xcode command-line tools (`xcode-select --install`)

### Build Steps

```bash
# 1. Build the frontend
cd frontend
npm ci
VITE_API_URL=/api npm run build
cd ..

# 2. Set up the Python environment
python3.11 -m venv backend/venv
source backend/venv/bin/activate
pip install -r backend/requirements-desktop-macos.txt

# 3. Build the .app bundle
bash backend/desktop/build_macos_bundle.sh

# 4. Build the DMG installer
bash backend/desktop/build_macos_dmg.sh
```

### Artifacts
- App bundle: `dist/StockScanner.app`
- DMG installer: `dist/StockScanner.dmg`

### Verify the Build

```bash
# Launch the app
open dist/StockScanner.app

# Or run the smoke test
python backend/desktop/smoke_test.py --base-url http://127.0.0.1:8765
```

## Background Refresh

The macOS bundle includes a `launchd` helper for background data refresh. This runs headless data updates so your stock data stays current without manual intervention.

To trigger a background refresh manually:
```bash
# From the app bundle's internal environment
backend/desktop/launcher.py --background-refresh
```

## Desktop Runtime Notes

### Data Location
- Database: `~/Library/Application Support/StockScanner/stockscanner.db`
- The app creates this on first launch from bundled seed data

### Starter Payload
The desktop build includes a lightweight starter payload:
- Stock universe metadata from `universe_seed.csv`
- IBD industry group mappings from `ibd_industry_seed.csv`
- Bundled baseline prices, fundamentals, breadth, and group rankings from `starter_manifest.json`

This makes a fresh install usable immediately. Regular refresh runs replace the starter baseline with live data.

### Bootstrap Options

By default, the desktop app skips heavy data refresh for fast first-run startup. To override:

```bash
DESKTOP_BOOTSTRAP_REFRESH_UNIVERSE=true \
DESKTOP_BOOTSTRAP_FUNDAMENTALS_LIMIT=25 \
open /Applications/StockScanner.app
```

### Key Files

| File | Purpose |
|------|---------|
| `backend/desktop/launcher.py` | Desktop app launcher |
| `backend/desktop/build_macos_bundle.sh` | macOS .app build script |
| `backend/desktop/build_macos_dmg.sh` | DMG creation script |
| `backend/desktop/StockScanner.macos.spec` | PyInstaller spec for macOS |
| `backend/desktop/smoke_test.py` | Post-build verification |

## Troubleshooting

### Gatekeeper blocks the app
Right-click the app, select **Open**, then confirm. Alternatively:
```bash
xattr -d com.apple.quarantine /Applications/StockScanner.app
```

### App shows empty data on first launch
This is normal. The starter payload provides a minimal dataset. Run a scan or wait for the background refresh to populate data from live sources.

### For local development from source (full stack)
If you want to run the full development stack with Redis, Celery workers, and hot-reload, see the [Development Guide](DEVELOPMENT.md).
