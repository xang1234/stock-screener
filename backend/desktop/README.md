# Desktop Bundle

This folder now contains both the legacy desktop packaging entrypoints and the new macOS-native bundle flow.

## macOS build flow

1. Build the frontend with `VITE_API_URL=/api`.
2. Install `backend/requirements-desktop-macos.txt`.
3. Build the app bundle:

```bash
bash backend/desktop/build_macos_bundle.sh
```

4. Build the DMG:

```bash
bash backend/desktop/build_macos_dmg.sh
```

The macOS bundle uses a native window shell around the local FastAPI + React app and installs a `launchd` helper for background refresh runs.

## Runtime entrypoints

- `launcher.py`: starts the local desktop runtime
- `launcher.py --background-refresh`: runs the headless desktop refresh helper
- `launcher.py --stop`: stops a running desktop instance
- `smoke_test.py`: exercises `/livez`, setup/bootstrap compatibility, and a sample scan

## Starter payload

The desktop starter payload intentionally stays lightweight:

- starter universe metadata from `universe_seed.csv`
- starter IBD industry mappings from `ibd_industry_seed.csv`
- bundled baseline prices, fundamentals, breadth, and group rankings from `starter_manifest.json`

The starter payload is only meant to make a fresh install usable immediately. Regular local refresh runs replace the starter baseline with live data over time.
