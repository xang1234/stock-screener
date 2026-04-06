# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path.cwd()
backend_root = project_root / "backend"
frontend_dist = project_root / "frontend" / "dist"
desktop_root = backend_root / "desktop"

api_route_modules = [
    "app.api.v1.app_runtime",
    "app.api.v1.auth",
    "app.api.v1.stocks",
    "app.api.v1.technical",
    "app.api.v1.scans",
    "app.api.v1.universe",
    "app.api.v1.breadth",
    "app.api.v1.groups",
    "app.api.v1.market_scan",
    "app.api.v1.user_watchlists",
    "app.api.v1.validation",
    "app.api.v1.digest",
    "app.api.v1.strategy_profiles",
    "app.api.v1.ticker_validation",
    "app.api.v1.filter_presets",
    "app.api.v1.features",
]

datas = [
    (str(desktop_root / "universe_seed.csv"), "backend/desktop"),
    (str(desktop_root / "ibd_industry_seed.csv"), "backend/desktop"),
    (str(desktop_root / "starter_manifest.json"), "backend/desktop"),
]

if frontend_dist.exists():
    datas.append((str(frontend_dist), "frontend/dist"))


a = Analysis(
    [str(desktop_root / "launcher.py")],
    pathex=[str(backend_root)],
    binaries=[],
    datas=datas,
    hiddenimports=sorted(
        {
            "app.main",
            "app.api.v1.router",
            "stop",
            "webview",
            "webview.platforms.cocoa",
            *api_route_modules,
        }
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StockScanner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="StockScanner",
)

app = BUNDLE(
    coll,
    name="StockScanner.app",
    bundle_identifier="com.stockscanner.desktop",
)
