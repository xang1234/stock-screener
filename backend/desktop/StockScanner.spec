# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules


project_root = Path.cwd()
backend_root = project_root / "backend"
frontend_dist = project_root / "frontend" / "dist"
desktop_root = backend_root / "desktop"

api_route_modules = collect_submodules("app.api.v1")

datas = [
    (str(desktop_root / "universe_seed.csv"), "backend/desktop"),
    (str(desktop_root / "ibd_industry_seed.csv"), "backend/desktop"),
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
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="StockScanner",
)
