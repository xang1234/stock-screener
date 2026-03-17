# Desktop Bundle

This folder contains the Windows desktop packaging entrypoints.

Build flow:

1. Build the frontend with `VITE_API_URL=/api`.
2. Install `backend/requirements-desktop.txt`.
3. Build the one-folder bundle with:

```bash
pyinstaller backend/desktop/StockScanner.spec --noconfirm --clean
```

4. Optionally compile `backend/desktop/windows_installer.iss` with Inno Setup.

Runtime entrypoints:

- `launcher.py`: starts the local FastAPI server and opens the browser.
- `launcher.py --stop`: stops a running desktop instance.
- `smoke_test.py`: exercises `/livez`, desktop bootstrap, and a sample scan.
