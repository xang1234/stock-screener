"""Desktop launcher for the local Stock Scanner bundle."""

from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(host: str, preferred_port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, preferred_port))
            return preferred_port
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def _configure_environment(args: argparse.Namespace) -> None:
    host = args.host
    explicit_port = args.port is not None
    preferred_port = args.port or int(os.getenv("API_PORT", "8000"))

    if explicit_port:
        if not _port_is_available(host, preferred_port):
            raise RuntimeError(f"Requested port {preferred_port} is unavailable on {host}")
        selected_port = preferred_port
    else:
        selected_port = _pick_port(host, preferred_port)

    os.environ.setdefault("DESKTOP_MODE", "true")
    os.environ.setdefault("FEATURE_THEMES", "false")
    os.environ.setdefault("FEATURE_CHATBOT", "false")
    os.environ.setdefault("FEATURE_TASKS", "false")
    os.environ.setdefault("XUI_ENABLED", "false")
    os.environ.setdefault("API_HOST", host)
    os.environ["API_PORT"] = str(selected_port)
    os.environ["DESKTOP_OPEN_BROWSER"] = "false" if args.no_browser else "true"


def _wait_for_server(base_url: str, timeout_seconds: int = 30) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/livez", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for desktop app at {base_url}")


def _open_browser_when_ready(base_url: str) -> None:
    try:
        _wait_for_server(base_url)
        webbrowser.open(base_url)
    except Exception:
        pass


def _write_pid_file(pid_file: Path, base_url: str) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "base_url": base_url,
        "started_at": time.time(),
    }
    pid_file.write_text(json.dumps(payload), encoding="utf-8")

    def _cleanup() -> None:
        try:
            if pid_file.exists():
                pid_file.unlink()
        except OSError:
            pass

    atexit.register(_cleanup)


def _load_runtime():
    from app.config import settings
    from app.main import app as fastapi_app

    return settings, fastapi_app


def _run_server(app, settings) -> None:
    import uvicorn

    config = uvicorn.Config(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        loop="asyncio",
        http="h11",
        ws="none",
        lifespan="on",
    )
    print("Desktop server entering run loop", flush=True)
    server = uvicorn.Server(config)
    server.run()
    print("Desktop server exited", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Stock Scanner desktop app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()

    if args.stop:
        from stop import main as stop_main

        os.environ.setdefault("DESKTOP_MODE", "true")
        stop_main()
        return

    _configure_environment(args)

    print("Initializing desktop runtime...", flush=True)
    settings, fastapi_app = _load_runtime()

    base_url = f"http://{settings.api_host}:{settings.api_port}"
    pid_file = settings.desktop_data_path / "stockscanner.pid"
    _write_pid_file(pid_file, base_url)

    if settings.desktop_open_browser:
        thread = threading.Thread(
            target=_open_browser_when_ready,
            args=(base_url,),
            daemon=True,
        )
        thread.start()

    print(f"Starting desktop server at {base_url}", flush=True)
    _run_server(fastapi_app, settings)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
