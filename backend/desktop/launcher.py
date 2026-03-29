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


def sys_platform_is_macos() -> bool:
    return sys.platform == "darwin"


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


def _should_use_native_window(args: argparse.Namespace) -> bool:
    env_value = os.getenv("DESKTOP_NATIVE_WINDOW")
    if env_value is not None:
        return env_value.lower() in {"1", "true", "yes", "on"}
    return sys_platform_is_macos() and not args.no_window and not args.background_refresh


def _configure_environment(args: argparse.Namespace) -> None:
    host = args.host
    explicit_port = args.port is not None
    preferred_port = args.port or int(os.getenv("API_PORT", "8000"))
    use_native_window = _should_use_native_window(args)

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
    os.environ["DESKTOP_NATIVE_WINDOW"] = "true" if use_native_window else "false"
    os.environ["DESKTOP_OPEN_BROWSER"] = "false" if use_native_window or args.no_browser else "true"


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
    from app.main import app as fastapi_app, initialize_runtime

    return settings, fastapi_app, initialize_runtime


class _RuntimeServerHandle:
    def __init__(self, app, settings) -> None:
        import uvicorn

        self._config = uvicorn.Config(
            app,
            host=settings.api_host,
            port=settings.api_port,
            reload=False,
            loop="asyncio",
            http="h11",
            ws="none",
            lifespan="on",
        )
        self._server = uvicorn.Server(self._config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


def _run_server(app, settings) -> None:
    handle = _RuntimeServerHandle(app, settings)
    handle.start()
    try:
        handle._thread.join()
    except KeyboardInterrupt:
        handle.stop()


def _current_program_arguments() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, str(Path(__file__).resolve())]


def _background_refresh_program_arguments() -> list[str]:
    return [
        *_current_program_arguments(),
        "--background-refresh",
        "--no-browser",
        "--no-window",
    ]


def _install_launch_agent() -> None:
    if not sys_platform_is_macos():
        return
    from app.config import settings

    if not settings.desktop_launch_agent_enabled:
        return

    from app.wiring.bootstrap import get_desktop_launch_agent_service

    get_desktop_launch_agent_service().install_or_update(_background_refresh_program_arguments())


def _launch_native_window(base_url: str, *, on_closed) -> bool:
    try:
        import webview
    except ModuleNotFoundError:
        return False

    _wait_for_server(base_url)
    window = webview.create_window(
        "Stock Scanner",
        url=base_url,
        width=1440,
        height=960,
        min_size=(1100, 720),
    )
    window.events.closed += on_closed
    webview.start()
    return True


def _run_background_refresh() -> None:
    print("Initializing desktop background helper...", flush=True)
    settings, _fastapi_app, initialize_runtime = _load_runtime()
    initialize_runtime()

    from app.wiring.bootstrap import get_desktop_update_service

    state = get_desktop_update_service().run_due_updates()
    message = state.get("message") or "Desktop background refresh finished"
    print(message, flush=True)
    if state.get("status") == "failed":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Stock Scanner desktop app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--background-refresh", action="store_true")
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()

    if args.stop:
        from stop import main as stop_main

        os.environ.setdefault("DESKTOP_MODE", "true")
        stop_main()
        return

    _configure_environment(args)

    if args.background_refresh:
        _run_background_refresh()
        return

    print("Initializing desktop runtime...", flush=True)
    settings, fastapi_app, _initialize_runtime = _load_runtime()
    _install_launch_agent()

    base_url = f"http://{settings.api_host}:{settings.api_port}"
    pid_file = settings.desktop_data_path / "stockscanner.pid"
    _write_pid_file(pid_file, base_url)

    if settings.desktop_native_window and sys_platform_is_macos() and not args.no_window:
        handle = _RuntimeServerHandle(fastapi_app, settings)
        handle.start()
        try:
            if not _launch_native_window(base_url, on_closed=handle.stop):
                if settings.desktop_open_browser:
                    thread = threading.Thread(
                        target=_open_browser_when_ready,
                        args=(base_url,),
                        daemon=True,
                    )
                    thread.start()
                handle._thread.join()
        finally:
            handle.stop()
        return

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
