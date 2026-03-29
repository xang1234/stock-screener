from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from desktop import launcher


def _args(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    no_browser: bool = True,
    no_window: bool = False,
    background_refresh: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        host=host,
        port=port,
        no_browser=no_browser,
        no_window=no_window,
        background_refresh=background_refresh,
        stop=False,
    )


def test_configure_environment_honors_explicit_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_PORT", raising=False)
    port = 8765
    monkeypatch.setattr(launcher, "_port_is_available", lambda host, requested_port: requested_port == port)

    launcher._configure_environment(_args(port=port))

    assert os.environ["API_PORT"] == str(port)


def test_configure_environment_rejects_unavailable_explicit_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_PORT", raising=False)
    port = 8765
    monkeypatch.setattr(launcher, "_port_is_available", lambda *_args, **_kwargs: False)

    with pytest.raises(RuntimeError, match=f"Requested port {port} is unavailable"):
        launcher._configure_environment(_args(port=port))


def test_configure_environment_falls_back_when_default_port_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    preferred_port = 8000
    fallback_port = 9100
    monkeypatch.setenv("API_PORT", str(preferred_port))
    monkeypatch.setattr(launcher, "_pick_port", lambda host, port: fallback_port)

    launcher._configure_environment(_args(port=None))

    assert os.environ["API_PORT"] == str(fallback_port)


def test_main_passes_preloaded_app_to_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = object()
    settings = SimpleNamespace(
        api_host="127.0.0.1",
        api_port=8765,
        desktop_data_path=tmp_path,
        desktop_native_window=False,
        desktop_open_browser=False,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: _args(port=8765))
    monkeypatch.setattr(launcher, "_configure_environment", lambda args: None)
    monkeypatch.setattr(launcher, "_load_runtime", lambda: (settings, app, lambda: None))
    monkeypatch.setattr(launcher, "_install_launch_agent", lambda: None)
    monkeypatch.setattr(launcher, "_write_pid_file", lambda pid_file, base_url: None)
    monkeypatch.setattr(
        launcher,
        "_run_server",
        lambda loaded_app, loaded_settings: captured.update(
            app=loaded_app,
            settings=loaded_settings,
        ),
    )

    launcher.main()

    assert captured["app"] is app
    assert captured["settings"] is settings


def test_main_opens_browser_when_native_window_fails_on_macos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = object()
    settings = SimpleNamespace(
        api_host="127.0.0.1",
        api_port=8765,
        desktop_data_path=tmp_path,
        desktop_native_window=True,
        desktop_open_browser=False,
    )
    opened: list[str] = []
    lifecycle: list[str] = []

    class _FakeServerThread:
        def join(self) -> None:
            lifecycle.append("join")

    class _FakeRuntimeServerHandle:
        def __init__(self, loaded_app, loaded_settings) -> None:
            assert loaded_app is app
            assert loaded_settings is settings
            self._thread = _FakeServerThread()

        def start(self) -> None:
            lifecycle.append("start")

        def stop(self) -> None:
            lifecycle.append("stop")

    class _ImmediateThread:
        def __init__(self, *, target, args=(), daemon=None) -> None:
            self._target = target
            self._args = args
            self._daemon = daemon

        def start(self) -> None:
            self._target(*self._args)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: _args(port=8765, no_browser=True, no_window=False))
    monkeypatch.setattr(launcher, "_configure_environment", lambda args: None)
    monkeypatch.setattr(launcher, "_load_runtime", lambda: (settings, app, lambda: None))
    monkeypatch.setattr(launcher, "_install_launch_agent", lambda: None)
    monkeypatch.setattr(launcher, "_write_pid_file", lambda pid_file, base_url: None)
    monkeypatch.setattr(launcher, "sys_platform_is_macos", lambda: True)
    monkeypatch.setattr(launcher, "_RuntimeServerHandle", _FakeRuntimeServerHandle)
    monkeypatch.setattr(launcher, "_launch_native_window", lambda base_url, on_closed: False)
    monkeypatch.setattr(launcher, "_open_browser_when_ready", lambda base_url: opened.append(base_url))
    monkeypatch.setattr(launcher.threading, "Thread", _ImmediateThread)

    launcher.main()

    assert opened == ["http://127.0.0.1:8765"]
    assert lifecycle == ["start", "join", "stop"]
