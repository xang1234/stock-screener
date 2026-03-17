from __future__ import annotations

import argparse
import os

import pytest

from desktop import launcher


def _args(*, host: str = "127.0.0.1", port: int | None = None, no_browser: bool = True) -> argparse.Namespace:
    return argparse.Namespace(host=host, port=port, no_browser=no_browser, stop=False)


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
