"""Process-wide curl_cffi session for yfinance calls.

Yahoo Finance's bot detection routinely 429s plain `requests`-based
clients. ``curl_cffi`` impersonates a real browser's TLS fingerprint and
keeps cookies/crumb across calls, which dramatically reduces 429 rate and
also avoids the per-process crumb refetch cost yfinance otherwise pays on
every cold call.

The session is a module-level singleton initialised lazily on first use.
Each Celery worker process gets its own session — safe because the
project runs Celery with ``--pool=solo`` (per CLAUDE.md), so there's no
fork-after-init concern.

When ``settings.yfinance_use_curl_cffi`` is False or ``curl_cffi`` is not
installed, ``get_session()`` returns ``None`` and yfinance falls back to
its default ``requests``-based path.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_session_lock = threading.Lock()
_session = None  # type: ignore[var-annotated]
_session_init_attempted = False


def get_session():
    """Return the shared curl_cffi session, or ``None`` to fall back.

    Thread-safe lazy init. The first call attempts to import ``curl_cffi``
    and build a session; subsequent calls return the cached object.
    Failure to import or construct is logged once and the function then
    consistently returns ``None`` so callers can safely
    ``kw["session"] = sess if sess is not None else …``.
    """
    global _session, _session_init_attempted
    if _session_init_attempted:
        return _session

    with _session_lock:
        if _session_init_attempted:
            return _session
        _session_init_attempted = True
        _session = _build_session()
        return _session


def reset_session() -> None:
    """Clear the cached session (test helper)."""
    global _session, _session_init_attempted
    with _session_lock:
        _session = None
        _session_init_attempted = False


def _build_session():
    try:
        from ..config import settings
    except Exception as exc:
        logger.debug("yf_session: settings unavailable (%s); skipping curl_cffi", exc)
        return None

    if not getattr(settings, "yfinance_use_curl_cffi", True):
        logger.info("yf_session: curl_cffi disabled by settings")
        return None

    impersonate = getattr(settings, "yfinance_curl_cffi_impersonate", "chrome") or "chrome"

    try:
        from curl_cffi import requests as curl_requests  # type: ignore
    except Exception as exc:
        logger.info(
            "yf_session: curl_cffi not available (%s); yfinance will use default session",
            exc,
        )
        return None

    try:
        session = curl_requests.Session(impersonate=impersonate)
        logger.info(
            "yf_session: built curl_cffi session (impersonate=%s)", impersonate,
        )
        return session
    except Exception as exc:
        logger.warning(
            "yf_session: failed to build curl_cffi session (%s); falling back",
            exc,
        )
        return None
