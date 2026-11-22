"""Single-user server authentication helpers."""

from __future__ import annotations

import base64
import functools
import hashlib
import hmac
import json
import logging
import os
import secrets
import stat
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import Response

from ..config import settings
from ..config.settings import get_project_root
from .errors import (
    AuthTokenDecodeError,
    AuthTokenError,
    AuthTokenExpiredError,
    AuthTokenSignatureError,
)

logger = logging.getLogger(__name__)

# In the persisted data dir (./data:/app/data in Docker) so restarts and every
# API process share one secret.
_SESSION_SECRET_PATH = get_project_root() / "data" / ".server_auth_session_secret"


@dataclass(frozen=True)
class ServerAuthStatus:
    required: bool
    configured: bool
    authenticated: bool
    mode: str = "session_cookie"
    message: str | None = None


def server_auth_required() -> bool:
    """Return True when server-mode auth should be enforced."""
    return bool(settings.server_auth_enabled)


def server_auth_configured() -> bool:
    """Return True when the shared-secret auth inputs are available."""
    return bool(_server_auth_password() and _session_secret())


def get_server_auth_status(request: Request | None = None) -> ServerAuthStatus:
    """Return effective auth state for the current client."""
    if not server_auth_required():
        return ServerAuthStatus(
            required=False,
            configured=True,
            authenticated=True,
            message="Server authentication is disabled.",
        )

    if not server_auth_configured():
        return ServerAuthStatus(
            required=True,
            configured=False,
            authenticated=False,
            message="Server authentication is required but not configured.",
        )

    authenticated = request_is_authenticated(request) if request is not None else False
    return ServerAuthStatus(
        required=True,
        configured=True,
        authenticated=authenticated,
        message="Authenticated." if authenticated else "Authentication required.",
    )


def issue_session_token() -> str:
    """Mint a signed, expiring session token."""
    now = int(time.time())
    payload = {
        "sub": "server-user",
        "iat": now,
        "exp": now + max(1, settings.server_auth_session_ttl_hours) * 3600,
    }
    return _sign_payload(payload)


def attach_auth_cookie(response: Response, token: str, request: Request | None = None) -> None:
    """Attach the authenticated session cookie to a response."""
    response.set_cookie(
        key=settings.server_auth_cookie_name,
        value=token,
        max_age=max(1, settings.server_auth_session_ttl_hours) * 3600,
        httponly=True,
        samesite="lax",
        secure=_request_is_secure(request),
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    """Clear the authenticated session cookie."""
    response.delete_cookie(
        key=settings.server_auth_cookie_name,
        httponly=True,
        samesite="lax",
        path="/",
    )


def password_matches(candidate: str | None) -> bool:
    """Return True when a supplied shared secret matches an accepted login secret."""
    if candidate is None:
        return False
    supplied = str(candidate)
    secret = _server_auth_password()
    return bool(secret) and hmac.compare_digest(supplied, secret)


def request_is_authenticated(request: Request | None) -> bool:
    """Return True when the request carries a valid auth cookie or shared-secret header."""
    if request is None:
        return False

    if _header_secret_matches(request):
        return True

    token = request.cookies.get(settings.server_auth_cookie_name)
    if not token:
        return False

    return _verify_session_token(token)


def require_server_session(request: Request) -> bool:
    """FastAPI dependency enforcing server auth on protected routes."""
    if not server_auth_required():
        return True
    if not server_auth_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Server authentication is required but not configured. "
                "Set SERVER_AUTH_PASSWORD before exposing the API."
            ),
        )
    if request_is_authenticated(request):
        return True
    raise HTTPException(status_code=401, detail="Authentication required")


def _server_auth_password() -> str:
    return str(settings.server_auth_password or "").strip()


def _session_secret() -> str:
    secret = str(settings.server_auth_session_secret or "").strip()
    if secret:
        return secret
    path = _SESSION_SECRET_PATH
    try:
        return _load_or_create_session_secret(path)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot read or create the session-signing secret at {path} ({exc}). "
            "Make that directory writable by the app user, or set SERVER_AUTH_SESSION_SECRET."
        ) from exc


_NO_FOLLOW = getattr(os, "O_NOFOLLOW", 0)
_SECRET_MAX_BYTES = 4096


def _read_secret_file(path: Path) -> str:
    """Read the secret, refusing symlinks, special files and other users' files.

    Returns "" for an empty or undecodable file.
    """
    # O_NONBLOCK keeps a planted FIFO from hanging startup.
    fd = os.open(path, os.O_RDONLY | _NO_FOLLOW | getattr(os, "O_NONBLOCK", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"Session-signing secret {path} is not a regular file.")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise RuntimeError(
                f"Session-signing secret {path} is owned by another user. "
                "Delete it to generate a new one, or set SERVER_AUTH_SESSION_SECRET."
            )
        raw = os.read(fd, _SECRET_MAX_BYTES)
    finally:
        os.close(fd)
    try:
        return raw.decode("ascii").strip()
    except UnicodeDecodeError:
        return ""


@functools.cache
def _load_or_create_session_secret(path: Path) -> str:
    # O_EXCL on the final name: racing processes see exactly one creator. It
    # needs no hard links, so it works on bind mounts and network filesystems.
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NO_FOLLOW, 0o600)
    except FileExistsError:
        secret = _read_secret_file(path)
        # The creator may not have written yet; give it a moment.
        for _ in range(50):
            if secret:
                return secret
            time.sleep(0.02)
            secret = _read_secret_file(path)
        raise RuntimeError(
            f"Session-signing secret file {path} is empty or unreadable. "
            "Delete it to generate a new one, or set SERVER_AUTH_SESSION_SECRET."
        )
    secret = secrets.token_urlsafe(32)
    with os.fdopen(fd, "w", encoding="ascii") as handle:
        handle.write(secret)
        handle.flush()
        os.fsync(handle.fileno())
    logger.info("Generated server auth session secret", extra={"path": str(path)})
    return secret


def _header_secret_matches(request: Request) -> bool:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        if password_matches(authorization.split(" ", 1)[1].strip()):
            return True

    x_server_password = request.headers.get("x-server-auth")
    return password_matches(x_server_password)


def _sign_payload(payload: dict[str, int | str]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(_session_secret().encode("utf-8"), raw, hashlib.sha256).digest()
    return f"{_b64_encode(raw)}.{_b64_encode(sig)}"


def _verify_session_token(token: str) -> bool:
    try:
        _decode_and_validate_session_token(token)
        return True
    except AuthTokenExpiredError as exc:
        logger.info(
            "Server auth token expired",
            extra={
                "event": "server_auth_token_validation_failed",
                "path": "server_auth._verify_session_token",
                "pipeline": None,
                "run_id": None,
                "symbol": None,
                "error_code": exc.error_code,
            },
        )
        return False
    except AuthTokenError as exc:
        logger.warning(
            "Server auth token rejected",
            extra={
                "event": "server_auth_token_validation_failed",
                "path": "server_auth._verify_session_token",
                "pipeline": None,
                "run_id": None,
                "symbol": None,
                "error_code": exc.error_code,
            },
        )
        return False


def _decode_and_validate_session_token(token: str) -> dict[str, int | str]:
    try:
        raw_payload, raw_sig = token.split(".", 1)
    except ValueError as exc:
        raise AuthTokenDecodeError("Malformed session token") from exc

    try:
        payload_bytes = _b64_decode(raw_payload)
        supplied_sig = _b64_decode(raw_sig)
    except (ValueError, TypeError) as exc:
        raise AuthTokenDecodeError("Invalid session token encoding") from exc

    expected_sig = hmac.new(
        _session_secret().encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(supplied_sig, expected_sig):
        raise AuthTokenSignatureError("Invalid session token signature")

    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthTokenDecodeError("Invalid session token payload") from exc

    if not isinstance(payload, dict):
        raise AuthTokenDecodeError("Invalid session token payload shape")
    if payload.get("sub") != "server-user":
        raise AuthTokenDecodeError("Invalid session token subject")

    try:
        exp = int(payload.get("exp") or 0)
    except (TypeError, ValueError) as exc:
        raise AuthTokenDecodeError("Invalid session token expiry") from exc
    now = int(time.time())
    if exp <= now:
        raise AuthTokenExpiredError("Session token expired")
    return payload


def _b64_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64_decode(value: str) -> bytes:
    padding = "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _request_is_secure(request: Request | None) -> bool:
    if settings.server_auth_secure_cookie:
        return True
    if request is None:
        return False
    return request.url.scheme == "https"
