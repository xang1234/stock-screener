"""Single-user server authentication helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request
from fastapi.responses import Response

from ..config import settings


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


async def require_server_session(request: Request) -> bool:
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
    return _server_auth_password()


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
        raw_payload, raw_sig = token.split(".", 1)
    except ValueError:
        return False

    try:
        payload_bytes = _b64_decode(raw_payload)
        supplied_sig = _b64_decode(raw_sig)
    except Exception:
        return False

    expected_sig = hmac.new(
        _session_secret().encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(supplied_sig, expected_sig):
        return False

    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return False

    exp = int(payload.get("exp") or 0)
    now = int(time.time())
    return exp > now and payload.get("sub") == "server-user"


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
