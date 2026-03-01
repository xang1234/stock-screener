"""Bridge service for importing X session cookies from a browser extension."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import importlib
import json
from pathlib import Path
import secrets
import sys
import threading
import time
from typing import Any, Callable
from urllib.parse import urlparse

from ..config import settings
from .redis_pool import get_redis_client


@dataclass(frozen=True)
class _XUIAuthBindings:
    probe_auth_status: Callable[..., Any]
    save_storage_state: Callable[..., Any]


@dataclass(frozen=True)
class SessionChallenge:
    challenge_id: str
    challenge_token: str
    expires_at: datetime
    ttl_seconds: int


@dataclass(frozen=True)
class TwitterSessionStatus:
    authenticated: bool
    status_code: str
    message: str
    profile: str
    storage_state_path: str
    provider: str = "xui"


class XUISessionBridgeError(RuntimeError):
    """Typed bridge error with HTTP status mapping."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class XUISessionBridgeService:
    """Issue one-time import challenges and persist XUI storage state."""

    _challenge_lock = threading.Lock()
    _memory_challenges: dict[str, tuple[float, dict[str, str]]] = {}
    _memory_rate_limits: dict[str, tuple[float, int]] = {}
    _challenge_prefix = "xui_bridge:challenge:"
    _rate_prefix = "xui_bridge:rate:"
    _rate_window_seconds = 60
    _rate_limit_per_window = 30

    def get_auth_status(self) -> TwitterSessionStatus:
        self._ensure_enabled()
        bindings = _load_xui_auth_bindings()
        result = _call_probe_auth_status(
            bindings.probe_auth_status,
            profile_name=settings.xui_profile,
            config_path=settings.xui_config_path,
        )
        return _to_twitter_session_status(result)

    def create_import_challenge(
        self,
        *,
        origin: str | None,
        client_key: str | None,
    ) -> SessionChallenge:
        self._ensure_enabled()
        normalized_origin = self._require_allowed_origin(origin)
        self._enforce_rate_limit("challenge", client_key or normalized_origin)

        ttl = int(max(30, settings.xui_bridge_challenge_ttl_seconds))
        challenge_id = secrets.token_urlsafe(18)
        challenge_token = secrets.token_urlsafe(36)
        token_hash = _token_hash(challenge_token)
        expires_ts = time.time() + ttl

        self._store_challenge(
            challenge_id,
            {
                "token_hash": token_hash,
                "origin": normalized_origin,
                "expires_ts": str(expires_ts),
            },
            ttl,
        )
        return SessionChallenge(
            challenge_id=challenge_id,
            challenge_token=challenge_token,
            expires_at=datetime.fromtimestamp(expires_ts, tz=timezone.utc),
            ttl_seconds=ttl,
        )

    def consume_import_challenge(
        self,
        *,
        challenge_id: str,
        challenge_token: str,
        origin: str | None,
    ) -> None:
        normalized_origin = self._require_allowed_origin(origin)
        challenge_id = str(challenge_id or "").strip()
        challenge_token = str(challenge_token or "").strip()
        if not challenge_id or not challenge_token:
            raise XUISessionBridgeError(401, "Missing challenge credentials.")

        payload = self._load_challenge(challenge_id)
        if not payload:
            raise XUISessionBridgeError(401, "Challenge is invalid or expired.")

        # Consume challenge on first validation attempt to prevent replay/brute-force.
        self._delete_challenge(challenge_id)

        if payload.get("origin") != normalized_origin:
            raise XUISessionBridgeError(
                403,
                "Challenge origin mismatch. Re-issue challenge from this browser tab.",
            )
        stored_hash = payload.get("token_hash", "")
        supplied_hash = _token_hash(challenge_token)
        if not hmac.compare_digest(stored_hash, supplied_hash):
            raise XUISessionBridgeError(401, "Challenge token is invalid or expired.")

    def import_browser_cookies(
        self,
        *,
        challenge_id: str,
        challenge_token: str,
        cookies: list[dict[str, Any]],
        origin: str | None,
        client_key: str | None,
        browser: str | None = None,
        extension_version: str | None = None,
    ) -> TwitterSessionStatus:
        self._ensure_enabled()
        normalized_origin = self._require_allowed_origin(origin)
        self._enforce_rate_limit("import", client_key or normalized_origin)
        self.consume_import_challenge(
            challenge_id=challenge_id,
            challenge_token=challenge_token,
            origin=normalized_origin,
        )

        normalized_cookies = _normalize_browser_cookies(
            cookies,
            max_cookies=max(1, settings.xui_bridge_max_cookies),
        )
        cookie_names = {cookie["name"] for cookie in normalized_cookies}
        required = {"auth_token", "ct0"}
        missing = sorted(required - cookie_names)
        if missing:
            raise XUISessionBridgeError(
                422,
                "Missing required X auth cookies after normalization: " + ", ".join(missing),
            )

        bindings = _load_xui_auth_bindings()
        storage_state = {"cookies": normalized_cookies, "origins": []}
        try:
            bindings.save_storage_state(
                profile_name=settings.xui_profile,
                config_path=settings.xui_config_path,
                storage_state=storage_state,
                create_profile_if_missing=True,
                init_config_if_missing=True,
            )
            status = _call_probe_auth_status(
                bindings.probe_auth_status,
                profile_name=settings.xui_profile,
                config_path=settings.xui_config_path,
            )
            return _to_twitter_session_status(status)
        except XUISessionBridgeError:
            raise
        except Exception as exc:
            raise XUISessionBridgeError(
                500,
                f"Failed to persist or validate imported X session state: {exc}",
            ) from exc

    def _ensure_enabled(self) -> None:
        if not settings.xui_bridge_enabled:
            raise XUISessionBridgeError(503, "XUI browser session bridge is disabled.")

    def _require_allowed_origin(self, origin: str | None) -> str:
        parsed_origin = _normalize_origin(origin)
        if not parsed_origin:
            raise XUISessionBridgeError(403, "Origin header is required for this operation.")
        if _origin_allowed(parsed_origin, settings.xui_bridge_allowed_origins):
            return parsed_origin
        raise XUISessionBridgeError(403, f"Origin '{parsed_origin}' is not allowed.")

    def _enforce_rate_limit(self, action: str, key: str) -> None:
        fingerprint = hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:20]
        redis_key = f"{self._rate_prefix}{action}:{fingerprint}"
        client = get_redis_client()
        if client is not None:
            try:
                count = int(client.incr(redis_key))
                if count == 1:
                    client.expire(redis_key, self._rate_window_seconds)
                if count > self._rate_limit_per_window:
                    raise XUISessionBridgeError(
                        429,
                        "Rate limit exceeded for X session bridge. Retry shortly.",
                    )
                return
            except XUISessionBridgeError:
                raise
            except Exception:
                pass

        now = time.time()
        with self._challenge_lock:
            window_start, count = self._memory_rate_limits.get(redis_key, (now, 0))
            if now - window_start >= self._rate_window_seconds:
                window_start, count = now, 0
            count += 1
            self._memory_rate_limits[redis_key] = (window_start, count)
            if count > self._rate_limit_per_window:
                raise XUISessionBridgeError(
                    429,
                    "Rate limit exceeded for X session bridge. Retry shortly.",
                )

    def _store_challenge(self, challenge_id: str, payload: dict[str, str], ttl: int) -> None:
        key = f"{self._challenge_prefix}{challenge_id}"
        serialized = json.dumps(payload, separators=(",", ":"))
        client = get_redis_client()
        if client is not None:
            try:
                client.setex(key, ttl, serialized.encode("utf-8"))
                return
            except Exception:
                pass

        expires_ts = time.time() + ttl
        with self._challenge_lock:
            self._prune_expired_memory_challenges()
            self._memory_challenges[key] = (expires_ts, payload)

    def _load_challenge(self, challenge_id: str) -> dict[str, str] | None:
        key = f"{self._challenge_prefix}{challenge_id}"
        client = get_redis_client()
        if client is not None:
            try:
                raw = client.get(key)
                if not raw:
                    return None
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return {str(k): str(v) for k, v in parsed.items()}
                return None
            except Exception:
                pass

        with self._challenge_lock:
            self._prune_expired_memory_challenges()
            entry = self._memory_challenges.get(key)
            if not entry:
                return None
            _expires, payload = entry
            return dict(payload)

    def _delete_challenge(self, challenge_id: str) -> None:
        key = f"{self._challenge_prefix}{challenge_id}"
        client = get_redis_client()
        if client is not None:
            try:
                client.delete(key)
            except Exception:
                pass

        with self._challenge_lock:
            self._memory_challenges.pop(key, None)

    def _prune_expired_memory_challenges(self) -> None:
        now = time.time()
        expired_keys = [key for key, (expires_ts, _payload) in self._memory_challenges.items() if expires_ts <= now]
        for key in expired_keys:
            self._memory_challenges.pop(key, None)


def _normalize_browser_cookies(
    cookies: list[dict[str, Any]],
    *,
    max_cookies: int,
) -> list[dict[str, Any]]:
    if not isinstance(cookies, list):
        raise XUISessionBridgeError(400, "Cookie payload must be a list.")
    if not cookies:
        raise XUISessionBridgeError(400, "Cookie payload is empty.")
    if len(cookies) > max_cookies:
        raise XUISessionBridgeError(
            400,
            f"Cookie payload has too many entries ({len(cookies)} > {max_cookies}).",
        )

    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw_cookie in cookies:
        if not isinstance(raw_cookie, dict):
            continue
        name = str(raw_cookie.get("name", "")).strip()
        value_obj = raw_cookie.get("value")
        value = str(value_obj) if value_obj is not None else ""
        domain = _normalize_cookie_domain(raw_cookie.get("domain"))
        if not name or not value or not domain:
            continue
        if not _is_x_domain(domain):
            continue
        path = str(raw_cookie.get("path") or "/").strip() or "/"
        secure = bool(raw_cookie.get("secure", False))
        http_only = bool(raw_cookie.get("httpOnly", False))
        same_site = _normalize_same_site(raw_cookie.get("sameSite"))
        expires = _normalize_cookie_expiry(
            raw_cookie.get("expirationDate"),
            raw_cookie.get("expires"),
        )
        normalized = {
            "name": name,
            "value": value,
            "domain": domain,
            "path": path,
            "secure": secure,
            "httpOnly": http_only,
            "sameSite": same_site,
            "expires": expires,
        }
        by_key[(name, domain, path)] = normalized

    normalized_cookies = list(by_key.values())
    if not normalized_cookies:
        raise XUISessionBridgeError(
            422,
            "No usable x.com/twitter.com cookies were provided by the extension.",
        )
    return normalized_cookies


def _normalize_cookie_domain(value: Any) -> str:
    domain = str(value or "").strip().lower()
    if not domain:
        return ""
    if domain.startswith("."):
        domain = domain[1:]
    return domain


def _is_x_domain(domain: str) -> bool:
    return domain == "x.com" or domain.endswith(".x.com") or domain == "twitter.com" or domain.endswith(".twitter.com")


def _normalize_same_site(value: Any) -> str:
    if value is None:
        return "Lax"
    same_site = str(value).strip().lower()
    mapping = {
        "strict": "Strict",
        "lax": "Lax",
        "none": "None",
        "no_restriction": "None",
        "unspecified": "Lax",
    }
    return mapping.get(same_site, "Lax")


def _normalize_cookie_expiry(expiration_date: Any, expires: Any) -> int:
    for candidate in (expiration_date, expires):
        if candidate is None:
            continue
        try:
            parsed = float(candidate)
            if parsed <= 0:
                continue
            return int(parsed)
        except (TypeError, ValueError):
            continue
    # Playwright storage_state uses -1 for session cookies.
    return -1


def _normalize_origin(origin: str | None) -> str | None:
    raw = str(origin or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.hostname:
        return None
    host = parsed.hostname.lower()
    resolved_port = _resolve_default_port(parsed.scheme, parsed.port)
    if resolved_port is None:
        return None
    port = f":{resolved_port}"
    return f"{parsed.scheme}://{host}{port}"


def _origin_allowed(origin: str, allowlist_raw: str) -> bool:
    normalized_origin = _normalize_origin(origin)
    if not normalized_origin:
        return False
    allow_entries = [entry.strip() for entry in str(allowlist_raw or "").split(",") if entry.strip()]
    if not allow_entries:
        return False
    for entry in allow_entries:
        candidate = _normalize_origin(entry)
        if candidate and candidate == normalized_origin:
            return True
    return False


def _resolve_default_port(scheme: str, port: int | None) -> int | None:
    if port is not None:
        return int(port)
    if scheme == "http":
        return 80
    if scheme == "https":
        return 443
    return None


def _to_twitter_session_status(auth_result: Any) -> TwitterSessionStatus:
    return TwitterSessionStatus(
        authenticated=bool(getattr(auth_result, "authenticated", False)),
        status_code=str(getattr(auth_result, "status_code", "")),
        message=str(getattr(auth_result, "message", "")),
        profile=str(getattr(auth_result, "profile", settings.xui_profile)),
        storage_state_path=str(getattr(auth_result, "storage_state_path", "")),
        provider="xui",
    )


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _call_probe_auth_status(probe_fn: Callable[..., Any], **kwargs: Any) -> Any:
    """
    Run Playwright sync probe outside an active asyncio loop.

    FastAPI async endpoints execute in an event-loop thread; calling
    Playwright Sync API there raises a runtime error. Offload probe to a
    worker thread only when needed.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return probe_fn(**kwargs)

    if not loop.is_running():
        return probe_fn(**kwargs)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(probe_fn, **kwargs).result()


def _repo_root_from_here() -> Path | None:
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        candidate = parent / "xui-reader" / "src"
        if candidate.exists():
            return parent
    return None


def _ensure_xui_module_path() -> None:
    repo_root = _repo_root_from_here()
    if repo_root is None:
        return
    xui_src = repo_root / "xui-reader" / "src"
    xui_src_text = str(xui_src)
    if xui_src.exists() and xui_src_text not in sys.path:
        sys.path.insert(0, xui_src_text)


def _load_xui_auth_bindings() -> _XUIAuthBindings:
    try:
        return _import_xui_auth_bindings()
    except ModuleNotFoundError:
        _ensure_xui_module_path()
        try:
            return _import_xui_auth_bindings()
        except ModuleNotFoundError as exc:
            raise XUISessionBridgeError(
                500,
                "xui-reader is not available. Install with `pip install -e ./xui-reader` "
                "and ensure Playwright Chromium is installed (`python -m playwright install chromium`).",
            ) from exc


def _import_xui_auth_bindings() -> _XUIAuthBindings:
    auth_mod = importlib.import_module("xui_reader.auth")
    return _XUIAuthBindings(
        probe_auth_status=getattr(auth_mod, "probe_auth_status"),
        save_storage_state=getattr(auth_mod, "save_storage_state"),
    )
