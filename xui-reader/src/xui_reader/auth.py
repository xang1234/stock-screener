"""Auth lifecycle helpers for login/status/logout and secure storage_state handling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

from .browser.session import PlaywrightBrowserSession
from .config import RuntimeConfig, load_runtime_config
from .errors import AuthError, BrowserError
from .profiles import profiles_root

DEFAULT_LOGIN_URL = "https://x.com/i/flow/login"
PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

StorageCaptureFn = Callable[[RuntimeConfig, str], dict[str, Any]]
StatusProbeFn = Callable[[RuntimeConfig, Path], "AuthProbeSnapshot"]


@dataclass(frozen=True)
class AuthBlockDetection:
    category: str
    message: str
    next_steps: tuple[str, ...]


@dataclass(frozen=True)
class AuthProbeSnapshot:
    current_url: str
    page_title: str
    body_text: str = ""


@dataclass(frozen=True)
class AuthStatusResult:
    profile: str
    storage_state_path: Path
    authenticated: bool
    status_code: str
    message: str
    next_steps: tuple[str, ...] = ()
    detection: AuthBlockDetection | None = None


@dataclass(frozen=True)
class AuthLogoutResult:
    profile: str
    storage_state_path: Path
    removed: bool
    message: str
    next_steps: tuple[str, ...] = ()


def auth_status_to_dict(result: AuthStatusResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["storage_state_path"] = str(result.storage_state_path)
    return payload


def auth_logout_to_dict(result: AuthLogoutResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["storage_state_path"] = str(result.storage_state_path)
    return payload


def storage_state_path(profile_name: str, config_path: str | Path | None = None) -> Path:
    """Return the canonical storage-state path for a profile."""
    normalized_name = _validate_profile_name(profile_name)
    return profiles_root(config_path) / normalized_name / "session" / "storage_state.json"


def login_and_save_storage_state(
    profile_name: str | None = None,
    config_path: str | Path | None = None,
    *,
    login_url: str = DEFAULT_LOGIN_URL,
    capture_fn: StorageCaptureFn | None = None,
) -> Path:
    """Capture storage_state via manual browser login and persist with restrictive permissions."""
    config = load_runtime_config(config_path)
    selected_profile = _resolve_profile_name(profile_name, config)
    profile_dir = _resolve_existing_profile_dir(selected_profile, config_path)
    session_dir = profile_dir / "session"
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AuthError(
            f"Could not prepare session directory '{session_dir}': {exc}. Check directory permissions."
        ) from exc

    collector = capture_fn or _capture_storage_state_via_playwright
    storage_state = collector(config, login_url)
    _validate_storage_state(storage_state)

    target_path = storage_state_path(selected_profile, config_path)
    _write_storage_state_secure(target_path, storage_state)
    return target_path


def probe_auth_status(
    profile_name: str | None = None,
    config_path: str | Path | None = None,
    *,
    probe_fn: StatusProbeFn | None = None,
) -> AuthStatusResult:
    """Fail-closed auth status probe with categorized login-wall/challenge messaging."""
    config = load_runtime_config(config_path)
    selected_profile = _resolve_profile_name(profile_name, config)
    storage_path = storage_state_path(selected_profile, config_path)

    if not _path_exists(storage_path):
        login_cmd = _login_command_hint(selected_profile, config_path)
        return AuthStatusResult(
            profile=selected_profile,
            storage_state_path=storage_path,
            authenticated=False,
            status_code="missing_storage_state",
            message=(
                f"No storage_state file found for profile '{selected_profile}'. "
                "Session validity cannot be confirmed."
            ),
            next_steps=(f"Run `{login_cmd}` to authenticate.",),
        )
    if _path_is_dir(storage_path):
        raise AuthError(f"Storage state path '{storage_path}' is a directory; expected a file.")

    storage_state = _read_storage_state_file(storage_path)
    _validate_storage_state(storage_state)

    detector = probe_fn or _probe_auth_with_playwright
    try:
        snapshot = detector(config, storage_path)
    except AuthError as exc:
        login_cmd = _login_command_hint(selected_profile, config_path)
        return AuthStatusResult(
            profile=selected_profile,
            storage_state_path=storage_path,
            authenticated=False,
            status_code="unconfirmed",
            message=(
                "Could not confirm an authenticated session (fail-closed). "
                f"Probe error: {exc}"
            ),
            next_steps=(f"Re-authenticate with `{login_cmd}` and retry `xui auth status`.",),
        )

    detection = detect_login_wall_or_challenge(
        snapshot.current_url,
        snapshot.page_title,
        snapshot.body_text,
        selected_profile,
        config_path,
    )
    if detection is not None:
        return AuthStatusResult(
            profile=selected_profile,
            storage_state_path=storage_path,
            authenticated=False,
            status_code=f"blocked_{detection.category}",
            message=detection.message,
            next_steps=detection.next_steps,
            detection=detection,
        )

    if _appears_authenticated(snapshot):
        return AuthStatusResult(
            profile=selected_profile,
            storage_state_path=storage_path,
            authenticated=True,
            status_code="authenticated",
            message="Authenticated session confirmed.",
        )

    login_cmd = _login_command_hint(selected_profile, config_path)
    return AuthStatusResult(
        profile=selected_profile,
        storage_state_path=storage_path,
        authenticated=False,
        status_code="unconfirmed",
        message=(
            "Session could not be confirmed from the current page state (fail-closed)."
        ),
        next_steps=(f"Re-authenticate with `{login_cmd}` and retry `xui auth status`.",),
    )


def logout_profile(
    profile_name: str | None = None, config_path: str | Path | None = None
) -> AuthLogoutResult:
    """Remove persisted session state and provide deterministic re-login guidance."""
    config = load_runtime_config(config_path)
    selected_profile = _resolve_profile_name(profile_name, config)
    target = storage_state_path(selected_profile, config_path)
    if _path_is_dir(target):
        raise AuthError(f"Storage state path '{target}' is a directory; expected a file.")

    next_steps = (_login_command_hint(selected_profile, config_path),)
    if not _path_exists(target):
        return AuthLogoutResult(
            profile=selected_profile,
            storage_state_path=target,
            removed=False,
            message=(
                f"No storage_state file found for profile '{selected_profile}'. "
                "Already logged out."
            ),
            next_steps=next_steps,
        )

    try:
        target.unlink()
    except OSError as exc:
        raise AuthError(
            f"Could not remove storage_state file '{target}': {exc}. Check file permissions."
        ) from exc
    return AuthLogoutResult(
        profile=selected_profile,
        storage_state_path=target,
        removed=True,
        message=f"Removed storage_state for profile '{selected_profile}'.",
        next_steps=next_steps,
    )


def detect_login_wall_or_challenge(
    current_url: str,
    page_title: str,
    body_text: str,
    profile_name: str,
    config_path: str | Path | None = None,
) -> AuthBlockDetection | None:
    """Categorize login-wall/challenge states and return explicit manual next actions."""
    login_cmd = _login_command_hint(profile_name, config_path)
    lowered_url = current_url.lower()
    lowered_title = page_title.lower()
    lowered_body = body_text.lower()

    likely_authenticated_url = lowered_url.startswith("https://x.com/home") or lowered_url.startswith(
        "https://twitter.com/home"
    )
    login_url_markers = ("/i/flow/login", "/login")
    login_title_markers = ("sign in", "log in")
    login_body_markers = (
        "sign in to x",
        "you need to log in",
        "please log in",
        "session expired",
        "your session has expired",
    )
    challenge_markers = (
        "/account/access",
        "/account/login_challenge",
        "/challenge",
        "confirm it",
        "verify your identity",
        "unusual activity",
        "enter the code",
        "suspicious login",
    )

    if any(marker in lowered_url for marker in login_url_markers) or any(
        marker in lowered_title for marker in login_title_markers
    ):
        return AuthBlockDetection(
            category="login_wall",
            message="Login wall detected. Session is not authenticated.",
            next_steps=(
                f"Run `{login_cmd}` and complete manual login.",
                "Re-run `xui auth status` to confirm session validity.",
            ),
        )

    if any(marker in lowered_url for marker in ("/account/access", "/challenge")) or any(
        marker in lowered_title for marker in ("challenge", "verify", "suspicious")
    ) or any(marker in lowered_body for marker in challenge_markers):
        return AuthBlockDetection(
            category="challenge",
            message="Account challenge detected. Manual verification is required.",
            next_steps=(
                f"Run `{login_cmd}` and complete the challenge flow in-browser.",
                "After verification, run `xui auth status` again.",
            ),
        )

    # Body-only checks are intentionally strict to avoid false positives from benign
    # text like "log in to another account" on authenticated pages.
    if not likely_authenticated_url and any(marker in lowered_body for marker in login_body_markers):
        return AuthBlockDetection(
            category="login_wall",
            message="Authentication prompt detected on page content.",
            next_steps=(
                f"Run `{login_cmd}` and complete manual login.",
                "Re-run `xui auth status` to confirm session validity.",
            ),
        )
    return None


def _resolve_profile_name(profile_name: str | None, config: RuntimeConfig) -> str:
    chosen = profile_name if profile_name else config.app.default_profile
    return _validate_profile_name(chosen)


def _validate_profile_name(profile_name: str) -> str:
    normalized_name = profile_name.strip()
    if not PROFILE_NAME_RE.match(normalized_name):
        raise AuthError(
            "Invalid profile name. Use 1-64 chars matching [A-Za-z0-9._-] and start with an alphanumeric character."
        )
    return normalized_name


def _resolve_existing_profile_dir(
    profile_name: str, config_path: str | Path | None = None
) -> Path:
    profile_dir = profiles_root(config_path) / profile_name
    try:
        exists = profile_dir.exists()
        is_dir = profile_dir.is_dir() if exists else False
    except OSError as exc:
        raise AuthError(
            f"Could not inspect profile path '{profile_dir}': {exc}. Check directory permissions."
        ) from exc

    if not exists or not is_dir:
        raise AuthError(
            f"Profile '{profile_name}' is not initialized at '{profile_dir}'. "
            f"Run `xui profiles create {profile_name}` before `xui auth login`."
        )
    return profile_dir


def _capture_storage_state_via_playwright(config: RuntimeConfig, login_url: str) -> dict[str, Any]:
    try:
        session = PlaywrightBrowserSession(config, headless=False)
        with session:
            page = session.new_page()
            page.goto(login_url, wait_until="load", timeout=session.options.navigation_timeout_ms)

            input(
                "Complete login in the opened browser window, then press Enter here to capture storage state..."
            )
            captured = session.storage_state()
    except BrowserError as exc:
        raise AuthError(str(exc)) from exc
    except Exception as exc:
        raise AuthError(
            f"Failed to capture auth storage_state from '{login_url}': {exc}. "
            "Complete login manually and retry."
        ) from exc

    if not isinstance(captured, dict):
        raise AuthError("Playwright returned invalid storage_state payload.")
    return captured


def _probe_auth_with_playwright(config: RuntimeConfig, storage_state_file: Path) -> AuthProbeSnapshot:
    try:
        session = PlaywrightBrowserSession(
            config,
            headless=True,
            storage_state=storage_state_file,
        )
        with session:
            page = session.new_page()
            page.goto(
                "https://x.com/home",
                wait_until="domcontentloaded",
                timeout=session.options.navigation_timeout_ms,
            )
            current_url = page.url
            page_title = page.title()
            try:
                body_text = page.inner_text("body", timeout=2_000)
            except Exception:
                body_text = ""
            return AuthProbeSnapshot(
                current_url=str(current_url),
                page_title=str(page_title),
                body_text=str(body_text),
            )
    except BrowserError as exc:
        raise AuthError(str(exc)) from exc
    except Exception as exc:
        raise AuthError(f"Auth status probe failed: {exc}") from exc


def _validate_storage_state(storage_state: dict[str, Any]) -> None:
    cookies = storage_state.get("cookies")
    origins = storage_state.get("origins")
    cookies_ok = isinstance(cookies, list)
    origins_ok = isinstance(origins, list)

    if not cookies_ok or not origins_ok:
        raise AuthError(
            "Captured storage_state is invalid: expected Playwright cookies/origins arrays."
        )
    if len(cookies) == 0 and len(origins) == 0:
        raise AuthError(
            "Captured storage_state is empty. Ensure login completed successfully before pressing Enter."
        )


def _write_storage_state_secure(path: Path, storage_state: dict[str, Any]) -> None:
    serialized = json.dumps(storage_state, indent=2, sort_keys=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    fd: int | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = None
            stream.write(serialized)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
    except OSError as exc:
        raise AuthError(
            f"Could not persist storage_state at '{path}': {exc}. Check directory permissions."
        ) from exc
    finally:
        if fd is not None:
            os.close(fd)
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def _appears_authenticated(snapshot: AuthProbeSnapshot) -> bool:
    lowered_url = snapshot.current_url.lower()
    if any(marker in lowered_url for marker in ("/i/flow/login", "/account/access", "/challenge")):
        return False
    return lowered_url.startswith("https://x.com/home") or lowered_url.startswith("https://twitter.com/home")


def _read_storage_state_file(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuthError(
            f"Could not read storage_state file '{path}': {exc}. Check file permissions."
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AuthError(
            f"Storage_state file '{path}' is not valid JSON: {exc}. Re-run `xui auth login`."
        ) from exc
    if not isinstance(data, dict):
        raise AuthError(
            f"Storage_state file '{path}' has invalid structure: expected JSON object."
        )
    return data


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError as exc:
        raise AuthError(f"Could not access path '{path}': {exc}. Check file permissions.") from exc


def _path_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError as exc:
        raise AuthError(f"Could not inspect path '{path}': {exc}. Check file permissions.") from exc


def _login_command_hint(profile_name: str, config_path: str | Path | None = None) -> str:
    parts = ["xui", "auth", "login", "--profile", profile_name]
    if config_path is not None:
        parts.extend(["--path", str(Path(config_path).expanduser())])
    return " ".join(parts)
