"""Resource-routing and overlay-dismissal policies for browser automation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from xui_reader.config import RuntimeConfig

DEFAULT_BLOCKED_RESOURCE_TYPES = frozenset({"image", "media", "font", "stylesheet"})
DEFAULT_ALLOWED_OVERLAY_SELECTORS = (
    'button:has-text("Not now")',
    'button:has-text("No thanks")',
    'button[aria-label="Close"]',
    'div[role="dialog"] button[aria-label="Close"]',
)

_LOGIN_URL_PATHS = frozenset({"/i/flow/login", "/login"})
_CHALLENGE_URL_MARKERS = ("/account/access", "/account/login_challenge", "/challenge")
_LOGIN_TITLE_MARKERS = ("sign in", "log in")
_CHALLENGE_TITLE_MARKERS = ("challenge", "verify", "suspicious")
_LOGIN_BODY_MARKERS = (
    "sign in to x",
    "you need to log in",
    "please log in",
    "session expired",
    "your session has expired",
)
_CHALLENGE_BODY_MARKERS = (
    "confirm it",
    "verify your identity",
    "unusual activity",
    "enter the code",
    "suspicious login",
)


class RequestRoute(Protocol):
    def abort(self) -> Any:
        """Abort current request."""

    def continue_(self) -> Any:
        """Continue current request."""


class RequestLike(Protocol):
    @property
    def resource_type(self) -> str:
        """Request resource type."""


class RoutablePage(Protocol):
    def route(
        self,
        url: str,
        handler: Callable[[RequestRoute, RequestLike], Any],
    ) -> Any:
        """Register request-routing handler."""


class OverlayElement(Protocol):
    def click(self, timeout: int | None = None) -> Any:
        """Click overlay element."""

    def is_visible(self) -> bool:
        """Return true when element is visible."""


class OverlayPage(Protocol):
    @property
    def url(self) -> str:
        """Current page URL."""

    def title(self) -> str:
        """Current page title."""

    def inner_text(self, selector: str, timeout: int | None = None) -> str:
        """Get text from selector."""

    def query_selector(self, selector: str) -> OverlayElement | None:
        """Query for single element."""


@dataclass(frozen=True)
class ResourceRoutingPolicy:
    enabled: bool
    blocked_resource_types: frozenset[str]


@dataclass(frozen=True)
class OverlayDismissResult:
    dismissed_count: int
    attempted_selectors: tuple[str, ...]
    blocked_category: str | None = None
    skipped_reason: str | None = None


def configure_resource_routing(
    page: RoutablePage,
    config: RuntimeConfig,
    *,
    blocked_resource_types: Iterable[str] | None = None,
) -> ResourceRoutingPolicy:
    """Install request interception using runtime configuration."""
    return install_resource_routing(
        page,
        block_resources=config.browser.block_resources,
        blocked_resource_types=blocked_resource_types,
    )


def install_resource_routing(
    page: RoutablePage,
    *,
    block_resources: bool,
    blocked_resource_types: Iterable[str] | None = None,
) -> ResourceRoutingPolicy:
    """Install request interception for configured blocked resource types."""
    if not block_resources:
        return ResourceRoutingPolicy(enabled=False, blocked_resource_types=frozenset())

    blocked = _normalize_resource_types(blocked_resource_types)
    if not blocked:
        return ResourceRoutingPolicy(enabled=False, blocked_resource_types=frozenset())

    handler = make_resource_route_handler(blocked)
    page.route("**/*", handler)
    return ResourceRoutingPolicy(enabled=True, blocked_resource_types=blocked)


def make_resource_route_handler(
    blocked_resource_types: Iterable[str],
) -> Callable[[RequestRoute, RequestLike], Any]:
    """Build a route handler that aborts blocked resource types."""
    blocked = _normalize_resource_types(blocked_resource_types)

    def _handler(route: RequestRoute, request: RequestLike) -> Any:
        resource_type = str(getattr(request, "resource_type", "")).strip().lower()
        if resource_type in blocked:
            return route.abort()
        return route.continue_()

    return _handler


def dismiss_allowed_overlays(
    page: OverlayPage,
    *,
    allowed_selectors: Sequence[str] = DEFAULT_ALLOWED_OVERLAY_SELECTORS,
    max_clicks: int = 3,
) -> OverlayDismissResult:
    """Dismiss known benign overlays; never interact during blocked auth/challenge states."""
    blocked_category = detect_unsupported_overlay_state(
        current_url=page.url,
        page_title=page.title(),
        body_text=_read_body_text(page),
    )
    if blocked_category is not None:
        return OverlayDismissResult(
            dismissed_count=0,
            attempted_selectors=(),
            blocked_category=blocked_category,
            skipped_reason="unsupported_blocked_state",
        )

    dismissed_count = 0
    attempted: list[str] = []
    for selector in allowed_selectors:
        if dismissed_count >= max_clicks:
            break
        attempted.append(selector)

        try:
            element = page.query_selector(selector)
        except Exception:
            continue
        if element is None:
            continue
        if not _is_visible(element):
            continue
        if _safe_click(element):
            dismissed_count += 1

    return OverlayDismissResult(
        dismissed_count=dismissed_count,
        attempted_selectors=tuple(attempted),
    )


def detect_unsupported_overlay_state(current_url: str, page_title: str, body_text: str) -> str | None:
    """Return blocked category if page appears to be login wall or account challenge."""
    lowered_url = str(current_url).lower()
    lowered_title = str(page_title).lower()
    lowered_body = str(body_text).lower()
    url_path = urlparse(lowered_url).path

    if any(marker in url_path for marker in _CHALLENGE_URL_MARKERS) or any(
        marker in lowered_title for marker in _CHALLENGE_TITLE_MARKERS
    ) or any(marker in lowered_body for marker in _CHALLENGE_BODY_MARKERS):
        return "challenge"

    if url_path in _LOGIN_URL_PATHS or any(marker in lowered_title for marker in _LOGIN_TITLE_MARKERS):
        return "login_wall"

    likely_authenticated_url = lowered_url.startswith("https://x.com/home") or lowered_url.startswith(
        "https://twitter.com/home"
    )
    if not likely_authenticated_url and any(marker in lowered_body for marker in _LOGIN_BODY_MARKERS):
        return "login_wall"
    return None


def _normalize_resource_types(resource_types: Iterable[str] | None) -> frozenset[str]:
    source = DEFAULT_BLOCKED_RESOURCE_TYPES if resource_types is None else resource_types
    normalized = {
        str(resource_type).strip().lower()
        for resource_type in source
        if str(resource_type).strip()
    }
    return frozenset(normalized)


def _read_body_text(page: OverlayPage) -> str:
    try:
        body_text = page.inner_text("body", timeout=2_000)
    except Exception:
        return ""
    return str(body_text)


def _is_visible(element: OverlayElement) -> bool:
    try:
        return bool(element.is_visible())
    except Exception:
        return False


def _safe_click(element: OverlayElement) -> bool:
    try:
        element.click(timeout=1_000)
    except TypeError:
        try:
            element.click()
        except Exception:
            return False
    except Exception:
        return False
    return True
