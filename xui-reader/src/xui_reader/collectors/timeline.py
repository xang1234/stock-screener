"""List/user timeline collector navigation and bounded scroll loops."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Protocol
from urllib.parse import urlparse

from xui_reader.browser.policy import (
    configure_resource_routing,
    dismiss_allowed_overlays,
)
from xui_reader.collectors.base import CollectionBatch, CollectionStats
from xui_reader.config import RuntimeConfig
from xui_reader.errors import CollectError
from xui_reader.models import Checkpoint, SourceKind, SourceRef, TweetItem

_TWEET_STATUS_RE = re.compile(r"/status/(\d+)")
_LIST_ID_RE = re.compile(r"^\d{3,32}$")
_HANDLE_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")
_USER_TAB_TEXT = {
    "posts": ("Posts",),
    "replies": ("Replies", "Posts & replies", "Posts and replies"),
    "media": ("Media",),
}
_USER_TAB_SUFFIX = {
    "posts": "",
    "replies": "/with_replies",
    "media": "/media",
}


class TimelinePage(Protocol):
    @property
    def url(self) -> str:
        """Current page URL."""

    def goto(self, url: str, **kwargs: Any) -> Any:
        """Navigate to a URL."""

    def content(self) -> str:
        """Return page HTML content."""

    def evaluate(self, expression: str) -> Any:
        """Run JavaScript expression on page."""

    def query_selector(self, selector: str) -> Any:
        """Return first matching element for selector."""

    def title(self) -> str:
        """Current page title."""

    def inner_text(self, selector: str, timeout: int | None = None) -> str:
        """Return text for selector."""

    def route(self, url: str, handler: object) -> Any:
        """Install request route."""


class BrowserSessionLike(Protocol):
    def new_page(self) -> TimelinePage:
        """Create and return a page."""


@dataclass(frozen=True)
class ScrollBounds:
    max_scrolls: int = 20
    stagnation_rounds: int = 3


@dataclass(frozen=True)
class TabSelectionResult:
    tab: str
    selected: bool
    strategy: str | None = None
    selector: str | None = None
    attempted_selectors: tuple[str, ...] = ()
    diagnostics: str = ""


class TimelineCollector:
    """Collect list/user timeline DOM payloads with explicit loop bounds."""

    def __init__(
        self,
        config: RuntimeConfig,
        session: BrowserSessionLike,
        *,
        bounds: ScrollBounds = ScrollBounds(),
        user_tab: str = "posts",
        overlay_max_clicks: int = 3,
        route_configurer: Callable[..., object] = configure_resource_routing,
        overlay_dismisser: Callable[..., object] = dismiss_allowed_overlays,
    ) -> None:
        self._config = config
        self._session = session
        self._bounds = _validate_bounds(bounds)
        self._user_tab = _normalize_tab(user_tab)
        self._overlay_max_clicks = max(1, int(overlay_max_clicks))
        self._route_configurer = route_configurer
        self._overlay_dismisser = overlay_dismisser

    def collect(self, source: SourceRef, limit: int | None = None) -> CollectionBatch:
        if limit is not None and limit <= 0:
            raise CollectError("Collection limit must be positive when provided.")

        page = self._session.new_page()
        try:
            self._route_configurer(page, self._config)
            target_url, handle = _canonical_target(source)
            _navigate(page, target_url, self._config.browser.navigation_timeout_ms)

            if source.kind is SourceKind.USER:
                selected_tab = source.tab if source.tab else self._user_tab
                tab_result = select_user_tab(page, handle=handle, tab=selected_tab)
                if not tab_result.selected:
                    raise CollectError(tab_result.diagnostics)

            return _collect_dom_loop(
                page,
                source_id=source.source_id,
                bounds=self._bounds,
                limit=limit,
                overlay_max_clicks=self._overlay_max_clicks,
                overlay_dismisser=self._overlay_dismisser,
            )
        except Exception as exc:
            _attach_collection_diagnostics(exc, page)
            raise


def select_user_tab(page: TimelinePage, *, handle: str, tab: str) -> TabSelectionResult:
    """Select user timeline tab using href-first and text fallback strategy."""
    normalized_handle = _normalize_handle(handle)
    normalized_tab = _normalize_tab(tab)
    attempted: list[str] = []

    for selector in _href_selectors_for_tab(normalized_handle, normalized_tab):
        attempted.append(selector)
        if _click_selector_if_visible(page, selector):
            return TabSelectionResult(
                tab=normalized_tab,
                selected=True,
                strategy="href",
                selector=selector,
                attempted_selectors=tuple(attempted),
            )

    for selector in _text_selectors_for_tab(normalized_tab):
        attempted.append(selector)
        if _click_selector_if_visible(page, selector):
            return TabSelectionResult(
                tab=normalized_tab,
                selected=True,
                strategy="text",
                selector=selector,
                attempted_selectors=tuple(attempted),
            )

    joined = ", ".join(attempted)
    return TabSelectionResult(
        tab=normalized_tab,
        selected=False,
        attempted_selectors=tuple(attempted),
        diagnostics=(
            f"Could not select user tab '{normalized_tab}' for @{normalized_handle}. "
            f"Tried selectors: {joined}. "
            "Verify tab markup and update href/text fallback selectors."
        ),
    )


def canonical_list_url(list_id_or_url: str) -> str:
    """Resolve a list id or URL to canonical X list URL."""
    return f"https://x.com/i/lists/{parse_list_id(list_id_or_url)}"


def canonical_user_url(handle_or_url: str) -> str:
    """Resolve handle or user URL to canonical user timeline URL."""
    return f"https://x.com/{parse_handle(handle_or_url)}"


def parse_list_id(list_id_or_url: str) -> str:
    """Parse and validate numeric list id from raw id or x.com list URL."""
    raw = list_id_or_url.strip()
    if _LIST_ID_RE.fullmatch(raw):
        return raw

    parsed = urlparse(raw)
    segments = [segment for segment in parsed.path.split("/") if segment]
    for index, segment in enumerate(segments):
        if segment == "lists" and index + 1 < len(segments):
            maybe_list_id = segments[index + 1]
            if _LIST_ID_RE.fullmatch(maybe_list_id):
                return maybe_list_id

    raise CollectError(
        f"Could not parse list id from '{list_id_or_url}'. Provide numeric list id or x.com list URL."
    )


def parse_handle(handle_or_url: str) -> str:
    """Parse and validate @handle from raw handle or profile URL."""
    return _normalize_handle(handle_or_url)


def _collect_dom_loop(
    page: TimelinePage,
    *,
    source_id: str,
    bounds: ScrollBounds,
    limit: int | None,
    overlay_max_clicks: int,
    overlay_dismisser: Callable[..., object],
) -> CollectionBatch:
    seen_ids: set[str] = set()
    ordered_ids: list[str] = []
    dom_snapshots: list[str] = []
    stagnation_rounds = 0
    scroll_rounds = 0
    stop_reason = "max_scrolls"

    while True:
        overlay_result = overlay_dismisser(page, max_clicks=overlay_max_clicks)
        blocked_category = _extract_blocked_category(overlay_result)
        if blocked_category is not None:
            raise CollectError(
                "Collection halted due to blocked session state "
                f"('{blocked_category}'). Re-authenticate and retry."
            )
        html = _read_page_content(page)
        dom_snapshots.append(html)
        new_ids = _extract_new_ids(html, seen_ids)
        if new_ids:
            ordered_ids.extend(new_ids)
            stagnation_rounds = 0
        else:
            stagnation_rounds += 1

        if limit is not None and len(ordered_ids) >= limit:
            stop_reason = "limit_reached"
            break
        if stagnation_rounds >= bounds.stagnation_rounds:
            stop_reason = "stagnation"
            break
        if scroll_rounds >= bounds.max_scrolls:
            stop_reason = "max_scrolls"
            break

        _scroll_once(page)
        scroll_rounds += 1

    if limit is not None:
        ordered_ids = ordered_ids[:limit]

    items = tuple(_to_item(tweet_id, source_id) for tweet_id in ordered_ids)
    checkpoint = Checkpoint(
        source_id=source_id,
        last_seen_id=ordered_ids[0] if ordered_ids else None,
    )
    stats = CollectionStats(
        source_id=source_id,
        dom_snapshots=len(dom_snapshots),
        observed_ids=len(ordered_ids),
        scroll_rounds=scroll_rounds,
        stagnation_rounds=stagnation_rounds,
        stop_reason=stop_reason,
    )
    return CollectionBatch(
        items=items,
        checkpoint=checkpoint,
        dom_snapshots=tuple(dom_snapshots),
        stats=stats,
    )


def _canonical_target(source: SourceRef) -> tuple[str, str]:
    if source.kind is SourceKind.LIST:
        return canonical_list_url(source.value), ""
    if source.kind is SourceKind.USER:
        handle = _normalize_handle(source.value)
        return canonical_user_url(handle), handle
    raise CollectError(f"Unsupported source kind '{source.kind}'.")


def _normalize_handle(value: str) -> str:
    raw = value.strip()
    if raw.startswith("@"):
        raw = raw[1:]
    elif "://" in raw:
        parsed = urlparse(raw)
        segments = [segment for segment in parsed.path.split("/") if segment]
        if not segments:
            raise CollectError(f"Could not parse handle from '{value}'.")
        raw = segments[0].lstrip("@")
    if not _HANDLE_RE.fullmatch(raw):
        raise CollectError(
            f"Invalid handle '{value}'. Use a handle like '@someuser' or user URL."
        )
    return raw


def _normalize_tab(tab: str) -> str:
    normalized = tab.strip().lower()
    if normalized not in _USER_TAB_TEXT:
        raise CollectError(
            f"Unsupported user tab '{tab}'. Supported tabs: posts, replies, media."
        )
    return normalized


def _validate_bounds(bounds: ScrollBounds) -> ScrollBounds:
    if bounds.max_scrolls < 0:
        raise CollectError("max_scrolls must be >= 0.")
    if bounds.stagnation_rounds <= 0:
        raise CollectError("stagnation_rounds must be > 0.")
    return bounds


def _href_selectors_for_tab(handle: str, tab: str) -> tuple[str, ...]:
    suffix = _USER_TAB_SUFFIX[tab]
    if suffix:
        return (
            f'nav a[href="/{handle}{suffix}"]',
            f'nav a[href="/{handle}{suffix}/"]',
            f'nav a[href^="/{handle}"][href*="{suffix}"]',
        )
    return (
        f'nav a[href="/{handle}"]',
        f'nav a[href="/{handle}/"]',
    )


def _text_selectors_for_tab(tab: str) -> tuple[str, ...]:
    selectors: list[str] = []
    for text in _USER_TAB_TEXT[tab]:
        selectors.append(f'nav a:has-text("{text}")')
    return tuple(selectors)


def _click_selector_if_visible(page: TimelinePage, selector: str) -> bool:
    try:
        element = page.query_selector(selector)
    except Exception:
        return False
    if element is None:
        return False
    try:
        visible = bool(element.is_visible())
    except Exception:
        return False
    if not visible:
        return False
    try:
        element.click(timeout=1_500)
    except TypeError:
        try:
            element.click()
        except Exception:
            return False
    except Exception:
        return False
    return True


def _extract_new_ids(html: str, seen_ids: set[str]) -> list[str]:
    new_ids: list[str] = []
    for tweet_id in _TWEET_STATUS_RE.findall(html):
        if tweet_id in seen_ids:
            continue
        seen_ids.add(tweet_id)
        new_ids.append(tweet_id)
    return new_ids


def _read_page_content(page: TimelinePage) -> str:
    try:
        return str(page.content())
    except Exception as exc:
        raise CollectError(f"Could not read page content during collection: {exc}") from exc


def _scroll_once(page: TimelinePage) -> None:
    try:
        page.evaluate("window.scrollBy(0, document.body.scrollHeight);")
    except Exception as exc:
        raise CollectError(f"Scroll operation failed: {exc}") from exc


def _navigate(page: TimelinePage, url: str, timeout_ms: int) -> None:
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    except Exception as exc:
        raise CollectError(f"Could not navigate to '{url}': {exc}") from exc


def _to_item(tweet_id: str, source_id: str) -> TweetItem:
    return TweetItem(
        tweet_id=tweet_id,
        created_at=datetime.now(timezone.utc),
        author_handle="",
        text="",
        source_id=source_id,
    )


def _extract_blocked_category(overlay_result: object) -> str | None:
    if isinstance(overlay_result, dict):
        candidate = overlay_result.get("blocked_category")
        return str(candidate) if candidate else None
    candidate = getattr(overlay_result, "blocked_category", None)
    return str(candidate) if candidate else None


def _attach_collection_diagnostics(exc: Exception, page: TimelinePage) -> None:
    try:
        setattr(exc, "dom_snapshot", _read_page_content(page))
    except Exception:
        pass

    screenshot = getattr(page, "screenshot", None)
    if not callable(screenshot):
        return
    try:
        payload = screenshot(type="png", full_page=True)
    except Exception:
        return
    if isinstance(payload, (bytes, bytearray)):
        setattr(exc, "screenshot_png", bytes(payload))
