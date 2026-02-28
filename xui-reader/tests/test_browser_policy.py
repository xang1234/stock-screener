"""Browser policy behavior for routing and overlay guardrails."""

from __future__ import annotations

from dataclasses import dataclass

from xui_reader.browser.policy import (
    configure_resource_routing,
    detect_unsupported_overlay_state,
    dismiss_allowed_overlays,
    install_resource_routing,
    make_resource_route_handler,
)
from xui_reader.config import BrowserConfig, RuntimeConfig


class FakeRoute:
    def __init__(self) -> None:
        self.aborted = 0
        self.continued = 0

    def abort(self) -> None:
        self.aborted += 1

    def continue_(self) -> None:
        self.continued += 1


@dataclass(frozen=True)
class FakeRequest:
    resource_type: str


class FakeRoutablePage:
    def __init__(self) -> None:
        self.route_calls: list[tuple[str, object]] = []

    def route(self, url: str, handler: object) -> None:
        self.route_calls.append((url, handler))


class FakeOverlayElement:
    def __init__(self, *, visible: bool = True, click_raises: bool = False) -> None:
        self._visible = visible
        self._click_raises = click_raises
        self.clicks = 0

    def is_visible(self) -> bool:
        return self._visible

    def click(self, timeout: int | None = None) -> None:
        _ = timeout
        if self._click_raises:
            raise RuntimeError("click failed")
        self.clicks += 1


class FakeOverlayPage:
    def __init__(
        self,
        *,
        url: str,
        title: str,
        body_text: str,
        selectors: dict[str, FakeOverlayElement | None | Exception] | None = None,
    ) -> None:
        self.url = url
        self._title = title
        self._body_text = body_text
        self._selectors = selectors or {}

    def title(self) -> str:
        return self._title

    def inner_text(self, selector: str, timeout: int | None = None) -> str:
        _ = timeout
        if selector != "body":
            raise RuntimeError("unexpected selector")
        return self._body_text

    def query_selector(self, selector: str) -> FakeOverlayElement | None:
        value = self._selectors.get(selector)
        if isinstance(value, Exception):
            raise value
        return value


def _config(*, block_resources: bool = True) -> RuntimeConfig:
    return RuntimeConfig(browser=BrowserConfig(block_resources=block_resources))


def test_make_resource_route_handler_aborts_blocked_resource_types() -> None:
    handler = make_resource_route_handler({"image", "media"})
    route = FakeRoute()

    handler(route, FakeRequest(resource_type="image"))
    handler(route, FakeRequest(resource_type="script"))

    assert route.aborted == 1
    assert route.continued == 1


def test_install_resource_routing_registers_default_policy() -> None:
    page = FakeRoutablePage()
    policy = install_resource_routing(page, block_resources=True)

    assert policy.enabled is True
    assert "image" in policy.blocked_resource_types
    assert page.route_calls
    assert page.route_calls[0][0] == "**/*"


def test_configure_resource_routing_respects_block_resources_flag() -> None:
    page = FakeRoutablePage()
    policy = configure_resource_routing(page, _config(block_resources=False))

    assert policy.enabled is False
    assert policy.blocked_resource_types == frozenset()
    assert page.route_calls == []


def test_dismiss_allowed_overlays_clicks_only_allowed_selectors() -> None:
    allowed = ("button:has-text('Not now')", "button[aria-label='Close']")
    first = FakeOverlayElement(visible=True)
    second = FakeOverlayElement(visible=False)
    page = FakeOverlayPage(
        url="https://x.com/home",
        title="Home / X",
        body_text="",
        selectors={
            allowed[0]: first,
            allowed[1]: second,
            "button:has-text('Verify identity')": FakeOverlayElement(visible=True),
        },
    )

    result = dismiss_allowed_overlays(page, allowed_selectors=allowed)

    assert result.dismissed_count == 1
    assert result.attempted_selectors == allowed
    assert first.clicks == 1


def test_dismiss_allowed_overlays_exits_safely_for_challenge_state() -> None:
    allowed = ("button:has-text('Not now')",)
    element = FakeOverlayElement(visible=True)
    page = FakeOverlayPage(
        url="https://x.com/account/login_challenge",
        title="Challenge",
        body_text="Confirm it's you",
        selectors={allowed[0]: element},
    )

    result = dismiss_allowed_overlays(page, allowed_selectors=allowed)

    assert result.dismissed_count == 0
    assert result.attempted_selectors == ()
    assert result.blocked_category == "challenge"
    assert result.skipped_reason == "unsupported_blocked_state"
    assert element.clicks == 0


def test_dismiss_allowed_overlays_exits_safely_for_login_wall_state() -> None:
    result = dismiss_allowed_overlays(
        FakeOverlayPage(
            url="https://x.com/i/flow/login",
            title="Log in / X",
            body_text="",
        ),
    )

    assert result.dismissed_count == 0
    assert result.blocked_category == "login_wall"
    assert result.skipped_reason == "unsupported_blocked_state"


def test_detect_unsupported_overlay_state_ignores_benign_home_login_copy() -> None:
    category = detect_unsupported_overlay_state(
        current_url="https://x.com/home",
        page_title="Home / X",
        body_text="Use this button to log in to another account.",
    )
    assert category is None

