"""Browser session lifecycle behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from xui_reader.browser.session import PlaywrightBrowserSession
from xui_reader.config import BrowserConfig, RuntimeConfig
from xui_reader.errors import BrowserError


class FakePage:
    def __init__(self) -> None:
        self.navigation_timeouts: list[int] = []

    def set_default_navigation_timeout(self, timeout: int) -> None:
        self.navigation_timeouts.append(timeout)


class FakeContext:
    def __init__(
        self,
        *,
        storage_state_payload: dict[str, object] | None = None,
        events: list[str] | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.default_timeout_ms: int | None = None
        self.new_page_calls = 0
        self.storage_state_payload = storage_state_payload or {"cookies": [], "origins": []}
        self.events = events
        self.close_error = close_error

    def set_default_timeout(self, timeout_ms: int) -> None:
        self.default_timeout_ms = timeout_ms

    def new_page(self) -> FakePage:
        self.new_page_calls += 1
        return FakePage()

    def storage_state(self) -> dict[str, object]:
        return self.storage_state_payload

    def close(self) -> None:
        if self.events is not None:
            self.events.append("context.close")
        if self.close_error is not None:
            raise self.close_error


class FakeBrowser:
    def __init__(
        self,
        context: FakeContext,
        *,
        events: list[str] | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.context = context
        self.new_context_kwargs: dict[str, object] | None = None
        self.events = events
        self.close_error = close_error

    def new_context(self, **kwargs: object) -> FakeContext:
        self.new_context_kwargs = kwargs
        return self.context

    def close(self) -> None:
        if self.events is not None:
            self.events.append("browser.close")
        if self.close_error is not None:
            raise self.close_error


class FakeLauncher:
    def __init__(self, browser: FakeBrowser) -> None:
        self.browser = browser
        self.launch_headless_values: list[bool] = []

    def launch(self, *, headless: bool) -> FakeBrowser:
        self.launch_headless_values.append(headless)
        return self.browser


class FakePlaywright:
    def __init__(self, **engines: FakeLauncher) -> None:
        for name, launcher in engines.items():
            setattr(self, name, launcher)


class FakePlaywrightContextManager:
    def __init__(
        self,
        playwright: FakePlaywright,
        *,
        events: list[str] | None = None,
        exit_error: Exception | None = None,
    ) -> None:
        self.playwright = playwright
        self.events = events
        self.exit_error = exit_error
        self.entered = 0
        self.exited = 0

    def __enter__(self) -> FakePlaywright:
        self.entered += 1
        return self.playwright

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.exited += 1
        if self.events is not None:
            self.events.append("playwright.exit")
        if self.exit_error is not None:
            raise self.exit_error
        return False


def _config(**browser_overrides: object) -> RuntimeConfig:
    browser = BrowserConfig(**browser_overrides)
    return RuntimeConfig(browser=browser)


def test_open_and_new_page_wire_defaults_from_config() -> None:
    context = FakeContext()
    browser = FakeBrowser(context)
    launcher = FakeLauncher(browser)
    manager = FakePlaywrightContextManager(FakePlaywright(chromium=launcher))

    session = PlaywrightBrowserSession(
        _config(
            engine="chromium",
            headless=True,
            navigation_timeout_ms=12_345,
            action_timeout_ms=4_321,
            viewport_width=1440,
            viewport_height=900,
            locale="en-GB",
        ),
        playwright_factory=lambda: manager,
    )

    session.open()
    page = session.new_page()

    assert launcher.launch_headless_values == [True]
    assert browser.new_context_kwargs == {
        "locale": "en-GB",
        "viewport": {"width": 1440, "height": 900},
    }
    assert context.default_timeout_ms == 4_321
    assert page.navigation_timeouts == [12_345]

    session.close()
    assert manager.entered == 1
    assert manager.exited == 1


def test_runtime_overrides_take_precedence_over_config() -> None:
    context = FakeContext()
    browser = FakeBrowser(context)
    chromium = FakeLauncher(browser)
    firefox = FakeLauncher(browser)
    manager = FakePlaywrightContextManager(
        FakePlaywright(chromium=chromium, firefox=firefox),
    )

    session = PlaywrightBrowserSession(
        _config(engine="chromium", headless=False, navigation_timeout_ms=500, action_timeout_ms=600),
        engine="firefox",
        headless=True,
        navigation_timeout_ms=700,
        action_timeout_ms=800,
        locale="fr-FR",
        viewport_width=1600,
        viewport_height=1000,
        storage_state=Path("/tmp/state.json"),
        playwright_factory=lambda: manager,
    )

    session.open()
    page = session.new_page()

    assert chromium.launch_headless_values == []
    assert firefox.launch_headless_values == [True]
    assert browser.new_context_kwargs == {
        "locale": "fr-FR",
        "viewport": {"width": 1600, "height": 1000},
        "storage_state": "/tmp/state.json",
    }
    assert context.default_timeout_ms == 800
    assert page.navigation_timeouts == [700]


def test_new_page_auto_opens_and_reuses_context() -> None:
    context = FakeContext()
    browser = FakeBrowser(context)
    launcher = FakeLauncher(browser)
    manager = FakePlaywrightContextManager(FakePlaywright(chromium=launcher))

    session = PlaywrightBrowserSession(_config(), playwright_factory=lambda: manager)

    first = session.new_page()
    second = session.new_page()

    assert first is not second
    assert manager.entered == 1
    assert launcher.launch_headless_values == [True]
    assert context.new_page_calls == 2


def test_storage_state_requires_open_session() -> None:
    session = PlaywrightBrowserSession(_config(), playwright_factory=lambda: FakePlaywrightContextManager(FakePlaywright()))
    with pytest.raises(BrowserError, match="not open"):
        session.storage_state()


def test_storage_state_returns_dict_payload() -> None:
    context = FakeContext(storage_state_payload={"cookies": [{"name": "auth"}], "origins": []})
    browser = FakeBrowser(context)
    launcher = FakeLauncher(browser)
    manager = FakePlaywrightContextManager(FakePlaywright(chromium=launcher))

    with PlaywrightBrowserSession(_config(), playwright_factory=lambda: manager) as session:
        assert session.storage_state() == {"cookies": [{"name": "auth"}], "origins": []}


def test_close_attempts_full_teardown_even_if_context_close_fails() -> None:
    events: list[str] = []
    context = FakeContext(events=events, close_error=RuntimeError("context boom"))
    browser = FakeBrowser(context, events=events)
    launcher = FakeLauncher(browser)
    manager = FakePlaywrightContextManager(FakePlaywright(chromium=launcher), events=events)

    session = PlaywrightBrowserSession(_config(), playwright_factory=lambda: manager)
    session.open()

    with pytest.raises(BrowserError, match="context boom"):
        session.close()

    assert events == ["context.close", "browser.close", "playwright.exit"]
    session.close()


def test_open_raises_for_unsupported_engine() -> None:
    context = FakeContext()
    browser = FakeBrowser(context)
    launcher = FakeLauncher(browser)
    manager = FakePlaywrightContextManager(FakePlaywright(chromium=launcher))

    session = PlaywrightBrowserSession(
        _config(engine="firefox"),
        playwright_factory=lambda: manager,
    )

    with pytest.raises(BrowserError, match="Unsupported browser engine"):
        session.open()
