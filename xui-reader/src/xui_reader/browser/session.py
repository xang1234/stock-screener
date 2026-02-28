"""Browser session lifecycle manager and protocols."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from xui_reader.config import RuntimeConfig
from xui_reader.errors import BrowserError


class BrowserPage(Protocol):
    def goto(self, url: str, **kwargs: Any) -> Any:
        """Navigate to a URL."""


class BrowserSessionManager(Protocol):
    def open(self) -> None:
        """Open browser resources."""

    def close(self) -> None:
        """Close browser resources."""

    def new_page(self) -> BrowserPage:
        """Create and return a page-like object."""


@dataclass(frozen=True)
class BrowserSessionOptions:
    engine: str
    headless: bool
    navigation_timeout_ms: int
    action_timeout_ms: int
    locale: str
    viewport_width: int
    viewport_height: int
    storage_state: str | dict[str, Any] | None = None


class PlaywrightBrowserSession:
    """Manage one browser/context lifecycle with deterministic teardown."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        engine: str | None = None,
        headless: bool | None = None,
        navigation_timeout_ms: int | None = None,
        action_timeout_ms: int | None = None,
        locale: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
        storage_state: str | Path | dict[str, Any] | None = None,
        playwright_factory: Callable[[], AbstractContextManager[Any]] | None = None,
    ) -> None:
        browser = config.browser
        resolved_storage_state: str | dict[str, Any] | None
        if isinstance(storage_state, Path):
            resolved_storage_state = str(storage_state)
        else:
            resolved_storage_state = storage_state

        self.options = BrowserSessionOptions(
            engine=engine if engine is not None else browser.engine,
            headless=headless if headless is not None else browser.headless,
            navigation_timeout_ms=(
                navigation_timeout_ms
                if navigation_timeout_ms is not None
                else browser.navigation_timeout_ms
            ),
            action_timeout_ms=(
                action_timeout_ms if action_timeout_ms is not None else browser.action_timeout_ms
            ),
            locale=locale if locale is not None else browser.locale,
            viewport_width=(
                viewport_width if viewport_width is not None else browser.viewport_width
            ),
            viewport_height=(
                viewport_height if viewport_height is not None else browser.viewport_height
            ),
            storage_state=resolved_storage_state,
        )
        self._playwright_factory = playwright_factory or _default_playwright_factory
        self._playwright_cm: AbstractContextManager[Any] | None = None
        self._browser: Any | None = None
        self._context: Any | None = None

    def open(self) -> None:
        if self._context is not None:
            return

        try:
            self._playwright_cm = self._playwright_factory()
            playwright = self._playwright_cm.__enter__()

            launcher = getattr(playwright, self.options.engine, None)
            if launcher is None:
                raise BrowserError(
                    f"Unsupported browser engine '{self.options.engine}' for Playwright session."
                )

            self._browser = launcher.launch(headless=self.options.headless)

            context_kwargs: dict[str, Any] = {
                "locale": self.options.locale,
                "viewport": {
                    "width": self.options.viewport_width,
                    "height": self.options.viewport_height,
                },
            }
            if self.options.storage_state is not None:
                context_kwargs["storage_state"] = self.options.storage_state
            self._context = self._browser.new_context(**context_kwargs)
            self._context.set_default_timeout(self.options.action_timeout_ms)
        except BrowserError:
            self._teardown(raise_on_error=False)
            raise
        except Exception as exc:
            self._teardown(raise_on_error=False)
            raise BrowserError(f"Failed to open browser session: {exc}") from exc

    def new_page(self) -> BrowserPage:
        if self._context is None:
            self.open()

        if self._context is None:
            raise BrowserError("Browser session is not open.")

        try:
            page = self._context.new_page()
            set_navigation_timeout = getattr(page, "set_default_navigation_timeout", None)
            if callable(set_navigation_timeout):
                set_navigation_timeout(self.options.navigation_timeout_ms)
            return page
        except Exception as exc:
            raise BrowserError(f"Failed to create browser page: {exc}") from exc

    def storage_state(self) -> dict[str, Any]:
        if self._context is None:
            raise BrowserError("Browser session is not open.")

        try:
            payload = self._context.storage_state()
        except Exception as exc:
            raise BrowserError(f"Failed to capture browser storage_state: {exc}") from exc

        if not isinstance(payload, dict):
            raise BrowserError("Playwright returned invalid storage_state payload.")
        return payload

    def close(self) -> None:
        self._teardown(raise_on_error=True)

    def __enter__(self) -> PlaywrightBrowserSession:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        try:
            self.close()
        except BrowserError:
            if exc_type is None:
                raise
        return False

    def _teardown(self, *, raise_on_error: bool) -> None:
        errors: list[str] = []

        if self._context is not None:
            try:
                self._context.close()
            except Exception as exc:
                errors.append(f"context close failed: {exc}")
            finally:
                self._context = None

        if self._browser is not None:
            try:
                self._browser.close()
            except Exception as exc:
                errors.append(f"browser close failed: {exc}")
            finally:
                self._browser = None

        if self._playwright_cm is not None:
            try:
                self._playwright_cm.__exit__(None, None, None)
            except Exception as exc:
                errors.append(f"playwright teardown failed: {exc}")
            finally:
                self._playwright_cm = None

        if raise_on_error and errors:
            raise BrowserError(
                "Errors occurred during browser session teardown: " + "; ".join(errors)
            )


def _default_playwright_factory() -> AbstractContextManager[Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise BrowserError(
            "Playwright is not available. Install dependencies and run "
            "`python -m playwright install chromium`."
        ) from exc
    return sync_playwright()
