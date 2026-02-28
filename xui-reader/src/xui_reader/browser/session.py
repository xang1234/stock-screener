"""Browser session interfaces."""

from __future__ import annotations

from typing import Protocol


class BrowserPage(Protocol):
    def goto(self, url: str) -> None:
        """Navigate to a URL."""


class BrowserSessionManager(Protocol):
    def open(self) -> None:
        """Open browser resources."""

    def close(self) -> None:
        """Close browser resources."""

    def new_page(self) -> BrowserPage:
        """Create and return a page-like object."""
