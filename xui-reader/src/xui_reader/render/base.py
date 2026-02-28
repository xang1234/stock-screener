"""Rendering interfaces."""

from __future__ import annotations

from typing import Protocol

from xui_reader.models import TweetItem


class Renderer(Protocol):
    def render(self, items: tuple[TweetItem, ...]) -> str:
        """Render a collection of items as text."""
