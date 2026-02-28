"""Extractor interfaces."""

from __future__ import annotations

from typing import Any, Protocol

from xui_reader.models import TweetItem


class Extractor(Protocol):
    def extract(self, raw_payload: Any) -> tuple[TweetItem, ...]:
        """Parse raw collector payload into normalized tweet items."""
