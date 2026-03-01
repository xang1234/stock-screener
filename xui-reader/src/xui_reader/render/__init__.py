"""Render contracts and concrete output formatters."""

from __future__ import annotations

from xui_reader.errors import RenderError
from xui_reader.models import TweetItem
from xui_reader.render.base import Renderer
from xui_reader.render.jsonout import render_json, render_jsonl, tweet_item_to_dict
from xui_reader.render.plain import render_plain
from xui_reader.render.pretty import render_pretty


def render_items(items: tuple[TweetItem, ...], output_format: str) -> str:
    if output_format == "pretty":
        return render_pretty(items)
    if output_format == "plain":
        return render_plain(items)
    if output_format == "json":
        return render_json(items)
    if output_format == "jsonl":
        return render_jsonl(items)
    raise RenderError(
        f"Unsupported output format '{output_format}'. Use one of: pretty, plain, json, jsonl."
    )


__all__ = [
    "Renderer",
    "render_items",
    "render_json",
    "render_jsonl",
    "render_plain",
    "render_pretty",
    "tweet_item_to_dict",
]
