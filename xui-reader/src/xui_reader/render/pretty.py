"""Human-friendly tweet rendering."""

from __future__ import annotations

from xui_reader.models import TweetItem


def render_pretty(items: tuple[TweetItem, ...]) -> str:
    if not items:
        return "(no items)"

    lines: list[str] = []
    for item in items:
        created = item.created_at.isoformat() if item.created_at else "-"
        author = item.author_handle or "-"
        text = item.text or ""
        lines.append(f"{created} {item.source_id} {item.tweet_id} {author}")
        if text:
            lines.append(f"  {text}")
    return "\n".join(lines)
