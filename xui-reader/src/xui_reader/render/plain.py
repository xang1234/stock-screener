"""Tab-separated tweet rendering for shell pipelines."""

from __future__ import annotations

from xui_reader.models import TweetItem


def render_plain(items: tuple[TweetItem, ...]) -> str:
    lines: list[str] = []
    for item in items:
        created = item.created_at.isoformat() if item.created_at else ""
        author = item.author_handle or ""
        text = (item.text or "").replace("\n", " ").replace("\t", " ")
        lines.append(
            "\t".join(
                (
                    item.source_id,
                    item.tweet_id,
                    created,
                    author,
                    text,
                )
            )
        )
    return "\n".join(lines)
