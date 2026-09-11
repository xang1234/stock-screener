"""Derived quantity views shared by evidence packets and offline replays."""

from dataclasses import asdict

from .quantity_display import normalize_quantities
from .review import _text


def normalized_segments(payload):
    """Keep each source and translation independently hash-bound and inspectable."""
    return [
        {
            "segment_index": index,
            "status": segment.status,
            "source": asdict(normalize_quantities(segment.original)),
            "translation": (
                asdict(normalize_quantities(segment.translated))
                if segment.translated is not None and payload.target_language == "en"
                else None
            ),
        }
        for index, segment in enumerate(payload.segments)
    ]


def render_normalized_segment(segment):
    """Show changed English text and all normalization holds, never approve it."""
    lines = []
    translated = segment["translation"]
    if translated and translated["text"] != translated["original"]:
        lines.extend(["**English quantity display (deterministic)**", ""])
        lines.extend("> " + _text(line) for line in translated["text"].splitlines())
        lines.append("")
    issues = {
        issue["code"]
        for view in (segment["source"], translated)
        if view is not None
        for issue in view["issues"]
    }
    if issues:
        lines.extend(
            ["Quantity review required: " + _text(", ".join(sorted(issues))), ""]
        )
    return lines
