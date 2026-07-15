"""Literal SQL LIKE patterns shared by query adapters."""


def literal_contains_pattern(value: str) -> str:
    """Wrap text for a contains search while escaping LIKE metacharacters."""

    escaped = (
        value.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )
    return f"%{escaped}%"


__all__ = ["literal_contains_pattern"]
