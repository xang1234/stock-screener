"""Dependency-free normalization helpers shared by domain workflows."""

from __future__ import annotations


def normalize_string_list(value: object) -> list[str]:
    """Normalize a scalar-or-sequence value into a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        normalized: list[str] = []
        for item in value:
            if item is None:
                continue
            text = item.strip() if isinstance(item, str) else str(item).strip()
            if text:
                normalized.append(text)
        return normalized
    text = str(value).strip()
    return [text] if text else []


__all__ = ["normalize_string_list"]
