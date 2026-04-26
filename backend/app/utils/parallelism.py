"""Helpers for bounded in-process parallelism."""

from __future__ import annotations

import os


def bounded_symbol_workers(requested: int) -> int:
    """Cap symbol-level workers to avoid excessive nested scan threads."""
    cpu_count = os.cpu_count() or 1
    cpu_cap = max(1, cpu_count // 2)
    return max(1, min(requested, cpu_cap))
