"""Dependency-free runtime diagnostics for long-running worker stages."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import resource
import sys
import time


def _max_rss_mb() -> float:
    raw_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return round(raw_rss / (1024 * 1024), 2)
    return round(raw_rss / 1024, 2)


@contextmanager
def log_runtime_stage(logger, name: str, **extra) -> Iterator[None]:
    started = time.perf_counter()
    logger.info(
        "Runtime stage started: %s",
        name,
        extra={"runtime_stage": name, **extra},
    )
    try:
        yield
    finally:
        elapsed = round(time.perf_counter() - started, 3)
        logger.info(
            "Runtime stage finished: %s",
            name,
            extra={
                "runtime_stage": name,
                "elapsed_seconds": elapsed,
                "max_rss_mb": _max_rss_mb(),
                **extra,
            },
        )
