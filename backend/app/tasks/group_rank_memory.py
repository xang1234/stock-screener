"""Memory cleanup for group-rank task orchestration."""

from __future__ import annotations

import gc
import logging
import sys


logger = logging.getLogger(__name__)


def release_group_rank_gapfill_memory() -> None:
    """Return freed pandas/SQLAlchemy heap pages before ranking today."""
    collected = gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            malloc_trim = getattr(ctypes.CDLL("libc.so.6"), "malloc_trim", None)
            if malloc_trim is not None:
                malloc_trim(0)
        except Exception:
            logger.debug("malloc_trim unavailable after group-ranking gap-fill", exc_info=True)
    logger.debug("Collected %d objects after group-ranking gap-fill", collected)
