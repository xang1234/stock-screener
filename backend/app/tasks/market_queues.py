"""
Per-market Celery queue topology.

Partitions data_fetch and user_scans work by market (US/HK/JP/TW) so that one
market's refresh pipeline cannot stall throughput for another market. Pairs
with the market-scoped DataFetchLock in data_fetch_lock.py.
"""
from __future__ import annotations

from typing import Iterable, List, Optional


SUPPORTED_MARKETS: tuple[str, ...] = ("US", "HK", "JP", "TW")

SHARED_SENTINEL = "SHARED"

DATA_FETCH_BASE = "data_fetch"
USER_SCANS_BASE = "user_scans"

SHARED_DATA_FETCH_QUEUE = f"{DATA_FETCH_BASE}_shared"
SHARED_USER_SCANS_QUEUE = f"{USER_SCANS_BASE}_shared"

LEGACY_DATA_FETCH_QUEUE = DATA_FETCH_BASE
LEGACY_USER_SCANS_QUEUE = USER_SCANS_BASE


def normalize_market(market: Optional[str]) -> str:
    """Return canonical upper-case market code, or SHARED for None/empty.

    Unknown markets raise ValueError so routing bugs surface early rather than
    silently piling tasks onto the shared queue.
    """
    if market is None:
        return SHARED_SENTINEL
    if not isinstance(market, str):
        raise ValueError(f"market must be a string, got {type(market).__name__}")
    upper = market.strip().upper()
    if upper in ("", SHARED_SENTINEL):
        return SHARED_SENTINEL
    if upper not in SUPPORTED_MARKETS:
        raise ValueError(
            f"Unsupported market {market!r}. Supported: {SUPPORTED_MARKETS} (or None/SHARED)."
        )
    return upper


def queue_for_market(market: Optional[str], base: str = DATA_FETCH_BASE) -> str:
    """Return the queue name for a given market.

    queue_for_market("HK")      -> "data_fetch_hk"
    queue_for_market(None)      -> "data_fetch_shared"
    queue_for_market("US", base="user_scans") -> "user_scans_us"
    """
    normalized = normalize_market(market)
    if normalized == SHARED_SENTINEL:
        return f"{base}_shared"
    return f"{base}_{normalized.lower()}"


def data_fetch_queue_for_market(market: Optional[str]) -> str:
    return queue_for_market(market, DATA_FETCH_BASE)


def user_scans_queue_for_market(market: Optional[str]) -> str:
    return queue_for_market(market, USER_SCANS_BASE)


def all_data_fetch_queues(markets: Optional[Iterable[str]] = None) -> List[str]:
    """Return all data_fetch queues for the given (or default) market set."""
    ms = list(markets) if markets is not None else list(SUPPORTED_MARKETS)
    return [queue_for_market(m, DATA_FETCH_BASE) for m in ms] + [SHARED_DATA_FETCH_QUEUE]


def all_user_scans_queues(markets: Optional[Iterable[str]] = None) -> List[str]:
    ms = list(markets) if markets is not None else list(SUPPORTED_MARKETS)
    return [queue_for_market(m, USER_SCANS_BASE) for m in ms] + [SHARED_USER_SCANS_QUEUE]


def log_extra(market: Optional[str]) -> dict:
    """Structured-log extra dict carrying the market label.

    Use with `logger.info("...", extra=log_extra(market))`. Downstream log
    formatters and the future observability pipeline (bead 9.2 / epic 10) can
    key off `market` without parsing log bodies.
    """
    return {"market": normalize_market(market).lower()}


def market_tag(market: Optional[str]) -> str:
    """Human-readable tag for log banners, e.g. '[market=hk]' or '[market=shared]'."""
    return f"[market={normalize_market(market).lower()}]"
