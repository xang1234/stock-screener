"""Read-through Redis cache for group-rankings API payloads.

Group rankings change at most once per trading day (plus manual
recalculations), but the rankings page refetches every 60 seconds and each
request recomputes rank changes from ~26K historical rows. This module caches
the computed service payloads (plain JSON-serializable dicts/lists) in Redis.

Invalidation is epoch-based: every cache key embeds a per-market epoch
counter, and the ranking-calculation Celery tasks bump the epoch after
writing new rankings. Old-epoch entries simply age out via TTL. Redis being
unavailable degrades to computing every request — never to an error.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from .redis_pool import get_redis_client

logger = logging.getLogger(__name__)

TTL_SECONDS = 60 * 60  # rankings change daily; epoch bump covers recalcs
_EPOCH_KEY_TEMPLATE = "groups:cache_epoch:{market}"
_PAYLOAD_KEY_TEMPLATE = "groups:{market}:e{epoch}:{name}:{params}"


def bump_group_rankings_epoch(market: str) -> None:
    """Invalidate cached payloads for a market after rankings are rewritten.

    Never raises: invalidation failure only extends staleness to the TTL.
    """
    try:
        client = get_redis_client()
        client.incr(_EPOCH_KEY_TEMPLATE.format(market=(market or "US").upper()))
    except Exception as exc:
        logger.warning("Could not bump group-rankings cache epoch for %s: %s", market, exc)


def cached_group_payload(
    *,
    market: str,
    name: str,
    params: str,
    compute: Callable[[], Any],
    should_cache: Callable[[Any], bool] = bool,
) -> Any:
    """Return the cached payload for (market, name, params), computing on miss.

    `compute` results are JSON round-tripped (dates become ISO strings, which
    the Pydantic response models parse back), so only cache payloads destined
    for JSON responses. Empty results (per `should_cache`) are not cached so
    a later backfill becomes visible immediately.
    """
    normalized_market = (market or "US").upper()
    client = None
    key = None
    try:
        client = get_redis_client()
        epoch = client.get(_EPOCH_KEY_TEMPLATE.format(market=normalized_market))
        epoch_str = epoch.decode() if isinstance(epoch, bytes) else str(epoch or 0)
        key = _PAYLOAD_KEY_TEMPLATE.format(
            market=normalized_market, epoch=epoch_str, name=name, params=params
        )
        raw = client.get(key)
        if raw is not None:
            return json.loads(raw)
    except Exception as exc:
        logger.warning("Group-rankings cache read failed (%s/%s): %s", market, name, exc)
        client = None

    value = compute()

    if client is not None and key is not None and should_cache(value):
        try:
            client.setex(key, TTL_SECONDS, json.dumps(value, default=str))
        except Exception as exc:
            logger.warning("Group-rankings cache write failed (%s/%s): %s", market, name, exc)

    return value
