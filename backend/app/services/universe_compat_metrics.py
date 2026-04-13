"""Compatibility telemetry counters for legacy universe request usage.

Provides Redis-backed counters so operators can track how many clients are
still hitting the legacy `universe` string path before the sunset date
(see docs/asia/asia_v2_legacy_universe_compat_deprecation_policy.md).

The counters degrade to no-ops when Redis is unavailable — legacy-path
requests continue working; only telemetry is lost.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .redis_pool import get_redis_client

logger = logging.getLogger(__name__)

LEGACY_TOTAL_KEY = "universe_compat:legacy_total"
LEGACY_VALUE_KEY_PREFIX = "universe_compat:legacy:"
LEGACY_LAST_SEEN_KEY = "universe_compat:legacy_last_seen_ts"


def _safe_sanitize(legacy_value: Optional[str]) -> str:
    """Clamp legacy values to a safe Redis key suffix.

    Unknown legacy strings can be arbitrary client-supplied text; we bucket
    anything unusual under "unknown" to keep the key space bounded.
    """
    if not legacy_value:
        return "unknown"
    cleaned = legacy_value.strip().lower()
    if not cleaned or len(cleaned) > 32:
        return "unknown"
    if any(c.isspace() for c in cleaned):
        return "unknown"
    return cleaned


def record_legacy_universe_usage(legacy_value: Optional[str]) -> None:
    """Increment legacy-path counters for a single request.

    Best-effort: any Redis failure is logged at debug level and swallowed so
    the request path is never broken by telemetry.
    """
    client = get_redis_client()
    if client is None:
        return

    bucket = _safe_sanitize(legacy_value)
    per_value_key = f"{LEGACY_VALUE_KEY_PREFIX}{bucket}"
    now = int(time.time())
    try:
        pipe = client.pipeline(transaction=False)
        pipe.incr(LEGACY_TOTAL_KEY)
        pipe.incr(per_value_key)
        pipe.set(LEGACY_LAST_SEEN_KEY, now)
        pipe.execute()
    except Exception as exc:
        logger.debug("Failed to record legacy universe telemetry: %s", exc)


def get_legacy_universe_counts() -> dict[str, Any]:
    """Return a snapshot of legacy-path counters for diagnostics/tests.

    Keys: ``total`` (int), ``by_value`` (dict[bucket, int]), and
    ``last_seen_ts`` (int unix timestamp or None). Returns an empty dict
    when Redis is unavailable.
    """
    client = get_redis_client()
    if client is None:
        return {}

    try:
        total_raw = client.get(LEGACY_TOTAL_KEY)
        total = int(total_raw) if total_raw is not None else 0

        by_value: dict[str, int] = {}
        for key in client.scan_iter(match=f"{LEGACY_VALUE_KEY_PREFIX}*"):
            key_str = key.decode() if isinstance(key, bytes) else key
            bucket = key_str[len(LEGACY_VALUE_KEY_PREFIX):]
            raw = client.get(key_str)
            by_value[bucket] = int(raw) if raw is not None else 0

        last_seen_raw = client.get(LEGACY_LAST_SEEN_KEY)
        last_seen = int(last_seen_raw) if last_seen_raw is not None else None
        return {"total": total, "by_value": by_value, "last_seen_ts": last_seen}
    except Exception as exc:
        logger.debug("Failed to read legacy universe telemetry: %s", exc)
        return {}
