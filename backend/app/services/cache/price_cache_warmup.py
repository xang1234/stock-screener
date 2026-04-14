"""Warmup metadata and heartbeat persistence for price cache operations.

The heartbeat key is scoped per market (``cache:warmup:heartbeat:hk``) so
parallel US+HK refreshes don't clobber each other's progress reporting.
The unsuffixed legacy key remains readable for one-shot post-deploy reads.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from ...tasks.market_queues import market_suffix
from ...theme_platform.contracts import WarmupHeartbeatState, WarmupStateSnapshot


def scoped_heartbeat_key(base: str, market: Optional[str]) -> str:
    """Return ``base:<market_lower>`` or ``base:shared`` when market is None.

    Public contract for constructing per-market heartbeat keys; used by
    both PriceCacheWarmupStore and DataFetchLock.get_current_task.
    """
    return f"{base}:{market_suffix(market)}"


class PriceCacheWarmupStore:
    """Owns warmup metadata + heartbeat persistence in Redis (per-market scoped)."""

    def __init__(
        self,
        *,
        logger,
        redis_client,
        metadata_key: str,
        heartbeat_key: str,
    ) -> None:
        self._logger = logger
        self._redis_client = redis_client
        # Base keys; per-market suffixes appended at call time. Retained as
        # ``_metadata_key`` / ``_heartbeat_key`` for back-compat with callers
        # that read them directly.
        self._metadata_key = metadata_key
        self._heartbeat_key = heartbeat_key

    def _hb_key(self, market: Optional[str]) -> str:
        return scoped_heartbeat_key(self._heartbeat_key, market)

    def _meta_key(self, market: Optional[str]) -> str:
        return scoped_heartbeat_key(self._metadata_key, market)

    def get_warmup_metadata(self, market: Optional[str] = None) -> Optional[WarmupStateSnapshot]:
        if not self._redis_client:
            return None
        key = self._meta_key(market)
        try:
            meta_json = self._redis_client.get(key)
            if not meta_json:
                return None
            metadata = json.loads(meta_json)
            if not isinstance(metadata, dict):
                self._logger.warning(
                    "Ignoring malformed warmup metadata payload for key=%s", key,
                )
                return None
            return metadata
        except Exception as exc:
            self._logger.error("Error getting warmup metadata: %s", exc)
            return None

    def save_warmup_metadata(
        self,
        status: str,
        count: int,
        total: int,
        error: str | None = None,
        market: Optional[str] = None,
    ) -> None:
        if not self._redis_client:
            return
        try:
            meta = {
                "status": status,
                "count": count,
                "total": total,
                "completed_at": datetime.now().isoformat(),
                "error": error,
                "market": (market or "shared").lower(),
            }
            self._redis_client.setex(self._meta_key(market), 86400 * 7, json.dumps(meta))
            self._logger.info(
                "Saved warmup metadata: %s (%s/%s) market=%s",
                status, count, total, (market or "shared").lower(),
            )
        except Exception as exc:
            self._logger.error("Error saving warmup metadata: %s", exc)

    def update_warmup_heartbeat(
        self,
        current: int,
        total: int,
        percent: float | None = None,
        market: Optional[str] = None,
    ) -> None:
        if not self._redis_client:
            return
        try:
            heartbeat = {
                "status": "running",
                "current": current,
                "total": total,
                "percent": percent if percent is not None else (round((current / total) * 100, 1) if total > 0 else 0),
                "updated_at": datetime.now().isoformat(),
                "market": (market or "shared").lower(),
            }
            self._redis_client.setex(self._hb_key(market), 3600, json.dumps(heartbeat))
        except Exception as exc:
            self._logger.error("Error updating warmup heartbeat: %s", exc)

    def get_heartbeat_info(self, market: Optional[str] = None) -> Optional[WarmupHeartbeatState]:
        if not self._redis_client:
            return None
        try:
            heartbeat_json = self._redis_client.get(self._hb_key(market))
            if not heartbeat_json:
                return None

            heartbeat = json.loads(heartbeat_json)
            hb_status = heartbeat.get("status", "running")
            ts_field = "completed_at" if hb_status in ("completed", "failed") else "updated_at"
            ts_str = heartbeat.get(ts_field, heartbeat.get("updated_at", ""))
            if ts_str:
                ts = datetime.fromisoformat(ts_str)
                minutes = (datetime.now() - ts).total_seconds() / 60
            else:
                minutes = None
            return {**heartbeat, "minutes": minutes, "status": hb_status}
        except Exception as exc:
            self._logger.error("Error getting heartbeat info: %s", exc)
            return None

    def get_minutes_since_heartbeat(self, market: Optional[str] = None) -> Optional[float]:
        info = self.get_heartbeat_info(market=market)
        if info is None:
            return None
        return info.get("minutes")

    def get_task_progress(self, market: Optional[str] = None) -> dict[str, int | float | None]:
        if not self._redis_client:
            return {}
        try:
            heartbeat_json = self._redis_client.get(self._hb_key(market))
            if not heartbeat_json:
                return {}
            heartbeat = json.loads(heartbeat_json)
            return {
                "current": heartbeat.get("current"),
                "total": heartbeat.get("total"),
                "progress": heartbeat.get("percent"),
            }
        except Exception as exc:
            self._logger.error("Error getting task progress: %s", exc)
            return {}

    def clear_warmup_heartbeat(self, market: Optional[str] = None) -> None:
        if not self._redis_client:
            return
        try:
            self._redis_client.delete(self._hb_key(market))
        except Exception as exc:
            self._logger.error("Error clearing warmup heartbeat: %s", exc)

    def complete_warmup_heartbeat(
        self,
        status: str = "completed",
        market: Optional[str] = None,
    ) -> None:
        if not self._redis_client:
            return
        terminal_status = status if status in {"completed", "failed"} else "completed"
        if terminal_status != status:
            self._logger.warning("Unsupported warmup terminal status %s; defaulting to completed", status)
        try:
            previous = self.get_heartbeat_info(market=market) or {}
            heartbeat = {
                "current": previous.get("current"),
                "total": previous.get("total"),
                "percent": 100.0 if terminal_status == "completed" else previous.get("percent"),
                "status": terminal_status,
                "completed_at": datetime.now().isoformat(),
                "market": (market or "shared").lower(),
            }
            self._redis_client.setex(self._hb_key(market), 3600, json.dumps(heartbeat))
        except Exception as exc:
            self._logger.error("Error completing warmup heartbeat: %s", exc)
