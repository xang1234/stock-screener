"""Warmup metadata and heartbeat persistence for price cache operations."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from ...theme_platform.contracts import WarmupHeartbeatState, WarmupStateSnapshot


class PriceCacheWarmupStore:
    """Owns warmup metadata + heartbeat persistence in Redis."""

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
        self._metadata_key = metadata_key
        self._heartbeat_key = heartbeat_key

    def get_warmup_metadata(self) -> Optional[WarmupStateSnapshot]:
        if not self._redis_client:
            return None
        try:
            meta_json = self._redis_client.get(self._metadata_key)
            if meta_json:
                return json.loads(meta_json)
            return None
        except Exception as exc:
            self._logger.error("Error getting warmup metadata: %s", exc)
            return None

    def save_warmup_metadata(self, status: str, count: int, total: int, error: str | None = None) -> None:
        if not self._redis_client:
            return
        try:
            meta = {
                "status": status,
                "count": count,
                "total": total,
                "completed_at": datetime.now().isoformat(),
                "error": error,
            }
            self._redis_client.setex(self._metadata_key, 86400 * 7, json.dumps(meta))
            self._logger.info("Saved warmup metadata: %s (%s/%s)", status, count, total)
        except Exception as exc:
            self._logger.error("Error saving warmup metadata: %s", exc)

    def update_warmup_heartbeat(self, current: int, total: int, percent: float | None = None) -> None:
        if not self._redis_client:
            return
        try:
            heartbeat = {
                "status": "running",
                "current": current,
                "total": total,
                "percent": percent if percent is not None else (round((current / total) * 100, 1) if total > 0 else 0),
                "updated_at": datetime.now().isoformat(),
            }
            self._redis_client.setex(self._heartbeat_key, 3600, json.dumps(heartbeat))
        except Exception as exc:
            self._logger.error("Error updating warmup heartbeat: %s", exc)

    def get_heartbeat_info(self) -> Optional[WarmupHeartbeatState]:
        if not self._redis_client:
            return None
        try:
            heartbeat_json = self._redis_client.get(self._heartbeat_key)
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

    def get_minutes_since_heartbeat(self) -> Optional[float]:
        info = self.get_heartbeat_info()
        if info is None:
            return None
        return info.get("minutes")

    def get_task_progress(self) -> dict[str, int | float | None]:
        if not self._redis_client:
            return {}
        try:
            heartbeat_json = self._redis_client.get(self._heartbeat_key)
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

    def clear_warmup_heartbeat(self) -> None:
        if not self._redis_client:
            return
        try:
            self._redis_client.delete(self._heartbeat_key)
        except Exception as exc:
            self._logger.error("Error clearing warmup heartbeat: %s", exc)

    def complete_warmup_heartbeat(self, status: str = "completed") -> None:
        if not self._redis_client:
            return
        try:
            heartbeat = {
                "status": status,
                "completed_at": datetime.now().isoformat(),
            }
            self._redis_client.setex(self._heartbeat_key, 3600, json.dumps(heartbeat))
        except Exception as exc:
            self._logger.error("Error completing warmup heartbeat: %s", exc)
