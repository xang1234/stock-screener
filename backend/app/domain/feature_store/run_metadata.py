"""FeatureRun metadata helpers."""

from __future__ import annotations

from typing import Any


def feature_run_market(run: Any) -> str | None:
    config = getattr(run, "config_json", None) or {}
    if not isinstance(config, dict):
        return None
    universe = config.get("universe")
    if isinstance(universe, dict):
        market = universe.get("market")
        if market is not None:
            normalized = str(market).strip().upper()
            return normalized or None
    return None
