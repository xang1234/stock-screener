"""FeatureRun metadata helpers."""

from __future__ import annotations

from typing import Any


def feature_run_market(run: Any) -> str | None:
    # ORM rows carry ``config_json``; the domain value object keeps the same payload under
    # ``config``, so both shapes answer the same question.
    config = getattr(run, "config_json", None)
    if config is None:
        config = getattr(run, "config", None)
    config = config or {}
    if not isinstance(config, dict):
        return None
    universe = config.get("universe")
    if isinstance(universe, dict):
        market = universe.get("market")
        if market is not None:
            normalized = str(market).strip().upper()
            return normalized or None
    return None
