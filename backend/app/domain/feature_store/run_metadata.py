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


def feature_run_publish_pointer_key(run: Any) -> str | None:
    """The pointer key the run was published to, as the run itself recorded it.

    ``build_daily_snapshot`` writes ``publish_pointer_key`` into the run config, so a run states
    whether it was published as a live market publication or as a staged rollout. That statement
    stays true after a retry moved the rollout pointer to a newer run and left this one behind no
    pointer at all - which is why the pointer table alone cannot answer it.
    """
    config = getattr(run, "config_json", None)
    if config is None:
        config = getattr(run, "config", None)
    if not isinstance(config, dict):
        return None
    key = config.get("publish_pointer_key")
    if key is None:
        return None
    normalized = str(key).strip()
    return normalized or None
