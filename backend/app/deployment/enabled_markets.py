"""Deployment policy for ENABLED_MARKETS-style configuration."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..domain.markets import normalize_market_codes


DEFAULT_DEPLOYMENT_ENABLED_MARKETS: tuple[str, ...] = ("US",)


def normalize_deployment_enabled_markets(
    raw: str | Iterable[object] | None,
) -> list[str]:
    """Parse and canonicalize deployment-enabled market codes.

    Blank deployment config intentionally defaults to US so single-market
    deployments keep their legacy worker and Beat topology.
    """
    if raw is None or isinstance(raw, str):
        raw_values = (raw or "").split(",")
    elif isinstance(raw, Iterable):
        raw_values = raw
    else:
        raw_values = [raw]

    values = [
        value
        for value in raw_values
        if value is not None and str(value).strip()
    ]
    if not values:
        values = list(DEFAULT_DEPLOYMENT_ENABLED_MARKETS)

    return normalize_market_codes(values)


def markets_outside_deployment_message(
    requested_markets: Iterable[object],
    deployment_markets: Sequence[str],
) -> str | None:
    """Explain which requested markets this deployment does not run, or None.

    The deployment only starts workers and Beat entries for its own
    ENABLED_MARKETS, so an app market outside that list would never run.
    """
    outside = [
        market
        for market in normalize_market_codes(requested_markets)
        if market not in deployment_markets
    ]
    if not outside:
        return None
    allowed = ", ".join(deployment_markets)
    verb, pronoun = ("is", "it") if len(outside) == 1 else ("are", "them")
    return (
        f"{', '.join(outside)} {verb} not in this deployment's ENABLED_MARKETS ({allowed}). "
        f"Add {pronoun} to ENABLED_MARKETS and restart the API, Beat and workers "
        f"(on Docker, recreate the containers), "
        f"or choose from {allowed}."
    )
