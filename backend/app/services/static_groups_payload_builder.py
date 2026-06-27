"""Static-site group payload assembly from a normalized group snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.schemas.groups import GroupRankResponse, GroupRankingsResponse, MoversResponse


@dataclass(frozen=True)
class StaticGroupsSnapshot:
    date: str
    rankings: list[dict[str, Any]]
    movers: dict[str, Any]
    group_details: dict[str, Any]
    market: str


def build_static_groups_payload(
    snapshot: StaticGroupsSnapshot,
    *,
    generated_at: str,
    schema_version: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "generated_at": generated_at,
        "available": True,
        "payload": {
            "rankings": GroupRankingsResponse(
                date=snapshot.date,
                total_groups=len(snapshot.rankings),
                rankings=[GroupRankResponse(**row) for row in snapshot.rankings],
            ).model_dump(mode="json"),
            "movers_period": snapshot.movers["period"],
            "movers": MoversResponse(
                period=snapshot.movers["period"],
                gainers=[
                    GroupRankResponse(**row)
                    for row in snapshot.movers.get("gainers", [])
                ],
                losers=[
                    GroupRankResponse(**row)
                    for row in snapshot.movers.get("losers", [])
                ],
            ).model_dump(mode="json"),
            "group_details": snapshot.group_details,
        },
    }
    payload["market"] = snapshot.market
    return payload
