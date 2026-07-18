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
    rs_formula_version: str
    market_rs_run_id: int | None
    rs_as_of_date: str
    rs_universe_size: int | None


def build_static_groups_payload(
    snapshot: StaticGroupsSnapshot,
    *,
    generated_at: str,
    schema_version: str,
) -> dict[str, Any]:
    rs_metadata = {
        "rs_formula_version": snapshot.rs_formula_version,
        "rs_as_of_date": snapshot.rs_as_of_date,
        "rs_universe_size": snapshot.rs_universe_size,
    }
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "generated_at": generated_at,
        "available": True,
        "payload": {
            "rankings": GroupRankingsResponse(
                date=snapshot.date,
                total_groups=len(snapshot.rankings),
                rankings=[GroupRankResponse(**row) for row in snapshot.rankings],
                **rs_metadata,
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
                **rs_metadata,
            ).model_dump(mode="json"),
            "group_details": snapshot.group_details,
        },
    }
    payload["market"] = snapshot.market
    payload.update(
        {
            **rs_metadata,
            "market_rs_run_id": snapshot.market_rs_run_id,
        }
    )
    return payload
