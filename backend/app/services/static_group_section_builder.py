from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy.orm import Session

from app.domain.relative_strength import (
    GroupSnapshotIdentity,
    RsPublicationIdentity,
)
from app.infra.db.models.feature_store import FeatureRun
from app.services.group_detail_payloads import (
    constituent_stock_payloads_from_group_rows,
)
from app.services.ibd_group_rank_service import GROUP_RANK_CHANGE_CALENDAR_DAYS
from app.services.group_ranking_history import build_group_detail_payload_from_parts
from app.services.group_ranking_payloads import group_snapshot_metadata
from app.services.group_rank_snapshot_reader import GroupSnapshotIntegrityError
from app.services.feature_run_rs_identity import (
    FeatureRunRsIdentityError,
    resolve_feature_run_rs_identity,
)
from app.services.static_groups_payload_builder import (
    StaticGroupsSnapshot,
    build_static_groups_payload,
)
from app.services.static_site_errors import StaticSiteSectionUnavailableError


STATIC_SITE_SCHEMA_VERSION = "static-site-v3"
STATIC_GROUP_HISTORY_RUNS = 40


class StaticGroupSectionBuilder:
    def __init__(
        self,
        *,
        snapshot_reader,
        rank_history_reader,
    ) -> None:
        self._snapshot_reader = snapshot_reader
        self._rank_history_reader = rank_history_reader

    def build(
        self,
        *,
        db: Session,
        generated_at: str,
        identity: GroupSnapshotIdentity,
        latest_run: FeatureRun,
        serialized_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            rankings = self._snapshot_reader.load_exact(db, identity=identity)
        except GroupSnapshotIntegrityError as exc:
            raise StaticSiteSectionUnavailableError(
                section="groups",
                reason=str(exc),
            ) from exc
        if not rankings:
            raise StaticSiteSectionUnavailableError(
                section="groups",
                reason=(
                    f"No exact stored Group snapshot is available for {identity.market} "
                    f"on {identity.as_of_date.isoformat()} with formula "
                    f"{identity.formula_version}."
                ),
            )
        try:
            metadata = group_snapshot_metadata(
                db,
                market=identity.market,
                rankings=rankings,
            )
        except RuntimeError as exc:
            raise StaticSiteSectionUnavailableError(
                section="groups",
                reason=str(exc),
            ) from exc
        self._validate_feature_run_group_source(
            latest_run=latest_run,
            identity=identity,
            market_rs_run_id=rankings[0].get("market_rs_run_id"),
            rs_universe_size=metadata["rs_universe_size"],
            serialized_rows=serialized_rows,
        )
        historical_ranks = self._rank_history_reader.get_historical_ranks_batch(
            db,
            [str(row["industry_group"]) for row in rankings],
            identity.as_of_date,
            GROUP_RANK_CHANGE_CALENDAR_DAYS,
            market=identity.market,
            formula_version=identity.formula_version,
        )
        for row in rankings:
            for period_name in GROUP_RANK_CHANGE_CALENDAR_DAYS:
                historical_rank = historical_ranks.get(
                    (str(row["industry_group"]), period_name)
                )
                row[f"rank_change_{period_name}"] = (
                    int(historical_rank) - int(row["rank"])
                    if historical_rank is not None
                    else None
                )
        historical = self._load_stored_group_history(db, identity=identity)
        details = self._build_stored_group_details(
            rankings=rankings,
            serialized_rows=serialized_rows,
            historical_rankings=historical,
        )
        return build_static_groups_payload(
            StaticGroupsSnapshot(
                date=identity.as_of_date.isoformat(),
                rankings=rankings,
                movers=self._build_group_movers(rankings),
                group_details=details,
                market=identity.market,
                rs_formula_version=identity.formula_version,
                market_rs_run_id=rankings[0].get("market_rs_run_id"),
                rs_as_of_date=metadata["rs_as_of_date"],
                rs_universe_size=metadata["rs_universe_size"],
            ),
            generated_at=generated_at,
            schema_version=STATIC_SITE_SCHEMA_VERSION,
        )

    @staticmethod
    def _validate_feature_run_group_source(
        *,
        latest_run: FeatureRun,
        identity: GroupSnapshotIdentity,
        market_rs_run_id: int | None,
        rs_universe_size: int | None,
        serialized_rows: list[dict[str, Any]],
    ) -> None:
        if (
            latest_run.status != "published"
            or latest_run.as_of_date != identity.as_of_date
        ):
            raise StaticSiteSectionUnavailableError(
                section="groups",
                reason=(
                    f"Feature run {latest_run.id} does not match published "
                    f"{identity.market} snapshot date {identity.as_of_date.isoformat()}."
                ),
            )
        try:
            resolved = resolve_feature_run_rs_identity(
                latest_run,
                ranking_date=identity.as_of_date,
            )
            expected = RsPublicationIdentity(
                snapshot=identity,
                market_rs_run_id=market_rs_run_id,
                universe_size=rs_universe_size,
            )
        except (FeatureRunRsIdentityError, ValueError) as exc:
            raise StaticSiteSectionUnavailableError(
                section="groups",
                reason=f"Feature run {latest_run.id} has invalid RS identity: {exc}",
            ) from exc
        if resolved.publication != expected:
            raise StaticSiteSectionUnavailableError(
                section="groups",
                reason=(
                    f"Feature run {latest_run.id} RS identity does not match the "
                    "stored Group publication."
                ),
            )
        for row in serialized_rows:
            row_formula = row.get("rs_formula_version")
            row_run_id = row.get("market_rs_run_id")
            if row_formula is not None and row_formula != identity.formula_version:
                raise StaticSiteSectionUnavailableError(
                    section="groups",
                    reason=f"Stock row {row.get('symbol')} uses a different RS formula.",
                )
            if row_run_id is not None and int(row_run_id) != market_rs_run_id:
                raise StaticSiteSectionUnavailableError(
                    section="groups",
                    reason=f"Stock row {row.get('symbol')} uses a different Market RS run.",
                )

    def _load_stored_group_history(
        self,
        db: Session,
        *,
        identity: GroupSnapshotIdentity,
    ) -> list[tuple[Any, list[dict[str, Any]]]]:
        available = self._snapshot_reader.available_dates(
            db,
            market=identity.market,
            formula_version=identity.formula_version,
            through_date=identity.as_of_date,
        )
        selected_dates = tuple(reversed(available[-STATIC_GROUP_HISTORY_RUNS:]))
        return [
            (
                ranking_date,
                self._snapshot_reader.load_exact(
                    db,
                    identity=GroupSnapshotIdentity(
                        identity.market,
                        ranking_date,
                        identity.formula_version,
                    ),
                    include_top_symbol_names=False,
                ),
            )
            for ranking_date in selected_dates
        ]

    @staticmethod
    def _build_stored_group_details(
        *,
        rankings: list[dict[str, Any]],
        serialized_rows: list[dict[str, Any]],
        historical_rankings: list[tuple[Any, list[dict[str, Any]]]],
    ) -> dict[str, Any]:
        current_rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in serialized_rows:
            group = row.get("ibd_industry_group")
            if group:
                current_rows_by_group[str(group)].append(row)
        historical_maps = [
            (ranking_date, {str(row["industry_group"]): row for row in rows})
            for ranking_date, rows in historical_rankings
        ]
        details: dict[str, Any] = {}
        for ranking in rankings:
            group = str(ranking["industry_group"])
            history = []
            for _ranking_date, rows_by_group in historical_maps:
                row = rows_by_group.get(group)
                if row is None:
                    continue
                history.append(
                    {
                        "date": row["date"],
                        "rank": row["rank"],
                        "avg_rs_rating": row["avg_rs_rating"],
                        "avg_rs_rating_1m": row.get("avg_rs_rating_1m"),
                        "avg_rs_rating_3m": row.get("avg_rs_rating_3m"),
                        "num_stocks": row["num_stocks"],
                    }
                )
            details[group] = build_group_detail_payload_from_parts(
                group,
                ranking=ranking,
                history=history,
                stocks=constituent_stock_payloads_from_group_rows(
                    current_rows_by_group.get(group, [])
                ),
            )
        return details

    @staticmethod
    def _build_group_movers(rankings: list[dict[str, Any]]) -> dict[str, Any]:
        gainers = sorted(
            [row for row in rankings if (row.get("rank_change_1w") or 0) > 0],
            key=lambda row: (-(row.get("rank_change_1w") or 0), row["rank"]),
        )[:10]
        losers = sorted(
            [row for row in rankings if (row.get("rank_change_1w") or 0) < 0],
            key=lambda row: ((row.get("rank_change_1w") or 0), row["rank"]),
        )[:10]
        return {"period": "1w", "gainers": gainers, "losers": losers}
