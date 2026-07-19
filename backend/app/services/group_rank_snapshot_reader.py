from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy.orm import Session

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    GroupSnapshotIdentity,
    RsPublicationIdentity,
    balanced_run_has_required_price_basis,
)
from app.infra.db.models.relative_strength import MarketRsRun
from app.models.industry import IBDGroupRank
from app.services.group_ranking_payloads import (
    annotate_top_symbol_names,
    rank_record_payload,
)


class GroupSnapshotIntegrityError(RuntimeError):
    """Persisted Group rows do not form one coherent snapshot."""


class GroupSnapshotUnavailable(LookupError):
    def __init__(self, identity: GroupSnapshotIdentity) -> None:
        self.identity = identity
        super().__init__(
            f"Group snapshot unavailable for {identity.market} on "
            f"{identity.as_of_date.isoformat()} ({identity.formula_version})"
        )


class GroupRankSnapshotReader:
    def load_publication(
        self,
        db: Session,
        *,
        publication: RsPublicationIdentity,
        include_top_symbol_names: bool = True,
    ) -> list[dict[str, Any]]:
        rows = self.load_exact(
            db,
            identity=publication.snapshot,
            include_top_symbol_names=include_top_symbol_names,
        )
        if not rows:
            return rows
        actual_run_ids = {row.get("market_rs_run_id") for row in rows}
        if actual_run_ids != {publication.market_rs_run_id}:
            raise GroupSnapshotIntegrityError(
                "Group rows do not reference the expected Market RS run"
            )
        if publication.market_rs_run_id is None:
            return rows
        run = db.get(MarketRsRun, publication.market_rs_run_id)
        if run is None or int(run.eligible_symbol_count) != publication.universe_size:
            raise GroupSnapshotIntegrityError(
                "Group rows do not reference the expected Market RS universe"
            )
        return rows

    def load_exact(
        self,
        db: Session,
        *,
        identity: GroupSnapshotIdentity,
        include_top_symbol_names: bool = True,
    ) -> list[dict[str, Any]]:
        records = (
            db.query(IBDGroupRank)
            .filter(
                IBDGroupRank.market == identity.market,
                IBDGroupRank.date == identity.as_of_date,
                IBDGroupRank.rs_formula_version == identity.formula_version,
            )
            .order_by(IBDGroupRank.rank, IBDGroupRank.industry_group)
            .all()
        )
        self._validate(db, identity=identity, records=records)
        payload = [
            rank_record_payload(
                record,
                pct_rs_above_80=(
                    round(
                        100.0
                        * int(record.num_stocks_rs_above_80 or 0)
                        / int(record.num_stocks),
                        1,
                    )
                    if record.num_stocks
                    else None
                ),
            )
            for record in records
        ]
        if include_top_symbol_names:
            annotate_top_symbol_names(db, payload)
        return payload

    def load_rank_map(
        self,
        db: Session,
        *,
        identity: GroupSnapshotIdentity,
    ) -> dict[str, int]:
        return {
            str(row["industry_group"]): int(row["rank"])
            for row in self.load_exact(
                db,
                identity=identity,
                include_top_symbol_names=False,
            )
        }

    def available_dates(
        self,
        db: Session,
        *,
        market: str,
        formula_version: str,
        through_date: date,
    ) -> tuple[date, ...]:
        rows = (
            db.query(IBDGroupRank.date)
            .filter(
                IBDGroupRank.market == str(market).strip().upper(),
                IBDGroupRank.rs_formula_version == str(formula_version).strip(),
                IBDGroupRank.date <= through_date,
            )
            .distinct()
            .order_by(IBDGroupRank.date)
            .all()
        )
        return tuple(row[0] for row in rows)

    @staticmethod
    def _validate(
        db: Session,
        *,
        identity: GroupSnapshotIdentity,
        records: list[IBDGroupRank],
    ) -> None:
        if not records:
            return
        ranks = [int(record.rank) for record in records]
        if ranks != list(range(1, len(records) + 1)):
            raise GroupSnapshotIntegrityError("Group ranks are not contiguous")
        if any(
            record.market != identity.market
            or record.date != identity.as_of_date
            or record.rs_formula_version != identity.formula_version
            for record in records
        ):
            raise GroupSnapshotIntegrityError("Group rows do not match their identity")
        if identity.formula_version != BALANCED_RS_FORMULA_VERSION:
            return
        run_ids = {record.market_rs_run_id for record in records}
        if None in run_ids or len(run_ids) != 1:
            raise GroupSnapshotIntegrityError("Balanced Group rows mix Market RS run IDs")
        run = db.get(MarketRsRun, int(next(iter(run_ids))))
        if (
            run is None
            or run.status != "completed"
            or run.market != identity.market
            or run.as_of_date != identity.as_of_date
            or run.formula_version != identity.formula_version
        ):
            raise GroupSnapshotIntegrityError(
                "Group rows reference the wrong Market RS run"
            )
        if not balanced_run_has_required_price_basis(run):
            raise GroupSnapshotIntegrityError(
                "Group rows reference a Market RS run with an incompatible price basis"
            )
