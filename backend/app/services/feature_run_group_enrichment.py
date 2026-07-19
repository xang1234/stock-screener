from __future__ import annotations

from datetime import date

from sqlalchemy import func

from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.models.industry import IBDIndustryGroup
from app.services.feature_run_rs_identity import resolve_feature_run_rs_identity
from app.services.group_rank_snapshot_reader import (
    GroupRankSnapshotReader,
    GroupSnapshotUnavailable,
)


class FeatureRunGroupEnrichmentService:
    def __init__(
        self,
        *,
        session_factory,
        taxonomy_service,
        snapshot_reader: GroupRankSnapshotReader,
        batch_size: int,
    ) -> None:
        self._session_factory = session_factory
        self._taxonomy_service = taxonomy_service
        self._snapshot_reader = snapshot_reader
        self._batch_size = max(1, int(batch_size))

    def enrich(
        self,
        *,
        feature_run_id: int,
        ranking_date: date,
    ) -> dict[str, int | str]:
        db = self._session_factory()
        try:
            feature_run = (
                db.query(FeatureRun).filter(FeatureRun.id == feature_run_id).first()
            )
            resolution = resolve_feature_run_rs_identity(
                feature_run,
                ranking_date=ranking_date,
            )
            identity = resolution.identity
            snapshot_rows = self._snapshot_reader.load_exact(
                db,
                identity=identity,
                include_top_symbol_names=False,
            )
            if not snapshot_rows:
                raise GroupSnapshotUnavailable(identity)
            ranks_by_group = {
                str(row["industry_group"]): int(row["rank"])
                for row in snapshot_rows
            }

            raw_total_rows = (
                db.query(func.count(StockFeatureDaily.symbol))
                .filter(StockFeatureDaily.run_id == feature_run_id)
                .scalar()
                or 0
            )
            total_rows = (
                int(raw_total_rows)
                if isinstance(raw_total_rows, (int, float))
                else 0
            )
            industries_by_symbol: dict[str, str | None] = {}
            market_themes_by_symbol: dict[str, list[str]] = {}
            sector_by_symbol: dict[str, str | None] = {}
            industry_by_symbol: dict[str, str | None] = {}

            if identity.market == "US":
                industries_by_symbol = {
                    symbol: industry_group
                    for symbol, industry_group in (
                        db.query(
                            StockFeatureDaily.symbol,
                            IBDIndustryGroup.industry_group,
                        )
                        .join(
                            IBDIndustryGroup,
                            IBDIndustryGroup.symbol == StockFeatureDaily.symbol,
                        )
                        .filter(StockFeatureDaily.run_id == feature_run_id)
                        .all()
                    )
                }
            else:
                for batch in self._iter_feature_rows(db, feature_run_id):
                    for row in batch:
                        entry = self._taxonomy_service.get(
                            row.symbol,
                            market=identity.market,
                        )
                        industries_by_symbol[row.symbol] = (
                            entry.industry_group if entry else None
                        )
                        sector_by_symbol[row.symbol] = entry.sector if entry else None
                        industry_by_symbol[row.symbol] = (
                            entry.industry if entry else None
                        )
                        market_themes_by_symbol[row.symbol] = (
                            entry.themes_list() if entry else []
                        )
                    db.expunge_all()

            updated_rows = 0
            missing_industry_rows = 0
            missing_rank_rows = 0
            for batch in self._iter_feature_rows(db, feature_run_id):
                batch_updated = False
                for row in batch:
                    details = dict(row.details_json or {})
                    industry_group = industries_by_symbol.get(row.symbol)
                    group_rank = (
                        ranks_by_group.get(industry_group) if industry_group else None
                    )
                    group_rank_date = (
                        ranking_date.isoformat() if group_rank is not None else None
                    )
                    market_themes = (
                        []
                        if identity.market == "US"
                        else list(market_themes_by_symbol.get(row.symbol) or [])
                    )
                    sector = sector_by_symbol.get(row.symbol)
                    industry = industry_by_symbol.get(row.symbol)
                    sector_changed = bool(
                        identity.market != "US"
                        and sector
                        and details.get("gics_sector") != sector
                    )
                    industry_changed = bool(
                        identity.market != "US"
                        and industry
                        and details.get("gics_industry") != industry
                    )
                    if industry_group is None:
                        missing_industry_rows += 1
                    elif group_rank is None:
                        missing_rank_rows += 1

                    if (
                        details.get("ibd_industry_group") != industry_group
                        or details.get("ibd_group_rank") != group_rank
                        or details.get("ibd_group_rank_date") != group_rank_date
                        or list(details.get("market_themes") or []) != market_themes
                        or sector_changed
                        or industry_changed
                    ):
                        details["ibd_industry_group"] = industry_group
                        details["ibd_group_rank"] = group_rank
                        details["ibd_group_rank_date"] = group_rank_date
                        details["market_themes"] = market_themes
                        if sector_changed:
                            details["gics_sector"] = sector
                        if industry_changed:
                            details["gics_industry"] = industry
                        row.details_json = details
                        updated_rows += 1
                        batch_updated = True
                if batch_updated:
                    db.flush()
                db.expunge_all()

            db.commit()
            return {
                "run_id": feature_run_id,
                "ranking_date": ranking_date.isoformat(),
                "total_rows": total_rows,
                "updated_rows": updated_rows,
                "missing_industry_rows": missing_industry_rows,
                "missing_rank_rows": missing_rank_rows,
                "rs_formula_version": identity.formula_version,
                "identity_source": resolution.identity_source,
            }
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _iter_feature_rows(self, db, feature_run_id: int):
        last_symbol: str | None = None
        while True:
            query = db.query(StockFeatureDaily).filter(
                StockFeatureDaily.run_id == feature_run_id
            )
            if last_symbol is not None:
                query = query.filter(StockFeatureDaily.symbol > last_symbol)
            batch = (
                query.order_by(StockFeatureDaily.symbol)
                .limit(self._batch_size)
                .all()
            )
            if not batch:
                break
            last_symbol = str(batch[-1].symbol)
            yield batch
