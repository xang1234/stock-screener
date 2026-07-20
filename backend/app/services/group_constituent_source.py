"""Resolve group constituent stocks from feature-run or legacy scan storage."""

from __future__ import annotations

import logging
from datetime import date, datetime, time
from typing import Callable

from sqlalchemy import desc, or_
from sqlalchemy.orm import Session

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    RsPublicationIdentity,
)
from app.domain.scanning.models import ScanResultItemDomain
from app.infra.db.models.feature_store import FeatureRun
from app.infra.db.repositories.feature_store_repo import SqlFeatureStoreRepository
from app.infra.db.repositories.scan_result_repo import SqlScanResultRepository
from app.models.scan_result import Scan
from app.services.feature_run_rs_identity import feature_run_matches_rs_source

logger = logging.getLogger(__name__)

FeatureRepoFactory = Callable[[Session], SqlFeatureStoreRepository]
ScanRepoFactory = Callable[[Session], SqlScanResultRepository]


class GroupConstituentPublicationUnavailable(LookupError):
    """No Feature run matches the exact canonical Group publication."""


class GroupConstituentSource:
    """Read group constituents while keeping source-selection out of services."""

    def __init__(
        self,
        *,
        feature_repo_factory: FeatureRepoFactory = SqlFeatureStoreRepository,
        scan_repo_factory: ScanRepoFactory = SqlScanResultRepository,
    ) -> None:
        self._feature_repo_factory = feature_repo_factory
        self._scan_repo_factory = scan_repo_factory

    def get_constituent_items(
        self,
        db: Session,
        industry_group: str,
        *,
        publication: RsPublicationIdentity,
    ) -> tuple[ScanResultItemDomain, ...]:
        feature_run = self._get_feature_run_for_publication(
            db,
            publication=publication,
        )
        if feature_run is not None:
            peers = self._feature_repo_factory(db).get_peers_by_industry_for_run(
                feature_run.id,
                industry_group,
                include_sparklines=True,
            )
            logger.info(
                "Found %d feature-run stocks for group %s (%s)",
                len(peers),
                industry_group,
                publication.snapshot.market,
            )
            return peers

        if publication.snapshot.formula_version == BALANCED_RS_FORMULA_VERSION:
            raise GroupConstituentPublicationUnavailable(
                "No published Feature run matches canonical Group publication "
                f"{publication.snapshot.market}/"
                f"{publication.snapshot.as_of_date.isoformat()}/"
                f"run-{publication.market_rs_run_id}."
            )

        latest_scan = self._get_latest_legacy_scan_for_market(
            db,
            market=publication.snapshot.market,
            as_of_date=publication.snapshot.as_of_date,
        )
        if not latest_scan:
            logger.warning("No completed scans found for constituent stocks")
            return ()

        peers = self._scan_repo_factory(db).get_peers_by_industry(
            latest_scan.scan_id,
            industry_group,
        )
        logger.info(
            "Found %d legacy scan stocks for group %s (%s)",
            len(peers),
            industry_group,
            publication.snapshot.market,
        )
        return peers

    @staticmethod
    def _scan_market_filter(normalized_market: str):
        if normalized_market == "US":
            return or_(Scan.universe_market == "US", Scan.universe_market.is_(None))
        return Scan.universe_market == normalized_market

    def _get_feature_run_for_publication(
        self,
        db: Session,
        *,
        publication: RsPublicationIdentity,
    ) -> FeatureRun | None:
        query = (
            db.query(FeatureRun)
            .filter(
                FeatureRun.status == "published",
                FeatureRun.as_of_date == publication.snapshot.as_of_date,
            )
            .order_by(
                desc(FeatureRun.published_at),
                desc(FeatureRun.id),
            )
        )
        for run in query.all():
            if feature_run_matches_rs_source(
                run,
                identity=publication.snapshot,
                market_rs_run_id=publication.market_rs_run_id,
                universe_size=publication.universe_size,
            ):
                return run
        return None

    def _get_latest_legacy_scan_for_market(
        self,
        db: Session,
        *,
        market: str,
        as_of_date: date | None,
    ) -> Scan | None:
        normalized_market = str(market or "US").strip().upper()
        query = db.query(Scan).filter(
            Scan.status == "completed",
            self._scan_market_filter(normalized_market),
        )
        if as_of_date is not None:
            query = query.filter(
                Scan.completed_at <= datetime.combine(as_of_date, time.max)
            )
        return query.filter(
            Scan.feature_run_id.is_(None),
        ).order_by(
            desc(Scan.completed_at),
            desc(Scan.id),
        ).first()
