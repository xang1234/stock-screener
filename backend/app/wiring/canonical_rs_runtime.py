"""Lazy runtime container for the canonical market-RS subsystem."""

from __future__ import annotations

from threading import RLock
from typing import Callable

from app.domain.scanning.ports import MarketRsReader
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.services.canonical_group_ranking_service import CanonicalGroupRankingService
from app.services.group_rank_snapshot_coordinator import GroupRankSnapshotCoordinator
from app.services.group_rank_snapshot_reader import GroupRankSnapshotReader
from app.services.market_calendar_service import MarketCalendarService
from app.services.market_rs_inputs import MarketRsInputLoader
from app.services.market_rs_rollout_service import MarketRsRolloutService
from app.services.market_rs_snapshot_service import MarketRsSnapshotService
from app.services.point_in_time_universe_service import PointInTimeUniverseService
from app.wiring.market_rs_services import MarketRsServices, build_market_rs_services


class CanonicalRsRuntime:
    """Own canonical RS wiring so the application bootstrap stays a facade."""

    def __init__(
        self,
        *,
        session_factory,
        market_calendar: MarketCalendarService,
        legacy_group_service_provider: Callable[[], object],
    ) -> None:
        self._session_factory = session_factory
        self._market_calendar = market_calendar
        self._legacy_group_service_provider = legacy_group_service_provider
        self._lock = RLock()
        self._point_in_time_universe: PointInTimeUniverseService | None = None
        self._services: MarketRsServices | None = None
        self._canonical_group_service: CanonicalGroupRankingService | None = None
        self._rollout_service: MarketRsRolloutService | None = None
        self._snapshot_reader: GroupRankSnapshotReader | None = None
        self._snapshot_coordinator: GroupRankSnapshotCoordinator | None = None

    def point_in_time_universe_service(self) -> PointInTimeUniverseService:
        if self._point_in_time_universe is None:
            with self._lock:
                if self._point_in_time_universe is None:
                    self._point_in_time_universe = PointInTimeUniverseService(
                        market_calendar=self._market_calendar
                    )
        return self._point_in_time_universe

    def market_rs_services(self) -> MarketRsServices:
        if self._services is None:
            with self._lock:
                if self._services is None:
                    self._services = build_market_rs_services(
                        session_factory=self._session_factory,
                        point_in_time_universe=self.point_in_time_universe_service(),
                        market_calendar=self._market_calendar,
                    )
        return self._services

    def input_loader(self) -> MarketRsInputLoader:
        return self.market_rs_services().input_loader

    def repository(self) -> MarketRsRunRepository:
        return self.market_rs_services().repository

    def snapshot_service(self) -> MarketRsSnapshotService:
        return self.market_rs_services().snapshot_service

    def reader(self) -> MarketRsReader:
        return self.market_rs_services().reader

    def canonical_group_service(self) -> CanonicalGroupRankingService:
        if self._canonical_group_service is None:
            with self._lock:
                if self._canonical_group_service is None:
                    self._canonical_group_service = CanonicalGroupRankingService(
                        repository=self.repository()
                    )
        return self._canonical_group_service

    def rollout_service(self) -> MarketRsRolloutService:
        if self._rollout_service is None:
            with self._lock:
                if self._rollout_service is None:
                    self._rollout_service = MarketRsRolloutService(
                        calendar_service=self._market_calendar,
                        input_loader=self.input_loader(),
                        market_rs_snapshot_service=self.snapshot_service(),
                        market_rs_repository=self.repository(),
                        canonical_group_service=self.canonical_group_service(),
                    )
        return self._rollout_service

    def group_rank_snapshot_reader(self) -> GroupRankSnapshotReader:
        if self._snapshot_reader is None:
            with self._lock:
                if self._snapshot_reader is None:
                    self._snapshot_reader = GroupRankSnapshotReader()
        return self._snapshot_reader

    def group_rank_snapshot_coordinator(self) -> GroupRankSnapshotCoordinator:
        if self._snapshot_coordinator is None:
            with self._lock:
                if self._snapshot_coordinator is None:
                    self._snapshot_coordinator = GroupRankSnapshotCoordinator(
                        reader=self.group_rank_snapshot_reader(),
                        market_rs_snapshot_service=self.snapshot_service(),
                        canonical_group_service=self.canonical_group_service(),
                        legacy_group_service=self._legacy_group_service_provider(),
                    )
        return self._snapshot_coordinator
