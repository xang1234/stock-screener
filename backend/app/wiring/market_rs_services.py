from dataclasses import dataclass

from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.services.market_rs_inputs import MarketRsInputLoader
from app.services.market_rs_reader import SqlMarketRsReader
from app.services.market_rs_snapshot_service import MarketRsSnapshotService


@dataclass(frozen=True)
class MarketRsServices:
    repository: MarketRsRunRepository
    input_loader: MarketRsInputLoader
    snapshot_service: MarketRsSnapshotService
    reader: SqlMarketRsReader


def build_market_rs_services(
    *,
    session_factory,
    point_in_time_universe,
    market_calendar,
) -> MarketRsServices:
    repository = MarketRsRunRepository()
    input_loader = MarketRsInputLoader(
        point_in_time_universe=point_in_time_universe,
        market_calendar=market_calendar,
    )
    snapshot_service = MarketRsSnapshotService(
        input_loader=input_loader,
        repository=repository,
    )
    return MarketRsServices(
        repository=repository,
        input_loader=input_loader,
        snapshot_service=snapshot_service,
        reader=SqlMarketRsReader(session_factory, repository=repository),
    )
