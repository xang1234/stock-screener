from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class GroupSnapshotIdentity:
    market: str
    as_of_date: date
    formula_version: str

    def __post_init__(self) -> None:
        market = str(self.market).strip().upper()
        formula = str(self.formula_version).strip()
        if not market:
            raise ValueError("market is required")
        if not formula:
            raise ValueError("formula_version is required")
        object.__setattr__(self, "market", market)
        object.__setattr__(self, "formula_version", formula)


@dataclass(frozen=True)
class RsPublicationIdentity:
    snapshot: GroupSnapshotIdentity
    market_rs_run_id: int | None
    universe_size: int | None

    def __post_init__(self) -> None:
        if self.market_rs_run_id is not None and int(self.market_rs_run_id) <= 0:
            raise ValueError("market_rs_run_id must be positive")
        if self.universe_size is not None and int(self.universe_size) <= 0:
            raise ValueError("universe_size must be positive")
        if (self.market_rs_run_id is None) != (self.universe_size is None):
            raise ValueError(
                "market_rs_run_id and universe_size must both be present or absent"
            )
        if self.market_rs_run_id is not None:
            object.__setattr__(self, "market_rs_run_id", int(self.market_rs_run_id))
            object.__setattr__(self, "universe_size", int(self.universe_size))
