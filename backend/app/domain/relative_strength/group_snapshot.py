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
