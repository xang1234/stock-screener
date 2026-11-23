"""Immutable domain values shared by all market-breadth adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Mapping

CURRENT_BREADTH_CALCULATION_REVISION = 3


@dataclass(frozen=True, slots=True)
class BreadthMarketPolicy:
    market: str
    currency: str
    min_adtv_local: float
    min_daily_volume: int
    min_month_reference_price_local: float


@dataclass(frozen=True, slots=True)
class BreadthFormulaPolicy:
    calculation_revision: int = CURRENT_BREADTH_CALCULATION_REVISION
    atr_period: int = 14
    atr_extension_threshold: float = 10.0


@dataclass(frozen=True, slots=True)
class BreadthDailyCount:
    date: date
    stocks_up_4pct: int
    stocks_down_4pct: int
    market: str | None = None
    calculation_revision: int = CURRENT_BREADTH_CALCULATION_REVISION


@dataclass(frozen=True, slots=True)
class BreadthRatios:
    ratio_5day: float | None = None
    ratio_10day: float | None = None


@dataclass(frozen=True, slots=True)
class BreadthUniverseMember:
    symbol: str
    currency: str
    is_common_stock: bool = True


@dataclass(frozen=True, slots=True)
class BreadthUniverseSnapshot:
    calculation_date: date
    members: tuple[BreadthUniverseMember, ...]
    broad_signature: str


@dataclass(frozen=True, slots=True)
class SymbolMetricEligibility:
    advance_decline: bool = False
    stockbee_liquidity: bool = False
    stockbee_daily: bool = False
    stockbee_month: bool = False
    stockbee_34day: bool = False
    stockbee_quarter: bool = False
    t2108: bool = False
    high_low_52week: bool = False
    atr_extension: bool = False
    sma200: bool = False
    off_high: bool = False


@dataclass(frozen=True, slots=True)
class SymbolBreadthSignals:
    eligibility: SymbolMetricEligibility
    advancing: bool = False
    declining: bool = False
    unchanged: bool = False
    up_4pct: bool = False
    down_4pct: bool = False
    up_25pct_quarter: bool = False
    down_25pct_quarter: bool = False
    up_25pct_month: bool = False
    down_25pct_month: bool = False
    up_50pct_month: bool = False
    down_50pct_month: bool = False
    up_13pct_34days: bool = False
    down_13pct_34days: bool = False
    new_high_52week: bool = False
    new_low_52week: bool = False
    t2108_above: bool = False
    atr_10x_extension: bool = False
    above_sma200: bool = False
    down_20pct_from_high: bool = False
    up_4pct_near_high: bool = False


@dataclass(frozen=True, slots=True)
class SymbolBreadthEvaluation:
    signals: SymbolBreadthSignals
    daily_change_pct: float | None
    qualifying_values: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class BreadthContributorMetadata:
    company_name: str | None = None
    ibd_industry_group: str = "No Group"


@dataclass(frozen=True, slots=True)
class BreadthContributor:
    symbol: str
    company_name: str | None
    ibd_industry_group: str
    daily_change_pct: float | None
    signals: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class BreadthContributorSnapshotResult:
    market: str
    calculation_date: date
    calculation_revision: int
    schema_id: str
    contributors: tuple[BreadthContributor, ...]


@dataclass(frozen=True, slots=True)
class BreadthEligibilityCounts:
    advance_decline_eligible_count: int = 0
    stockbee_daily_eligible_count: int = 0
    stockbee_month_eligible_count: int = 0
    stockbee_34day_eligible_count: int = 0
    stockbee_quarter_eligible_count: int = 0
    t2108_eligible_count: int = 0
    high_low_52week_eligible_count: int = 0
    atr_extension_eligible_count: int = 0
    sma200_eligible_count: int = 0
    off_high_eligible_count: int = 0


@dataclass(frozen=True, slots=True)
class BreadthIndicatorValues:
    stocks_up_4pct: int = 0
    stocks_down_4pct: int = 0
    ratio_5day: float | None = None
    ratio_10day: float | None = None
    stocks_up_25pct_quarter: int = 0
    stocks_down_25pct_quarter: int = 0
    stocks_up_25pct_month: int = 0
    stocks_down_25pct_month: int = 0
    stocks_up_50pct_month: int = 0
    stocks_down_50pct_month: int = 0
    stocks_up_13pct_34days: int = 0
    stocks_down_13pct_34days: int = 0
    advancing_count: int = 0
    declining_count: int = 0
    unchanged_count: int = 0
    new_high_52week_count: int = 0
    new_low_52week_count: int = 0
    t2108_count: int = 0
    t2108_pct: float | None = None
    atr_10x_extension_count: int = 0
    above_sma200_count: int = 0
    above_sma200_pct: float | None = None
    down_20pct_from_high_count: int = 0
    down_20pct_from_high_pct: float | None = None
    up_4pct_near_high_count: int = 0


# (pct field, count field, eligible denominator field) on one breadth row.
BREADTH_SHARE_FIELDS = (
    ("t2108_pct", "t2108_count", "t2108_eligible_count"),
    ("above_sma200_pct", "above_sma200_count", "sma200_eligible_count"),
    (
        "down_20pct_from_high_pct",
        "down_20pct_from_high_count",
        "off_high_eligible_count",
    ),
)


def share_pct(count: int | None, eligible: int | None) -> float | None:
    """Percent of an eligible denominator, None when nothing was eligible."""
    if count is None or not eligible:
        return None
    return round(count / eligible * 100.0, 2)


@dataclass(frozen=True, slots=True)
class BreadthDailyResult:
    market: str
    calculation_date: date
    values: BreadthIndicatorValues
    eligibility: BreadthEligibilityCounts
    broad_universe_count: int
    eligibility_signature: str
    stockbee_eligibility_signature: str
    calculation_revision: int = CURRENT_BREADTH_CALCULATION_REVISION

    def to_record_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "market": self.market,
            "date": self.calculation_date,
            **asdict(self.values),
            **asdict(self.eligibility),
            "broad_universe_count": self.broad_universe_count,
            "total_stocks_scanned": self.broad_universe_count,
            "eligibility_signature": self.eligibility_signature,
            "stockbee_eligibility_signature": self.stockbee_eligibility_signature,
            "calculation_revision": self.calculation_revision,
        }
        return result


@dataclass(frozen=True, slots=True)
class BreadthEngineBatchResult:
    daily_results: Mapping[date, BreadthDailyResult]
    contributor_snapshots: Mapping[date, BreadthContributorSnapshotResult]
