"""Canonical range engine for all market-breadth calculation paths."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from types import MappingProxyType

import pandas as pd

from app.services.point_in_time_universe_service import (
    hash_point_in_time_universe_symbols,
)

from .contributors import (
    BREADTH_CONTRIBUTOR_SIGNALS,
    CONTRIBUTOR_SCHEMA_ID,
    NO_GROUP_LABEL,
    reconcile_contributor_counts,
)
from .formulas import evaluate_symbol_at, prepare_feature_frame, validate_price_frame
from .ratios import calculate_inclusive_ratios
from .types import (
    BREADTH_SHARE_FIELDS,
    CURRENT_BREADTH_CALCULATION_REVISION,
    BreadthContributor,
    BreadthContributorMetadata,
    BreadthContributorSnapshotResult,
    BreadthDailyCount,
    BreadthDailyResult,
    BreadthEligibilityCounts,
    BreadthEngineBatchResult,
    BreadthFormulaPolicy,
    BreadthIndicatorValues,
    BreadthMarketPolicy,
    BreadthUniverseMember,
    BreadthUniverseSnapshot,
    SymbolBreadthEvaluation,
    share_pct,
)
from .universe import breadth_eligibility_signature

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BreadthEngineRequest:
    market: str
    dates: tuple[date, ...]
    universes_by_date: Mapping[date, BreadthUniverseSnapshot]
    prices_by_symbol: Mapping[str, pd.DataFrame]
    market_policy: BreadthMarketPolicy
    seed_counts: tuple[BreadthDailyCount, ...] = ()
    contributor_metadata_by_date: Mapping[
        date, Mapping[str, BreadthContributorMetadata]
    ] = field(default_factory=dict)
    policy: BreadthFormulaPolicy = field(default_factory=BreadthFormulaPolicy)


class BreadthEngine:
    def calculate(
        self, request: BreadthEngineRequest
    ) -> Mapping[date, BreadthDailyResult]:
        """Compatibility projection for aggregate-only callers."""
        return self.calculate_with_contributors(request).daily_results

    def accumulator(
        self,
        *,
        market: str,
        dates: tuple[date, ...],
        universes_by_date: Mapping[date, BreadthUniverseSnapshot],
        market_policy: BreadthMarketPolicy,
        policy: BreadthFormulaPolicy | None = None,
    ) -> BreadthAccumulator:
        """Start a streaming calculation that accepts prices in batches."""
        return BreadthAccumulator(
            market=market,
            dates=dates,
            universes_by_date=universes_by_date,
            market_policy=market_policy,
            policy=policy or BreadthFormulaPolicy(),
        )

    def calculate_with_contributors(
        self, request: BreadthEngineRequest
    ) -> BreadthEngineBatchResult:
        accumulator = self.accumulator(
            market=request.market,
            dates=request.dates,
            universes_by_date=request.universes_by_date,
            market_policy=request.market_policy,
            policy=request.policy,
        )
        accumulator.add_prices(request.prices_by_symbol)
        return accumulator.finish(
            seed_counts=request.seed_counts,
            contributor_metadata_by_date=request.contributor_metadata_by_date,
        )


class _DateTotals:
    """Running per-date sums; per-symbol evaluations are not retained.

    Only symbols with qualifying contributor signals are kept, since the
    contributor snapshot lists them individually.
    """

    __slots__ = (
        "eligibility",
        "values",
        "stockbee_symbols",
        "qualifying",
    )

    def __init__(self) -> None:
        self.eligibility = dict.fromkeys(BreadthEligibilityCounts.__dataclass_fields__, 0)
        self.values = dict.fromkeys(_COUNTED_VALUE_FIELDS, 0)
        self.stockbee_symbols: list[str] = []
        self.qualifying: dict[str, SymbolBreadthEvaluation] = {}

    def add(self, symbol: str, evaluation: SymbolBreadthEvaluation) -> None:
        signals = evaluation.signals
        flags = signals.eligibility
        for field_name, attribute in _ELIGIBILITY_ATTRIBUTES:
            self.eligibility[field_name] += getattr(flags, attribute)
        for field_name, attribute in _VALUE_ATTRIBUTES:
            self.values[field_name] += getattr(signals, attribute)
        if flags.stockbee_liquidity:
            self.stockbee_symbols.append(symbol)
        if evaluation.qualifying_values:
            self.qualifying[symbol] = evaluation


_ELIGIBILITY_ATTRIBUTES = (
    ("advance_decline_eligible_count", "advance_decline"),
    ("stockbee_daily_eligible_count", "stockbee_daily"),
    ("stockbee_month_eligible_count", "stockbee_month"),
    ("stockbee_34day_eligible_count", "stockbee_34day"),
    ("stockbee_quarter_eligible_count", "stockbee_quarter"),
    ("t2108_eligible_count", "t2108"),
    ("high_low_52week_eligible_count", "high_low_52week"),
    ("atr_extension_eligible_count", "atr_extension"),
    ("sma200_eligible_count", "sma200"),
    ("off_high_eligible_count", "off_high"),
)
_VALUE_ATTRIBUTES = (
    *(
        (definition.aggregate_field, definition.signal_attribute)
        for definition in BREADTH_CONTRIBUTOR_SIGNALS.values()
    ),
    ("advancing_count", "advancing"),
    ("declining_count", "declining"),
    ("unchanged_count", "unchanged"),
    ("new_high_52week_count", "new_high_52week"),
    ("new_low_52week_count", "new_low_52week"),
    ("t2108_count", "t2108_above"),
    ("above_sma200_count", "above_sma200"),
    ("down_20pct_from_high_count", "down_20pct_from_high"),
    ("up_4pct_near_high_count", "up_4pct_near_high"),
)
_COUNTED_VALUE_FIELDS = tuple(field_name for field_name, _ in _VALUE_ATTRIBUTES)


class BreadthAccumulator:
    """Streaming breadth calculation over one ordered set of dates.

    Callers feed price histories in batches with ``add_prices`` and may drop
    each batch afterwards: every symbol is reduced to per-date counts at once,
    so peak memory is one batch of frames rather than the whole universe.
    Each symbol may be supplied at most once. The result is identical to
    evaluating every symbol in one call, because all aggregates are sums and
    every ordered output (contributors, signatures) is sorted at ``finish``.
    """

    def __init__(
        self,
        *,
        market: str,
        dates: tuple[date, ...],
        universes_by_date: Mapping[date, BreadthUniverseSnapshot],
        market_policy: BreadthMarketPolicy,
        policy: BreadthFormulaPolicy,
    ) -> None:
        market = market.strip().upper()
        if market_policy.market != market:
            raise ValueError(
                f"Breadth policy market {market_policy.market} "
                f"does not match request market {market}"
            )

        dates = tuple(dates)
        if dates != tuple(sorted(set(dates))):
            raise ValueError("Breadth calculation dates must be ordered and unique")

        currencies_by_symbol: dict[str, str] = {}
        memberships: dict[str, list[tuple[date, BreadthUniverseMember]]] = {}
        snapshots: dict[date, BreadthUniverseSnapshot] = {}
        for calculation_date in dates:
            snapshot = universes_by_date.get(calculation_date)
            if snapshot is None:
                raise ValueError(
                    f"Missing breadth universe for {calculation_date.isoformat()}"
                )
            if snapshot.calculation_date != calculation_date:
                raise ValueError("Breadth universe date does not match request date")
            snapshots[calculation_date] = snapshot
            for member in snapshot.members:
                prior_currency = currencies_by_symbol.setdefault(
                    member.symbol,
                    member.currency.upper(),
                )
                if prior_currency != member.currency.upper():
                    raise ValueError(
                        f"Currency changed within breadth range for {member.symbol}"
                    )
                memberships.setdefault(member.symbol, []).append(
                    (calculation_date, member)
                )

        self._market = market
        self._dates = dates
        self._snapshots = snapshots
        self._market_policy = market_policy
        self._policy = policy
        self._memberships = memberships
        self._supplied: set[str] = set()
        self._totals = {calculation_date: _DateTotals() for calculation_date in dates}

    def add_prices(self, prices_by_symbol: Mapping[str, pd.DataFrame]) -> None:
        """Evaluate one batch of symbols on every date they are members."""
        for symbol, prices in prices_by_symbol.items():
            memberships = self._memberships.get(symbol)
            if memberships is None:
                continue
            if symbol in self._supplied:
                raise ValueError(f"Breadth prices supplied twice for {symbol}")
            self._supplied.add(symbol)
            if prices is None or prices.empty:
                continue
            try:
                validate_price_frame(prices)
            except ValueError as exc:
                logger.warning(
                    "Skipping malformed breadth prices for %s: %s", symbol, exc
                )
                continue
            features = prepare_feature_frame(
                prices,
                atr_period=self._policy.atr_period,
            )
            for calculation_date, member in memberships:
                if not member.is_common_stock:
                    continue
                self._totals[calculation_date].add(
                    symbol,
                    evaluate_symbol_at(
                        features,
                        calculation_date,
                        self._policy,
                        self._market_policy,
                        stockbee_currency_matches=(
                            member.currency.upper() == self._market_policy.currency
                        ),
                    ),
                )

    def finish(
        self,
        *,
        dates: tuple[date, ...] | None = None,
        seed_counts: tuple[BreadthDailyCount, ...] = (),
        contributor_metadata_by_date: Mapping[
            date, Mapping[str, BreadthContributorMetadata]
        ]
        | None = None,
    ) -> BreadthEngineBatchResult:
        """Build results for ``dates`` (default: every accumulated date).

        A subset may be requested when some dates turn out to be unusable
        after all batches are seen; ratios are then computed over the subset
        only, exactly as if only those dates had been requested.
        """
        policy = self._policy
        if dates is None:
            dates = self._dates
        dates = tuple(dates)
        if dates != tuple(sorted(set(dates))) or not set(dates) <= set(self._dates):
            raise ValueError("Breadth result dates must be ordered accumulated dates")
        contributor_metadata_by_date = contributor_metadata_by_date or {}
        market = self._market

        partial_results: dict[date, BreadthDailyResult] = {}
        contributor_snapshots: dict[date, BreadthContributorSnapshotResult] = {}
        daily_counts: list[BreadthDailyCount] = []
        for calculation_date in dates:
            snapshot = self._snapshots[calculation_date]
            totals = self._totals[calculation_date]
            eligibility = BreadthEligibilityCounts(**totals.eligibility)
            for _, count_field, eligible_field in BREADTH_SHARE_FIELDS:
                count = totals.values[count_field]
                if not 0 <= count <= totals.eligibility[eligible_field]:
                    raise AssertionError(f"{count_field} exceeds its eligible denominator")
            values = BreadthIndicatorValues(
                **totals.values,
                **{
                    pct_field: share_pct(
                        totals.values[count_field], totals.eligibility[eligible_field]
                    )
                    for pct_field, count_field, eligible_field in BREADTH_SHARE_FIELDS
                },
            )

            if (
                values.advancing_count + values.declining_count + values.unchanged_count
                != eligibility.advance_decline_eligible_count
            ):
                raise AssertionError("Advance/decline counts do not reconcile")
            if values.up_4pct_near_high_count > values.stocks_up_4pct:
                raise AssertionError("Near-high 4% gainers exceed all 4% gainers")
            if policy.calculation_revision != CURRENT_BREADTH_CALCULATION_REVISION:
                raise AssertionError(
                    "Canonical breadth engine must produce the current revision"
                )

            result = BreadthDailyResult(
                market=market,
                calculation_date=calculation_date,
                values=values,
                eligibility=eligibility,
                broad_universe_count=len(snapshot.members),
                eligibility_signature=breadth_eligibility_signature(
                    member.symbol for member in snapshot.members
                ),
                stockbee_eligibility_signature=hash_point_in_time_universe_symbols(
                    tuple(sorted(totals.stockbee_symbols))
                ),
                calculation_revision=policy.calculation_revision,
            )
            partial_results[calculation_date] = result
            metadata_by_symbol = contributor_metadata_by_date.get(
                calculation_date,
                {},
            )
            contributors: list[BreadthContributor] = []
            for symbol in sorted(totals.qualifying):
                evaluation = totals.qualifying[symbol]
                metadata = metadata_by_symbol.get(
                    symbol,
                    BreadthContributorMetadata(),
                )
                company_name = (
                    str(metadata.company_name).strip()
                    if metadata.company_name is not None
                    and str(metadata.company_name).strip()
                    else None
                )
                group = str(metadata.ibd_industry_group or "").strip() or NO_GROUP_LABEL
                contributors.append(
                    BreadthContributor(
                        symbol=symbol,
                        company_name=company_name,
                        ibd_industry_group=group,
                        daily_change_pct=evaluation.daily_change_pct,
                        signals=MappingProxyType(dict(evaluation.qualifying_values)),
                    )
                )
            contributor_snapshot = BreadthContributorSnapshotResult(
                market=market,
                calculation_date=calculation_date,
                calculation_revision=policy.calculation_revision,
                schema_id=CONTRIBUTOR_SCHEMA_ID,
                contributors=tuple(contributors),
            )
            reconcile_contributor_counts(contributor_snapshot, result)
            contributor_snapshots[calculation_date] = contributor_snapshot
            daily_counts.append(
                BreadthDailyCount(
                    date=calculation_date,
                    stocks_up_4pct=values.stocks_up_4pct,
                    stocks_down_4pct=values.stocks_down_4pct,
                    market=result.market,
                    calculation_revision=result.calculation_revision,
                )
            )

        ratios_by_date = calculate_inclusive_ratios(
            daily_counts,
            seed_counts,
            market=market,
            calculation_revision=policy.calculation_revision,
        )
        daily_results = {
            calculation_date: replace(
                result,
                values=replace(
                    result.values,
                    ratio_5day=ratios_by_date[calculation_date].ratio_5day,
                    ratio_10day=ratios_by_date[calculation_date].ratio_10day,
                ),
            )
            for calculation_date, result in partial_results.items()
        }
        return BreadthEngineBatchResult(
            daily_results=MappingProxyType(daily_results),
            contributor_snapshots=MappingProxyType(contributor_snapshots),
        )
