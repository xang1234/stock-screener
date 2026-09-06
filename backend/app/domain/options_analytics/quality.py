"""Publication quality gate for Options Analytics Runs."""

from __future__ import annotations

import math
from collections.abc import Iterable

from .models import (
    CandidateKind,
    ChainObservation,
    OptionCandidate,
    OptionSide,
    PublicationDecision,
)

MIN_CURRENT_COVERAGE = 0.90


def has_core_chain_coverage(observation: ChainObservation) -> bool:
    if (
        not math.isfinite(float(observation.source_spot_price))
        or observation.source_spot_price <= 0
    ):
        return False
    by_side = {
        side: [contract for contract in observation.contracts if contract.side is side]
        for side in (OptionSide.CALL, OptionSide.PUT)
    }
    if any(len(contracts) < 5 for contracts in by_side.values()):
        return False
    for contracts in by_side.values():
        total_open_interest = sum(
            contract.open_interest
            for contract in contracts
            if contract.open_interest is not None
            and math.isfinite(float(contract.open_interest))
            and contract.open_interest >= 0
        )
        if total_open_interest <= 0:
            return False
    usable_strikes = {
        float(contract.strike)
        for contract in observation.contracts
        if math.isfinite(float(contract.strike))
        and contract.strike > 0
        and contract.open_interest is not None
        and math.isfinite(float(contract.open_interest))
        and contract.open_interest >= 0
    }
    return len(usable_strikes) >= 3


def evaluate_publication(
    cohort: Iterable[OptionCandidate],
    *,
    core_valid_symbols: set[str],
) -> PublicationDecision:
    current = [row for row in cohort if row.kind is CandidateKind.CURRENT]
    current_symbols = {row.symbol for row in current}
    valid_symbols = {symbol.strip().upper() for symbol in core_valid_symbols}
    valid_count = len(current_symbols & valid_symbols)
    if not current:
        return PublicationDecision(
            publish=False,
            current_count=0,
            core_valid_current_count=0,
            coverage=0.0,
            reason_codes=("empty_current_cohort",),
        )
    coverage = valid_count / len(current)
    publish = coverage >= MIN_CURRENT_COVERAGE
    return PublicationDecision(
        publish=publish,
        current_count=len(current),
        core_valid_current_count=valid_count,
        coverage=coverage,
        reason_codes=() if publish else ("insufficient_core_coverage",),
    )
