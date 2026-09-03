"""Publication quality gate for Options Analytics Runs."""

from __future__ import annotations

from collections.abc import Iterable, Set

from .models import CandidateKind, OptionCandidate, PublicationDecision

MIN_CURRENT_COVERAGE = 0.90


def evaluate_publication(
    cohort: Iterable[OptionCandidate],
    *,
    core_valid_symbols: Set[str],
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

