from __future__ import annotations

from datetime import date

from app.domain.options_analytics.models import OptionCandidateInput
from app.domain.options_analytics.ports import (
    CandidateSourceSnapshot,
    OptionsCandidateSource,
)


class CompleteCandidateSource:
    def read(self, source_feature_run_id: int) -> CandidateSourceSnapshot:
        return CandidateSourceSnapshot(
            source_feature_run_id=source_feature_run_id,
            as_of_date=date(2026, 9, 4),
            top_candidate_inputs=(),
            leader_inputs=(),
        )

    def read_continuity_inputs(
        self,
        symbols: tuple[str, ...],
        as_of_date: date,
    ) -> dict[str, OptionCandidateInput]:
        del symbols, as_of_date
        return {}


class IncompleteCandidateSource:
    def read(self, source_feature_run_id: int) -> CandidateSourceSnapshot:
        raise AssertionError(source_feature_run_id)


def test_candidate_source_protocol_describes_both_real_operations() -> None:
    assert isinstance(CompleteCandidateSource(), OptionsCandidateSource)
    assert not isinstance(IncompleteCandidateSource(), OptionsCandidateSource)
