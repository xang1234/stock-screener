from __future__ import annotations

import threading
import time
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.domain.options_analytics.history import HistoricalObservation
from app.domain.options_analytics.models import (
    CandidateKind,
    ChainObservation,
    DividendSource,
    NormalizedOptionContract,
    ObservationState,
    OptionCandidate,
    OptionCandidateInput,
    OptionSide,
)
from app.domain.options_analytics.ports import (
    LastCurrentMembership,
    OptionsProviderError,
    TransientOptionsProviderError,
)
from app.infra.providers.yahoo_options import ThrottledOptionsProviderError
from app.use_cases.options_analytics.analysis_models import OptionsMetricValues
from app.use_cases.options_analytics.refresh import (
    RefreshOptionsAnalyticsCommand,
    RefreshOptionsAnalyticsUseCase,
)


def _candidate(symbol: str, kind=CandidateKind.CURRENT) -> OptionCandidate:
    return OptionCandidate(
        symbol=symbol,
        kind=kind,
        composite_score=90,
        daily_dollar_volume=200_000_000,
        spot_price=100,
        dividend_yield=0.01,
        price_closes=tuple(float(value) for value in range(100, 121)),
        candidate_rank=1 if kind is CandidateKind.CURRENT else None,
    )


def _observation(symbol: str) -> ChainObservation:
    def contract(side, iv, strike):
        return NormalizedOptionContract(
            side=side,
            strike=strike,
            bid=1,
            ask=2,
            last_price=1.5,
            volume=150,
            open_interest=200,
            implied_volatility=iv,
            last_trade_at=None,
            contract_size="REGULAR",
            multiplier=100,
        )

    return ChainObservation(
        symbol=symbol,
        expiration=date(2026, 9, 18),
        source_spot_price=100,
        fetched_at=datetime(2026, 9, 4, 1, tzinfo=timezone.utc),
        contracts=tuple(
            contract(side, iv, strike)
            for side, iv in ((OptionSide.CALL, 0.25), (OptionSide.PUT, 0.30))
            for strike in (90, 95, 100, 105, 110)
        ),
    )


class _Calendar:
    def is_session(self, value):
        return value.weekday() < 5

    def sessions_ending_on(self, value, count):
        sessions = []
        cursor = value
        while len(sessions) < count:
            if cursor.weekday() < 5:
                sessions.append(cursor)
            cursor -= timedelta(days=1)
        return tuple(reversed(sessions))


class _Provider:
    def __init__(self, failures=None, delay=0):
        self.failures = dict(failures or {})
        self.delay = delay
        self.fetch_counts = {}
        self.risk_free_calls = 0
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def risk_free_rate(self, _as_of):
        self.risk_free_calls += 1
        return 0.04

    def list_expirations(self, _symbol):
        return (date(2026, 9, 18),)

    def fetch_chain(self, symbol, _expiration, *, source_spot_price):
        self.fetch_counts[symbol] = self.fetch_counts.get(symbol, 0) + 1
        remaining = self.failures.get(symbol, 0)
        if remaining:
            self.failures[symbol] = remaining - 1
            raise TransientOptionsProviderError("temporary")
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        if self.delay:
            time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        assert source_spot_price == 100
        return _observation(symbol)


class _Repository:
    def __init__(self, memberships=None, history=()):
        self.run = SimpleNamespace(
            id=17,
            market="US",
            calculation_version="v1",
            status="staged",
            source_feature_run_id=33,
            expected_count=0,
            completed_count=0,
            core_valid_current_count=0,
            failed_count=0,
            retried_count=0,
            coverage=0.0,
            warnings_json=[],
            risk_free_rate=None,
            assumptions_json=None,
        )
        self.memberships = dict(memberships or {})
        self.history = tuple(history)
        self.staged = {}
        self.saved = {}
        self.unavailable = {}
        self.published = None
        self.failed_quality = None
        self.cancelled = False
        self.prune_calls = 0
        self.persistence_threads = []
        self.history_threads = []
        self.activity_ranks = None
        self.run_assumptions = None
        self.commit_count = 0
        self.events = []
        self.start_kwargs = None

    def start_or_reuse(self, **kwargs):
        self.start_kwargs = kwargs
        return self.run

    def last_current_memberships(self, _market, _calculation_version):
        return self.memberships

    def stage_candidates(self, _run_id, candidates):
        self.staged = {candidate.symbol: candidate for candidate in candidates}

    def incomplete_symbols(self, _run_id):
        terminal = set(self.saved) | set(self.unavailable)
        return tuple(symbol for symbol in self.staged if symbol not in terminal)

    def save_item_result(self, _run_id, symbol, **values):
        self.persistence_threads.append(threading.get_ident())
        self.saved[symbol] = values

    def save_unavailable(self, _run_id, symbol, **values):
        self.persistence_threads.append(threading.get_ident())
        self.unavailable[symbol] = values

    def save_analysis(self, run_id, analysis):
        values = {
            "evidence": analysis.evidence,
            "assumptions": analysis.assumptions,
            "warnings": analysis.warnings,
            "reason_codes": analysis.reason_codes,
            "retry_count": analysis.retry_count,
        }
        if hasattr(analysis, "observation"):
            values["historical_metrics"] = getattr(analysis, "historical_metrics", None)
            self.save_item_result(
                run_id,
                analysis.candidate.symbol,
                observation=analysis.observation,
                core_valid=analysis.core_valid,
                metric_values=analysis.metric_values,
                strike_points=analysis.strike_points,
                history_readiness=analysis.history_readiness,
                **values,
            )
        else:
            self.save_unavailable(run_id, analysis.candidate.symbol, **values)

    def items_for_run(self, _run_id):
        rows = []
        for symbol, candidate in self.staged.items():
            if symbol in self.saved:
                metric_values = self.saved[symbol].get(
                    "metric_values"
                ) or OptionsMetricValues(
                    max_pain=100,
                    atm_iv=0.25,
                    activity_intensity=0.5,
                )
                rows.append(
                    SimpleNamespace(
                        security_symbol=symbol,
                        candidate_kind=candidate.kind.value,
                        observation_state="available",
                        core_valid=self.saved[symbol].get("core_valid", True),
                        max_pain=metric_values.max_pain,
                        atm_iv=metric_values.atm_iv,
                        activity_intensity=metric_values.activity_intensity,
                        retry_count=self.saved[symbol].get("retry_count", 0),
                    )
                )
            elif symbol in self.unavailable:
                rows.append(
                    SimpleNamespace(
                        security_symbol=symbol,
                        candidate_kind=candidate.kind.value,
                        observation_state="unavailable",
                        core_valid=False,
                        max_pain=None,
                        atm_iv=None,
                        activity_intensity=None,
                        retry_count=self.unavailable[symbol].get("retry_count", 0),
                    )
                )
        return tuple(rows)

    def analysis_history(self, _symbol, **_kwargs):
        self.history_threads.append(threading.get_ident())
        return self.history

    def save_activity_ranks(self, _run_id, ranks):
        self.activity_ranks = ranks

    def save_run_assumptions(self, _run_id, *, risk_free_rate, assumptions):
        self.run_assumptions = (risk_free_rate, assumptions)
        self.run.risk_free_rate = risk_free_rate
        self.run.assumptions_json = dict(assumptions)

    def publish(self, _run_id, summary):
        self.events.append("publish")
        self.published = summary

    def mark_failed_quality(self, _run_id, *, reason_codes):
        self.failed_quality = tuple(reason_codes)

    def cancel(self, _run_id):
        self.cancelled = True
        self.run.status = "cancelled"

    def prune(self, **_kwargs):
        self.events.append("prune")
        self.prune_calls += 1

    def commit(self):
        self.events.append("commit")
        self.commit_count += 1


class _Source:
    def __init__(self, candidates, continuity_inputs=None):
        inputs = tuple(
            OptionCandidateInput(
                symbol=candidate.symbol,
                composite_score=candidate.composite_score,
                daily_dollar_volume=candidate.daily_dollar_volume,
                spot_price=candidate.spot_price,
                dividend_yield=candidate.dividend_yield,
                price_closes=candidate.price_closes,
            )
            for candidate in candidates
        )
        self.snapshot = SimpleNamespace(
            source_feature_run_id=33,
            as_of_date=date(2026, 9, 4),
            top_candidate_inputs=inputs,
            leader_inputs=(),
        )
        self.continuity_inputs = dict(continuity_inputs or {})

    def read(self, source_run_id):
        assert source_run_id == 33
        return self.snapshot

    def read_continuity_inputs(self, symbols, as_of_date):
        assert as_of_date == self.snapshot.as_of_date
        return {
            symbol: self.continuity_inputs[symbol]
            for symbol in symbols
            if symbol in self.continuity_inputs
        }


class _Cancellation:
    def __init__(self, cancelled=False):
        self.cancelled = cancelled

    def is_cancelled(self):
        return self.cancelled


class _CancelAfterChecks:
    def __init__(self, allowed_checks: int) -> None:
        self._allowed_checks = allowed_checks
        self._checks = 0

    def is_cancelled(self) -> bool:
        self._checks += 1
        return self._checks > self._allowed_checks


def _use_case(
    candidates,
    *,
    repo=None,
    provider=None,
    cancellation=None,
    continuity_inputs=None,
    throttle_backoff=lambda _attempt: None,
):
    repositories = repo or _Repository()
    return RefreshOptionsAnalyticsUseCase(
        candidate_source=_Source(candidates, continuity_inputs),
        run_writer=repositories,
        published_reader=repositories,
        retention=repositories,
        provider=provider or _Provider(),
        calendar=_Calendar(),
        cancellation=cancellation or _Cancellation(),
        calculation_version="v1",
        schema_version="v1",
        max_workers=2,
        throttle_backoff=throttle_backoff,
    )


def test_exactly_90_percent_current_coverage_publishes_and_prunes() -> None:
    candidates = [_candidate(f"S{index}") for index in range(10)]
    repo = _Repository(
        memberships={
            "OLD": LastCurrentMembership(
                symbol="OLD",
                as_of_date=date(2026, 9, 3),
                prior_best_rank=4,
                dividend_yield=None,
                dividend_source=None,
            )
        }
    )
    provider = _Provider(failures={"S9": 3, "OLD": 3})
    continuity_inputs = {"OLD": OptionCandidateInput("OLD", 80, 200_000_000, 100)}

    result = _use_case(
        candidates,
        repo=repo,
        provider=provider,
        continuity_inputs=continuity_inputs,
    ).execute(RefreshOptionsAnalyticsCommand(source_run_id=33))

    assert repo.published.coverage == 0.9
    assert repo.published.core_valid_current_count == 9
    assert repo.failed_quality is None
    assert repo.prune_calls == 1
    publish_index = repo.events.index("publish")
    prune_index = repo.events.index("prune")
    assert publish_index < prune_index
    assert "OLD" in repo.unavailable
    assert result["status"] == "published"
    assert result["coverage"] == 0.9
    assert provider.risk_free_calls == 1
    assert repo.run_assumptions == (
        0.04,
        {"risk_free_source": "Yahoo ^IRX close on or before source date"},
    )


def test_retention_failure_does_not_fail_an_already_published_refresh() -> None:
    repo = _Repository()

    def fail_prune(**_kwargs):
        repo.events.append("prune")
        raise RuntimeError("temporary retention failure")

    repo.prune = fail_prune

    result = _use_case([_candidate("AAPL")], repo=repo).execute(
        RefreshOptionsAnalyticsCommand(source_run_id=33)
    )

    assert repo.published is not None
    assert result["status"] == "published"
    assert repo.events == ["publish", "prune"]


def test_below_90_percent_keeps_pointer_unpublished_and_skips_retention() -> None:
    candidates = [_candidate(f"S{index}") for index in range(9)]
    repo = _Repository()
    provider = _Provider(failures={"S7": 3, "S8": 3})

    result = _use_case(candidates, repo=repo, provider=provider).execute(
        RefreshOptionsAnalyticsCommand(source_run_id=33)
    )

    assert repo.published is None
    assert repo.failed_quality == ("insufficient_core_coverage",)
    assert repo.prune_calls == 0
    assert result["status"] == "failed_quality"


def test_empty_current_cohort_fails_quality_even_with_continuity() -> None:
    repo = _Repository(
        memberships={
            "OLD": LastCurrentMembership(
                symbol="OLD",
                as_of_date=date(2026, 9, 3),
                prior_best_rank=4,
                dividend_yield=None,
                dividend_source=None,
            )
        }
    )

    result = _use_case(
        [],
        repo=repo,
        continuity_inputs={"OLD": OptionCandidateInput("OLD", 80, 200_000_000, 100)},
    ).execute(RefreshOptionsAnalyticsCommand(source_run_id=33))

    assert repo.failed_quality == ("empty_current_cohort",)
    assert result["coverage"] == 0.0


def test_transient_symbol_retries_three_times_but_saves_one_observation() -> None:
    provider = _Provider(failures={"AAPL": 2})
    repo = _Repository()

    _use_case([_candidate("AAPL")], repo=repo, provider=provider).execute(
        RefreshOptionsAnalyticsCommand(source_run_id=33)
    )

    assert provider.fetch_counts["AAPL"] == 3
    assert list(repo.saved) == ["AAPL"]
    assert repo.saved["AAPL"]["retry_count"] == 2
    assert repo.saved["AAPL"]["metric_values"].call_open_interest == 1000
    assert repo.saved["AAPL"]["metric_values"].put_open_interest == 1000
    strike = repo.saved["AAPL"]["strike_points"][0]
    assert strike.estimated_call_gex > 0
    assert strike.estimated_put_gex < 0


def test_yahoo_throttling_applies_backoff_before_each_retry() -> None:
    class ThrottledProvider(_Provider):
        def fetch_chain(self, symbol, _expiration, *, source_spot_price):
            self.fetch_counts[symbol] = self.fetch_counts.get(symbol, 0) + 1
            if self.fetch_counts[symbol] < 3:
                raise ThrottledOptionsProviderError("429")
            return _observation(symbol)

    provider = ThrottledProvider()
    backoff_attempts = []

    _use_case(
        [_candidate("AAPL")],
        provider=provider,
        throttle_backoff=backoff_attempts.append,
    ).execute(RefreshOptionsAnalyticsCommand(source_run_id=33))

    assert provider.fetch_counts == {"AAPL": 3}
    assert backoff_attempts == [1, 2]


def test_non_transient_provider_error_is_not_retried() -> None:
    class InvalidProvider(_Provider):
        def fetch_chain(self, symbol, _expiration, *, source_spot_price):
            self.fetch_counts[symbol] = self.fetch_counts.get(symbol, 0) + 1
            raise OptionsProviderError("invalid chain")

    provider = InvalidProvider()
    repo = _Repository()

    _use_case([_candidate("AAPL")], repo=repo, provider=provider).execute(
        RefreshOptionsAnalyticsCommand(source_run_id=33)
    )

    assert provider.fetch_counts == {"AAPL": 1}
    assert repo.unavailable["AAPL"]["reason_codes"] == ("provider_unavailable",)
    assert repo.unavailable["AAPL"]["evidence"]["quality"] == {
        "source_spot_price": 100,
        "provider_spot_price": None,
        "spot_disagreement_ratio": None,
        "latest_contract_trade_at": None,
        "days_to_expiration": None,
        "normalized_call_count": 0,
        "normalized_put_count": 0,
        "distinct_strike_count": 0,
        "open_interest_coverage": 0.0,
        "iv_coverage": 0.0,
        "volume_coverage": 0.0,
        "two_sided_quote_coverage": 0.0,
    }
    assert repo.unavailable["AAPL"]["assumptions"] == {
        "dividend_yield": 0.01,
        "dividend_source": "pinned_feature_run",
    }


def test_missing_source_spot_is_unavailable_without_calling_provider() -> None:
    provider = _Provider()
    repo = _Repository()

    _use_case(
        [replace(_candidate("AAPL"), spot_price=None)],
        repo=repo,
        provider=provider,
    ).execute(RefreshOptionsAnalyticsCommand(source_run_id=33))

    assert provider.fetch_counts == {}
    assert repo.unavailable["AAPL"]["reason_codes"] == ("source_spot_unavailable",)


def test_successful_but_thin_chain_is_saved_as_insufficient_core_quality() -> None:
    class ThinProvider(_Provider):
        def fetch_chain(self, symbol, _expiration, *, source_spot_price):
            full = _observation(symbol)
            return replace(full, contracts=full.contracts[:2])

    repo = _Repository()

    result = _use_case(
        [_candidate("AAPL")], repo=repo, provider=ThinProvider()
    ).execute(RefreshOptionsAnalyticsCommand(source_run_id=33))

    assert result["status"] == "failed_quality"
    assert repo.saved["AAPL"]["core_valid"] is False
    assert "insufficient_core_quality" in repo.saved["AAPL"]["reason_codes"]


def test_structurally_valid_chain_without_atm_iv_remains_core_valid() -> None:
    class MissingIvProvider(_Provider):
        def fetch_chain(self, symbol, _expiration, *, source_spot_price):
            full = _observation(symbol)
            return replace(
                full,
                contracts=tuple(
                    replace(contract, implied_volatility=None)
                    for contract in full.contracts
                ),
            )

    repo = _Repository()

    result = _use_case(
        [_candidate("AAPL")], repo=repo, provider=MissingIvProvider()
    ).execute(RefreshOptionsAnalyticsCommand(source_run_id=33))

    assert result["status"] == "published"
    assert repo.saved["AAPL"]["core_valid"] is True
    assert repo.saved["AAPL"]["metric_values"].atm_iv is None


def test_invalid_observation_does_not_increment_compatible_history() -> None:
    class ThinProvider(_Provider):
        def fetch_chain(self, symbol, _expiration, *, source_spot_price):
            full = _observation(symbol)
            return replace(full, contracts=full.contracts[:2])

    repo = _Repository()

    _use_case([_candidate("AAPL")], repo=repo, provider=ThinProvider()).execute(
        RefreshOptionsAnalyticsCommand(source_run_id=33)
    )

    readiness = repo.saved["AAPL"]["history_readiness"]
    assert readiness.lifetime_observation_count == 0
    assert readiness.short_observation_count == 0


def test_risk_free_failure_only_makes_model_dependent_metrics_unavailable() -> None:
    class MissingRateProvider(_Provider):
        def risk_free_rate(self, _as_of):
            self.risk_free_calls += 1
            raise OptionsProviderError("IRX unavailable")

    repo = _Repository()

    result = _use_case(
        [_candidate("AAPL")], repo=repo, provider=MissingRateProvider()
    ).execute(RefreshOptionsAnalyticsCommand(source_run_id=33))

    assert result["status"] == "published"
    assert repo.saved["AAPL"]["metric_values"].net_gex is None
    assert repo.saved["AAPL"]["metric_values"].atm_iv == 0.275
    assert repo.run_assumptions[0] is None


def test_resumed_run_reuses_persisted_risk_free_assumption() -> None:
    repo = _Repository()
    repo.run.status = "failed_quality"
    repo.run.risk_free_rate = 0.031
    repo.run.assumptions_json = {"risk_free_source": "persisted ^IRX close"}
    repo.saved["AAPL"] = {"core_valid": True}
    provider = _Provider()

    _use_case(
        [_candidate("AAPL"), _candidate("MSFT")],
        repo=repo,
        provider=provider,
    ).execute(RefreshOptionsAnalyticsCommand(source_run_id=33))

    assert provider.risk_free_calls == 0
    assert repo.run_assumptions is None
    assert repo.saved["MSFT"]["assumptions"]["risk_free_rate"] == 0.031


def test_fetches_at_most_two_concurrently_but_persists_on_caller_thread() -> None:
    provider = _Provider(delay=0.02)
    repo = _Repository()
    caller_thread = threading.get_ident()

    _use_case(
        [_candidate("AAPL"), _candidate("MSFT"), _candidate("NVDA")],
        repo=repo,
        provider=provider,
    ).execute(RefreshOptionsAnalyticsCommand(source_run_id=33))

    assert provider.max_active == 2
    assert repo.history_threads == [caller_thread, caller_thread, caller_thread]
    assert repo.persistence_threads == [caller_thread, caller_thread, caller_thread]


def test_resume_skips_successful_items_and_cancellation_persists_terminal_state() -> (
    None
):
    repo = _Repository()
    repo.staged = {"AAPL": _candidate("AAPL")}
    repo.saved = {"AAPL": {}}
    provider = _Provider()

    result = _use_case(
        [_candidate("AAPL")],
        repo=repo,
        provider=provider,
        cancellation=_Cancellation(cancelled=True),
    ).execute(RefreshOptionsAnalyticsCommand(source_run_id=33))

    assert provider.fetch_counts == {}
    assert repo.published is None
    assert repo.failed_quality is None
    assert repo.cancelled is True
    assert repo.run.status == "cancelled"
    assert result["status"] == "cancelled"


def test_cancellation_during_collection_persists_completed_work_then_cancels() -> None:
    repo = _Repository()

    result = _use_case(
        [_candidate("AAPL")],
        repo=repo,
        cancellation=_CancelAfterChecks(allowed_checks=1),
    ).execute(RefreshOptionsAnalyticsCommand(source_run_id=33))

    assert tuple(repo.saved) == ("AAPL",)
    assert repo.published is None
    assert repo.failed_quality is None
    assert repo.cancelled is True
    assert result["status"] == "cancelled"


def test_completed_symbol_is_persisted_before_later_worker_exception(monkeypatch) -> None:
    repo = _Repository()
    use_case = _use_case([_candidate("AAPL"), _candidate("MSFT")], repo=repo)
    original_analyze = use_case._analyzer.analyze

    def analyze(candidate, context):
        if candidate.symbol == "MSFT":
            raise RuntimeError("unexpected candidate failure")
        return original_analyze(candidate, context)

    monkeypatch.setattr(use_case._analyzer, "analyze", analyze)
    use_case._max_workers = 1

    with pytest.raises(RuntimeError, match="unexpected candidate failure"):
        use_case.execute(RefreshOptionsAnalyticsCommand(source_run_id=33))

    assert tuple(repo.saved) == ("AAPL",)


def test_completed_workers_are_drained_when_another_worker_fails_first(
    monkeypatch,
) -> None:
    repo = _Repository()
    use_case = _use_case([_candidate("AAPL"), _candidate("MSFT")], repo=repo)
    failure_started = threading.Event()
    original_analyze = use_case._analyzer.analyze

    def analyze(candidate, context):
        if candidate.symbol == "MSFT":
            failure_started.set()
            raise RuntimeError("unexpected candidate failure")
        assert failure_started.wait(timeout=1)
        return original_analyze(candidate, context)

    monkeypatch.setattr(use_case._analyzer, "analyze", analyze)

    with pytest.raises(RuntimeError, match="unexpected candidate failure"):
        use_case.execute(RefreshOptionsAnalyticsCommand(source_run_id=33))

    assert tuple(repo.saved) == ("AAPL",)


def test_resume_counts_previously_saved_items_toward_publication() -> None:
    repo = _Repository()
    repo.saved = {"AAPL": {}}
    provider = _Provider()

    result = _use_case(
        [_candidate("AAPL"), _candidate("MSFT")], repo=repo, provider=provider
    ).execute(RefreshOptionsAnalyticsCommand(source_run_id=33))

    assert provider.fetch_counts == {"MSFT": 1}
    assert result["completed_count"] == 2
    assert result["core_valid_current_count"] == 2
    assert result["status"] == "published"


def test_history_readiness_uses_compatible_observations_with_real_gaps() -> None:
    sessions = _Calendar().sessions_ending_on(date(2026, 9, 4), 30)
    history = tuple(
        HistoricalObservation(
            session=session, calculation_version="v1", state=ObservationState.AVAILABLE
        )
        for session in sessions[::2]
    )
    repo = _Repository(history=history)

    _use_case([_candidate("AAPL")], repo=repo).execute(
        RefreshOptionsAnalyticsCommand(source_run_id=33)
    )

    readiness = repo.saved["AAPL"]["history_readiness"]
    assert readiness.lifetime_observation_count == 16
    assert readiness.short_observation_count in (3, 4)
    assert readiness.reason_codes == ("building_history",)


def test_ready_history_calculates_iv_context_and_five_observation_changes() -> None:
    sessions = _Calendar().sessions_ending_on(date(2026, 9, 4), 30)
    history = tuple(
        HistoricalObservation(
            session=session,
            calculation_version="v1",
            state=ObservationState.AVAILABLE,
            max_pain=90 + index,
            net_gex=1_000 + index,
            gamma_flip=95 + index,
            atm_iv=0.10 + index / 100,
            skew_25_delta=-0.05 + index / 1000,
            realized_volatility=0.08 + index / 100,
            vrp=0.02 + index / 1000,
            activity_intensity=0.20 + index / 100,
        )
        for index, session in enumerate(sessions[-20:-1])
    )
    repo = _Repository(history=history)

    _use_case([_candidate("AAPL")], repo=repo).execute(
        RefreshOptionsAnalyticsCommand(source_run_id=33)
    )

    metrics = repo.saved["AAPL"]["historical_metrics"]
    assert metrics.iv_percentile.available is True
    assert metrics.iv_rank.available is True
    assert metrics.max_pain_change_5.available is True


def test_continuity_is_derived_from_last_current_membership_and_expires_after_five_sessions() -> (
    None
):
    repo = _Repository(
        memberships={
            "RECENT": LastCurrentMembership(
                symbol="RECENT",
                as_of_date=date(2026, 8, 28),
                prior_best_rank=2,
                dividend_yield=0.0,
                dividend_source=DividendSource.ZERO_ASSUMPTION,
            ),
            "EXPIRED": LastCurrentMembership(
                symbol="EXPIRED",
                as_of_date=date(2026, 8, 27),
                prior_best_rank=1,
                dividend_yield=None,
                dividend_source=None,
            ),
        }
    )
    inputs = {
        symbol: OptionCandidateInput(symbol, 80, 200_000_000, 100)
        for symbol in ("RECENT", "EXPIRED")
    }

    _use_case([_candidate("AAPL")], repo=repo, continuity_inputs=inputs).execute(
        RefreshOptionsAnalyticsCommand(source_run_id=33)
    )

    assert repo.staged["RECENT"].kind is CandidateKind.CONTINUITY
    assert repo.staged["RECENT"].sessions_since_current == 5
    assert repo.staged["RECENT"].dividend_yield == 0.0
    assert repo.staged["RECENT"].dividend_source == "zero_assumption"
    assert repo.saved["RECENT"]["assumptions"]["dividend_source"] == "zero_assumption"
    assert "zero_dividend_assumption" in repo.saved["RECENT"]["warnings"]
    assert "EXPIRED" not in repo.staged
    assert repo.activity_ranks == {"AAPL": 1}


def test_input_signature_changes_with_calculation_or_schema_version() -> None:
    cohort = (_candidate("AAPL"),)

    first = RefreshOptionsAnalyticsUseCase._input_signature(
        33, cohort, calculation_version="calc-v1", schema_version="schema-v1"
    )
    changed_calculation = RefreshOptionsAnalyticsUseCase._input_signature(
        33, cohort, calculation_version="calc-v2", schema_version="schema-v1"
    )
    changed_schema = RefreshOptionsAnalyticsUseCase._input_signature(
        33, cohort, calculation_version="calc-v1", schema_version="schema-v2"
    )

    assert len({first, changed_calculation, changed_schema}) == 3


def test_quality_evidence_keeps_provider_spot_and_stale_trade_warning() -> None:
    class DisagreementProvider(_Provider):
        def fetch_chain(self, symbol, _expiration, *, source_spot_price):
            full = _observation(symbol)
            stale_time = datetime(2026, 8, 28, 20, tzinfo=timezone.utc)
            return replace(
                full,
                provider_spot_price=103,
                contracts=tuple(
                    replace(contract, last_trade_at=stale_time)
                    for contract in full.contracts
                ),
            )

    repo = _Repository()

    _use_case([_candidate("AAPL")], repo=repo, provider=DisagreementProvider()).execute(
        RefreshOptionsAnalyticsCommand(source_run_id=33)
    )

    saved = repo.saved["AAPL"]
    assert saved["evidence"]["quality"]["provider_spot_price"] == 103
    assert saved["evidence"]["quality"]["spot_disagreement_ratio"] == 0.03
    assert saved["evidence"]["quality"]["days_to_expiration"] == 14
    assert saved["evidence"]["quality"]["latest_contract_trade_at"].startswith(
        "2026-08-28"
    )
    assert saved["evidence"]["quality"]["normalized_call_count"] == 5
    assert saved["evidence"]["quality"]["normalized_put_count"] == 5
    assert saved["evidence"]["quality"]["distinct_strike_count"] == 5
    assert saved["evidence"]["quality"]["open_interest_coverage"] == 1.0
    assert saved["evidence"]["quality"]["iv_coverage"] == 1.0
    assert saved["evidence"]["quality"]["volume_coverage"] == 1.0
    assert saved["evidence"]["quality"]["two_sided_quote_coverage"] == 1.0
    assert set(saved["warnings"]) == {
        "provider_spot_disagreement",
        "stale_contract_trades",
    }


def test_quality_evidence_keeps_explicit_null_when_provider_spot_is_missing() -> None:
    repo = _Repository()

    _use_case([_candidate("AAPL")], repo=repo).execute(
        RefreshOptionsAnalyticsCommand(source_run_id=33)
    )

    quality = repo.saved["AAPL"]["evidence"]["quality"]
    assert quality["provider_spot_price"] is None
    assert quality["spot_disagreement_ratio"] is None


def test_missing_or_invalid_dividend_uses_disclosed_zero_assumption() -> None:
    repo = _Repository()

    _use_case(
        [replace(_candidate("AAPL"), dividend_yield=float("nan"))],
        repo=repo,
    ).execute(RefreshOptionsAnalyticsCommand(source_run_id=33))

    saved = repo.saved["AAPL"]
    assert saved["assumptions"]["dividend_yield"] == 0.0
    assert saved["assumptions"]["dividend_source"] == "zero_assumption"
    assert "zero_dividend_assumption" in saved["warnings"]


def test_unavailable_row_keeps_zero_dividend_warning() -> None:
    repo = _Repository()
    candidate = replace(
        _candidate("AAPL"),
        spot_price=None,
        dividend_yield=float("nan"),
    )

    _use_case([candidate], repo=repo).execute(
        RefreshOptionsAnalyticsCommand(source_run_id=33)
    )

    assert repo.unavailable["AAPL"]["warnings"] == ("zero_dividend_assumption",)


def test_repeated_delivery_of_published_run_returns_without_mutation() -> None:
    repo = _Repository()
    repo.run = SimpleNamespace(
        id=17,
        market="US",
        calculation_version="v1",
        status="published",
        source_feature_run_id=33,
        expected_count=1,
        completed_count=1,
        core_valid_current_count=1,
        failed_count=0,
        retried_count=0,
        coverage=1.0,
        warnings_json=[],
    )
    provider = _Provider()

    result = _use_case([_candidate("AAPL")], repo=repo, provider=provider).execute(
        RefreshOptionsAnalyticsCommand(source_run_id=33)
    )

    assert result == {
        "run_id": 17,
        "source_run_id": 33,
        "status": "published",
        "expected_count": 1,
        "completed_count": 1,
        "core_valid_current_count": 1,
        "failed_count": 0,
        "retried_count": 0,
        "coverage": 1.0,
        "reason_codes": [],
    }
    assert repo.staged == {}
    assert repo.run_assumptions is None
    assert provider.risk_free_calls == 0
