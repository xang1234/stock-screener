"""Independent SQLite connections exercise the persisted installation ledger."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from threading import Barrier

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.app_settings import AppSetting
from app.infra.db.models.social_signals import SocialSourceRegistry

NOW = datetime(2026, 9, 7, 15, 59, tzinfo=timezone.utc)


@pytest.fixture
def ledger(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'ledger.sqlite'}", connect_args={"timeout": 15})
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory.begin() as db:
        db.add(SocialSourceRegistry(id=1))
    yield factory
    engine.dispose()


def service(factory):
    from app.services.social_llm_budget_service import SocialLLMBudgetService
    return SocialLLMBudgetService(factory)


def test_singapore_budget_day_changes_at_1600_utc():
    from app.services.social_llm_budget_service import social_budget_date
    assert social_budget_date(NOW, "Asia/Singapore") == date(2026, 9, 7)
    assert social_budget_date(NOW + timedelta(minutes=1), "Asia/Singapore") == date(2026, 9, 8)


def test_deployment_budget_policy_overrides_seeded_database_defaults(ledger):
    from app.services.social_llm_budget_service import SocialLLMBudgetService

    budget = SocialLLMBudgetService(
        ledger,
        daily_limit_usd=Decimal("0.75"),
        budget_timezone="UTC",
    )

    assert budget.status(NOW).remaining_usd == Decimal("0.75")
    with ledger() as db:
        values = {
            row.key: row.value
            for row in db.scalars(select(AppSetting).where(AppSetting.key.in_({
                "social_llm_daily_limit_usd", "social_llm_budget_timezone",
            })))
        }
    assert values == {
        "social_llm_daily_limit_usd": "0.750000000000",
        "social_llm_budget_timezone": "UTC",
    }


def test_concurrent_reservations_share_last_quarter_dollar(ledger):
    budget = service(ledger)
    budget.reserve("spent", (1,), Decimal("1.75"), NOW)
    barrier = Barrier(2)
    def reserve(key):
        barrier.wait()
        return service(ledger).reserve(key, (2,), Decimal("0.20"), NOW)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, ("a", "b")))
    assert sum(result is not None for result in results) == 1


@pytest.mark.parametrize("actual,remaining", [("0.10", "1.90"), ("2.10", "0")])
def test_actual_cost_replaces_reservation_and_reconcile_is_idempotent(ledger, actual, remaining):
    budget = service(ledger)
    attempt = budget.reserve("a", (1,), Decimal("0.20"), NOW)
    budget.mark_dispatched(attempt)
    budget.reconcile(attempt, Decimal(actual), "request-a")
    budget.reconcile(attempt, Decimal(actual), "request-a")
    assert budget.status(NOW).remaining_usd == Decimal(remaining)


def test_late_completion_uses_original_day_and_restart_does_not_refund(ledger):
    budget = service(ledger)
    attempt = budget.reserve("a", (1,), Decimal("2"), NOW)
    budget.mark_dispatched(attempt)
    assert service(ledger).reserve("b", (2,), Decimal(".01"), NOW) is None
    budget.reconcile(attempt, Decimal("1.50"), "a")
    assert service(ledger).status(NOW).remaining_usd == Decimal(".50")
    assert service(ledger).status(NOW + timedelta(minutes=1)).remaining_usd == Decimal("2")


def test_missing_charge_retains_reservation_until_known(ledger):
    budget = service(ledger)
    attempt = budget.reserve("a", (1,), Decimal("2"), NOW)
    budget.mark_dispatched(attempt)
    budget.reconcile(attempt, None, None)
    assert budget.status(NOW).remaining_usd == 0
    assert not budget.release(attempt)
    budget.reconcile(attempt, Decimal(".30"), "late-id")
    assert budget.status(NOW).remaining_usd == Decimal("1.70")


def test_logical_operation_attempts_are_distinct_and_monotonic(ledger):
    budget = service(ledger)
    first = budget.reserve(
        "dispatch-1",
        (1,),
        Decimal(".20"),
        NOW,
        logical_operation_key="request:42",
        operation_kind="economic_extract",
    )
    budget.mark_dispatched(first)
    budget.reconcile(first, Decimal("0"), "provider-1")
    second = budget.reserve(
        "dispatch-2",
        (1,),
        Decimal(".20"),
        NOW,
        logical_operation_key="request:42",
        operation_kind="economic_extract",
    )

    with ledger() as db:
        from app.infra.db.models.social_analysis import SocialLLMAttempt

        attempts = db.scalars(
            select(SocialLLMAttempt).order_by(SocialLLMAttempt.attempt_number)
        ).all()
        assert [attempt.id for attempt in attempts] == [first, second]
        assert [attempt.attempt_number for attempt in attempts] == [1, 2]
        assert len({attempt.idempotency_key for attempt in attempts}) == 2


def test_uncertain_logical_operation_blocks_immediate_new_dispatch(ledger):
    budget = service(ledger)
    first = budget.reserve(
        "dispatch-1",
        (1,),
        Decimal(".20"),
        NOW,
        logical_operation_key="request:uncertain",
        operation_kind="economic_extract",
    )
    budget.mark_dispatched(first)
    budget.reconcile(first, None, None)

    assert budget.reserve(
        "dispatch-2",
        (1,),
        Decimal(".20"),
        NOW,
        logical_operation_key="request:uncertain",
        operation_kind="economic_extract",
    ) is None
    assert budget.reservation_is_open(first) is True


def _economic_request(ledger):
    from app.infra.db.repositories.economic_taxonomy_work_repo import (
        EconomicTaxonomyWorkRepository,
    )
    from app.models.economic_taxonomy_runtime import TaxonomyAuthority
    from app.services.economic_source_admission import (
        EconomicSourceAdmissionService,
        EvidenceAdmission,
    )

    with ledger.begin() as db:
        db.add(
            TaxonomyAuthority(
                id=1,
                mode="shadow",
                processing_head_revision=1,
                authority_epoch=1,
                writes_fenced=False,
                rollback_state="ready",
            )
        )
        db.flush()
        admitted = EconomicSourceAdmissionService(db).admit_social_work(
            EvidenceAdmission(
                provider="x",
                canonical_item_id="social-budget-post",
                capture_route="social",
                original_text="Memory pricing rose.",
                preparation_version="social-v1",
                captured_at=NOW,
                available_at=NOW,
                evidence_channels=("narrative",),
            )
        )
        request = EconomicTaxonomyWorkRepository(db).enqueue_request(
            source_lineage_id=admitted.source_lineage_id,
            evidence_packet_id=admitted.packet_id,
            policy_bundle_version="bundle-v1",
            available_at=NOW,
        )
        return request.id


class _EconomicProvider:
    def __init__(self, *effects):
        self.effects = list(effects)
        self.call_count = 0

    def extract(self, **_kwargs):
        self.call_count += 1
        effect = self.effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


def _accepted_economic_payload():
    return {
        "status": "accepted_candidates",
        "candidates": [
            {
                "candidate_key": "memory",
                "display_name": "Memory",
                "raw_facets": {"industry": "Memory"},
                "mechanism": "Pricing changes producer margins.",
                "evidence_spans": ["Memory pricing rose."],
                "relationship_evidence": [],
                "exposure_support": "direct",
                "development_support": "present",
                "securities": [],
            }
        ],
        "provider_request_id": "provider-success",
        "actual_cost": Decimal("0.01"),
    }


def test_exhausted_social_budget_prevents_economic_provider_dispatch(ledger):
    from app.services.economic_exposure_extraction import (
        BudgetExhausted,
        EconomicExposureExtractor,
    )
    from app.services.social_llm_budget_service import (
        SocialEconomicReservationManager,
        SocialLLMBudgetService,
    )

    request_id = _economic_request(ledger)
    provider = _EconomicProvider(_accepted_economic_payload())
    budget = SocialLLMBudgetService(ledger, daily_limit_usd=Decimal("0"))
    reservations = SocialEconomicReservationManager(
        budget,
        work_ids=(1,),
        maximum_usd=Decimal("0.20"),
        now=NOW,
        pricing_version="test-v1",
        input_token_limit=100,
        output_token_limit=100,
    )
    extractor = EconomicExposureExtractor(
        ledger,
        provider=provider,
        extraction_policy_version="extract-v1",
        approved_dimensions={"industry"},
        reservations=reservations,
    )

    with pytest.raises(BudgetExhausted, match="budget_exhausted"):
        extractor.extract(request_id)

    assert provider.call_count == 0


def test_social_retry_uses_two_attempts_then_reuses_successful_artifact(ledger):
    from app.infra.db.models.social_analysis import SocialLLMAttempt
    from app.services.economic_exposure_extraction import (
        EconomicExposureExtractor,
        RetryableProviderError,
        RetryableProviderFailure,
    )
    from app.services.social_llm_budget_service import (
        SocialEconomicReservationManager,
        SocialLLMBudgetService,
    )

    request_id = _economic_request(ledger)
    provider = _EconomicProvider(
        RetryableProviderError(
            "temporary",
            provider_request_id="provider-retry",
            actual_cost=Decimal("0"),
        ),
        _accepted_economic_payload(),
    )
    budget = SocialLLMBudgetService(ledger)
    reservations = SocialEconomicReservationManager(
        budget,
        work_ids=(1,),
        maximum_usd=Decimal("0.20"),
        now=NOW,
        pricing_version="test-v1",
        input_token_limit=100,
        output_token_limit=100,
    )
    extractor = EconomicExposureExtractor(
        ledger,
        provider=provider,
        extraction_policy_version="extract-v1",
        approved_dimensions={"industry"},
        reservations=reservations,
    )

    with pytest.raises(RetryableProviderFailure):
        extractor.extract(request_id)
    success = extractor.extract(request_id)
    repeated = extractor.extract(request_id)

    with ledger.begin() as db:
        from app.models.economic_taxonomy_runtime import ProcessingRequest
        from app.services.economic_source_admission import (
            EconomicSourceAdmissionService,
        )

        request = db.get(ProcessingRequest, request_id)
        eligibility = EconomicSourceAdmissionService(db).revise_lens_eligibility(
            request.evidence_packet_id,
            add="fundamental",
            reason="additional Social lens",
        )

    with ledger() as db:
        attempts = db.scalars(
            select(SocialLLMAttempt).where(
                SocialLLMAttempt.logical_operation_key == str(request_id)
            )
        ).all()
        assert [attempt.state for attempt in attempts] == ["reconciled", "reconciled"]
        assert len({attempt.idempotency_key for attempt in attempts}) == 2
    assert success.id == repeated.id
    assert eligibility.evidence_channels == ("fundamental", "narrative")
    assert provider.call_count == 2


def test_cancel_before_dispatch_and_idempotency(ledger):
    budget = service(ledger)
    attempt = budget.reserve("a", (1,), Decimal("2"), NOW)
    assert budget.reserve("a", (1,), Decimal("2"), NOW) == attempt
    assert budget.release(attempt)
    assert not budget.mark_dispatched(attempt)
    assert budget.status(NOW).remaining_usd == 2
    with pytest.raises(ValueError, match="idempotency_conflict"):
        budget.reserve("a", (2,), Decimal("2"), NOW)


def test_economic_pre_dispatch_release_refunds_only_reserved_or_dispatched_attempts(ledger):
    from app.services.social_llm_budget_service import SocialEconomicReservationManager

    budget = service(ledger)
    manager = SocialEconomicReservationManager(budget, work_ids=(1,), maximum_usd=Decimal("0.50"), now=NOW,
        pricing_version="test-v1", input_token_limit=100, output_token_limit=100)
    reserved, dispatched, uncertain = (
        manager.reserve(attempt_key=key, logical_request_id=key, operation_kind="extract")
        for key in ("reserved", "dispatched", "uncertain")
    )
    dispatched.mark_dispatched()
    uncertain.mark_dispatched()
    budget.reconcile(uncertain.attempt_id, None, None)

    reserved.release_pre_dispatch()
    dispatched.release_pre_dispatch()
    for unreleasable in (uncertain, dispatched):
        with pytest.raises(ValueError, match="social_attempt_not_releasable"):
            unreleasable.release_pre_dispatch()

    assert (reserved.state, dispatched.state, uncertain.state) == ("released", "released", "uncertain")
    assert budget.status(NOW).remaining_usd == Decimal("1.50")


def test_timezone_change_carries_overlap_without_double_counting(ledger):
    budget = service(ledger)
    attempt = budget.reserve("a", (1,), Decimal("1.80"), NOW)
    budget.mark_dispatched(attempt)
    with ledger.begin() as db:
        setting = db.scalar(select(AppSetting).where(AppSetting.key == "social_llm_budget_timezone"))
        setting.value = "UTC"
    assert service(ledger).reserve("b", (2,), Decimal(".30"), NOW) is None
    budget.reconcile(attempt, Decimal("1.70"), "a")
    assert budget.reserve("c", (3,), Decimal(".20"), NOW) is not None
    assert budget.status(NOW).remaining_usd == Decimal(".10")
    with ledger.begin() as db:
        db.scalar(select(AppSetting).where(AppSetting.key == "social_llm_budget_timezone")).value = "Asia/Singapore"
    assert budget.status(NOW).remaining_usd == Decimal(".10")


def test_money_rejects_float_and_nonfinite_values(ledger):
    for value in (0.2, Decimal("NaN"), Decimal("-1")):
        with pytest.raises(ValueError, match="invalid_money"):
            service(ledger).reserve("bad", (1,), value, NOW)


def test_out_of_order_old_price_block_cannot_reenable_current_or_previous_version(ledger):
    import json
    config = {"version": "v2", "models": {"synthetic/requested": {
        "provider": "synthetic", "actual_models": ["actual"],
        "input_usd_per_million": "1", "output_usd_per_million": "2"}}}
    with ledger.begin() as db:
        db.add(AppSetting(key="social_llm_pricing", value=json.dumps(config)))
    budget = service(ledger)
    assert budget.price("synthetic/requested") is not None
    budget.block_price("synthetic/requested", "v2", "billing_model_mismatch")
    budget.block_price("synthetic/requested", "v1", "billing_model_mismatch")
    assert service(ledger).price("synthetic/requested") is None
    with ledger.begin() as db:
        config["version"] = "v1"
        db.scalar(select(AppSetting).where(AppSetting.key == "social_llm_pricing")).value = json.dumps(config)
    assert service(ledger).price("synthetic/requested") is None
    with ledger.begin() as db:
        config["version"] = "v3"
        db.scalar(select(AppSetting).where(AppSetting.key == "social_llm_pricing")).value = json.dumps(config)
    assert service(ledger).price("synthetic/requested") is not None


def test_legacy_price_block_is_preserved_when_another_version_finishes(ledger):
    import json
    config = {"version": "v2", "models": {"synthetic/requested": {
        "provider": "synthetic", "actual_models": ["actual"],
        "input_usd_per_million": "1", "output_usd_per_million": "2"}}}
    with ledger.begin() as db:
        db.add(AppSetting(key="social_llm_pricing", value=json.dumps(config)))
        db.add(AppSetting(key="social_llm_pricing_blocks", value=json.dumps({
            "synthetic/requested": {"version": "v2", "reason": "billing_model_mismatch"}})))
    budget = service(ledger)
    assert budget.price("synthetic/requested") is None
    budget.block_price("synthetic/requested", "v1", "billing_model_mismatch")
    assert budget.price("synthetic/requested") is None


def test_metered_transport_performs_one_http_attempt_and_redacts_errors(monkeypatch, caplog):
    import asyncio
    import httpx
    from app.services.llm.llm_service import LLMService, LLMError
    from litellm.llms.openai.openai import OpenAIChatCompletion
    requests = []
    async def respond(request):
        requests.append(request)
        return httpx.Response(500, json={"error": {"message": "synthetic-private-response", "type": "server_error"}})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            monkeypatch.setattr(OpenAIChatCompletion, "_get_async_http_client", staticmethod(lambda **kw: client))
            def overrides(self, params):
                params.update(api_key="synthetic-only", api_base="https://synthetic.invalid/v1")
                return None, None, None
            monkeypatch.setattr(LLMService, "_apply_provider_overrides", overrides)
            with pytest.raises(LLMError) as error:
                await LLMService(use_case="extraction").completion(model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": "fixture"}], max_tokens=20,
                    allow_fallbacks=False, num_retries=0, metered=True)
            return str(error.value)
    message = asyncio.run(run())
    assert len(requests) == 1
    assert "synthetic-private-response" not in message
    assert "synthetic-private-response" not in caplog.text


@pytest.mark.parametrize("status,refused", [
    (400, True), (401, True), (404, True), (408, False), (429, False), (500, False),
])
def test_metered_transport_separates_refused_requests_from_unknown_outcomes(
    monkeypatch, caplog, status, refused,
):
    import asyncio
    import httpx
    from app.services.llm.llm_service import LLMError, LLMProviderRejectedError, LLMService
    from litellm.llms.openai.openai import OpenAIChatCompletion
    async def respond(request):
        return httpx.Response(status, json={"error": {"message": "synthetic-private-response"}})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            monkeypatch.setattr(OpenAIChatCompletion, "_get_async_http_client", staticmethod(lambda **kw: client))
            def overrides(self, params):
                params.update(api_key="synthetic-only", api_base="https://synthetic.invalid/v1")
                return None, None, None
            monkeypatch.setattr(LLMService, "_apply_provider_overrides", overrides)
            with pytest.raises(LLMError) as error:
                await LLMService(use_case="extraction").completion(model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": "fixture"}], max_tokens=20,
                    allow_fallbacks=False, num_retries=0, metered=True)
            return error.value
    error = asyncio.run(run())
    assert (isinstance(error, LLMProviderRejectedError), getattr(error, "status_code", None)) == (
        refused, status if refused else None)
    assert f"status={status}" in caplog.text
    assert "synthetic-private-response" not in caplog.text
