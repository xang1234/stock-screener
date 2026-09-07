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


def test_cancel_before_dispatch_and_idempotency(ledger):
    budget = service(ledger)
    attempt = budget.reserve("a", (1,), Decimal("2"), NOW)
    assert budget.reserve("a", (1,), Decimal("2"), NOW) == attempt
    assert budget.release(attempt)
    assert not budget.mark_dispatched(attempt)
    assert budget.status(NOW).remaining_usd == 2
    with pytest.raises(ValueError, match="idempotency_conflict"):
        budget.reserve("a", (2,), Decimal("2"), NOW)


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
