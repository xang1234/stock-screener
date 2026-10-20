"""Saved-source processing with real durable transactions and synthetic LLM only."""
import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.domain.social_signals.records import SocialAttachmentRef, SocialPostRecord
from app.models.app_settings import AppSetting
from app.models.theme import ContentItem
from app.infra.db.models.social_signals import SocialSourceRegistry

NOW = datetime(2026, 9, 7, 15, 59, tzinfo=timezone.utc)


@pytest.fixture
def backlog(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'backlog.sqlite'}", connect_args={"timeout": 15})
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory.begin() as db:
        db.add(SocialSourceRegistry(id=1, mode="live", provider="official"))
        db.add(AppSetting(key="social_llm_pricing", value=json.dumps({
            "version": "fixture-v1", "models": {"synthetic/requested": {
                "provider": "synthetic", "actual_models": ["actual-model"],
                "input_usd_per_million": "0", "output_usd_per_million": "244.140625"}}})))
    yield factory
    engine.dispose()


def post(number, age=1):
    return SocialPostRecord(provider="official", provider_post_id=str(number), source_id="list",
        text="$AAA supplies cooling equipment", url=f"https://x.com/a/status/{number}",
        author_handle="a", created_at=NOW - timedelta(days=age), observed_at=NOW)


class FakeLLM:
    def __init__(self, *, malformed=False, usage=True, model="actual-model", poison=()):
        self.seen = []
        self.calls = 0
        self.malformed, self.usage, self.model = malformed, usage, model
        self.poison = set(poison)

    async def completion(self, **kwargs):
        self.calls += 1
        sources = json.loads(kwargs["messages"][1]["content"])["posts"]
        self.seen.extend(p["post_id"] for p in sources)
        assert kwargs["max_tokens"] == 8192
        assert kwargs["num_retries"] == 0 and not kwargs["allow_fallbacks"]
        payload = {"posts": [dict(post_id=p["post_id"], claims=[], has_new_thesis=False,
            canonical_claim_key=None) for p in sources]}
        malformed = self.malformed or bool(self.poison & {p["post_id"] for p in sources})
        return SimpleNamespace(model=self.model, id="safe-request", _hidden_params={"custom_llm_provider": "synthetic"},
            choices=[SimpleNamespace(message=SimpleNamespace(content="bad" if malformed else json.dumps(payload)))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=8192) if self.usage else None)


def processor(factory, llm):
    from app.services.social_signal_backlog_service import ProcessSocialBacklog
    return ProcessSocialBacklog(factory, llm=llm)


def configure_budget(backlog, *, dollars="20", output_rate="0"):
    with backlog.begin() as db:
        daily = db.scalar(select(AppSetting).where(
            AppSetting.key == "social_llm_daily_limit_usd"
        ))
        if daily is None:
            db.add(AppSetting(
                key="social_llm_daily_limit_usd", value=dollars, category="social",
            ))
        else:
            daily.value = dollars
        row = db.scalar(select(AppSetting).where(AppSetting.key == "social_llm_pricing"))
        value = json.loads(row.value)
        value["models"]["synthetic/requested"]["output_usd_per_million"] = output_rate
        row.value = json.dumps(value)


def enqueue(factory, worker, value):
    with factory.begin() as db:
        item = ContentItem(source_type="twitter", external_id=value.provider_post_id,
            url=value.url, content=value.text, published_at=value.created_at)
        db.add(item)
        db.flush()
        item_id = item.id
    return worker.enqueue(item_id, value, selected_model="synthetic/requested", now=NOW)


def test_batches_twenty_posts_into_one_metered_provider_call(backlog):
    from app.infra.db.models.social_analysis import SocialLLMAttempt
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    configure_budget(backlog)
    llm = FakeLLM()
    worker = ProcessSocialBacklog(backlog, llm=llm, batch_size=20)
    work_ids = [enqueue(backlog, worker, post(number)) for number in range(1, 21)]

    result = asyncio.run(worker.execute(NOW, 20))

    assert result.succeeded == 20
    assert llm.calls == 1
    with backlog() as db:
        assert db.scalar(select(SocialLLMAttempt)).work_ids == work_ids


def test_per_run_call_cap_leaves_remaining_work_pending(backlog):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    configure_budget(backlog)
    llm = FakeLLM()
    worker = ProcessSocialBacklog(
        backlog, llm=llm, batch_size=2, max_calls_per_run=2,
    )
    work_ids = [enqueue(backlog, worker, post(number)) for number in range(1, 6)]

    result = asyncio.run(worker.execute(NOW, 5))

    assert result.succeeded == 4
    assert llm.calls == 2
    with backlog() as db:
        assert db.get(SocialExtractionWork, work_ids[-1]).state == "pending"


def test_daily_call_cap_defers_without_a_third_provider_call(backlog):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    configure_budget(backlog)
    llm = FakeLLM()
    worker = ProcessSocialBacklog(
        backlog, llm=llm, batch_size=1, max_calls_per_day=2,
    )
    work_ids = [enqueue(backlog, worker, post(number)) for number in range(1, 4)]

    result = asyncio.run(worker.execute(NOW, 3))

    assert (result.succeeded, result.deferred) == (2, 1)
    assert llm.calls == 2
    with backlog() as db:
        deferred = db.get(SocialExtractionWork, work_ids[-1])
        assert (deferred.state, deferred.error_code) == (
            "waiting_budget", "daily_call_limit_exhausted",
        )


def test_daily_call_cap_counts_attempts_from_overlapping_timezone_buckets(backlog):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    configure_budget(backlog)
    llm = FakeLLM()
    worker = ProcessSocialBacklog(
        backlog, llm=llm, batch_size=1, max_calls_per_day=1,
    )
    enqueue(backlog, worker, post(1))
    deferred_id = enqueue(backlog, worker, post(2))

    assert asyncio.run(worker.execute(NOW, 1)).succeeded == 1
    with backlog.begin() as db:
        timezone_setting = db.scalar(select(AppSetting).where(
            AppSetting.key == "social_llm_budget_timezone"
        ))
        timezone_setting.value = "UTC"

    result = asyncio.run(worker.execute(NOW, 1))

    assert (result.succeeded, result.deferred) == (0, 1)
    assert llm.calls == 1
    with backlog() as db:
        deferred = db.get(SocialExtractionWork, deferred_id)
        assert (deferred.state, deferred.error_code) == (
            "waiting_budget", "daily_call_limit_exhausted",
        )


def test_waits_between_sequential_provider_calls(backlog):
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    class Gate:
        def __init__(self):
            self.waits = iter((0, 5))

        def acquire(self, owner, ttl):
            return True

        def wait_seconds(self):
            return next(self.waits)

        def mark_started(self, seconds):
            assert seconds == 5

        def release(self, owner):
            pass

    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    configure_budget(backlog)
    llm = FakeLLM()
    worker = ProcessSocialBacklog(
        backlog, llm=llm, batch_size=1, request_gate=Gate(),
        min_interval_seconds=5, sleep=sleep,
    )
    enqueue(backlog, worker, post(1))
    enqueue(backlog, worker, post(2))

    assert asyncio.run(worker.execute(NOW, 2)).succeeded == 2
    assert llm.calls == 2
    assert slept == [5]


def test_shared_in_flight_lease_defers_without_provider_call(backlog):
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialLLMAttempt
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    class BusyGate:
        def acquire(self, owner, ttl):
            return False

        def release(self, owner):
            raise AssertionError("unowned lease must not be released")

    configure_budget(backlog)
    llm = FakeLLM()
    worker = ProcessSocialBacklog(backlog, llm=llm, request_gate=BusyGate())
    work_id = enqueue(backlog, worker, post(1))

    result = asyncio.run(worker.execute(NOW, 1))

    assert result.deferred == 1 and llm.calls == 0
    with backlog() as db:
        work = db.get(SocialExtractionWork, work_id)
        assert (work.state, work.error_code) == ("pending", "llm_request_in_flight")
        assert db.scalar(select(SocialLLMAttempt)).state == "released"


@pytest.mark.parametrize("failure_method", ["acquire", "wait", "mark"])
def test_request_gate_outage_fails_closed_and_preserves_work(backlog, failure_method):
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialLLMAttempt
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    class BrokenGate:
        def acquire(self, owner, ttl):
            if failure_method == "acquire":
                raise ConnectionError("redis unavailable")
            return True

        def wait_seconds(self):
            if failure_method == "wait":
                raise ConnectionError("redis unavailable")
            return 0

        def mark_started(self, seconds):
            if failure_method == "mark":
                raise ConnectionError("redis unavailable")

        def release(self, owner):
            pass

    configure_budget(backlog)
    llm = FakeLLM()
    worker = ProcessSocialBacklog(backlog, llm=llm, request_gate=BrokenGate())
    work_id = enqueue(backlog, worker, post(1))

    result = asyncio.run(worker.execute(NOW, 1))

    assert result.deferred == 1 and llm.calls == 0
    with backlog() as db:
        work = db.get(SocialExtractionWork, work_id)
        assert (work.state, work.error_code) == (
            "pending", "llm_request_gate_unavailable",
        )
        assert db.scalar(select(SocialLLMAttempt)).state == "released"


def test_exhaust_two_dollars_restart_resume_without_collection(backlog):
    llm = FakeLLM()
    worker = processor(backlog, llm)
    newer = enqueue(backlog, worker, post(1, 1))
    older = enqueue(backlog, worker, post(2, 2))
    result = asyncio.run(worker.execute(NOW, 10))
    assert (result.succeeded, result.deferred) == (1, 1)
    assert llm.seen == ["2"]
    resumed = processor(backlog, llm)
    result = asyncio.run(resumed.execute(NOW + timedelta(minutes=2), 10))
    assert result.succeeded == 1 and llm.seen == ["2", "1"]
    assert asyncio.run(resumed.execute(NOW + timedelta(minutes=3), 10)).succeeded == 0


def test_boundary_retention_and_admin_old_analysis(backlog, monkeypatch):
    from app.services import social_signal_backlog_service as process_backlog
    # Hold the logical dispatch exactly on the boundary, rather than allowing
    # test execution time to move it a few milliseconds outside the window.
    monkeypatch.setattr(process_backlog, "monotonic", lambda: 0)
    llm = FakeLLM()
    worker = processor(backlog, llm)
    old = enqueue(backlog, worker, post(1, 14 + 1/86400))
    boundary = enqueue(backlog, worker, post(2, 14))
    result = asyncio.run(worker.execute(NOW, 10))
    assert (result.succeeded, result.outside_window) == (1, 1)
    assert llm.seen == ["2"]
    from app.infra.db.models.social_analysis import SocialExtractionWork
    with backlog() as db:
        assert db.get(SocialExtractionWork, old).state == "outside_window"
    result = asyncio.run(worker.execute(NOW + timedelta(minutes=2), 10, admin_work_ids=(old,)))
    assert result.succeeded == 1 and llm.seen == ["2", "1"]


@pytest.mark.parametrize("malformed,usage,model,error", [(True, True, "actual-model", "malformed_json"),
    (False, False, "actual-model", None), (False, True, "different-model", None)])
def test_parse_failure_and_ambiguous_billing_are_not_refunded(backlog, malformed, usage, model, error):
    llm = FakeLLM(malformed=malformed, usage=usage, model=model)
    worker = processor(backlog, llm)
    work_id = enqueue(backlog, worker, post(1))
    result = asyncio.run(worker.execute(NOW, 10))
    from app.services.social_llm_budget_service import SocialLLMBudgetService
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialLLMAttempt
    assert SocialLLMBudgetService(backlog).status(NOW).remaining_usd == 0
    with backlog() as db:
        work = db.get(SocialExtractionWork, work_id)
        assert work.error_code == error
        assert work.actual_model == model and work.actual_provider == "synthetic"
        assert work.state == ("failed_terminal" if malformed else "succeeded")
        attempt = db.scalar(select(SocialLLMAttempt))
        assert attempt.state == ("reconciled" if usage and model == "actual-model" else "uncertain")


def test_unparseable_batch_is_split_so_only_the_offending_post_fails(backlog):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    configure_budget(backlog)
    llm = FakeLLM(poison={"3"})
    worker = ProcessSocialBacklog(backlog, llm=llm, batch_size=4)
    work_ids = [enqueue(backlog, worker, post(number)) for number in range(1, 5)]

    result = asyncio.run(worker.execute(NOW, 4))

    assert (result.succeeded, result.failed) == (3, 1)
    with backlog() as db:
        states = [(db.get(SocialExtractionWork, work_id).state,
                   db.get(SocialExtractionWork, work_id).error_code) for work_id in work_ids]
    assert states == [("succeeded", None), ("succeeded", None),
                      ("failed_terminal", "malformed_json"), ("succeeded", None)]
    assert llm.calls == 5


def test_unparseable_posts_in_both_halves_are_each_isolated(backlog):
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    configure_budget(backlog)
    llm = FakeLLM(poison={"1", "3"})
    worker = ProcessSocialBacklog(backlog, llm=llm, batch_size=4)
    for number in range(1, 5):
        enqueue(backlog, worker, post(number))

    result = asyncio.run(worker.execute(NOW, 4))

    assert (result.succeeded, result.failed, llm.calls) == (2, 2, 7)


def test_batch_splitting_stays_within_the_per_run_call_cap(backlog):
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    configure_budget(backlog)
    llm = FakeLLM(malformed=True)
    worker = ProcessSocialBacklog(backlog, llm=llm, batch_size=8, max_calls_per_run=4)
    for number in range(1, 9):
        enqueue(backlog, worker, post(number))

    result = asyncio.run(worker.execute(NOW, 8))

    assert (result.succeeded, result.failed, llm.calls) == (0, 8, 3)


class FlakyLLM(FakeLLM):
    def __init__(self, failures):
        super().__init__()
        self.failures = failures

    async def completion(self, **kwargs):
        if self.failures:
            self.failures -= 1
            self.calls += 1
            raise TimeoutError("provider timed out")
        return await super().completion(**kwargs)


@pytest.mark.parametrize("failures,max_calls_per_run,expected", [
    (1, 20, ((1, 0), 2, ("succeeded", None), ["reconciled", "uncertain"])),
    (2, 20, ((0, 1), 2, ("failed_terminal", "completion_unknown"), ["uncertain", "uncertain"])),
    (1, 1, ((0, 1), 1, ("failed_terminal", "completion_unknown"), ["uncertain"])),
])
def test_request_with_unknown_outcome_is_retried_once_within_the_run_cap(
    backlog, failures, max_calls_per_run, expected,
):
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialLLMAttempt
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    configure_budget(backlog)
    llm = FlakyLLM(failures)
    worker = ProcessSocialBacklog(backlog, llm=llm, max_calls_per_run=max_calls_per_run)
    work_id = enqueue(backlog, worker, post(1))

    result = asyncio.run(worker.execute(NOW, 1))

    with backlog() as db:
        work = db.get(SocialExtractionWork, work_id)
        attempts = sorted(attempt.state for attempt in db.scalars(select(SocialLLMAttempt)))
        assert ((result.succeeded, result.failed), llm.calls, (work.state, work.error_code), attempts) == expected


def test_deferred_retry_keeps_the_unknown_outcome_terminal(backlog):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    configure_budget(backlog)
    llm = FlakyLLM(1)
    worker = ProcessSocialBacklog(backlog, llm=llm, max_calls_per_day=1)
    work_id = enqueue(backlog, worker, post(1))

    result = asyncio.run(worker.execute(NOW, 1))
    later = asyncio.run(ProcessSocialBacklog(backlog, llm=llm).execute(
        NOW + timedelta(days=1), 1, work_ids=(work_id,)))

    with backlog() as db:
        work = db.get(SocialExtractionWork, work_id)
        assert ((result.deferred, result.failed), later.succeeded, llm.calls, work.state, work.error_code) == (
            (0, 1), 0, 1, "failed_terminal", "completion_unknown")


def test_bookkeeping_failure_after_a_response_is_not_retried(backlog, monkeypatch):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.services.social_llm_budget_service import SocialLLMBudgetService
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    configure_budget(backlog)
    llm = FakeLLM()
    worker = ProcessSocialBacklog(backlog, llm=llm)
    work_id = enqueue(backlog, worker, post(1))
    reconcile = SocialLLMBudgetService.reconcile
    def ledger_outage(self, attempt, *args, **kwargs):
        if args[0] is not None:
            raise RuntimeError("ledger unavailable")
        return reconcile(self, attempt, *args, **kwargs)
    monkeypatch.setattr(SocialLLMBudgetService, "reconcile", ledger_outage)

    result = asyncio.run(worker.execute(NOW, 1))

    with backlog() as db:
        work = db.get(SocialExtractionWork, work_id)
        assert ((result.failed, llm.calls), (work.state, work.error_code)) == (
            (1, 1), ("failed_terminal", "completion_unknown"))


def test_explicit_generation_counts_preexisting_terminal_work_as_failed(backlog):
    from app.infra.db.models.social_analysis import SocialExtractionWork

    llm = FakeLLM()
    worker = processor(backlog, llm)
    work_id = enqueue(backlog, worker, post(1))
    with backlog.begin() as db:
        work = db.get(SocialExtractionWork, work_id)
        work.state = "failed_terminal"
        work.error_code = "malformed_json"

    result = asyncio.run(worker.execute(NOW, 10, work_ids=(work_id,)))

    assert result.failed == 1
    assert llm.seen == []


def test_pricing_mismatch_pauses_later_dispatch_until_version_corrected(backlog):
    llm = FakeLLM(model="different-model")
    worker = processor(backlog, llm)
    enqueue(backlog, worker, post(1))
    enqueue(backlog, worker, post(2))
    asyncio.run(worker.execute(NOW, 1))
    result = asyncio.run(processor(backlog, llm).execute(NOW + timedelta(minutes=2), 10))
    assert result.deferred == 1 and llm.seen == ["1"]
    with backlog.begin() as db:
        row = db.scalar(select(AppSetting).where(AppSetting.key == "social_llm_pricing"))
        config = json.loads(row.value)
        config["version"] = "fixture-v2"
        config["models"]["synthetic/requested"]["actual_models"].append("different-model")
        row.value = json.dumps(config)
    assert asyncio.run(worker.execute(NOW + timedelta(minutes=3), 10)).succeeded == 1


def test_engagement_only_updates_reuse_completed_work(backlog):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    worker, value = processor(backlog, FakeLLM()), post(1)
    work_id = enqueue(backlog, worker, value)
    asyncio.run(worker.execute(NOW, 1))
    with backlog() as db:
        item_id = db.get(SocialExtractionWork, work_id).content_item_id
    assert worker.enqueue(item_id, replace(value, likes=1000, source_id="other"),
        selected_model="synthetic/requested", now=NOW) == work_id
    assert asyncio.run(worker.execute(NOW, 10)).succeeded == 0


def test_generation_scoped_execution_does_not_claim_unrelated_pending_work(backlog):
    llm = FakeLLM()
    worker = processor(backlog, llm)
    unrelated = enqueue(backlog, worker, post(1, 2))
    selected = enqueue(backlog, worker, post(2, 1))
    result = asyncio.run(worker.execute(NOW, 10, work_ids=(selected,)))
    assert result.succeeded == 1 and llm.seen == ["2"]
    from app.infra.db.models.social_analysis import SocialExtractionWork
    with backlog() as db:
        assert db.get(SocialExtractionWork, unrelated).state == "pending"


@pytest.mark.parametrize("dispatched", [False, True])
def test_expired_claim_recovery_retains_ambiguous_charge(backlog, dispatched):
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialLLMAttempt
    from decimal import Decimal
    worker = processor(backlog, FakeLLM())
    work_id = enqueue(backlog, worker, post(1))
    with backlog.begin() as db:
        row = db.get(SocialExtractionWork, work_id)
        row.state, row.claim_token, row.claim_expires_at = "running", "dead", NOW - timedelta(seconds=1)
    attempt = worker.budget.reserve("dead", (work_id,), Decimal("2"), NOW)
    if dispatched:
        worker.budget.mark_dispatched(attempt)
    result = asyncio.run(worker.execute(NOW, 10))
    with backlog() as db:
        assert db.get(SocialLLMAttempt, attempt).state == ("uncertain" if dispatched else "released")
        assert db.get(SocialExtractionWork, work_id).state == ("failed_terminal" if dispatched else "succeeded")
    assert result.succeeded == (0 if dispatched else 1)


def test_duplicate_delivery_during_request_does_not_dispatch_again(backlog):
    worker = processor(backlog, FakeLLM())
    enqueue(backlog, worker, post(1))
    original = worker.llm.completion
    async def concurrent_delivery(**kwargs):
        duplicate = await processor(backlog, FakeLLM()).execute(NOW, 10)
        assert duplicate.succeeded == 0
        # Also proves another real transaction can commit during provider I/O.
        return await original(**kwargs)
    worker.llm.completion = concurrent_delivery
    assert asyncio.run(worker.execute(NOW, 10)).succeeded == 1


def test_unknown_pricing_defers_without_any_attempt(backlog):
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialLLMAttempt
    with backlog.begin() as db:
        db.delete(db.scalar(select(AppSetting).where(AppSetting.key == "social_llm_pricing")))
    llm = FakeLLM()
    worker = processor(backlog, llm)
    work_id = enqueue(backlog, worker, post(1))
    assert asyncio.run(worker.execute(NOW, 1)).deferred == 1
    assert not llm.seen
    with backlog() as db:
        assert db.scalar(select(SocialLLMAttempt)) is None
        assert db.get(SocialExtractionWork, work_id).error_code == "pricing_not_configured"


def test_migration_upgrade_downgrade_matches_runtime_schema(backlog):
    import importlib.util
    from pathlib import Path
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import inspect
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialRunWork, SocialLLMBudgetDay, SocialLLMAttempt
    tables = [SocialRunWork.__table__, SocialLLMAttempt.__table__, SocialExtractionWork.__table__, SocialLLMBudgetDay.__table__]
    engine = backlog.kw["bind"]
    for table in tables:
        table.drop(engine)
    path = Path(__file__).resolve().parents[2] / "alembic/versions/20260907_0036_add_social_analysis_work.py"
    assert path.exists(), "durable backlog migration missing"
    spec = importlib.util.spec_from_file_location("social_analysis_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            migration.upgrade()
            for table in tables:
                expected = set(table.c.keys())
                if table is SocialLLMAttempt.__table__:
                    expected -= {
                        "logical_operation_key",
                        "operation_kind",
                        "attempt_number",
                    }
                assert set(c["name"] for c in inspect(conn).get_columns(table.name)) == expected
            migration.downgrade()
        assert not set(t.name for t in tables) & set(inspect(conn).get_table_names())


def test_late_response_after_claim_expiry_can_save_result_without_second_call(backlog):
    llm = FakeLLM()
    worker = processor(backlog, llm)
    work_id = enqueue(backlog, worker, post(1))
    original = llm.completion
    async def expired_during_request(**kwargs):
        assert (await processor(backlog, FakeLLM()).execute(NOW + timedelta(minutes=11), 10)).succeeded == 0
        return await original(**kwargs)
    llm.completion = expired_during_request
    assert asyncio.run(worker.execute(NOW, 1)).succeeded == 1


def test_pinned_work_identity_and_unsuccessful_aging_remain_auditable(backlog):
    from app.infra.db.models.social_signals import SocialSignalRun
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialRunWork
    worker = processor(backlog, FakeLLM())
    value = post(1, 15)
    work_id = enqueue(backlog, worker, value)
    with backlog.begin() as db:
        item_id = db.get(SocialExtractionWork, work_id).content_item_id
        db.add(SocialSignalRun(id="historical", registry_id=1, registry_version=1,
            mode="validation", provider="official", status="running", source_outcomes_json={},
            application_progress_json={}, feature_run_ids_json={}, exposure_dates_json={}, coverage_json={}))
    for _ in range(2):
        assert worker.enqueue(item_id, value, selected_model="synthetic/requested", now=NOW,
            run_id="historical") == work_id
    asyncio.run(worker.execute(NOW, 10))
    with backlog() as db:
        links = db.scalars(select(SocialRunWork)).all()
        work = db.get(SocialExtractionWork, work_id)
        assert len(links) == 1 and links[0].input_hash == work.input_hash
        assert work.state == "outside_window" and work.result_json is None


def test_dispatches_crossing_midnight_use_distinct_dispatch_day_buckets(backlog, monkeypatch):
    from app.services import social_signal_backlog_service as process_backlog
    from app.infra.db.models.social_analysis import SocialLLMAttempt
    elapsed = [0]
    monkeypatch.setattr(process_backlog, "monotonic", lambda: elapsed[0], raising=False)
    llm = FakeLLM()
    original = llm.completion
    async def slow_first(**kwargs):
        response = await original(**kwargs)
        elapsed[0] = 120
        return response
    llm.completion = slow_first
    worker = processor(backlog, llm)
    enqueue(backlog, worker, post(1))
    enqueue(backlog, worker, post(2))
    assert asyncio.run(worker.execute(NOW, 10)).succeeded == 2
    with backlog() as db:
        attempts = db.scalars(select(SocialLLMAttempt)).all()
        assert len({a.budget_day_id for a in attempts}) == 2


def test_expired_queued_claim_does_not_dispatch_after_long_previous_call(backlog, monkeypatch):
    from app.services import social_signal_backlog_service as process_backlog
    with backlog.begin() as db:
        db.add(AppSetting(key="social_llm_daily_limit_usd", value="4"))
    elapsed = [0]
    monkeypatch.setattr(process_backlog, "monotonic", lambda: elapsed[0], raising=False)
    llm = FakeLLM()
    original = llm.completion
    async def slow_first(**kwargs):
        response = await original(**kwargs)
        elapsed[0] = 700
        return response
    llm.completion = slow_first
    worker = processor(backlog, llm)
    enqueue(backlog, worker, post(1))
    enqueue(backlog, worker, post(2))
    asyncio.run(worker.execute(NOW, 10))
    assert llm.seen == ["1"]


def test_runtime_off_mid_batch_stops_new_model_dispatch_and_releases_reservation(backlog):
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialLLMAttempt
    with backlog.begin() as db:
        db.add(AppSetting(key="social_llm_daily_limit_usd", value="4"))
    llm = FakeLLM()
    original = llm.completion

    async def disable_after_first(**kwargs):
        response = await original(**kwargs)
        with backlog.begin() as db:
            registry = db.get(SocialSourceRegistry, 1)
            registry.mode, registry.provider = "off", "disabled"
        return response

    llm.completion = disable_after_first
    worker = processor(backlog, llm)
    first = enqueue(backlog, worker, post(1, 2))
    second = enqueue(backlog, worker, post(2, 1))
    result = asyncio.run(worker.execute(NOW, 10))

    assert result.succeeded == 1 and result.deferred == 1
    assert llm.seen == ["1"]
    with backlog() as db:
        assert db.get(SocialExtractionWork, first).state == "succeeded"
        assert db.get(SocialExtractionWork, second).state == "pending"
        attempts = db.scalars(select(SocialLLMAttempt).order_by(SocialLLMAttempt.id)).all()
        assert [attempt.state for attempt in attempts] == ["reconciled", "released"]


def test_pricing_change_between_reservation_and_dispatch_releases_without_call(backlog):
    from app.infra.db.models.social_analysis import SocialLLMAttempt
    llm = FakeLLM()
    worker = processor(backlog, llm)
    enqueue(backlog, worker, post(1))
    original = worker.budget.reserve
    def changed_policy(*args, **kwargs):
        attempt = original(*args, **kwargs)
        worker.budget.block_price("synthetic/requested", "fixture-v1", "billing_model_mismatch")
        return attempt
    worker.budget.reserve = changed_policy
    assert asyncio.run(worker.execute(NOW, 1)).deferred == 1
    assert llm.seen == []
    with backlog() as db:
        assert db.scalar(select(SocialLLMAttempt)).state == "released"


def test_cancelled_provider_attempt_is_uncertain_and_never_automatically_retried(backlog):
    from app.infra.db.models.social_analysis import SocialLLMAttempt
    llm = FakeLLM()
    worker = processor(backlog, llm)
    enqueue(backlog, worker, post(1))
    async def cancelled(**kwargs):
        raise asyncio.CancelledError
    llm.completion = cancelled
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(worker.execute(NOW, 1))
    with backlog() as db:
        assert db.scalar(select(SocialLLMAttempt)).state == "uncertain"
    assert asyncio.run(processor(backlog, FakeLLM()).execute(NOW + timedelta(minutes=2), 10)).succeeded == 0


def test_cancellation_before_dispatch_releases_budget_and_keeps_work_claimable(backlog):
    from app.infra.db.models.social_analysis import (
        SocialExtractionWork, SocialLLMAttempt, SocialLLMBudgetDay,
    )
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    class SpacingGate:
        def acquire(self, owner, ttl):
            return True

        def wait_seconds(self):
            return 5

        def mark_started(self, seconds):
            pass

        def release(self, owner):
            pass

    async def cancelled_while_waiting(_seconds):
        raise asyncio.CancelledError

    configure_budget(backlog)
    llm = FakeLLM()
    worker = ProcessSocialBacklog(backlog, llm=llm, request_gate=SpacingGate(), sleep=cancelled_while_waiting)
    work_id = enqueue(backlog, worker, post(1))

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(worker.execute(NOW, 1))

    with backlog() as db:
        work = db.get(SocialExtractionWork, work_id)
        assert (llm.calls, db.scalar(select(SocialLLMAttempt)).state, work.state) == (0, "released", "pending")
        assert db.scalar(select(SocialLLMBudgetDay)).reserved_usd == 0
    later = asyncio.run(processor(backlog, FakeLLM()).execute(NOW + timedelta(minutes=2), 1))
    assert later.succeeded == 1


def test_cancellation_survives_a_failed_refund(backlog, monkeypatch):
    from app.services.social_llm_budget_service import SocialLLMBudgetService
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    class SpacingGate:
        def acquire(self, owner, ttl):
            return True

        def wait_seconds(self):
            return 5

        def mark_started(self, seconds):
            pass

        def release(self, owner):
            pass

    async def cancelled_while_waiting(_seconds):
        raise asyncio.CancelledError

    def ledger_down(self, attempt_id):
        raise RuntimeError("ledger unavailable")

    configure_budget(backlog)
    monkeypatch.setattr(SocialLLMBudgetService, "release", ledger_down)
    llm = FakeLLM()
    worker = ProcessSocialBacklog(backlog, llm=llm, batch_size=1, request_gate=SpacingGate(),
                                  sleep=cancelled_while_waiting)
    enqueue(backlog, worker, post(1))
    enqueue(backlog, worker, post(2))

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(worker.execute(NOW, 2))
    assert llm.calls == 0

    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialLLMAttempt
    with backlog() as db:
        assert {work.state for work in db.scalars(select(SocialExtractionWork))} == {"running"}
    # Claim expiry is the recovery path: it releases the reservation and requeues the posts.
    later = asyncio.run(processor(backlog, FakeLLM()).execute(NOW + timedelta(minutes=11), 2))
    assert later.succeeded == 2
    with backlog() as db:
        assert db.scalars(select(SocialLLMAttempt.state).order_by(SocialLLMAttempt.id)).first() == "released"


def test_cancelled_retry_keeps_the_first_unknown_outcome_terminal(backlog):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    class SecondCallWaits:
        def __init__(self):
            self.waits = iter((0, 5))

        def acquire(self, owner, ttl):
            return True

        def wait_seconds(self):
            return next(self.waits)

        def mark_started(self, seconds):
            pass

        def release(self, owner):
            pass

    async def cancelled_while_waiting(_seconds):
        raise asyncio.CancelledError

    configure_budget(backlog)
    llm = FlakyLLM(1)
    worker = ProcessSocialBacklog(backlog, llm=llm, request_gate=SecondCallWaits(), sleep=cancelled_while_waiting)
    work_id = enqueue(backlog, worker, post(1))

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(worker.execute(NOW, 1))

    with backlog() as db:
        work = db.get(SocialExtractionWork, work_id)
        assert (llm.calls, work.state, work.error_code) == (1, "failed_terminal", "completion_unknown")


def test_cancelled_retry_with_a_failed_refund_is_settled_by_claim_expiry(backlog, monkeypatch):
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialLLMAttempt
    from app.services.social_llm_budget_service import SocialLLMBudgetService
    from app.services.social_signal_backlog_service import ProcessSocialBacklog

    class SecondCallWaits:
        def __init__(self):
            self.waits = iter((0, 5))

        def acquire(self, owner, ttl):
            return True

        def wait_seconds(self):
            return next(self.waits)

        def mark_started(self, seconds):
            pass

        def release(self, owner):
            pass

    async def cancelled_while_waiting(_seconds):
        raise asyncio.CancelledError

    configure_budget(backlog)
    release = SocialLLMBudgetService.release
    monkeypatch.setattr(SocialLLMBudgetService, "release",
                        lambda self, attempt_id: (_ for _ in ()).throw(RuntimeError("ledger unavailable")))
    llm = FlakyLLM(1)
    worker = ProcessSocialBacklog(backlog, llm=llm, request_gate=SecondCallWaits(), sleep=cancelled_while_waiting)
    work_id = enqueue(backlog, worker, post(1))

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(worker.execute(NOW, 1))
    monkeypatch.setattr(SocialLLMBudgetService, "release", release)
    asyncio.run(processor(backlog, FakeLLM()).execute(NOW + timedelta(minutes=11), 1))

    with backlog() as db:
        work = db.get(SocialExtractionWork, work_id)
        states = sorted(attempt.state for attempt in db.scalars(select(SocialLLMAttempt)))
        assert (llm.calls, work.state, work.error_code, states) == (
            1, "failed_terminal", "completion_unknown", ["released", "uncertain"])


def test_confirmed_pre_dispatch_failure_releases_budget_without_retry(backlog):
    from app.infra.db.models.social_analysis import (
        SocialExtractionWork,
        SocialLLMAttempt,
        SocialLLMBudgetDay,
    )
    from app.services.llm.llm_service import LLMPreDispatchError

    class MissingConfigurationLLM:
        async def completion(self, **_kwargs):
            raise LLMPreDispatchError("metered_provider_configuration_error")

    worker = processor(backlog, MissingConfigurationLLM())
    work_id = enqueue(backlog, worker, post(1))

    result = asyncio.run(worker.execute(NOW, 1))

    assert result.failed == 1
    with backlog() as db:
        attempt = db.scalar(select(SocialLLMAttempt))
        day = db.scalar(select(SocialLLMBudgetDay))
        work = db.get(SocialExtractionWork, work_id)
        assert attempt.state == "released"
        assert day.reserved_usd == day.actual_usd == 0
        assert work.state == "failed_terminal"
        assert work.error_code == "llm_configuration_error"


@pytest.mark.parametrize("admin_requested", [False, True])
def test_queued_post_aging_out_before_dispatch_releases_only_unused_reservation(backlog, monkeypatch, admin_requested):
    from app.services import social_signal_backlog_service as process_backlog
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialLLMAttempt
    elapsed = [0]
    monkeypatch.setattr(process_backlog, "monotonic", lambda: elapsed[0])
    llm = FakeLLM()
    original = llm.completion
    async def slow_first(**kwargs):
        response = await original(**kwargs)
        elapsed[0] = 120
        return response
    llm.completion = slow_first
    worker = processor(backlog, llm)
    enqueue(backlog, worker, post(1, 14 - 10/86400))
    second_id = enqueue(backlog, worker, post(2, 14 - 60/86400))
    result = asyncio.run(worker.execute(NOW, 10, admin_work_ids=(second_id,) if admin_requested else ()))
    assert llm.seen == (["1", "2"] if admin_requested else ["1"])
    assert result.outside_window == (0 if admin_requested else 1)
    with backlog() as db:
        second = db.get(SocialExtractionWork, second_id)
        assert second.state == ("succeeded" if admin_requested else "outside_window")
        assert (second.result_json is not None) == admin_requested
        attempts = db.scalars(select(SocialLLMAttempt).order_by(SocialLLMAttempt.id)).all()
        assert attempts[0].state == "reconciled" and attempts[0].actual_usd == 2
        assert attempts[1].state == ("reconciled" if admin_requested else "released")
        assert attempts[1].actual_usd == (2 if admin_requested else None)


@pytest.mark.parametrize("value", [
    replace(post(1), text="RT $AAA big news", is_repost=True),
    replace(post(1), text="Could you recommend some coffee shops in San Jose? 👀"),
    replace(post(1), text="Paid $10K for $50M of hype"),
])
def test_no_signal_posts_finish_without_an_llm_call(backlog, value):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.services.social_theme_projection_service import _decode

    configure_budget(backlog)
    llm = FakeLLM()
    worker = processor(backlog, llm)
    work_id = enqueue(backlog, worker, value)

    asyncio.run(worker.execute(NOW, 1))

    assert llm.calls == 0
    with backlog() as db:
        work = db.get(SocialExtractionWork, work_id)
        assert (work.state, work.error_code) == ("succeeded", "skipped:no_signal")
        _, result = _decode(work)
        assert result.claims == () and result.judgments[0].has_new_thesis is False


@pytest.mark.parametrize("value", [
    replace(post(1), text="$AAA supplies cooling"),
    replace(post(1), text="nvidia just guided above the street"),
    replace(post(1), text="Look at MU here"),
    replace(post(1), text="Tencent 0700.HK looks heavy"),
    replace(post(1), text="F cuts EV spending"),
    replace(post(1), text="MU.It is breaking out"),
    replace(post(1), text="Foxconn guides AI server revenue higher"),
    replace(post(1), text="hon hai capacity is sold out"),
    replace(post(1), text="3008.TWO rallies"),
    replace(post(1), text="雷军说小米汽车订单很多"),
    replace(post(1), text="this chart", attachments=(SocialAttachmentRef("image", "https://pbs.twimg.com/a.jpg"),)),
    replace(post(1), text="agreed", quoted_text="$AAA beats"),
])
def test_posts_with_a_ticker_name_or_attachment_still_reach_the_llm(backlog, value):
    from app.models.stock_universe import StockUniverse

    with backlog.begin() as db:
        db.add(StockUniverse(symbol="NVDA", name="NVIDIA Corp", market="US", exchange="NASDAQ", is_active=True))
        db.add(StockUniverse(symbol="MU", name="Micron Technology Inc", market="US", exchange="NASDAQ", is_active=True))
        db.add(StockUniverse(symbol="0700.HK", name="TENCENT", market="HK", exchange="HKEX", is_active=True))
        db.add(StockUniverse(symbol="F", name="Ford Motor Co", market="US", exchange="NYSE", is_active=True))
        db.add(StockUniverse(symbol="2317.TW", name="HON HAI PRECISION", market="TW", exchange="TWSE", is_active=True))
        db.add(StockUniverse(symbol="3008.TWO", name="LARGAN", market="TW", exchange="TPEX", is_active=True))
    configure_budget(backlog)
    llm = FakeLLM()
    worker = processor(backlog, llm)
    enqueue(backlog, worker, value)

    asyncio.run(worker.execute(NOW, 1))

    assert llm.calls == 1


def test_a_no_signal_skip_yields_new_work_once_the_universe_lists_the_ticker(backlog):
    """A skip made before a listing loaded must not stick, nor change under runs that saved it."""
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.models.stock_universe import StockUniverse

    configure_budget(backlog)
    llm = FakeLLM()
    value = replace(post(1), text="ZZZ supplies cooling")
    work_id = enqueue(backlog, processor(backlog, llm), value)
    with backlog.begin() as db:
        db.add(StockUniverse(symbol="ZZZ", name="Zeta Cooling Inc", market="US", exchange="NASDAQ", is_active=True))
    worker = processor(backlog, llm)

    new_id = worker.enqueue(1, value, selected_model="synthetic/requested", now=NOW)
    asyncio.run(worker.execute(NOW, 1))

    assert new_id != work_id and llm.calls == 1
    with backlog() as db:
        skipped, extracted = db.get(SocialExtractionWork, work_id), db.get(SocialExtractionWork, new_id)
        assert (skipped.state, skipped.error_code) == ("succeeded", "skipped:no_signal")
        assert (extracted.state, extracted.error_code) == ("succeeded", None)
    # Still no signal for a post that names nothing: the saved skip is reused.
    quiet = replace(post(2), text="good morning")
    first = enqueue(backlog, worker, quiet)
    assert worker.enqueue(2, quiet, selected_model="synthetic/requested", now=NOW) == first


def test_alias_of_a_listing_outside_the_active_universe_does_not_reach_the_llm(backlog):
    """An alias can only yield a usable claim when its listing is active; otherwise the
    resolver rejects it and the call just spends the shared budget."""
    from app.infra.db.models.social_analysis import SocialExtractionWork

    configure_budget(backlog)
    llm = FakeLLM()
    worker = processor(backlog, llm)
    work_id = enqueue(backlog, worker, replace(post(1), text="Foxconn guides AI server revenue higher"))

    asyncio.run(worker.execute(NOW, 1))

    assert llm.calls == 0
    with backlog() as db:
        assert db.get(SocialExtractionWork, work_id).error_code == "skipped:no_signal"


def test_enqueue_stores_a_post_with_prepared_evidence_as_json(backlog):
    from hashlib import sha256

    from app.domain.social_signals.records import SocialPreparedEvidence
    from app.infra.db.models.social_analysis import SocialExtractionWork

    text = "chart shows $AAA backlog doubling"
    digest = sha256(text.encode()).hexdigest()
    evidence = SocialPreparedEvidence(id=digest, kind="image", url="https://pbs.twimg.com/a.jpg", text=text,
        original_text_sha256=digest, text_sha256=digest, available_at=NOW, provenance_json="{}")
    value = replace(post(1), attachments=(SocialAttachmentRef("image", "https://pbs.twimg.com/a.jpg"),),
        prepared_evidence=(evidence,), evidence_digest=digest)

    enqueue(backlog, processor(backlog, FakeLLM()), value)

    with backlog() as db:
        snapshot = dict(db.scalar(select(SocialExtractionWork)).input_snapshot_json)
    for field in ("created_at", "observed_at"):
        snapshot[field] = datetime.fromisoformat(snapshot[field])
    assert SocialPostRecord(**snapshot) == value
