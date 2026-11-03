import asyncio
from datetime import timedelta
from decimal import Decimal

from sqlalchemy import select

from tests.integration.test_social_analysis_backlog import (  # noqa: F401
    NOW, FakeLLM, backlog, configure_budget, enqueue, post, processor,
)


def strand(factory, work_id, *attempt_states, admin=False):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.services.social_llm_budget_service import SocialLLMBudgetService
    budget = SocialLLMBudgetService(factory)
    for number, state in enumerate(attempt_states):
        attempt = budget.reserve(f"fixture:{work_id}:{number}", (work_id,), Decimal("0.1"), NOW)
        if state == "released":
            budget.release(attempt)
            continue
        budget.mark_dispatched(attempt)
        if state != "dispatched":
            budget.reconcile(attempt, Decimal("0.1") if state == "reconciled" else None, None)
    with factory.begin() as db:
        work = db.get(SocialExtractionWork, work_id)
        work.state, work.error_code, work.claim_token = "failed_terminal", "completion_unknown", "expired-claim"
        work.requested_by_admin = admin


LATER = NOW + timedelta(hours=2)


def states(factory, *work_ids):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    with factory() as db:
        return [(work.state, work.error_code, work.claim_token)
                for work in (db.get(SocialExtractionWork, work_id) for work_id in work_ids)]


def test_requeue_returns_only_unbilled_settled_in_window_work_to_the_backlog(backlog):
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialLLMAttempt
    from scripts.requeue_social_completion_unknown import requeue

    configure_budget(backlog)
    worker = processor(backlog, FakeLLM())
    unknown, retried, billed, in_flight, aged, admin_aged, other = (
        enqueue(backlog, worker, post(number, age=20 if number in {5, 6} else 1))
        for number in range(1, 8)
    )
    strand(backlog, unknown, "uncertain")
    strand(backlog, retried, "uncertain", "released")
    strand(backlog, billed, "reconciled")
    strand(backlog, in_flight, "dispatched")
    strand(backlog, aged, "uncertain")
    strand(backlog, admin_aged, "uncertain", admin=True)
    with backlog.begin() as db:
        work = db.get(SocialExtractionWork, other)
        work.state, work.error_code = "failed_terminal", "provider_rejected"
    everything = (unknown, retried, billed, in_flight, aged, admin_aged, other)
    before = states(backlog, *everything)
    counts = {"requeue": 3, "billed": 1, "in_flight": 1, "outside_window": 1}

    assert requeue(backlog, LATER) == counts
    assert states(backlog, *everything) == before

    assert requeue(backlog, LATER, apply=True) == counts
    assert states(backlog, unknown, retried, admin_aged) == [("pending", None, None)] * 3
    assert states(backlog, billed, in_flight, aged, other) == before[2:5] + before[6:]
    assert requeue(backlog, LATER, apply=True) == {
        "billed": 1, "in_flight": 1, "outside_window": 1,
    }

    llm = FakeLLM()
    result = asyncio.run(processor(backlog, llm).execute(NOW, 1, work_ids=(unknown,)))
    assert (result.succeeded, llm.calls, states(backlog, unknown)) == (1, 1, [("succeeded", None, None)])
    with backlog() as db:
        assert sorted(attempt.state for attempt in db.scalars(select(SocialLLMAttempt))
                      if unknown in attempt.work_ids) == ["reconciled", "uncertain"]


def refuse(factory, work_id, at):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.services.social_llm_budget_service import SocialLLMBudgetService
    budget = SocialLLMBudgetService(factory)
    budget.release(budget.reserve(f"refused:{work_id}", (work_id,), Decimal("0.1"), at))
    with factory.begin() as db:
        work = db.get(SocialExtractionWork, work_id)
        work.state, work.error_code = "failed_terminal", "provider_rejected"


def test_requeue_targets_one_error_code_within_an_attempt_window(backlog):
    from scripts.requeue_social_completion_unknown import requeue

    configure_budget(backlog)
    worker = processor(backlog, FakeLLM())
    earlier, outage, unknown = (enqueue(backlog, worker, post(number)) for number in range(1, 4))
    refuse(backlog, earlier, NOW - timedelta(days=2))
    refuse(backlog, outage, NOW - timedelta(hours=1))
    strand(backlog, unknown, "uncertain")
    window = {"since": NOW - timedelta(days=1), "until": (NOW - timedelta(minutes=1)).replace(tzinfo=None)}

    assert requeue(backlog, LATER, apply=True, error_code="provider_rejected", **window) == {"requeue": 1}
    assert states(backlog, earlier, outage, unknown) == [
        ("failed_terminal", "provider_rejected", None),
        ("pending", None, None),
        ("failed_terminal", "completion_unknown", "expired-claim"),
    ]


def test_requeue_leaves_work_whose_request_may_still_be_running(backlog):
    from scripts.requeue_social_completion_unknown import requeue

    configure_budget(backlog)
    work_id = enqueue(backlog, processor(backlog, FakeLLM()), post(1))
    strand(backlog, work_id, "uncertain")
    stranded = states(backlog, work_id)

    assert requeue(backlog, NOW + timedelta(minutes=30), apply=True) == {"in_flight": 1}
    assert states(backlog, work_id) == stranded
    assert requeue(backlog, LATER, apply=True) == {"requeue": 1}


def test_requeue_moves_requeued_and_waiting_work_to_another_model(backlog):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from scripts.requeue_social_completion_unknown import requeue

    configure_budget(backlog)
    worker = processor(backlog, FakeLLM())
    unknown, waiting, taken = (enqueue(backlog, worker, post(number)) for number in range(1, 4))
    strand(backlog, unknown, "uncertain")
    with backlog.begin() as db:
        db.get(SocialExtractionWork, waiting).state = "waiting_budget"
        source = db.get(SocialExtractionWork, taken)
        db.add(SocialExtractionWork(
            content_item_id=source.content_item_id, input_hash=source.input_hash,
            prompt_version=source.prompt_version, schema_version=source.schema_version,
            selected_model="anthropic/claude-haiku-5-5", input_snapshot_json=source.input_snapshot_json,
            state="pending", created_at=NOW, updated_at=NOW))
    model = "anthropic/claude-haiku-5-5"

    assert requeue(backlog, LATER, model=model) == {"requeue": 1, "retarget": 1, "retarget_duplicate": 1}
    assert requeue(backlog, LATER, apply=True, model=model) == {"requeue": 1, "retarget": 1, "retarget_duplicate": 1}
    with backlog() as db:
        models = {work_id: (db.get(SocialExtractionWork, work_id).selected_model,
                            db.get(SocialExtractionWork, work_id).state) for work_id in (unknown, waiting, taken)}
    assert models == {unknown: (model, "pending"), waiting: (model, "waiting_budget"),
                      taken: ("synthetic/requested", "pending")}
    assert requeue(backlog, LATER, apply=True, model=model) == {"retarget_duplicate": 1}


def test_requeue_with_the_current_model_still_requeues(backlog):
    from scripts.requeue_social_completion_unknown import requeue

    configure_budget(backlog)
    work_id = enqueue(backlog, processor(backlog, FakeLLM()), post(1))
    strand(backlog, work_id, "uncertain")

    assert requeue(backlog, LATER, apply=True, model="synthetic/requested") == {"requeue": 1}
    assert states(backlog, work_id) == [("pending", None, None)]


def test_requeue_moves_only_one_of_two_old_models_onto_the_same_post(backlog):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from scripts.requeue_social_completion_unknown import requeue

    configure_budget(backlog)
    first = enqueue(backlog, processor(backlog, FakeLLM()), post(1))
    with backlog.begin() as db:
        source = db.get(SocialExtractionWork, first)
        source.state = "waiting_budget"
        twin = SocialExtractionWork(
            content_item_id=source.content_item_id, input_hash=source.input_hash,
            prompt_version=source.prompt_version, schema_version=source.schema_version,
            selected_model="other/old", input_snapshot_json=source.input_snapshot_json,
            state="pending", created_at=NOW, updated_at=NOW)
        db.add(twin)
        db.flush()
        second = twin.id

    model = "anthropic/claude-haiku-5-5"
    assert requeue(backlog, LATER, apply=True, model=model) == {"retarget": 1, "retarget_duplicate": 1}
    with backlog() as db:
        assert sorted(db.get(SocialExtractionWork, i).selected_model for i in (first, second)) == [model, "other/old"]
