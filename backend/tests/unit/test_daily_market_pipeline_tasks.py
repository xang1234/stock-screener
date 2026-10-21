"""Daily per-market pipeline orchestration tests."""

from __future__ import annotations

from datetime import date
from datetime import datetime
from types import SimpleNamespace

import pytest


class _FakeSignature:
    def __init__(self, task: str, *, args=None, kwargs=None):
        self.task = task
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.queue = None

    def set(self, queue=None, **_kwargs):
        self.queue = queue
        return self


class _FakeTask:
    def __init__(self, task: str):
        self.task = task

    def si(self, *args, **kwargs):
        return _FakeSignature(self.task, args=args, kwargs=kwargs)


def test_daily_market_pipeline_orders_refresh_compute_and_scan(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    monkeypatch.setattr(
        "app.tasks.cache_tasks.smart_refresh_cache",
        _FakeTask("app.tasks.cache_tasks.smart_refresh_cache"),
    )
    monkeypatch.setattr(
        "app.tasks.market_rs_tasks.calculate_market_rs_snapshot",
        _FakeTask("app.tasks.market_rs_tasks.calculate_market_rs_snapshot"),
    )
    monkeypatch.setattr(
        "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill",
        _FakeTask("app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill"),
    )
    monkeypatch.setattr(
        "app.tasks.breadth_tasks.calculate_market_exposure",
        _FakeTask("app.tasks.breadth_tasks.calculate_market_exposure"),
    )
    monkeypatch.setattr(
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill",
        _FakeTask("app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill"),
    )
    monkeypatch.setattr(
        "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
        _FakeTask("app.interfaces.tasks.feature_store_tasks.build_daily_snapshot"),
    )

    signatures = module._build_daily_market_pipeline_signatures("hk", date(2026, 3, 16))

    assert [signature.task for signature in signatures] == [
        "app.tasks.cache_tasks.smart_refresh_cache",
        # On the data_fetch queue: refetches the benchmark candidates through
        # the provider plan when the primary has no bar for the as-of date.
        "app.tasks.daily_market_pipeline_tasks.ensure_benchmark_prices",
        "app.tasks.daily_market_pipeline_tasks.guard_price_refresh",
        "app.tasks.market_rs_tasks.calculate_market_rs_snapshot",
        "app.tasks.daily_market_pipeline_tasks.guard_market_rs_result",
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill",
        "app.tasks.daily_market_pipeline_tasks.guard_group_result",
        "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
        "app.tasks.daily_market_pipeline_tasks.guard_snapshot_result",
        # Last: a breadth failure (guarded, or an uncatchable OOM kill) must
        # not cost the day's group rankings or scan.
        "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill",
        "app.tasks.daily_market_pipeline_tasks.guard_breadth_result",
        "app.tasks.breadth_tasks.calculate_market_exposure",
        "app.tasks.daily_market_pipeline_tasks.guard_exposure_result",
    ]
    assert signatures[0].kwargs == {"mode": "delta", "market": "HK"}
    assert signatures[1].kwargs == {"market": "HK", "as_of_date": "2026-03-16"}
    assert signatures[1].options["queue"] == "data_fetch_hk"
    assert signatures[2].kwargs == {"market": "HK", "as_of_date": "2026-03-16", "rerun_attempt": 0}
    assert signatures[2].options["queue"] == "market_jobs_hk"
    assert signatures[3].kwargs == {
        "market": "HK",
        "calculation_date": "2026-03-16",
        "formula_version": "balanced-horizon-percentile-v2",
    }
    assert signatures[5].kwargs == {
        "market": "HK",
        "calculation_date": "2026-03-16",
        "execution_policy": "refresh_guarded",
    }
    assert signatures[9].kwargs == {
        "market": "HK",
        "calculation_date": "2026-03-16",
        "execution_policy": "refresh_guarded",
    }
    assert signatures[7].kwargs == {
        "market": "HK",
        "as_of_date_str": "2026-03-16",
        "universe_name": "market:HK",
        "publish_pointer_key": "latest_published_market:HK",
        "static_daily_mode": True,
    }


@pytest.mark.parametrize("market", ("AU", "SG", "MY"))
def test_daily_pipeline_omits_breadth_tasks_and_guards_for_disabled_market(
    monkeypatch,
    market,
):
    from app.tasks import daily_market_pipeline_tasks as module

    monkeypatch.setattr(
        "app.tasks.cache_tasks.smart_refresh_cache",
        _FakeTask("app.tasks.cache_tasks.smart_refresh_cache"),
    )
    monkeypatch.setattr(
        "app.tasks.market_rs_tasks.calculate_market_rs_snapshot",
        _FakeTask("app.tasks.market_rs_tasks.calculate_market_rs_snapshot"),
    )
    monkeypatch.setattr(
        "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill",
        _FakeTask("app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill"),
    )
    monkeypatch.setattr(
        "app.tasks.breadth_tasks.calculate_market_exposure",
        _FakeTask("app.tasks.breadth_tasks.calculate_market_exposure"),
    )
    monkeypatch.setattr(
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill",
        _FakeTask("app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill"),
    )
    monkeypatch.setattr(
        "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
        _FakeTask("app.interfaces.tasks.feature_store_tasks.build_daily_snapshot"),
    )

    tasks = [
        signature.task
        for signature in module._build_daily_market_pipeline_signatures(
            market,
            date(2026, 3, 16),
        )
    ]

    assert "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill" not in tasks
    assert "app.tasks.daily_market_pipeline_tasks.guard_breadth_result" not in tasks
    assert "app.tasks.breadth_tasks.calculate_market_exposure" not in tasks
    assert "app.tasks.daily_market_pipeline_tasks.guard_exposure_result" not in tasks
    assert "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot" in tasks


def test_queue_daily_market_pipeline_skips_disabled_market(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _market: False,
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.get_runtime_bootstrap_status",
        lambda _db: SimpleNamespace(
            bootstrap_required=False,
            bootstrap_state="ready",
            primary_market="US",
            enabled_markets=["US"],
        ),
    )
    monkeypatch.setattr(module, "SessionLocal", lambda: SimpleNamespace(close=lambda: None))

    result = module.queue_daily_market_pipeline.run("HK")

    assert result["status"] == "skipped"
    assert result["reason"] == "market HK is disabled in local runtime preferences"


def test_queue_daily_market_pipeline_prefers_bootstrap_blocker_over_disabled_market(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _market: False,
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.get_runtime_bootstrap_status",
        lambda _db: SimpleNamespace(
            bootstrap_required=True,
            bootstrap_state="not_started",
            primary_market="US",
            enabled_markets=["US"],
        ),
    )
    monkeypatch.setattr(module, "SessionLocal", lambda: SimpleNamespace(close=lambda: None))

    result = module.queue_daily_market_pipeline.run("HK")

    assert result["status"] == "skipped"
    assert result["reason"] == "local_runtime_bootstrap_not_ready"
    assert result["market"] == "HK"


def test_queue_daily_market_pipeline_skips_until_local_bootstrap_ready(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    class _FakeCalendar:
        def market_now(self, _market):
            return datetime(2026, 3, 16, 10, 0)

        def is_trading_day(self, _market, _today):
            return True

        def last_completed_trading_day(self, _market):
            return date(2026, 3, 16)

    chain_called = False

    class _FakeChain:
        def apply_async(self):
            nonlocal chain_called
            chain_called = True
            return SimpleNamespace(id="queued-task")

    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _market: True,
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.get_runtime_bootstrap_status",
        lambda _db: SimpleNamespace(
            bootstrap_required=True,
            bootstrap_state="not_started",
            primary_market="US",
            enabled_markets=["US"],
        ),
    )
    monkeypatch.setattr(module, "SessionLocal", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(module, "_market_pipeline_active", lambda _market: None)
    monkeypatch.setattr(module, "MarketCalendarService", lambda: _FakeCalendar())
    monkeypatch.setattr(module, "_build_daily_market_pipeline_signatures", lambda *_args: [])
    monkeypatch.setattr(module, "chain", lambda *_signatures: _FakeChain())

    result = module.queue_daily_market_pipeline.run("US")

    assert result["status"] == "skipped"
    assert result["reason"] == "local_runtime_bootstrap_not_ready"
    assert result["bootstrap_state"] == "not_started"
    assert result["bootstrap_required"] is True
    assert chain_called is False


class _ChainClaims:
    """Stands in for WorkloadCoordination's per-market pipeline-chain lease."""

    def __init__(self, *, fail_claim: bool = False):
        self.holders: dict[str, dict] = {}
        self.task_ids: dict[str, list[str]] = {}
        self.missing_since: dict[str, float] = {}
        self.clock = 0.0
        self.renewed: list[tuple[str, str]] = []
        self.released: list[tuple[str, str]] = []
        self.fail_claim = fail_claim
        self.heartbeat_holder: str | None = None

    def claim_pipeline_chain(self, token, *, market, task_ids):
        if self.fail_claim:
            raise ConnectionError("redis down")
        if market in self.holders:
            return False
        self.holders[market] = {"task_id": token, "started_at": "2026-03-16T20:30:00+00:00"}
        self.task_ids[token] = list(task_ids)
        return True

    def get_pipeline_chain_holder(self, market):
        return self.holders.get(market)

    def get_pipeline_chain_holders(self):
        return dict(self.holders)

    def get_pipeline_chain_task_ids(self, token):
        return self.task_ids.get(token)

    def renew_pipeline_chain(self, token, *, market):
        self.renewed.append((market, token))
        holder = self.holders.get(market)
        return bool(holder and holder["task_id"] == token)

    def acquire_pipeline_chain_heartbeat(self, token):
        if self.heartbeat_holder is not None:
            return False
        self.heartbeat_holder = token
        return True

    def release_pipeline_chain_heartbeat(self, token):
        if self.heartbeat_holder == token:
            self.heartbeat_holder = None

    def pipeline_chain_missing(self, token):
        since = self.missing_since.setdefault(token, self.clock)
        return self.clock - since, repr(since)

    def release_missing_pipeline_chain(self, token, *, market, missing_since, heartbeat):
        self.released.append((market, token))
        holder = self.holders.get(market)
        if not (holder and holder["task_id"] == token):
            return "lost"
        if self.heartbeat_holder != heartbeat:
            return "kept"
        if token not in self.missing_since or repr(self.missing_since[token]) != missing_since:
            return "kept"
        del self.holders[market]
        return "released"

    def clear_pipeline_chain_missing(self, token):
        self.missing_since.pop(token, None)

    def release_pipeline_chain(self, token, *, market):
        self.released.append((market, token))
        holder = self.holders.get(market)
        if holder and holder["task_id"] == token:
            del self.holders[market]
            return True
        return False


def _ready_to_dispatch(monkeypatch, module, claims, published, *, market_open=False):
    class _FakeCalendar:
        def market_now(self, _market):
            return datetime(2026, 3, 16, 16, 30)

        def is_trading_day(self, _market, _today):
            return True

        def last_completed_trading_day(self, _market):
            return date(2026, 3, 16)

        def is_market_open(self, _market):
            return market_open

    class _FakeChain:
        def __init__(self, signatures):
            self.signatures = signatures

        def freeze(self):
            node = None
            for index, _signature in enumerate(self.signatures):
                node = SimpleNamespace(id=f"stage-{index}", parent=node)
            return node

        def apply_async(self, **options):
            published.append((self.signatures, options))
            return SimpleNamespace(id=f"chain-{len(published)}")

    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now", lambda _market: True,
    )
    monkeypatch.setattr(module, "_local_runtime_bootstrap_blocker", lambda _market: None)
    monkeypatch.setattr(module, "_market_pipeline_active", lambda _market: None)
    monkeypatch.setattr(module, "MarketCalendarService", lambda: _FakeCalendar())
    monkeypatch.setattr(
        module,
        "_build_daily_market_pipeline_signatures",
        lambda *_args, **_kwargs: ["prices", "groups", "snapshot"],
    )
    monkeypatch.setattr(module, "chain", lambda *signatures: _FakeChain(list(signatures)))
    monkeypatch.setattr(module, "_pipeline_chain_claims", lambda: claims)


def test_dispatch_holds_the_chain_until_its_last_stage_or_first_failure(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    claims, published = _ChainClaims(), []
    _ready_to_dispatch(monkeypatch, module, claims, published)

    result = module.queue_daily_market_pipeline.run("US")

    assert result["status"] == "queued"
    token = claims.holders["US"]["task_id"]
    [(signatures, options)] = published
    # Released by a final stage on success...
    release = signatures[-1]
    assert signatures[:-1] == ["prices", "groups", "snapshot"]
    assert release.task == "app.tasks.daily_market_pipeline_tasks.release_daily_market_pipeline_claim"
    assert release.kwargs == {"market": "US", "token": token}
    assert release.immutable
    # ...and by the same task as every stage's error callback.
    assert options["link_error"] is release
    # The claim records the ids the chain is published with, release included.
    assert claims.task_ids[token] == ["stage-0", "stage-1", "stage-2", "stage-3"]


def test_second_dispatch_between_stages_is_refused(monkeypatch):
    # The per-stage workload lease is free between stages; the chain claim
    # is what stops a second chain for the same market.
    from app.tasks import daily_market_pipeline_tasks as module

    claims, published = _ChainClaims(), []
    _ready_to_dispatch(monkeypatch, module, claims, published)

    first = module.queue_daily_market_pipeline.run("US")
    second = module.queue_daily_market_pipeline.run("US")

    assert first["status"] == "queued"
    assert second["status"] == "skipped"
    assert second["reason"] == "market_pipeline_already_active"
    # A claim token, not a Celery task id: named so nobody looks it up as one.
    assert second["active_chain_token"] == claims.holders["US"]["task_id"]
    assert "active_task_id" not in second
    assert second["active_since"] == "2026-03-16T20:30:00+00:00"
    assert len(published) == 1


def test_claim_freed_between_claim_and_holder_lookup_is_retried(monkeypatch):
    # The first chain releases right after the second's SET NX lost: the
    # second dispatch must not skip with nothing running.
    from app.tasks import daily_market_pipeline_tasks as module

    claims, published = _ChainClaims(), []
    _ready_to_dispatch(monkeypatch, module, claims, published)
    attempts = []
    real_claim = claims.claim_pipeline_chain

    def flaky_claim(token, *, market, task_ids):
        attempts.append(token)
        return False if len(attempts) == 1 else real_claim(token, market=market, task_ids=task_ids)

    monkeypatch.setattr(claims, "claim_pipeline_chain", flaky_claim)

    result = module.queue_daily_market_pipeline.run("US")

    assert result["status"] == "queued"
    assert len(attempts) == 2
    assert len(published) == 1


def test_dispatch_fails_open_when_the_coordination_cannot_be_built(monkeypatch, caplog):
    from app.tasks import daily_market_pipeline_tasks as module

    claims, published = _ChainClaims(), []
    _ready_to_dispatch(monkeypatch, module, claims, published)

    def no_coordination():
        raise RuntimeError("Redis package is not installed")

    monkeypatch.setattr(module, "_pipeline_chain_claims", no_coordination)

    result = module.queue_daily_market_pipeline.run("US")

    assert result["status"] == "queued"
    assert "Could not claim the US pipeline chain" in caplog.text


def test_real_celery_chain_attaches_the_release_to_every_stage():
    # Uses Celery's own chain preparation, not the fakes above.
    from celery import chain
    from celery.utils.functional import arity_greater

    from app.tasks import daily_market_pipeline_tasks as module

    release = module.release_daily_market_pipeline_claim.si(market="US", token="tok")
    stages = [
        module.guard_price_refresh.s(market="US"),
        module.guard_group_result.s(market="US"),
        release,
    ]

    tasks, _results = chain(*stages).prepare_steps(
        (), {}, chain(*stages).tasks, link_error=release, app=module.celery_app,
    )

    assert len(tasks) == 3
    for task in tasks:
        [errback] = task.options["link_error"]
        assert errback["task"] == release.task
        assert errback["kwargs"] == {"market": "US", "token": "tok"}
    # Published as a task (immutable signature drops the failed task id),
    # not called inline with (request, exc, traceback).
    assert arity_greater(module.release_daily_market_pipeline_claim.__header__, 1) is False


def test_other_markets_are_not_blocked_by_a_running_chain(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    claims, published = _ChainClaims(), []
    _ready_to_dispatch(monkeypatch, module, claims, published)

    assert module.queue_daily_market_pipeline.run("US")["status"] == "queued"
    assert module.queue_daily_market_pipeline.run("HK")["status"] == "queued"


def test_a_failed_publish_releases_the_claim(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    claims, published = _ChainClaims(), []
    _ready_to_dispatch(monkeypatch, module, claims, published)

    class _BrokenChain:
        def freeze(self):
            return SimpleNamespace(id="stage-0", parent=None)

        def apply_async(self, **_options):
            raise ConnectionError("broker unavailable")

    monkeypatch.setattr(module, "chain", lambda *_signatures: _BrokenChain())

    with pytest.raises(ConnectionError):
        module.queue_daily_market_pipeline.run("US")

    assert claims.holders == {}


def test_a_failed_release_after_a_failed_publish_keeps_the_publish_error(monkeypatch, caplog):
    from app.tasks import daily_market_pipeline_tasks as module

    claims, published = _ChainClaims(), []
    _ready_to_dispatch(monkeypatch, module, claims, published)

    class _BrokenChain:
        def freeze(self):
            return SimpleNamespace(id="stage-0", parent=None)

        def apply_async(self, **_options):
            raise ConnectionError("broker unavailable")

    def broken_release(_token, *, market):
        raise TimeoutError("redis down")

    monkeypatch.setattr(module, "chain", lambda *_signatures: _BrokenChain())
    monkeypatch.setattr(claims, "release_pipeline_chain", broken_release)

    with pytest.raises(ConnectionError, match="broker unavailable"):
        module.queue_daily_market_pipeline.run("US")
    assert "Could not release the US pipeline chain claim" in caplog.text


def test_dispatch_still_runs_when_the_claim_store_is_down(monkeypatch, caplog):
    # The daily pipeline matters more than duplicate protection: fail open,
    # loudly, as the existing workload-holder check already does.
    from app.tasks import daily_market_pipeline_tasks as module

    claims, published = _ChainClaims(fail_claim=True), []
    _ready_to_dispatch(monkeypatch, module, claims, published)

    result = module.queue_daily_market_pipeline.run("US")

    assert result["status"] == "queued"
    [(signatures, options)] = published
    assert signatures == ["prices", "groups", "snapshot"]
    assert "link_error" not in options
    assert "Could not claim the US pipeline chain" in caplog.text


def test_release_only_frees_its_own_claim(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    claims = _ChainClaims()
    claims.holders["US"] = {"task_id": "newer-chain", "started_at": "x"}
    monkeypatch.setattr(module, "_pipeline_chain_claims", lambda: claims)

    result = module.release_daily_market_pipeline_claim.run(market="US", token="older-chain")

    assert result == {"status": "ok", "market": "US", "stage": "release", "released": False}
    assert claims.holders["US"]["task_id"] == "newer-chain"


def test_release_task_takes_no_positional_args():
    # Celery calls an errback inline with (request, exc, traceback) when its
    # task takes more than one positional argument; keyword-only arguments
    # keep it a published task whose immutable signature ignores the task id.
    import inspect

    from app.tasks import daily_market_pipeline_tasks as module

    params = inspect.signature(module.release_daily_market_pipeline_claim.run).parameters.values()
    assert all(param.kind is inspect.Parameter.KEYWORD_ONLY for param in params)


# ── Heartbeat (#605): keep the claim while the chain lives ───────────────


def _dispatched_chain(monkeypatch, states, live_ids=()):
    """Dispatch a US chain, then report ``states`` for its stage ids (by
    index; unlisted stages read PENDING) and ``live_ids`` as found in the
    broker. Returns the set the broker search reads, to change between beats,
    and the list of searches made."""
    from app.tasks import daily_market_pipeline_tasks as module

    claims, published = _ChainClaims(), []
    _ready_to_dispatch(monkeypatch, module, claims, published)
    assert module.queue_daily_market_pipeline.run("US")["status"] == "queued"
    token = claims.holders["US"]["task_id"]
    stage_ids = claims.task_ids[token]
    by_id = {stage_ids[index]: state for index, state in states.items()}
    monkeypatch.setattr(
        module.celery_app,
        "AsyncResult",
        lambda task_id: SimpleNamespace(state=by_id.get(task_id, "PENDING")),
    )
    live = {stage_ids[index] for index in live_ids}
    searches = []

    def still_in_broker(task_ids, *, cache=True):
        searches.append((list(task_ids), cache))
        return {task_id: task_id in live for task_id in task_ids}

    monkeypatch.setattr(
        "app.services.task_registry_service.still_in_broker", still_in_broker
    )
    return module, claims, token, (live, searches)


def test_a_chain_queued_behind_a_backlog_keeps_its_claim(monkeypatch):
    # Prices finished long ago (its result expired, so it reads PENDING) and
    # the next stage still waits in the data-fetch queue: the chain is alive.
    module, claims, token, _ = _dispatched_chain(monkeypatch, states={}, live_ids=(1,))

    for _beat in range(30):  # 2.5 hours of heartbeats
        assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "renewed"}

    assert claims.holders["US"]["task_id"] == token
    assert claims.released == []
    assert module.queue_daily_market_pipeline.run("US")["reason"] == "market_pipeline_already_active"


def test_a_running_or_retrying_stage_keeps_the_claim_without_reading_the_broker(monkeypatch):
    # PROGRESS: a running price refresh reports progress through update_state,
    # and its acked message is in no queue, so only its state shows it runs.
    for state in ("STARTED", "RETRY", "PROGRESS"):
        module, claims, _token, (_live, searches) = _dispatched_chain(
            monkeypatch, states={0: "SUCCESS", 1: state}
        )

        assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "renewed"}
        assert searches == []


def test_a_revoked_stage_frees_the_claim_within_one_heartbeat(monkeypatch):
    module, claims, token, _ = _dispatched_chain(
        monkeypatch, states={0: "SUCCESS", 1: "REVOKED"}, live_ids=(1,)
    )

    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "released_dead"}
    assert claims.released == [("US", token)]
    assert module.queue_daily_market_pipeline.run("US")["status"] == "queued"


def test_a_failed_stage_or_a_finished_chain_frees_the_claim(monkeypatch):
    module, claims, _token, _ = _dispatched_chain(monkeypatch, states={0: "FAILURE"})
    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "released_dead"}

    module, claims, _token, _ = _dispatched_chain(
        monkeypatch, states={0: "SUCCESS", 1: "SUCCESS", 2: "SUCCESS", 3: "SUCCESS"}
    )
    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "released_done"}


def test_a_chain_with_no_trace_is_freed_on_the_next_heartbeat_only(monkeypatch):
    # One check can fall between a worker taking a stage and storing its state.
    from app.tasks.workload_coordination import PIPELINE_CHAIN_HEARTBEAT_SECONDS

    module, claims, token, _ = _dispatched_chain(monkeypatch, states={0: "SUCCESS"})

    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "missing"}
    assert claims.holders["US"]["task_id"] == token
    # Renewed while unconfirmed, so a claim near its TTL (after a stretch of
    # failed checks) survives until the confirming check.
    assert claims.renewed == [("US", token)]

    claims.clock += PIPELINE_CHAIN_HEARTBEAT_SECONDS
    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "released_missing"}
    assert claims.holders == {}


def test_overlapping_heartbeats_cannot_both_count_one_gap(monkeypatch):
    # A slow heartbeat and the next one, both landing in the same handoff gap.
    module, claims, token, _ = _dispatched_chain(monkeypatch, states={0: "SUCCESS"})

    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "missing"}
    claims.clock += 5
    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "missing"}
    assert claims.holders["US"]["task_id"] == token


def test_a_stale_missing_release_keeps_a_chain_seen_alive_meanwhile(monkeypatch):
    # #627: heartbeat A reads an aged missing marker; before it releases, an
    # overlapping heartbeat B sees a live stage, clears the marker and renews.
    # A's release must then leave the claim alone.
    from app.tasks.workload_coordination import PIPELINE_CHAIN_HEARTBEAT_SECONDS

    module, claims, token, _ = _dispatched_chain(monkeypatch, states={0: "SUCCESS"})
    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "missing"}
    claims.clock += PIPELINE_CHAIN_HEARTBEAT_SECONDS

    read_marker = claims.pipeline_chain_missing

    def read_then_beat_b_sees_it_alive(chain_token):
        result = read_marker(chain_token)
        claims.clear_pipeline_chain_missing(chain_token)
        claims.renew_pipeline_chain(chain_token, market="US")
        return result

    monkeypatch.setattr(claims, "pipeline_chain_missing", read_then_beat_b_sees_it_alive)

    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "kept"}
    assert claims.holders["US"]["task_id"] == token

    # A marker replaced by a later miss (B cleared it, C started a new one)
    # is not the one A read either.
    def read_then_marker_replaced(chain_token):
        result = read_marker(chain_token)
        claims.missing_since[chain_token] = claims.clock
        return result

    claims.missing_since[token] = claims.clock - PIPELINE_CHAIN_HEARTBEAT_SECONDS
    monkeypatch.setattr(claims, "pipeline_chain_missing", read_then_marker_replaced)
    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "kept"}
    assert claims.holders["US"]["task_id"] == token


def test_heartbeats_never_overlap(monkeypatch):
    # #627: a heartbeat reads a chain's states seconds before it acts on them,
    # so a second one acting on an older read could free a claim the first
    # just saw alive. A heartbeat that finds another running skips the beat.
    module, claims, token, _ = _dispatched_chain(monkeypatch, states={1: "REVOKED"})
    claims.heartbeat_holder = "other-heartbeat"

    assert module.renew_daily_market_pipeline_claims.run() == {
        "status": "skipped", "reason": "heartbeat_running", "claims": {}
    }
    assert claims.holders["US"]["task_id"] == token
    assert claims.released == [] and claims.renewed == []

    claims.heartbeat_holder = None
    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "released_dead"}
    assert claims.heartbeat_holder is None  # released after the beat


def test_a_heartbeat_whose_lock_expired_cannot_free_a_missing_chain(monkeypatch):
    # A heartbeat slower than its lock (e.g. stuck in result-backend retries)
    # overlaps the next one again; the next may have just seen the chain
    # alive without clearing the marker yet. Releases are fenced on the lock.
    from app.tasks.workload_coordination import PIPELINE_CHAIN_HEARTBEAT_SECONDS

    module, claims, token, _ = _dispatched_chain(monkeypatch, states={0: "SUCCESS"})
    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "missing"}
    claims.clock += PIPELINE_CHAIN_HEARTBEAT_SECONDS

    read_marker = claims.pipeline_chain_missing

    def read_then_lock_taken_over(chain_token):
        result = read_marker(chain_token)
        claims.heartbeat_holder = "next-heartbeat"  # ours expired; the next took it
        return result

    monkeypatch.setattr(claims, "pipeline_chain_missing", read_then_lock_taken_over)

    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "kept"}
    assert claims.holders["US"]["task_id"] == token
    assert claims.heartbeat_holder == "next-heartbeat"  # not ours to release


def test_a_heartbeat_that_raises_still_frees_the_heartbeat_lock(monkeypatch):
    module, claims, _token, _ = _dispatched_chain(monkeypatch, states={})

    def holders_down():
        raise ConnectionError("redis down")

    monkeypatch.setattr(claims, "get_pipeline_chain_holders", holders_down)
    with pytest.raises(ConnectionError):
        module.renew_daily_market_pipeline_claims.run()
    assert claims.heartbeat_holder is None


def test_a_live_heartbeat_resets_the_missing_clock(monkeypatch):
    from app.tasks.workload_coordination import PIPELINE_CHAIN_HEARTBEAT_SECONDS

    module, claims, token, (live, _searches) = _dispatched_chain(monkeypatch, states={})

    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "missing"}
    claims.clock += PIPELINE_CHAIN_HEARTBEAT_SECONDS
    live.add(claims.task_ids[token][1])
    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "renewed"}
    claims.clock += PIPELINE_CHAIN_HEARTBEAT_SECONDS
    live.clear()
    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "missing"}
    assert claims.holders["US"]["task_id"] == token


def test_a_broker_search_out_of_budget_renews_the_claim(monkeypatch):
    # still_in_broker answers None when the broker is unreadable or the
    # backlog is too big to search in its budget. A stage buried that deep is
    # likely still queued, so renew: only renewing keeps a backlog longer than
    # the 48h TTL from letting a second chain start. A conclusive search once
    # the backlog drains frees a dead chain's claim.
    from app.tasks.workload_coordination import PIPELINE_CHAIN_HEARTBEAT_SECONDS

    from app.services import task_registry_service as registry

    module, claims, token, _ = _dispatched_chain(monkeypatch, states={})
    real_search = registry.still_in_broker  # the fake that finds nothing
    inconclusive = {"on": False}

    def search(task_ids, *, cache=True):
        if inconclusive["on"]:
            return {task_id: None for task_id in task_ids}
        return real_search(task_ids, cache=cache)

    monkeypatch.setattr("app.services.task_registry_service.still_in_broker", search)

    # One miss starts the missing clock...
    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "missing"}
    inconclusive["on"] = True
    for _beat in range(3):
        claims.clock += PIPELINE_CHAIN_HEARTBEAT_SECONDS
        assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "renewed_unknown"}
    # The unconfirmed miss renews too, then each inconclusive beat.
    assert claims.renewed == [("US", token)] * 4
    # ...which an inconclusive search clears, since it proved nothing gone.
    assert claims.missing_since == {}
    assert claims.holders["US"]["task_id"] == token


def test_undecided_chains_share_one_fresh_broker_search(monkeypatch):
    module, claims, token, (_live, searches) = _dispatched_chain(monkeypatch, states={})
    claims.holders["HK"] = {"task_id": "hk-token", "started_at": "x"}
    claims.task_ids["hk-token"] = ["hk-0", "hk-1"]

    module.renew_daily_market_pipeline_claims.run()

    # One bounded search for every undecided chain, never from the cache: a
    # cached "not found" from before a stage was published would read a live
    # chain as missing.
    [(searched, cache)] = searches
    assert sorted(searched) == sorted([*claims.task_ids[token], "hk-0", "hk-1"])
    assert cache is False


def test_a_failing_heartbeat_keeps_the_claim(monkeypatch, caplog):
    module, claims, token, _ = _dispatched_chain(monkeypatch, states={})

    def backend_down(_task_id):
        raise ConnectionError("result backend down")

    monkeypatch.setattr(module.celery_app, "AsyncResult", backend_down)

    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "error"}
    assert claims.holders["US"]["task_id"] == token
    assert claims.released == []
    assert "Could not check the US pipeline chain claim" in caplog.text


def test_a_claim_taken_before_stage_ids_were_recorded_is_left_to_expire(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    claims = _ChainClaims()
    claims.holders["US"] = {"task_id": "pre-605", "started_at": "x"}
    monkeypatch.setattr(module, "_pipeline_chain_claims", lambda: claims)

    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "untracked"}
    assert claims.holders["US"]["task_id"] == "pre-605"


def test_the_heartbeat_never_renews_or_frees_a_newer_chains_claim(monkeypatch):
    # The claim changed hands after the holder was read.
    module, claims, token, _ = _dispatched_chain(monkeypatch, states={1: "REVOKED"})
    claims.holders["US"] = {"task_id": "newer", "started_at": "y"}
    monkeypatch.setattr(
        claims, "get_pipeline_chain_holders", lambda: {"US": {"task_id": token}}
    )

    assert module.renew_daily_market_pipeline_claims.run()["claims"] == {"US": "lost"}
    assert claims.holders["US"]["task_id"] == "newer"


@pytest.mark.allow_real_celery_dispatch
def test_real_celery_chain_publishes_with_the_recorded_ids(monkeypatch):
    # Celery's own chain preparation and message building, with the network
    # seams stubbed: the broker producer is in memory, the result backend's
    # subscribe is a no-op, and send_task_message is captured. CI has no
    # Redis, so a test reaching either service fails there.
    from contextlib import nullcontext

    from celery import chain
    from kombu import Connection

    from app.tasks import daily_market_pipeline_tasks as module

    memory = Connection("memory://")
    monkeypatch.setattr(
        module.celery_app, "producer_or_acquire", lambda _producer=None: nullcontext(memory.Producer())
    )
    monkeypatch.setattr(module.celery_app.backend, "on_task_call", lambda *_args, **_kwargs: None)

    release = module.release_daily_market_pipeline_claim.si(market="US", token="tok")
    workflow = chain(
        module.guard_price_refresh.s(market="US"),
        module.guard_group_result.s(market="US"),
        release,
    )
    task_ids = module._chain_task_ids(workflow)

    published = []
    monkeypatch.setattr(
        module.celery_app.amqp,
        "send_task_message",
        lambda _producer, _name, message, *args, **kwargs: published.append(message),
    )
    workflow.apply_async(link_error=release)
    memory.release()

    [first] = published
    _args, _kwargs, embed = first.body
    later = [signature["options"]["task_id"] for signature in reversed(embed["chain"])]
    assert [first.headers["id"], *later] == task_ids
    assert len(set(task_ids)) == 3


def test_guard_price_refresh_fails_lock_contention_skip_result():
    from app.tasks import daily_market_pipeline_tasks as module

    result = {
        "status": "already_running",
        "skipped": True,
        "reason": "data_fetch_lock_active",
    }

    with pytest.raises(RuntimeError, match="Daily price refresh failed for HK"):
        module.guard_price_refresh.run(result, market="HK")


def test_snapshot_guard_preserves_source_identity_for_options_follow_on():
    from app.tasks import daily_market_pipeline_tasks as module

    result = module.guard_snapshot_result.run(
        {
            "auto_scan_id": "scan-1",
            "run_id": 33,
            "as_of_date": "2026-03-16",
        },
        market="US",
    )

    assert result["auto_scan_id"] == "scan-1"
    assert result["run_id"] == 33
    assert result["as_of_date"] == "2026-03-16"


def test_options_follow_on_is_fire_and_report(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    dispatched = []
    monkeypatch.setattr(
        "app.interfaces.tasks.options_analytics_tasks.refresh_options_analytics.apply_async",
        lambda **values: dispatched.append(values) or SimpleNamespace(id="options-1"),
    )

    result = module.dispatch_options_after_snapshot.run(
        {"run_id": 33, "as_of_date": "2026-03-16", "auto_scan_id": "scan-1"},
        market="US",
    )

    assert result["options_dispatch"] == {"status": "queued", "task_id": "options-1"}
    assert dispatched[0]["queue"] == "data_fetch_us"


def test_options_follow_on_uses_already_published_snapshot_identity(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    dispatched = []
    monkeypatch.setattr(
        "app.interfaces.tasks.options_analytics_tasks.refresh_options_analytics.apply_async",
        lambda **values: dispatched.append(values) or SimpleNamespace(id="options-1"),
    )
    guarded = module.guard_snapshot_result.run(
        {
            "status": "skipped",
            "reason": "already_published",
            "existing_run_id": 123,
            "auto_scan_id": 456,
        },
        market="US",
    )

    module.dispatch_options_after_snapshot.run(guarded, market="US")

    assert dispatched[0]["kwargs"] == {"source_run_id": 123, "market": "US"}


def test_guard_price_refresh_accepts_high_coverage_partial_result():
    from app.tasks import daily_market_pipeline_tasks as module

    result = {
        "status": "partial",
        "refreshed": 900,
        "failed": 100,
        "total": 1000,
    }

    assert module.guard_price_refresh.run(result, market="HK") == {
        "status": "ok",
        "market": "HK",
        "stage": "prices",
    }


def test_guard_price_refresh_uses_coverage_denominator_when_present():
    from app.tasks import daily_market_pipeline_tasks as module

    result = {
        "status": "partial",
        "refreshed": 50,
        "failed": 34,
        "total": 84,
        "coverage_refreshed": 9949,
        "coverage_failed": 34,
        "coverage_total": 9983,
        "live_top_up_refreshed": 50,
        "live_top_up_failed": 34,
        "live_top_up_total": 84,
    }

    assert module.guard_price_refresh.run(result, market="US") == {
        "status": "ok",
        "market": "US",
        "stage": "prices",
    }


def test_guard_price_refresh_falls_back_when_coverage_values_are_null():
    from app.tasks import daily_market_pipeline_tasks as module

    result = {
        "status": "partial",
        "refreshed": 900,
        "failed": 100,
        "total": 1000,
        "coverage_refreshed": None,
        "coverage_total": None,
    }

    assert module.guard_price_refresh.run(result, market="HK") == {
        "status": "ok",
        "market": "HK",
        "stage": "prices",
    }


def test_guard_price_refresh_fails_low_coverage_partial_result():
    from app.tasks import daily_market_pipeline_tasks as module

    result = {
        "status": "partial",
        "refreshed": 899,
        "failed": 101,
        "total": 1000,
    }

    with pytest.raises(RuntimeError, match="Daily price refresh failed for HK"):
        module.guard_price_refresh.run(result, market="HK")


def test_guard_snapshot_result_accepts_already_published_scan_result():
    from app.tasks import daily_market_pipeline_tasks as module

    result = {
        "status": "skipped",
        "reason": "already_published",
        "existing_run_id": 123,
        "auto_scan_id": 456,
    }

    assert module.guard_snapshot_result.run(result, market="HK") == {
        "status": "ok",
        "market": "HK",
        "stage": "scan",
        "auto_scan_id": 456,
        "run_id": 123,
    }


def test_guard_group_result_accepts_no_taxonomy_market_skip():
    from app.tasks import daily_market_pipeline_tasks as module

    result = {
        "status": "skipped",
        "reason": "no_taxonomy_for_market",
        "market": "JP",
    }

    assert module.guard_group_result.run(result, market="JP") == {
        "status": "ok",
        "market": "JP",
        "stage": "groups",
    }


def test_guard_exposure_result_skips_without_aborting_pipeline():
    # Exposure is a non-critical leaf: a failed/missing exposure result must NOT
    # raise (which would abort groups + snapshot); it returns a skipped status.
    from app.tasks import daily_market_pipeline_tasks as module

    result = {"error": "benchmark_not_current", "market": "US", "date": "2026-06-19"}
    assert module.guard_exposure_result.run(result, market="US") == {
        "status": "skipped",
        "market": "US",
        "stage": "exposure",
    }


def test_guard_market_rs_blocks_failed_result():
    from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
    from app.tasks import daily_market_pipeline_tasks as module

    with pytest.raises(RuntimeError, match="Canonical Market RS failed for US"):
        module.guard_market_rs_result.run(
            {
                "status": "failed",
                "market": "US",
                "as_of_date": "2026-04-10",
                "formula_version": BALANCED_RS_FORMULA_VERSION,
                "reason_code": "benchmark_anchor_missing",
            },
            market="US",
            calculation_date="2026-04-10",
        )


def test_guard_market_rs_accepts_exact_completed_balanced_run():
    from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
    from app.tasks import daily_market_pipeline_tasks as module

    assert module.guard_market_rs_result.run(
        {
            "status": "completed",
            "market": "US",
            "as_of_date": "2026-04-10",
            "formula_version": BALANCED_RS_FORMULA_VERSION,
            "market_rs_run_id": 42,
            "eligible_symbol_count": 5000,
        },
        market="US",
        calculation_date="2026-04-10",
    ) == {
        "status": "ok",
        "market": "US",
        "stage": "market_rs",
        "as_of_date": "2026-04-10",
        "formula_version": BALANCED_RS_FORMULA_VERSION,
        "market_rs_run_id": 42,
    }


# --- The primary benchmark's as-of bar: refetch once, then a bounded re-run ---


def _price_rows(monkeypatch, module, rows):
    """Point the module's SessionLocal at an in-memory StockPrice table.

    ``rows`` are (symbol, date, adj_close). Every ``SessionLocal()`` opens a new
    connection, so the engine shares one connection across them.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.database import Base
    from app.models.stock import StockPrice

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine, tables=[StockPrice.__table__])
    session_factory = sessionmaker(bind=engine)
    monkeypatch.setattr(module, "SessionLocal", session_factory)

    def insert(symbol, day, adj_close):
        with session_factory() as session:
            session.add(StockPrice(
                symbol=symbol, date=day, open=1.0, high=1.0, low=1.0, close=1.0, volume=1,
                adj_close=adj_close,
            ))
            session.commit()

    for symbol, day, adj_close in rows:
        insert(symbol, day, adj_close)
    return insert


def test_ensure_benchmark_prices_passes_a_present_bar_through_without_a_refetch(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    _price_rows(monkeypatch, module, [("SPY", date(2026, 3, 16), 500.0)])
    fetches = []
    monkeypatch.setattr(module, "_refetch_prices", lambda market, symbols: fetches.append((market, symbols)))

    result = module.ensure_benchmark_prices.run(
        {"status": "completed", "refreshed": 10, "total": 10}, market="US", as_of_date="2026-03-16",
    )

    assert result == {
        "status": "completed",
        "refreshed": 10,
        "total": 10,
        "benchmark": {"status": "present", "symbol": "SPY", "as_of_date": "2026-03-16", "refetched": False},
    }
    assert fetches == []


def test_ensure_benchmark_prices_refetches_the_candidates_when_the_primary_bar_is_missing(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    insert = _price_rows(monkeypatch, module, [("SPY", date(2026, 3, 13), 499.0)])
    fetches = []

    def refetch_supplies_the_bar(market, symbols):
        fetches.append((market, symbols))
        insert("SPY", date(2026, 3, 16), 500.0)

    monkeypatch.setattr(module, "_refetch_prices", refetch_supplies_the_bar)

    result = module.ensure_benchmark_prices.run(
        {"status": "completed", "refreshed": 10, "total": 10}, market="US", as_of_date="2026-03-16",
    )

    assert result["benchmark"] == {
        "status": "present", "symbol": "SPY", "as_of_date": "2026-03-16", "refetched": True,
    }
    assert fetches == [("US", ("SPY", "IVV", "^VIX"))]


def test_ensure_benchmark_prices_reports_a_bar_still_missing_after_the_refetch(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    _price_rows(monkeypatch, module, [("SPY", date(2026, 3, 13), 499.0)])
    monkeypatch.setattr(module, "_refetch_prices", lambda market, symbols: None)

    result = module.ensure_benchmark_prices.run(
        {"status": "completed", "refreshed": 10, "total": 10}, market="US", as_of_date="2026-03-16",
    )

    assert result["benchmark"] == {
        "status": "missing", "symbol": "SPY", "as_of_date": "2026-03-16", "refetched": True,
    }


def test_ensure_benchmark_prices_ignores_a_bar_without_a_usable_adjusted_close(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    _price_rows(monkeypatch, module, [("SPY", date(2026, 3, 16), None)])
    fetches = []
    monkeypatch.setattr(module, "_refetch_prices", lambda market, symbols: fetches.append(symbols))

    result = module.ensure_benchmark_prices.run(
        {"status": "completed", "refreshed": 10, "total": 10}, market="US", as_of_date="2026-03-16",
    )

    assert result["benchmark"]["status"] == "missing"
    assert fetches == [("SPY", "IVV", "^VIX")]


def test_ensure_benchmark_prices_reports_a_refetch_that_raises_instead_of_ending_the_chain(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    _price_rows(monkeypatch, module, [])

    def yahoo_down(market, symbols):
        raise ConnectionError("yahoo down")

    monkeypatch.setattr(module, "_refetch_prices", yahoo_down)

    result = module.ensure_benchmark_prices.run(
        {"status": "completed", "refreshed": 10, "total": 10}, market="US", as_of_date="2026-03-16",
    )

    assert result["benchmark"] == {
        "status": "missing",
        "symbol": "SPY",
        "as_of_date": "2026-03-16",
        "refetched": True,
        "error": "yahoo down",
    }


def test_ensure_benchmark_prices_leaves_a_failed_refresh_for_its_guard(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    _price_rows(monkeypatch, module, [])
    fetches = []
    monkeypatch.setattr(module, "_refetch_prices", lambda market, symbols: fetches.append(symbols))
    failed = {"status": "failed", "error": "boom", "refreshed": 0, "failed": 10}

    result = module.ensure_benchmark_prices.run(failed, market="US", as_of_date="2026-03-16")

    assert result == {"status": "failed", "error": "boom", "refreshed": 0, "failed": 10}
    assert fetches == []


def test_refetch_prices_runs_the_candidates_through_the_bulk_price_path(monkeypatch):
    import pandas as pd

    from app.tasks import daily_market_pipeline_tasks as module

    spy = pd.DataFrame({"Close": [500.0]}, index=pd.to_datetime(["2026-03-16"]))
    vix = pd.DataFrame({"Close": [15.0]}, index=pd.to_datetime(["2026-03-16"]))
    fetched, stored = [], []

    class _FakeBulkFetcher:
        def fetch_prices_in_batches(self, symbols, *, period, market):
            fetched.append((symbols, period, market))
            return {
                "SPY": {"symbol": "SPY", "price_data": spy, "has_error": False},
                "IVV": {"symbol": "IVV", "price_data": None, "has_error": True, "error": "No data"},
                "^VIX": {"symbol": "^VIX", "price_data": vix, "has_error": False},
            }

    class _FakePriceCache:
        def store_refreshed_batch(self, batch, *, period, market_by_symbol):
            stored.append((batch, period, market_by_symbol))
            return len(batch)

    monkeypatch.setattr("app.services.bulk_data_fetcher.BulkDataFetcher", _FakeBulkFetcher)
    monkeypatch.setattr("app.wiring.bootstrap.get_price_cache", lambda: _FakePriceCache())

    module._refetch_prices("US", ("SPY", "IVV", "^VIX"))

    assert fetched == [(["SPY", "IVV", "^VIX"], "1y", "US")]
    [(batch, period, market_by_symbol)] = stored
    assert batch == {"SPY": spy, "^VIX": vix}
    assert period == "1y"
    assert market_by_symbol == {"SPY": "US", "^VIX": "US"}


def _guard_with_benchmark(monkeypatch, module, *, status, rerun_attempt, market_open=False):
    scheduled = []
    monkeypatch.setattr(
        module.queue_daily_market_pipeline,
        "apply_async",
        lambda **options: scheduled.append(options) or SimpleNamespace(id="rerun-1"),
    )
    monkeypatch.setattr(
        module, "_preflight_calendar", lambda: SimpleNamespace(is_market_open=lambda _market: market_open),
    )
    result = {
        "status": "completed",
        "refreshed": 10,
        "total": 10,
        "benchmark": {"status": status, "symbol": "SPY", "as_of_date": "2026-03-16", "refetched": True},
    }
    return (
        lambda: module.guard_price_refresh.run(
            result, market="US", as_of_date="2026-03-16", rerun_attempt=rerun_attempt,
        ),
        scheduled,
    )


def test_guard_price_refresh_passes_when_the_primary_benchmark_bar_is_present(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    guard, scheduled = _guard_with_benchmark(monkeypatch, module, status="present", rerun_attempt=0)

    assert guard() == {"status": "ok", "market": "US", "stage": "prices"}
    assert scheduled == []


def test_guard_price_refresh_schedules_one_rerun_when_the_primary_benchmark_is_missing(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    guard, scheduled = _guard_with_benchmark(monkeypatch, module, status="missing", rerun_attempt=0)

    with pytest.raises(module.PrimaryBenchmarkMissing, match="primary_benchmark_missing: US has no SPY bar for 2026-03-16"):
        guard()

    assert scheduled == [{
        "kwargs": {"market": "US", "rerun_attempt": 1, "rerun_as_of_date": "2026-03-16"},
        "countdown": 45 * 60,
        "queue": "market_jobs_us",
    }]


def test_guard_price_refresh_stops_rerunning_after_the_last_attempt(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    last, scheduled_by_last = _guard_with_benchmark(monkeypatch, module, status="missing", rerun_attempt=1)
    with pytest.raises(module.PrimaryBenchmarkMissing):
        last()
    assert [options["kwargs"]["rerun_attempt"] for options in scheduled_by_last] == [2]

    exhausted, scheduled_by_exhausted = _guard_with_benchmark(monkeypatch, module, status="missing", rerun_attempt=2)
    with pytest.raises(module.PrimaryBenchmarkMissing, match="exhausted"):
        exhausted()
    assert scheduled_by_exhausted == []


def test_guard_price_refresh_never_schedules_a_rerun_while_the_market_is_open(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    guard, scheduled = _guard_with_benchmark(
        monkeypatch, module, status="missing", rerun_attempt=0, market_open=True,
    )

    with pytest.raises(module.PrimaryBenchmarkMissing, match="market_open"):
        guard()
    assert scheduled == []


def test_guard_price_refresh_accepts_a_chain_published_before_the_benchmark_stage(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    scheduled = []
    monkeypatch.setattr(
        module.queue_daily_market_pipeline, "apply_async", lambda **options: scheduled.append(options),
    )

    result = module.guard_price_refresh.run({"status": "completed", "refreshed": 10, "total": 10}, market="US")

    assert result == {"status": "ok", "market": "US", "stage": "prices"}
    assert scheduled == []


def test_a_benchmark_rerun_dispatches_with_its_attempt_number(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    claims, published = _ChainClaims(), []
    _ready_to_dispatch(monkeypatch, module, claims, published)
    built = []
    monkeypatch.setattr(
        module,
        "_build_daily_market_pipeline_signatures",
        lambda market, trading_day, *, rerun_attempt: built.append((market, trading_day, rerun_attempt)) or ["prices"],
    )

    result = module.queue_daily_market_pipeline.run("US", rerun_attempt=1, rerun_as_of_date="2026-03-16")

    assert result["status"] == "queued"
    assert result["as_of_date"] == "2026-03-16"
    assert built == [("US", date(2026, 3, 16), 1)]
    assert len(published) == 1


def test_a_benchmark_rerun_skips_once_a_newer_session_completed(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    claims, published = _ChainClaims(), []
    _ready_to_dispatch(monkeypatch, module, claims, published)

    result = module.queue_daily_market_pipeline.run("US", rerun_attempt=1, rerun_as_of_date="2026-03-13")

    assert result["status"] == "skipped"
    assert result["reason"] == "rerun_session_advanced"
    assert result["as_of_date"] == "2026-03-16"
    assert result["rerun_as_of_date"] == "2026-03-13"
    assert published == []
    assert claims.holders == {}


def test_a_benchmark_rerun_never_runs_during_a_session(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    claims, published = _ChainClaims(), []
    _ready_to_dispatch(monkeypatch, module, claims, published, market_open=True)

    result = module.queue_daily_market_pipeline.run("US", rerun_attempt=1, rerun_as_of_date="2026-03-16")

    assert result == {
        "status": "skipped",
        "reason": "market_open",
        "market": "US",
        "as_of_date": "2026-03-16",
        "timestamp": result["timestamp"],
    }
    assert published == []
    assert claims.holders == {}


def test_a_benchmark_rerun_cannot_double_run_a_pipeline_still_holding_the_claim(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    claims, published = _ChainClaims(), []
    _ready_to_dispatch(monkeypatch, module, claims, published)

    first = module.queue_daily_market_pipeline.run("US")
    rerun = module.queue_daily_market_pipeline.run("US", rerun_attempt=1, rerun_as_of_date="2026-03-16")

    assert first["status"] == "queued"
    assert rerun["status"] == "skipped"
    assert rerun["reason"] == "market_pipeline_already_active"
    assert rerun["active_chain_token"] == claims.holders["US"]["task_id"]
    assert len(published) == 1
