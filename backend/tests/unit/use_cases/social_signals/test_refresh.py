from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.domain.social_signals.records import (
    BacklogResult,
    ConfirmationFacts,
    ConfirmationInput,
    DailyFreshness,
    GroupConfirmationContext,
    MarketConfirmationBatch,
    MarketConfirmationContext,
    PinnedFeatureRun,
    ReplayInput,
    SavedSocialRunInputs,
    SocialCurrentInputManifest,
    SocialCollectionProgress,
    SocialEvidenceInput,
    SocialPostRecord,
    SocialReadRequest,
    SocialRunResult,
    SocialSourceBatch,
    SocialSourceOutcome,
    SocialSourceView,
)
from app.services.social_source_admin_service import SocialRuntimeState


NOW = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)


def source(source_id: str, list_id: str) -> SocialSourceView:
    return SocialSourceView(
        source_id, f"List {source_id}", f"https://x.com/i/lists/{list_id}", list_id,
        "enabled", "system_seed", None, None, 1, NOW, NOW,
    )


SOURCES = (source("1", "111"), source("2", "222"))


class Catalog:
    def __init__(self, mode="live", provider="official", progress=(), current=True):
        self.snapshot = (SocialRuntimeState(mode, provider, 7), SOURCES, "model-x")
        self.progress = dict(progress)
        self.current = current
        self.reads = 0

    def read(self):
        self.reads += 1
        return self.snapshot

    def admitted(self, *, version, mode, provider):
        return self.current and (version, mode, provider) == (7, self.snapshot[0].mode, self.snapshot[0].provider)


class Lease:
    def __init__(self, acquired=True, events=None, renew_results=None):
        self.acquired = acquired
        self.events = events if events is not None else []
        self.renew_results = list(renew_results or ())

    def acquire(self, owner, ttl_seconds):
        self.events.append("lease.acquire")
        return self.acquired

    def renew(self, owner, ttl_seconds):
        self.events.append("lease.renew")
        return self.renew_results.pop(0) if self.renew_results else self.acquired

    def release(self, owner):
        self.events.append("lease.release")


class Provider:
    def __init__(self, events, *, fail_sources=()):
        self.events = events
        self.fail_sources = set(fail_sources)
        self.requests = []

    def read_source(self, request):
        self.events.append(f"provider.{request.source_id}")
        self.requests.append(request)
        if request.source_id in self.fail_sources:
            outcome = SocialSourceOutcome(
                "failed", "failed", "limited", ("provider_error",), (), None, None,
                0, None, "provider_error",
            )
            return SocialSourceBatch(request, (), outcome)
        post = SocialPostRecord(
            "official", f"p-{request.source_id}", request.source_id, "$AAA cooling",
            f"https://x.com/a/status/p-{request.source_id}", "author",
            NOW - timedelta(days=2), NOW, likes=10, reposts=1, replies=1,
        )
        outcome = SocialSourceOutcome(
            "success", "pending", "limited", ("bounded_provider_read",), (),
            post.created_at, post.created_at, 1, None, None,
            proposed_progress=f"cursor-{request.source_id}",
        )
        return SocialSourceBatch(request, (post,), outcome)


class Writer:
    def __init__(self, events, mode="live"):
        self.events = events
        self.mode = mode
        self.batches = []
        self.created = []
        self.prepared = None
        self.collection_progress = {
            "2": SocialCollectionProgress(initial_complete=True, cursor="prior-cursor")
        }
        self.failures = []
        self.analysis_failed = False
        self.retained = ()

    def create_run(self, run_id, as_of):
        self.events.append("writer.create")
        self.created.append((run_id, as_of))
        return run_id

    def create_replay_run(self, run_id, saved_run_id, as_of):
        self.events.append("writer.replay")
        self.created.append((run_id, as_of))
        return run_id

    def resume_existing_run(self, run_id, expected_version):
        return None

    def latest_collection_progress(self, source_id, provider):
        return self.collection_progress.get(source_id)

    def persist_observations(self, batch, *, run_id=None):
        self.events.append(f"writer.persist.{batch.request.source_id}")
        self.batches.append(batch)
        return replace(batch, outcome=replace(batch.outcome, committed_progress=batch.outcome.proposed_progress))

    def select_current_inputs(self, run_id):
        self.events.append("writer.select")
        content_ids = tuple((batch.posts[0].provider_post_id, index + 10)
                            for index, batch in enumerate(self.batches) if batch.posts)
        required = tuple(ReplayInput(content_id, f"hash-{post_id}", "current")
                         for post_id, content_id in content_ids)
        manifest = (SocialCurrentInputManifest((), (), (), ("analysis_unavailable",),
                                               scoring_work_ids=tuple(range(len(self.retained))),
                                               unavailable_inputs=required)
                    if self.analysis_failed else SocialCurrentInputManifest(required, (), ()))
        return SavedSocialRunInputs(
            run_id, NOW, 7, tuple(self.batches), content_ids, (), current_manifest=manifest,
        )

    def current_inputs_ready(self, run_id):
        return True

    def fail_run(self, run_id, reason_code):
        self.events.append("writer.fail_run")
        self.failures.append((run_id, reason_code))

    def prepare_run(self, run_id, rows, as_of, *, theme_evidence=(), context=None):
        self.events.append("writer.prepare")
        self.prepared = (rows, theme_evidence, context)

    def publish(self, run_id, expected_mode_version):
        self.events.append("writer.publish")
        return SocialRunResult(run_id, self.mode, "complete", self.mode == "live",
                               (("1", "limited"), ("2", "limited")))


class Backlog:
    def __init__(self, events, *, deferred=0, failed=0):
        self.events = events
        self.deferred = deferred
        self.failed = failed
        self.enqueued = []

    def enqueue(self, content_item_id, post, **kwargs):
        self.events.append("backlog.enqueue")
        self.enqueued.append((content_item_id, post, kwargs))
        return content_item_id

    async def execute(self, now, limit, admin_work_ids=(), work_ids=()):
        self.events.append("backlog.execute")
        assert tuple(value[0] for value in self.enqueued) == work_ids
        return BacklogResult(
            max(0, limit - self.deferred - self.failed),
            self.deferred,
            self.failed,
            0,
            NOW + timedelta(hours=12),
        )


class EvidenceReader:
    def __init__(self, events):
        self.events = events

    def read(self, run_id, as_of):
        self.events.append("evidence.read")
        posts = () if self.writer.analysis_failed else tuple(
            post for source_batch in self.writer.batches for post in source_batch.posts)
        posts += self.writer.retained
        return (SocialEvidenceInput("US:AAA", "AAA", "US", posts, ("1", "2"), False,
                                    ("bounded_provider_read",)),)


class ThemeService:
    class Projection:
        proposals = ()

    class Application:
        baskets = ()

    def __init__(self, events):
        self.events = events

    def prepare(self, run_id, now):
        self.events.append("theme.prepare")
        return self.Projection()

    def prepare_application(self, projection, *, theme_keys=()):
        self.events.append("theme.application")
        return self.Application()


class ConfirmationReader:
    def __init__(self, events):
        self.events = events

    def pin_feature_run(self, market):
        return PinnedFeatureRun(market, 3)

    def read_market(self, market, symbols, now, *, theme_keys=None, membership_reader=None,
                    pinned_run=None):
        self.events.append(f"confirmation.{market}")
        fresh = DailyFreshness(date(2026, 9, 4), date(2026, 9, 4), True)
        facts = ConfirmationFacts(3, 4, fresh, fresh, "SPY", Decimal("80"), Decimal("60"), (),
                                  Decimal("85"), True, Decimal("70"), Decimal("80"), True,
                                  Decimal("10000000"))
        value = ConfirmationInput("US:AAA", "US", now, Decimal("85"), Decimal("70"),
                                  Decimal("80"), 1, 2, market_benchmark="SPY")
        return MarketConfirmationBatch(
            pinned_run or PinnedFeatureRun("US", 3),
            MarketConfirmationContext("US", now, 4, fresh, Decimal("60"), "SPY", ("SPY",), "v1"),
            GroupConfirmationContext(date(2026, 9, 4), "g1", 9, (("Software", 1), ("Banks", 2))),
            (value,), (("AAA", facts),),
        )


def use_case(*, mode="live", fail_sources=(), deferred=0, failed=0, current=True):
    from app.use_cases.social_signals.refresh import RefreshSocialSignals

    events = []
    catalog = Catalog(mode=mode, current=current)
    provider = Provider(events, fail_sources=fail_sources)
    writer = Writer(events, mode=mode)
    writer.analysis_failed = bool(failed)
    backlog = Backlog(events, deferred=deferred, failed=failed)
    evidence = EvidenceReader(events)
    evidence.writer = writer
    lease = Lease(events=events)
    refresh = RefreshSocialSignals(
        catalog=catalog,
        providers={"official": provider},
        writer=writer,
        backlog=backlog,
        evidence_reader=evidence,
        theme_service=ThemeService(events),
        confirmation_reader=ConfirmationReader(events),
        provider_lease=lease,
        run_id_factory=lambda origin, now: f"{origin}-{now:%Y%m%d%H%M}",
        input_hash=lambda post: f"hash-{post.provider_post_id}",
    )
    return refresh, events, provider, writer, backlog, catalog


@pytest.mark.asyncio
async def test_off_and_disabled_do_no_automatic_io():
    refresh, events, provider, writer, _, _ = use_case(mode="off")
    result = await refresh.execute("scheduled", NOW)
    assert result.processing_status == "skipped"
    assert not events and not provider.requests and not writer.created


@pytest.mark.asyncio
async def test_refresh_orders_collection_before_metered_work_and_publishes_all_windows():
    refresh, events, provider, writer, backlog, _ = use_case()
    result = await refresh.execute("scheduled", NOW)

    assert result.published is True
    assert [(r.intent, r.limit, r.application_progress) for r in provider.requests] == [
        ("initial", 1000, None), ("incremental", 200, None)
    ]
    assert events.index("lease.release") < events.index("backlog.execute")
    assert events.index("evidence.read") < events.index("confirmation.US") < events.index("writer.prepare")
    rows, _, context = writer.prepared
    assert {(row.candidate_key, row.window_days) for row in rows} == {
        ("US:AAA", 1), ("US:AAA", 7), ("US:AAA", 14)
    }
    assert len(context.candidates) == 3
    assert all(item[2]["run_id"] == "scheduled-202609071200" for item in backlog.enqueued)


@pytest.mark.asyncio
async def test_one_failed_list_is_a_coverage_gap_and_the_other_lists_still_publish():
    refresh, events, _, writer, backlog, _ = use_case(fail_sources={"2"})
    result = await refresh.execute("scheduled", NOW)
    assert result.published is True
    assert len(writer.batches) == 2
    assert [post.source_id for _, post, _ in backlog.enqueued] == ["1"]
    assert "writer.publish" in events


@pytest.mark.asyncio
async def test_every_list_failing_blocks_analysis_and_publication_but_keeps_audit_batches():
    refresh, events, _, writer, backlog, _ = use_case(fail_sources={"1", "2"})
    result = await refresh.execute("scheduled", NOW)
    assert result.published is False
    assert result.reason_codes == ("source_participation_failed",)
    assert len(writer.batches) == 2
    assert not backlog.enqueued
    assert "writer.prepare" not in events and "writer.publish" not in events


def test_read_intent_distinguishes_completed_cursorless_initial_from_paged_initial():
    refresh, _, _, writer, _, _ = use_case()
    writer.collection_progress["1"] = SocialCollectionProgress(True, None)
    completed = refresh._request("run", SOURCES[0], "xui", NOW)
    writer.collection_progress["1"] = SocialCollectionProgress(False, "page-2")
    paged = refresh._request("run", SOURCES[0], "official", NOW)

    assert (completed.intent, completed.application_progress) == ("incremental", None)
    assert (paged.intent, paged.application_progress) == ("initial", "page-2")


@pytest.mark.asyncio
@pytest.mark.parametrize("last_read_ago,expected_window", [
    (timedelta(hours=6), timedelta(hours=54)),
    (timedelta(days=20), timedelta(days=14)),
])
async def test_incremental_read_starts_shortly_before_the_last_successful_read(last_read_ago, expected_window):
    refresh, _, provider, writer, _, _ = use_case()
    writer.collection_progress["2"] = SocialCollectionProgress(
        initial_complete=True, cursor="prior-cursor", last_read_at=NOW - last_read_ago)

    await refresh.execute("scheduled", NOW)

    request = next(value for value in provider.requests if value.source_id == "2")
    assert (request.intent, request.target_published_after) == ("incremental", NOW - expected_window)


@pytest.mark.asyncio
async def test_provider_lease_is_renewed_before_each_source_read():
    refresh, events, provider, writer, backlog, catalog = use_case()

    await refresh.execute("scheduled", NOW)

    assert events.count("lease.renew") == len(SOURCES)


@pytest.mark.asyncio
async def test_budget_pause_keeps_work_and_does_not_prepare():
    refresh, events, _, writer, backlog, _ = use_case(deferred=1)
    result = await refresh.execute("scheduled", NOW)
    assert result.processing_status == "deferred"
    assert backlog.enqueued
    assert "writer.prepare" not in events
    assert writer.failures == []


def lose_lease(refresh, *renewals):
    refresh.provider_lease.renew_results = list(renewals)


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt,status,reason,closes", [
    (lambda refresh: setattr(refresh.provider_lease, "acquired", False),
     "deferred", "provider_lease_unavailable", False),
    (lambda refresh: lose_lease(refresh, True, False), "deferred", "provider_lease_lost", False),
    (lambda refresh: setattr(refresh.catalog, "current", False), "blocked", "social_runtime_changed", True),
    (lambda refresh: (setattr(refresh.catalog, "current", False), lose_lease(refresh, False)),
     "deferred", "provider_lease_lost", False),
    (lambda refresh: refresh.providers["official"].fail_sources.update({"1", "2"}),
     "blocked", "source_participation_failed", True),
    (lambda refresh: (refresh.providers["official"].fail_sources.update({"1", "2"}), lose_lease(refresh, True, True, False)),
     "deferred", "provider_lease_lost", False),
])
async def test_only_the_lease_holder_closes_a_generation_nothing_can_resume(interrupt, status, reason, closes):
    refresh, _, _, writer, backlog, _ = use_case()
    interrupt(refresh)
    result = await refresh.execute("scheduled", NOW)
    assert (result.processing_status, result.reason_codes) == (status, (reason,))
    assert writer.failures == ([("scheduled-202609071200", reason)] if closes else [])
    assert not backlog.enqueued


@pytest.mark.asyncio
async def test_duplicate_delivery_without_the_lease_leaves_the_generation_to_its_collector():
    refresh, _, provider, writer, _, _ = use_case()

    def already_exists(run_id, as_of):
        raise ValueError("run_exists")

    writer.create_run = already_exists
    writer.read_run_inputs = lambda run_id: SavedSocialRunInputs(run_id, NOW, 7, (), (), ())
    refresh.provider_lease.acquired = False

    result = await refresh.execute("scheduled", NOW)

    assert (result.processing_status, result.reason_codes) == ("deferred", ("provider_lease_unavailable",))
    assert writer.failures == [] and not provider.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("retained_age", [None, timedelta(days=14, hours=12)])
async def test_terminal_analysis_failure_fails_generation_without_budget_resume_state(retained_age):
    refresh, events, _, writer, backlog, _ = use_case(failed=1)
    if retained_age is not None:
        # Retained beyond the 14-day scoring window, so it cannot stand in for analysis.
        writer.retained = (SocialPostRecord(
            "official", "old", "1", "$AAA", "https://x.com/a/status/old", "author",
            NOW - retained_age, NOW - retained_age,
        ),)

    result = await refresh.execute("scheduled", NOW)

    assert result.processing_status == "failed"
    assert result.reason_codes == ("analysis_failed",)
    assert writer.failures == [(result.run_id, "analysis_failed")]
    assert backlog.enqueued
    assert "writer.prepare" not in events


@pytest.mark.asyncio
async def test_current_parent_wins_over_retained_evidence_to_prevent_duplicate_weight():
    """Adding a retained revision beside the fresh post would process one parent twice."""
    refresh, _, _, _, backlog, _ = use_case()
    retained = SocialPostRecord(
        "official", "old-parent", "1", "$AAA evidence", "https://x.com/a/status/old-parent",
        "author", NOW - timedelta(days=2), NOW,
    )
    refresh.evidence_reader.retained_posts = lambda run_id, as_of: ((10, retained),)

    await refresh.execute("scheduled", NOW)

    assert [content_id for content_id, _, _ in backlog.enqueued] == [10, 11]


@pytest.mark.asyncio
async def test_validation_stages_admin_result_without_live_publication():
    refresh, events, _, writer, _, _ = use_case(mode="validation")
    result = await refresh.execute("scheduled", NOW)
    assert result.processing_status == "complete" and result.published is False
    assert "writer.prepare" in events and "writer.publish" in events


@pytest.mark.asyncio
async def test_context_and_unresolved_mentions_are_preserved_as_separate_states():
    refresh, _, _, writer, _, _ = use_case()
    base = refresh.evidence_reader.read

    def evidence(run_id, as_of):
        ranked = base(run_id, as_of)[0]
        post = ranked.posts[0]
        return (
            ranked,
            SocialEvidenceInput("US:SPY", "SPY", "US", (post,), ("1", "2"),
                                False, (), True, "broad_etf"),
            SocialEvidenceInput("unresolved:$zzz", "$ZZZ", None, (post,), ("1", "2"),
                                False, ("unresolved_security",), False, "stock"),
        )

    refresh.evidence_reader.read = evidence
    await refresh.execute("scheduled", NOW)
    rows = writer.prepared[0]
    assert {(row.candidate_key, row.candidate_state) for row in rows} >= {
        ("US:SPY", "context"), ("unresolved:$zzz", "unresolved")
    }


@pytest.mark.asyncio
async def test_replay_generation_uses_saved_inputs_without_provider_or_lease():
    refresh, events, provider, writer, _, _ = use_case()
    result = await refresh.execute("budget-resume", NOW, saved_run_id="old-run")
    assert result.published is True
    assert "writer.replay" in events
    assert not provider.requests
    assert "lease.acquire" not in events


@pytest.mark.asyncio
async def test_same_generation_retry_reuses_committed_source_without_refetching_it():
    refresh, events, provider, writer, _, _ = use_case()
    first_request = refresh._request("scheduled-202609071200", SOURCES[0], "official", NOW)
    first_batch = provider.read_source(first_request)
    writer.batches.append(first_batch)
    provider.requests.clear()
    events.clear()

    def already_exists(run_id, as_of):
        raise ValueError("run_exists")

    writer.create_run = already_exists
    writer.read_run_inputs = lambda run_id: SavedSocialRunInputs(
        run_id, NOW, 7, (first_batch,), ((first_batch.posts[0].provider_post_id, 10),), ()
    )
    result = await refresh.execute("scheduled", NOW)
    assert result.published is True
    assert [request.source_id for request in provider.requests] == ["2"]


@pytest.mark.asyncio
async def test_duplicate_delivery_returns_terminal_generation_without_replaying_io():
    refresh, events, provider, writer, backlog, _ = use_case()

    def already_exists(run_id, as_of):
        raise ValueError("run_exists")

    writer.create_run = already_exists
    writer.resume_existing_run = lambda run_id, expected_version: SocialRunResult(
        run_id, "live", "complete", True, (("1", "limited"), ("2", "limited")),
    )

    result = await refresh.execute("scheduled", NOW)

    assert result.published is True
    assert not provider.requests and not backlog.enqueued
    assert events == []


def test_production_factory_has_explicit_lazy_official_and_xui_adapters():
    from app.infra.providers.official_x_social_provider import OfficialXSocialProvider
    from app.infra.providers.xui_cli_social_provider import XuiCliSocialProvider
    from app.wiring.use_case_factories import get_refresh_social_signals_use_case

    lease = Lease()
    request_gate = object()
    refresh = get_refresh_social_signals_use_case(
        session_factory=lambda: None,
        provider_lease=lease,
        llm_request_gate=request_gate,
        official_client=object(),
    )
    official = refresh._provider("official")
    assert isinstance(official, OfficialXSocialProvider)
    assert official._release is not None, "unused official read capacity must be returned"
    xui = refresh._provider("xui")
    assert isinstance(xui, XuiCliSocialProvider)
    assert xui._timeout == 300, "a 1000-post backfill scroll needs more than the 60s default"
    # One xui read can take auth + read + 30s retry delay + read before the next renewal.
    worst_xui_read = 3 * xui._timeout + 30
    assert refresh.lease_ttl_seconds > worst_xui_read

    from app.wiring.use_case_factories import get_validate_social_source_use_case
    validation = get_validate_social_source_use_case(session_factory=lambda: None, provider_lease=lease)
    assert validation.lease_ttl_seconds > worst_xui_read
    assert refresh.catalog.__class__.__name__ == "SqlSocialRefreshCatalog"
    assert refresh.evidence_reader.__class__.__name__ == "SocialScoringEvidenceReader"
    assert refresh.theme_service.__class__.__name__ == "SqlThemeProjectionFacade"
    assert refresh.confirmation_reader.__class__.__name__ == "SqlConfirmationReaderFacade"
    assert refresh.backlog.request_gate is request_gate
    assert refresh.backlog.batch_size == 20
    assert refresh.backlog.min_interval_seconds == 5
    assert refresh.backlog.max_calls_per_run == 20
    assert refresh.backlog.max_calls_per_day == 80
    with pytest.raises(ValueError, match="not_wired"):
        refresh._provider("disabled")


def test_scoring_evidence_retains_older_in_window_post_after_incremental_cap(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database import Base
    from app.infra.db.models.social_signals import SocialSourceRegistry
    from app.infra.db.repositories.social_refresh_support import SocialScoringEvidenceReader
    from app.infra.db.repositories.social_signal_writer import SocialSignalWriter
    from app.models.stock_universe import StockUniverse
    from app.services.social_source_admin_service import SocialSourceAdminService

    engine = create_engine(f"sqlite:///{tmp_path / 'rolling-social.sqlite'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    with sessions() as db:
        SocialSourceAdminService(db).ensure_seed_sources()
    with sessions.begin() as db:
        registry = db.get(SocialSourceRegistry, 1)
        registry.mode, registry.provider = "live", "official"
        db.add(StockUniverse(symbol="AAA", market="US", exchange="NASDAQ", is_active=True))

    writer = SocialSignalWriter(sessions, clock=lambda: NOW)
    list_ids = ("1522014550211457024", "1986290701492232693")

    def saved_batch(source_id, run_at, post_id, created_at, intent, *, likes=10):
        request = SocialReadRequest(
            f"{run_at.isoformat()}:{source_id}", str(source_id), list_ids[source_id - 1],
            intent, run_at, 1000 if intent == "initial" else 200,
            run_at - timedelta(days=14), None if intent == "initial" else "cursor",
        )
        value = SocialPostRecord(
            "official", post_id, str(source_id), "$AAA thesis", f"https://x.com/a/{post_id}",
            f"a{source_id}", created_at, run_at, likes=likes, reposts=1, replies=1,
        )
        history_status = "observed_window" if intent == "initial" else "limited"
        reasons = () if intent == "initial" else ("bounded_provider_read",)
        gaps = () if intent == "initial" else ((old_at, run_at),)
        outcome = SocialSourceOutcome(
            "success", "pending", history_status, reasons, gaps,
            created_at, created_at, 1, None, None, proposed_progress="cursor",
        )
        return SocialSourceBatch(request, (value,), outcome)

    old_at = NOW - timedelta(days=3)
    writer.clock = lambda: old_at
    writer.create_run("initial", old_at)
    writer.persist_observations(saved_batch(1, old_at, "old", NOW - timedelta(days=8), "initial"), run_id="initial")
    writer.persist_observations(saved_batch(2, old_at, "peer-old", NOW - timedelta(days=4), "initial"), run_id="initial")
    writer.clock = lambda: NOW
    writer.create_run("incremental", NOW)
    writer.persist_observations(saved_batch(1, NOW, "new", NOW - timedelta(days=1), "incremental"), run_id="incremental")
    writer.persist_observations(saved_batch(
        2, NOW, "new", NOW - timedelta(hours=3), "incremental", likes=None,
    ), run_id="incremental")
    writer.select_current_inputs("incremental")

    evidence = SocialScoringEvidenceReader(sessions).read("incremental", NOW)
    aaa = next(item for item in evidence if item.candidate_key == "US:AAA")
    assert {post.provider_post_id for post in aaa.posts} == {"old", "peer-old", "new"}
    assert len(aaa.posts) == 4
    assert {post.likes for post in aaa.posts if post.provider_post_id == "new"} == {10}
    assert aaa.enabled_source_ids == ("1", "2")
    assert aaa.history_complete is False
    # A list that fails now must not inherit its earlier complete window.
    writer.create_run("partial", NOW)
    writer.persist_observations(saved_batch(1, NOW, "fresh", NOW - timedelta(hours=1), "initial"), run_id="partial")
    failed_request = SocialReadRequest("partial:2", "2", list_ids[1], "incremental", NOW, 200,
                                       NOW - timedelta(days=14), "cursor")
    writer.persist_observations(SocialSourceBatch(failed_request, (), SocialSourceOutcome(
        "failed", "failed", "limited", ("provider_timeout",), (), None, None, 0, None,
        "provider_timeout",
    )), run_id="partial")
    partial = SocialScoringEvidenceReader(sessions).read("partial", NOW)
    assert all(item.history_complete is False for item in partial)
    assert all("provider_timeout" in item.coverage_reasons for item in partial)
    writer.create_replay_run("partial-replay", "partial", NOW)
    replayed = SocialScoringEvidenceReader(sessions).read("partial-replay", NOW)
    assert replayed and all(item.history_complete is False for item in replayed)
    # A retained semantic judgment can be explicitly pinned to the new
    # generation without pretending the old post was recollected.
    from dataclasses import asdict
    from app.domain.social_signals.records import ExtractionPostJudgment, ExtractionResult
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialRunWork
    from app.services.social_extraction_service import SocialExtractionService
    old_post = next(post for post in aaa.posts if post.provider_post_id == "old")
    old_content_id = dict(writer.read_run_inputs("initial").content_ids)["old"]
    old_hash = SocialExtractionService.input_hash((old_post,))
    result = ExtractionResult(old_hash, "synthetic", "actual", "v1", "v1", (), 1, 1,
                              (ExtractionPostJudgment("old", False, None),))
    with sessions.begin() as db:
        work = SocialExtractionWork(
            content_item_id=old_content_id, input_hash=old_hash, prompt_version="v1",
            schema_version="v1", selected_model="requested", actual_provider="synthetic",
            actual_model="actual", result_json=asdict(result), state="succeeded",
            input_snapshot_json={**asdict(old_post), "created_at": old_post.created_at.isoformat(),
                                 "observed_at": old_post.observed_at.isoformat()},
            created_at=NOW, updated_at=NOW,
        )
        db.add(work)
        db.flush()
        db.add(SocialRunWork(run_id="incremental", work_id=work.id,
                             input_hash=old_hash, included_at=NOW))
        retained_work_id = work.id
    selected = writer.select_current_inputs("incremental")
    assert selected.current_manifest.scoring_work_ids == (retained_work_id,)
    with sessions() as db:
        from app.infra.db.models.social_signals import SocialSignalRun
        assert [pin["name"] for pin in db.get(SocialSignalRun, "incremental")
                .application_progress_json["sources"].values()] == [
            "Minervini Research List", "Asia-Pacific Growth List"
        ]
    engine.dispose()


class FixtureLLM:
    def __init__(self, malformed_post_ids=()):
        self.malformed_post_ids = set(malformed_post_ids)
        self.seen = []

    async def completion(self, **kwargs):
        import json
        from types import SimpleNamespace

        sources = json.loads(kwargs["messages"][1]["content"])["posts"]
        self.seen.append(tuple(source["post_id"] for source in sources))
        payload = {"posts": [{"post_id": source["post_id"], "claims": [],
                              "has_new_thesis": False, "canonical_claim_key": None}
                             for source in sources]}
        content = ("bad" if self.malformed_post_ids & {s["post_id"] for s in sources}
                   else json.dumps(payload))
        return SimpleNamespace(
            model="actual", id=f"fixture-{sources[0]['post_id']}",
            _hidden_params={"custom_llm_provider": "synthetic"},
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )


def real_refresh(tmp_path, llm, *, fail_sources=(), batch_size=1, run_id_factory=None):
    import json

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database import Base
    from app.infra.db.models.social_signals import SocialSourceRegistry
    from app.infra.db.repositories.social_refresh_support import (
        SocialScoringEvidenceReader, SqlSocialRefreshCatalog, SqlThemeProjectionFacade,
    )
    from app.infra.db.repositories.social_signal_writer import SocialSignalWriter
    from app.models.app_settings import AppSetting
    from app.models.stock_universe import StockUniverse
    from app.services.social_extraction_service import SocialExtractionService
    from app.services.social_source_admin_service import SocialSourceAdminService
    from app.services.social_signal_backlog_service import ProcessSocialBacklog
    from app.use_cases.social_signals.refresh import RefreshSocialSignals

    engine = create_engine(f"sqlite:///{tmp_path / 'refresh-flow.sqlite'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    with sessions() as db:
        SocialSourceAdminService(db).ensure_seed_sources()
    with sessions.begin() as db:
        registry = db.get(SocialSourceRegistry, 1)
        registry.mode, registry.provider = "live", "official"
        db.add(StockUniverse(symbol="AAA", market="US", exchange="NASDAQ", is_active=True))
        db.add(AppSetting(key="social_llm_extraction_model", value="synthetic/requested"))
        db.add(AppSetting(key="social_llm_daily_limit_usd", value="2"))
        db.add(AppSetting(key="social_llm_pricing", value=json.dumps({
            "version": "fixture-v1", "models": {"synthetic/requested": {
                "provider": "synthetic", "actual_models": ["actual"],
                "input_usd_per_million": "0", "output_usd_per_million": "0.01",
            }},
        })))

    events = []
    provider = Provider(events, fail_sources=fail_sources)
    confirmation = ConfirmationReader(events)
    writer = SocialSignalWriter(
        sessions, clock=lambda: NOW, confirmation_reader_factory=lambda db: confirmation
    )
    refresh = RefreshSocialSignals(
        catalog=SqlSocialRefreshCatalog(sessions), providers={"official": provider},
        writer=writer, backlog=ProcessSocialBacklog(sessions, llm=llm, batch_size=batch_size),
        evidence_reader=SocialScoringEvidenceReader(sessions),
        theme_service=SqlThemeProjectionFacade(sessions), confirmation_reader=confirmation,
        provider_lease=Lease(events=events),
        run_id_factory=run_id_factory or (lambda origin, now: "fixture-run"),
        input_hash=lambda post: SocialExtractionService.input_hash((post,)),
        lease_owner_factory=lambda: "attempt",
    )
    return engine, sessions, writer, refresh


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_sources", [(), ("2",)])
async def test_refresh_runs_through_real_writer_backlog_and_publication_with_fixture_io(tmp_path, fail_sources):
    from app.infra.db.models.social_signals import SocialSignalRunPointer

    engine, sessions, writer, refresh = real_refresh(tmp_path, FixtureLLM(), fail_sources=fail_sources)
    result = await refresh.execute("fixture", NOW)
    assert result.published is True
    with sessions() as db:
        assert db.get(SocialSignalRunPointer, "latest_published").run_id == "fixture-run"
    # Replays rebuild every saved outcome, including a failed list's.
    assert len(writer.read_run_inputs("fixture-run").batches) == 2
    engine.dispose()


@pytest.mark.asyncio
async def test_one_malformed_llm_response_does_not_block_later_generations(tmp_path):
    from sqlalchemy import select

    from app.infra.db.models.social_signals import SocialSignalRunPointer, SocialSignalSnapshot

    llm = FixtureLLM(malformed_post_ids={"p-1"})
    engine, sessions, writer, refresh = real_refresh(
        tmp_path, llm, run_id_factory=lambda origin, now: f"run-{now.hour}",
    )
    first = await refresh.execute("scheduled", NOW)

    later = NOW + timedelta(hours=6)
    writer.clock = lambda: later
    llm.malformed_post_ids.clear()
    second = await refresh.execute("scheduled", later)

    assert (second.published, second.reason_codes) == (True, ())
    assert (first.published, first.reason_codes) == (True, ())
    with sessions() as db:
        assert db.get(SocialSignalRunPointer, "latest_published").run_id == "run-18"
        rows = db.scalars(select(SocialSignalSnapshot).where(
            SocialSignalSnapshot.run_id == "run-12", SocialSignalSnapshot.window_days == 14,
        )).all()
    assert [(row.candidate_key, row.mention_count) for row in rows] == [("US:AAA", 1)]
    assert "analysis_unavailable" in str(rows[0].coverage_json)
    assert llm.seen.count(("p-1",)) == 1
    engine.dispose()


@pytest.mark.asyncio
async def test_resumed_generation_scores_a_post_whose_analysis_later_succeeded(tmp_path):
    from sqlalchemy import select

    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.infra.db.models.social_signals import SocialSignalSnapshot

    llm = FixtureLLM(malformed_post_ids={"p-1"})
    engine, sessions, writer, refresh = real_refresh(tmp_path, llm)
    read = refresh.evidence_reader.read
    def interrupted(run_id, as_of):
        refresh.evidence_reader.read = read
        raise RuntimeError("worker lost")
    refresh.evidence_reader.read = interrupted
    with pytest.raises(RuntimeError, match="worker lost"):
        await refresh.execute("fixture", NOW)
    with sessions.begin() as db:
        work = db.scalar(select(SocialExtractionWork).where(
            SocialExtractionWork.state == "failed_terminal"))
        work.state, work.error_code = "pending", None
    llm.malformed_post_ids.clear()

    result = await refresh.execute("fixture", NOW)

    assert result.published is True
    with sessions() as db:
        rows = db.scalars(select(SocialSignalSnapshot).where(
            SocialSignalSnapshot.window_days == 14)).all()
    assert [(row.candidate_key, row.mention_count) for row in rows] == [("US:AAA", 2)]
    engine.dispose()


@pytest.mark.asyncio
async def test_unavailable_post_leaves_mention_history_incomplete(tmp_path):
    from app.infra.db.repositories.social_refresh_support import SocialScoringEvidenceReader

    engine, sessions, writer, refresh = real_refresh(tmp_path, FixtureLLM(malformed_post_ids={"p-1"}))
    provider = refresh.providers["official"]
    read_source = provider.read_source
    provider.read_source = lambda request: (lambda batch: replace(batch, outcome=replace(
        batch.outcome, history_status="observed_window", coverage_reason_codes=(),
    )))(read_source(request))

    assert (await refresh.execute("fixture", NOW)).published is True
    evidence = SocialScoringEvidenceReader(sessions).read("fixture-run", NOW)

    assert [(item.candidate_key, item.history_complete) for item in evidence] == [("US:AAA", False)]
    engine.dispose()


def test_scoring_evidence_never_reads_generations_older_than_its_window(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database import Base
    from app.infra.db.models.social_signals import SocialSignalRun, SocialSourceRegistry
    from app.infra.db.repositories.social_refresh_support import SocialScoringEvidenceReader
    from app.infra.db.repositories.social_signal_writer import SocialSignalWriter
    from app.services.social_source_admin_service import SocialSourceAdminService

    engine = create_engine(f"sqlite:///{tmp_path / 'aged-social.sqlite'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    with sessions() as db:
        SocialSourceAdminService(db).ensure_seed_sources()
    with sessions.begin() as db:
        registry = db.get(SocialSourceRegistry, 1)
        registry.mode, registry.provider = "live", "official"
    SocialSignalWriter(sessions, clock=lambda: NOW).create_run("current", NOW)
    with sessions.begin() as db:
        db.add(SocialSignalRun(
            id="aged", registry_id=1, registry_version=db.get(SocialSourceRegistry, 1).version,
            mode="live", provider="official", status="failed", created_at=NOW - timedelta(days=16),
            source_outcomes_json={"1": {"read_status": "success"}},
            application_progress_json={"sources": {}, "observations": {
                "1": {"inputs": [{"content_item_id": 1, "post": {"unreadable": True}}]},
            }},
            feature_run_ids_json={}, exposure_dates_json={}, coverage_json={},
        ))

    reader = SocialScoringEvidenceReader(sessions)
    assert reader.retained_posts("current", NOW) == ()
    assert reader.read("current", NOW) == ()
