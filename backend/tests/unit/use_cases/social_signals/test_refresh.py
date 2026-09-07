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
    def __init__(self, acquired=True, events=None):
        self.acquired = acquired
        self.events = events if events is not None else []

    def acquire(self, owner, ttl_seconds):
        self.events.append("lease.acquire")
        return self.acquired

    def release(self, owner):
        self.events.append("lease.release")


class Provider:
    def __init__(self, events, *, fail_source=None):
        self.events = events
        self.fail_source = fail_source
        self.requests = []

    def read_source(self, request):
        self.events.append(f"provider.{request.source_id}")
        self.requests.append(request)
        if request.source_id == self.fail_source:
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

    def latest_committed_progress(self, source_id, provider):
        return {"2": "prior-cursor"}.get(source_id)

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
        return SavedSocialRunInputs(
            run_id, NOW, 7, tuple(self.batches), content_ids, (),
            current_manifest=SocialCurrentInputManifest(required, (), ()),
        )

    def current_inputs_ready(self, run_id):
        return True

    def prepare_run(self, run_id, rows, as_of, *, theme_evidence=(), context=None):
        self.events.append("writer.prepare")
        self.prepared = (rows, theme_evidence, context)

    def publish(self, run_id, expected_mode_version):
        self.events.append("writer.publish")
        return SocialRunResult(run_id, self.mode, "complete", self.mode == "live",
                               (("1", "limited"), ("2", "limited")))


class Backlog:
    def __init__(self, events, *, deferred=0):
        self.events = events
        self.deferred = deferred
        self.enqueued = []

    def enqueue(self, content_item_id, post, **kwargs):
        self.events.append("backlog.enqueue")
        self.enqueued.append((content_item_id, post, kwargs))
        return content_item_id

    async def execute(self, now, limit, admin_work_ids=(), work_ids=()):
        self.events.append("backlog.execute")
        assert tuple(value[0] for value in self.enqueued) == work_ids
        return BacklogResult(max(0, limit - self.deferred), self.deferred, 0, 0, NOW + timedelta(hours=12))


class EvidenceReader:
    def __init__(self, events):
        self.events = events

    def read(self, run_id, as_of):
        self.events.append("evidence.read")
        posts = tuple(post for source_batch in self.writer.batches for post in source_batch.posts)
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


def use_case(*, mode="live", fail_source=None, deferred=0, current=True):
    from app.use_cases.social_signals.refresh import RefreshSocialSignals

    events = []
    catalog = Catalog(mode=mode, current=current)
    provider = Provider(events, fail_source=fail_source)
    writer = Writer(events, mode=mode)
    backlog = Backlog(events, deferred=deferred)
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
        ("initial", 1000, None), ("incremental", 200, "prior-cursor")
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
async def test_failed_enabled_source_blocks_analysis_and_publication_but_keeps_audit_batch():
    refresh, events, _, writer, backlog, _ = use_case(fail_source="2")
    result = await refresh.execute("scheduled", NOW)
    assert result.published is False
    assert result.reason_codes == ("source_participation_failed",)
    assert len(writer.batches) == 2
    assert not backlog.enqueued
    assert "writer.prepare" not in events and "writer.publish" not in events


@pytest.mark.asyncio
async def test_budget_pause_keeps_work_and_does_not_prepare():
    refresh, events, _, _, backlog, _ = use_case(deferred=1)
    result = await refresh.execute("scheduled", NOW)
    assert result.processing_status == "deferred"
    assert backlog.enqueued
    assert "writer.prepare" not in events


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
    refresh = get_refresh_social_signals_use_case(
        session_factory=lambda: None,
        provider_lease=lease,
        official_client=object(),
    )
    assert isinstance(refresh._provider("official"), OfficialXSocialProvider)
    assert isinstance(refresh._provider("xui"), XuiCliSocialProvider)
    assert refresh.catalog.__class__.__name__ == "SqlSocialRefreshCatalog"
    assert refresh.evidence_reader.__class__.__name__ == "SocialScoringEvidenceReader"
    assert refresh.theme_service.__class__.__name__ == "SqlThemeProjectionFacade"
    assert refresh.confirmation_reader.__class__.__name__ == "SqlConfirmationReaderFacade"
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

    def saved_batch(source_id, run_at, post_id, created_at, intent):
        request = SocialReadRequest(
            f"{run_at.isoformat()}:{source_id}", str(source_id), list_ids[source_id - 1],
            intent, run_at, 1000 if intent == "initial" else 200,
            run_at - timedelta(days=14), None if intent == "initial" else "cursor",
        )
        value = SocialPostRecord(
            "official", post_id, str(source_id), "$AAA thesis", f"https://x.com/a/{post_id}",
            f"a{source_id}", created_at, run_at, likes=10, reposts=1, replies=1,
        )
        outcome = SocialSourceOutcome(
            "success", "pending", "limited", ("bounded_provider_read",), (),
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
    writer.persist_observations(saved_batch(2, NOW, "peer-new", NOW - timedelta(hours=3), "incremental"), run_id="incremental")
    writer.select_current_inputs("incremental")

    evidence = SocialScoringEvidenceReader(sessions).read("incremental", NOW)
    aaa = next(item for item in evidence if item.candidate_key == "US:AAA")
    assert {post.provider_post_id for post in aaa.posts} == {"old", "peer-old", "new", "peer-new"}
    assert aaa.enabled_source_ids == ("1", "2")
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


@pytest.mark.asyncio
async def test_refresh_runs_through_real_writer_backlog_and_publication_with_fixture_io(tmp_path):
    import json
    from types import SimpleNamespace

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database import Base
    from app.infra.db.models.social_signals import SocialSignalRunPointer, SocialSourceRegistry
    from app.infra.db.repositories.social_refresh_support import (
        SocialScoringEvidenceReader, SqlSocialRefreshCatalog, SqlThemeProjectionFacade,
    )
    from app.infra.db.repositories.social_signal_writer import SocialSignalWriter
    from app.models.app_settings import AppSetting
    from app.models.stock_universe import StockUniverse
    from app.services.social_extraction_service import SocialExtractionService
    from app.services.social_source_admin_service import SocialSourceAdminService
    from app.use_cases.social_signals.process_backlog import ProcessSocialBacklog
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
        db.add(AppSetting(key="llm_extraction_model", value="synthetic/requested"))
        db.add(AppSetting(key="social_llm_daily_limit_usd", value="2"))
        db.add(AppSetting(key="social_llm_pricing", value=json.dumps({
            "version": "fixture-v1", "models": {"synthetic/requested": {
                "provider": "synthetic", "actual_models": ["actual"],
                "input_usd_per_million": "0", "output_usd_per_million": "0.01",
            }},
        })))

    class LLM:
        async def completion(self, **kwargs):
            source = json.loads(kwargs["messages"][1]["content"])["posts"][0]
            payload = {"posts": [{"post_id": source["post_id"], "claims": [],
                                    "has_new_thesis": False, "canonical_claim_key": None}]}
            return SimpleNamespace(
                model="actual", id=f"fixture-{source['post_id']}",
                _hidden_params={"custom_llm_provider": "synthetic"},
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    events = []
    provider = Provider(events)
    confirmation = ConfirmationReader(events)
    writer = SocialSignalWriter(
        sessions, clock=lambda: NOW, confirmation_reader_factory=lambda db: confirmation
    )
    refresh = RefreshSocialSignals(
        catalog=SqlSocialRefreshCatalog(sessions), providers={"official": provider},
        writer=writer, backlog=ProcessSocialBacklog(sessions, llm=LLM()),
        evidence_reader=SocialScoringEvidenceReader(sessions),
        theme_service=SqlThemeProjectionFacade(sessions), confirmation_reader=confirmation,
        provider_lease=Lease(events=events), run_id_factory=lambda origin, now: "fixture-run",
        input_hash=lambda post: SocialExtractionService.input_hash((post,)),
        lease_owner_factory=lambda: "attempt",
    )
    result = await refresh.execute("fixture", NOW)
    assert result.published is True
    with sessions() as db:
        assert db.get(SocialSignalRunPointer, "latest_published").run_id == "fixture-run"
    engine.dispose()
