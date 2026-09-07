"""Short, owned transactions for canonical observations and immutable publication."""
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import re

from sqlalchemy import select

from app.domain.social_signals.records import (PreparedSocialPublication, SavedSocialRunInputs, SocialPostRecord,
    SocialReadRequest, SocialSourceOutcome, SocialSourceBatch, SocialRunResult, validate_utc_timestamp)
from app.infra.db.models.social_signals import (
    ContentPipelineEligibility, SocialContentMetrics, SocialPostSource,
    SocialSignalRun, SocialSourceConfiguration, SocialSignalSnapshot, SocialSignalRunPointer, SocialPostTicker,
)
from app.models.theme import ContentItem
from app.services.social_theme_projection_service import _lock_registry
from app.services.social_theme_projection_service import SocialThemeProjectionService, _decode
from app.infra.db.models.social_analysis import SocialRunWork, SocialExtractionWork
from app.services.social_extraction_service import SocialExtractionService

METRICS = ("likes", "reposts", "replies", "quotes", "bookmarks", "views")


def utc(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def serialized(value):
    return json.loads(json.dumps(asdict(value), default=lambda v: v.isoformat() if isinstance(v, datetime) else str(v)))


class SocialSignalWriter:
    def __init__(self, session_factory, *, clock=None):
        self.session_factory = session_factory
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def create_run(self, run_id, as_of):
        validate_utc_timestamp(as_of, "as_of")
        with self.session_factory.begin() as db:
            registry = _lock_registry(db)
            if registry is None or registry.mode not in {"live", "validation"} or registry.provider == "disabled":
                raise ValueError("social_runtime_unavailable")
            sources = db.scalars(select(SocialSourceConfiguration).where(SocialSourceConfiguration.lifecycle_state == "enabled").order_by(SocialSourceConfiguration.content_source_id)).all()
            if len(sources) < 2:
                raise ValueError("minimum_two_enabled")
            if db.get(SocialSignalRun, run_id) is not None:
                raise ValueError("run_exists")
            db.add(SocialSignalRun(id=run_id, registry_id=1, registry_version=registry.version,
                mode=registry.mode, provider=registry.provider, status="running", created_at=as_of,
                source_outcomes_json={}, application_progress_json={"sources": {
                    str(s.content_source_id): {"list_id": s.x_list_id, "version": s.version} for s in sources}, "observations": {}},
                feature_run_ids_json={}, exposure_dates_json={}, coverage_json={}))
        return run_id

    def persist_observations(self, batch, *, run_id=None):
        if batch.request.intent == "test":
            raise ValueError("diagnostic_observations_forbidden")
        if len(batch.posts) > batch.request.limit or batch.outcome.received_count != len(batch.posts):
            raise ValueError("invalid_bounded_batch")
        with self.session_factory.begin() as db:
            registry = _lock_registry(db)
            source = db.get(SocialSourceConfiguration, int(batch.request.source_id))
            if source is None or source.x_list_id != batch.request.list_id:
                raise ValueError("source_identity_mismatch")
            if source.lifecycle_state != "enabled":
                raise ValueError("source_not_enabled")
            if registry is None or registry.mode not in {"live", "validation"}:
                raise ValueError("social_runtime_unavailable")
            run = db.get(SocialSignalRun, run_id) if run_id else None
            if run_id and (run is None or run.status != "running"):
                raise ValueError("run_not_collecting")
            if run is not None:
                pin = run.application_progress_json["sources"].get(batch.request.source_id)
                if (pin != {"list_id": source.x_list_id, "version": source.version}
                        or registry.version != run.registry_version or registry.provider != run.provider):
                    raise ValueError("source_configuration_changed")
                existing = run.application_progress_json["observations"].get(batch.request.source_id)
                if existing:
                    if existing["request_id"] != batch.request.request_id:
                        raise ValueError("source_already_collected")
                    if (existing["request"] != serialized(batch.request)
                            or [i["post"] for i in existing["inputs"]] != [serialized(p) for p in batch.posts]
                            or run.source_outcomes_json[batch.request.source_id] != serialized(replace(batch.outcome,
                                committed_progress=existing["committed_progress"]))):
                        raise ValueError("duplicate_delivery_mismatch")
                    return replace(batch, outcome=replace(batch.outcome, committed_progress=existing["committed_progress"]))
            observations = []
            for post in batch.posts:
                if post.source_id != batch.request.source_id or post.provider != registry.provider or (run and post.provider != run.provider):
                    raise ValueError("post_source_identity_mismatch")
                item = db.scalar(select(ContentItem).where(ContentItem.source_type == "twitter", ContentItem.external_id == post.provider_post_id))
                if item is None:
                    item = ContentItem(source_id=source.content_source_id, source_type="twitter", external_id=post.provider_post_id,
                        content=post.text, url=post.url, author=post.author_handle, published_at=post.created_at, fetched_at=post.observed_at)
                    db.add(item)
                    db.flush()
                membership = db.get(SocialPostSource, (item.id, source.content_source_id))
                if membership is None:
                    db.add(SocialPostSource(content_item_id=item.id, content_source_id=source.content_source_id, observed_at=post.observed_at))
                elif utc(membership.observed_at) < post.observed_at:
                    membership.observed_at = post.observed_at
                for pipeline in ("technical", "fundamental"):
                    if db.get(ContentPipelineEligibility, (item.id, pipeline, "social")) is None:
                        db.add(ContentPipelineEligibility(content_item_id=item.id, pipeline=pipeline, channel="social",
                            originating_source_id=source.content_source_id, observed_at=post.observed_at))
                metrics = db.get(SocialContentMetrics, item.id)
                if metrics is None:
                    metrics = SocialContentMetrics(content_item_id=item.id, provider=post.provider,
                        provider_post_id=post.provider_post_id, observed_at=post.observed_at)
                    db.add(metrics)
                    newer = True
                else:
                    newer = utc(metrics.observed_at) < post.observed_at
                if newer:
                    for field in METRICS:
                        if getattr(post, field) is not None:
                            setattr(metrics, field, getattr(post, field))
                    metrics.observed_at, metrics.provider = post.observed_at, post.provider
                observations.append({"content_item_id": item.id, "post": serialized(post)})
                self._map_cashtags(db, item.id, post.text)
            progress = batch.outcome.proposed_progress if batch.outcome.read_status == "success" else None
            outcome = replace(batch.outcome, committed_progress=progress)
            if run is not None:
                payload = deepcopy(run.application_progress_json)
                payload["observations"][batch.request.source_id] = {"request_id": batch.request.request_id,
                    "observed_at": batch.request.observed_at.isoformat(), "intent": batch.request.intent,
                    "committed_progress": progress, "request": serialized(batch.request), "inputs": observations}
                run.application_progress_json = payload
                run.source_outcomes_json = {**run.source_outcomes_json, batch.request.source_id: serialized(outcome)}
            if outcome.read_status == "success" and (source.last_successful_collection_at is None or utc(source.last_successful_collection_at) < batch.request.observed_at):
                source.last_successful_collection_at = batch.request.observed_at
        return replace(batch, outcome=outcome)

    @staticmethod
    def _map_cashtags(db, content_id, text):
        """Explicit observed tokens only; semantic claims remain saved work."""
        from app.services.social_ticker_resolver import SocialTickerResolver
        resolver = SocialTickerResolver(db)
        for token in sorted(set(re.findall(r"\$[A-Za-z0-9][A-Za-z0-9.\-]{0,19}", text))):
            resolution = resolver.resolve(token)
            key = f"{resolution.market}:{resolution.symbol}" if resolution.status == "resolved" else f"unresolved:{token.casefold()}"
            existing = db.scalar(select(SocialPostTicker).where(SocialPostTicker.content_item_id == content_id, SocialPostTicker.candidate_key == key))
            if existing is None:
                db.add(SocialPostTicker(content_item_id=content_id, candidate_key=key, raw_token=token,
                    stock_universe_id=int(resolution.security_id) if resolution.security_id else None,
                    canonical_symbol=resolution.symbol, market=resolution.market, resolution_state=resolution.status,
                    resolution_policy_version="social-resolution-v1", explanation_json=serialized(resolution)))

    def latest_committed_progress(self, source_id, provider):
        with self.session_factory() as db:
            observations = []
            for run in db.scalars(select(SocialSignalRun).where(SocialSignalRun.provider == provider)):
                outcome = run.source_outcomes_json.get(str(source_id), {})
                observation = run.application_progress_json.get("observations", {}).get(str(source_id))
                if observation and outcome.get("read_status") == "success" and observation.get("intent") == "initial":
                    observations.append((observation["observed_at"], utc(run.created_at), run.id, observation["committed_progress"]))
            return max(observations)[-1] if observations else None

    def read_run_inputs(self, run_id):
        """Frozen historical read evidence; not a claim of a new provider read."""
        with self.session_factory() as db:
            run = db.get(SocialSignalRun, run_id)
            if run is None:
                raise ValueError("run_unavailable")
            batches, content_ids = [], set()
            for source_id, observation in sorted(run.application_progress_json["observations"].items()):
                request = dict(observation["request"])
                for field in ("observed_at", "target_published_after"):
                    request[field] = datetime.fromisoformat(request[field])
                posts = []
                for entry in observation["inputs"]:
                    post = dict(entry["post"])
                    for field in ("created_at", "observed_at"):
                        post[field] = datetime.fromisoformat(post[field])
                    posts.append(SocialPostRecord(**post))
                    content_ids.add((post["provider_post_id"], entry["content_item_id"]))
                outcome = dict(run.source_outcomes_json[source_id])
                for field in ("observed_oldest_at", "observed_newest_at", "rate_limit_reset_at"):
                    outcome[field] = datetime.fromisoformat(outcome[field]) if outcome[field] else None
                outcome["coverage_reason_codes"] = tuple(outcome["coverage_reason_codes"])
                outcome["known_gap_intervals"] = tuple(tuple(datetime.fromisoformat(v) for v in gap) for gap in outcome["known_gap_intervals"])
                batches.append(SocialSourceBatch(SocialReadRequest(**request), tuple(posts), SocialSourceOutcome(**outcome)))
            if run.status == "running":
                work_ids = tuple(db.scalars(select(SocialRunWork.work_id).where(SocialRunWork.run_id == run_id).order_by(SocialRunWork.work_id)))
            else:
                manifest = run.application_progress_json.get("prepared", {}).get("work_ids")
                if manifest is None:
                    raise ValueError("terminal_run_manifest_missing")
                work_ids = tuple(manifest)
            return SavedSocialRunInputs(run_id, utc(run.created_at), run.registry_version, tuple(batches), tuple(sorted(content_ids)), work_ids)

    def _validate_inputs(self, db, run):
        pins = run.application_progress_json["sources"]
        observations = run.application_progress_json["observations"]
        if len(pins) < 2 or set(run.source_outcomes_json) != set(pins) or set(observations) != set(pins):
            raise ValueError("source_participation_incomplete")
        if any(outcome["read_status"] != "success" for outcome in run.source_outcomes_json.values()):
            raise ValueError("source_participation_failed")
        required = set()
        for source in observations.values():
            for observation in source["inputs"]:
                value = dict(observation["post"])
                value.update(created_at=datetime.fromisoformat(value["created_at"]), observed_at=datetime.fromisoformat(value["observed_at"]))
                required.add((observation["content_item_id"], SocialExtractionService.input_hash((SocialPostRecord(**value),))))
        actual, ids = set(), []
        for link in db.scalars(select(SocialRunWork).where(SocialRunWork.run_id == run.id).order_by(SocialRunWork.work_id)):
            work = db.get(SocialExtractionWork, link.work_id)
            if work is None or link.input_hash != work.input_hash:
                raise ValueError("pinned_input_mismatch")
            _decode(work)
            actual.add((work.content_item_id, work.input_hash))
            ids.append(work.id)
        if actual != required:
            raise ValueError("pinned_inputs_incomplete")
        return tuple(ids)

    def prepare_run(self, run_id, rows, as_of, *, theme_evidence=()):
        validate_utc_timestamp(as_of, "as_of")
        # Read/validate expensive saved input and measurement work before locking.
        with self.session_factory() as db:
            run = db.get(SocialSignalRun, run_id)
            if run is None or run.status != "running":
                raise ValueError("run_not_collecting")
            if utc(run.created_at) != as_of:
                raise ValueError("generation_time_mismatch")
            work_ids = self._validate_inputs(db, run)
            service = SocialThemeProjectionService(db)
            projection = service.prepare(run_id, as_of)
            application = service.prepare_application(projection, theme_keys=tuple(e.theme_key for e in theme_evidence))
            for evidence in theme_evidence:
                basket = application.read(evidence.theme_key, evidence.market)
                version = sha256(json.dumps(asdict(basket), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if (evidence.basket_version != version or evidence.membership != basket.membership
                        or evidence.registry_version != basket.registry_version
                        or evidence.identity_version != basket.identity_version
                        or evidence.accepted_company_count != len({m.company_key for m in basket.membership if m.company_count_eligible})):
                    raise ValueError("measurement_basket_changed")
            result = PreparedSocialPublication(run_id, run.registry_version, as_of, work_ids, rows, theme_evidence)
            frozen_input = deepcopy(run.application_progress_json)
        with self.session_factory.begin() as db:
            registry = _lock_registry(db)
            run = db.get(SocialSignalRun, run_id)
            if run.status != "running" or run.application_progress_json != frozen_input or registry.version != result.registry_version:
                raise ValueError("publication_preparation_changed")
            keys = set()
            for record in rows:
                if record.run_id != run_id or (record.window_days, record.candidate_key) in keys:
                    raise ValueError("snapshot_identity_mismatch")
                keys.add((record.window_days, record.candidate_key))
                db.add(SocialSignalSnapshot(run_id=run_id, window_days=record.window_days,
                    candidate_key=record.candidate_key, canonical_symbol=record.canonical_symbol, market=record.market,
                    state=record.candidate_state, social_score=record.social_score, confirmation_score=record.confirmation_score,
                    queue_score=record.queue_score, explanation_json={"record": serialized(record), "evidence_run_id": run_id},
                    coverage_json={"evidence_run_id": run_id, "reasons": list(record.coverage)},
                    resolution_policy_version="social-resolution-v1", formula_version=record.formula_version,
                    latest_mention=record.latest_mention, mention_count=record.mention_count,
                    observed_list_count=record.observed_list_count, enabled_list_count=len(frozen_input["sources"]),
                    normalization_scope=record.normalization_scope))
            run.application_progress_json = {**frozen_input, "prepared": {"work_ids": list(work_ids),
                "projection": serialized(projection), "theme_evidence": [serialized(e) for e in theme_evidence],
                "basket_fingerprint": application.fingerprint, "baskets": [serialized(b) for b in application.baskets]}}
            run.completed_at, run.status = as_of, "staged"
            run.coverage_json = deepcopy(run.source_outcomes_json)
        return result

    def publish(self, run_id, expected_mode_version):
        with self.session_factory() as db:
            run = db.get(SocialSignalRun, run_id)
            if run is None or run.status not in {"staged", "published"}:
                raise ValueError("run_not_prepared")
            summary = tuple((key, value["history_status"]) for key, value in sorted(run.source_outcomes_json.items()))
            if run.mode == "validation":
                return SocialRunResult(run_id, "validation", "complete", False, summary, ("validation_only",))
            if run.status == "published":
                return SocialRunResult(run_id, "live", "complete", True, summary)
            self._validate_inputs(db, run)
            service = SocialThemeProjectionService(db)
            projection = service.prepare(run_id, utc(run.created_at))
            if serialized(projection) != run.application_progress_json["prepared"]["projection"]:
                raise ValueError("publication_version_changed")
            prepared_data = run.application_progress_json["prepared"]
            application = service.prepare_application(projection, theme_keys=tuple(e["theme_key"] for e in prepared_data["theme_evidence"]))
            if (application.fingerprint != prepared_data["basket_fingerprint"]
                    or [serialized(b) for b in application.baskets] != prepared_data["baskets"]):
                raise ValueError("publication_basket_changed")
        with self.session_factory.begin() as db:
            registry = _lock_registry(db)
            run = db.get(SocialSignalRun, run_id)
            if registry.mode != "live" or registry.version != expected_mode_version or run.registry_version != expected_mode_version:
                raise ValueError("publication_live_version_changed")
            sources = {str(s.content_source_id): {"list_id": s.x_list_id, "version": s.version}
                for s in db.scalars(select(SocialSourceConfiguration).where(SocialSourceConfiguration.lifecycle_state == "enabled"))}
            if sources != run.application_progress_json["sources"] or registry.provider != run.provider:
                raise ValueError("publication_source_version_changed")
            pointer = db.scalar(select(SocialSignalRunPointer).where(SocialSignalRunPointer.key == "latest_published").with_for_update())
            if pointer and pointer.run_id != run_id:
                prior = db.get(SocialSignalRun, pointer.run_id)
                if utc(prior.created_at) >= utc(run.created_at):
                    raise ValueError("publication_generation_superseded")
            SocialThemeProjectionService(db).apply_live(projection, expected_mode_version, prepared=application)
            from app.services.social_theme_market_service import LiveAcceptedBasketReader
            basket_reader = LiveAcceptedBasketReader(db)
            for expected in application.baskets:
                if basket_reader.read(expected.theme_key, expected.market) != expected:
                    raise ValueError("publication_basket_changed")
            now = self.clock()
            if pointer is None:
                db.add(SocialSignalRunPointer(key="latest_published", run_id=run_id, updated_at=now))
            else:
                pointer.run_id, pointer.updated_at = run_id, now
            run.status, run.published_at = "published", now
        return SocialRunResult(run_id, "live", "complete", True, summary)
