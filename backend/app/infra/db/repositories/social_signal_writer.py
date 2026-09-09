"""Short, owned transactions for canonical observations and immutable publication."""
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
import re
from types import UnionType
from typing import get_args, get_origin, get_type_hints

from sqlalchemy import select

from app.domain.social_signals.records import (PreparedSocialPublication, SavedSocialRunInputs, SocialPostRecord,
    SocialReplayManifest, ReplayInput, SocialPublicationContext, SocialCurrentInputManifest,
    SocialCollectionProgress, SocialReadRequest, SocialSourceOutcome, SocialSourceBatch,
    SocialRunResult, validate_utc_timestamp)
from app.infra.db.models.social_signals import (
    ContentPipelineEligibility, SocialContentMetrics, SocialPostSource,
    SocialSignalRun, SocialSourceConfiguration, SocialSignalSnapshot, SocialSignalRunPointer, SocialPostTicker,
)
from app.models.theme import ContentItem, ContentSource
from app.services.social_theme_projection_service import _lock_registry
from app.services.social_theme_projection_service import SocialThemeProjectionService, _decode
from app.infra.db.models.social_analysis import SocialRunWork, SocialExtractionWork
from app.services.social_extraction_service import SocialExtractionService
from app.services.twitter_content_identity import twitter_external_id

METRICS = ("likes", "reposts", "replies", "quotes", "bookmarks", "views")


def utc(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def serialized(value):
    return json.loads(json.dumps(asdict(value), default=lambda v: v.isoformat() if isinstance(v, datetime) else str(v)))


def _restore_record(kind, value):
    """Decode the trusted, typed run-context schema, including nested tuple evidence."""
    if value is None:
        return None
    if get_origin(kind) is UnionType:
        return _restore_record(next(k for k in get_args(kind) if k is not type(None)), value)
    if get_origin(kind) is tuple:
        args = get_args(kind)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_restore_record(args[0], v) for v in value)
        if len(args) != len(value):
            raise ValueError("invalid_context_tuple")
        return tuple(_restore_record(k, v) for k, v in zip(args, value))
    if kind in (datetime, date):
        return kind.fromisoformat(value)
    if kind is Decimal:
        return Decimal(value)
    if is_dataclass(kind):
        hints = get_type_hints(kind)
        return kind(**{key: _restore_record(hints[key], val) for key, val in value.items()})
    return value


class SocialSignalWriter:
    def __init__(self, session_factory, *, clock=None, confirmation_reader_factory=None):
        self.session_factory = session_factory
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        if confirmation_reader_factory is None:
            from app.services.social_confirmation_reader import SocialConfirmationReader
            from app.services.market_calendar_service import MarketCalendarService
            calendar = MarketCalendarService(use_shared_cache=False)
            confirmation_reader_factory = lambda db: SocialConfirmationReader(db, calendar=calendar)
        self.confirmation_reader_factory = confirmation_reader_factory

    def create_run(self, run_id, as_of):
        validate_utc_timestamp(as_of, "as_of")
        with self.session_factory.begin() as db:
            self._create_run(db, run_id, as_of)
        return run_id

    @staticmethod
    def _create_run(db, run_id, as_of):
        registry = _lock_registry(db)
        if registry is None or registry.mode not in {"live", "validation"} or registry.provider == "disabled":
            raise ValueError("social_runtime_unavailable")
        sources = db.scalars(select(SocialSourceConfiguration).where(SocialSourceConfiguration.lifecycle_state == "enabled").order_by(SocialSourceConfiguration.content_source_id)).all()
        if len(sources) < 2:
            raise ValueError("minimum_two_enabled")
        if db.get(SocialSignalRun, run_id) is not None:
            raise ValueError("run_exists")
        run = SocialSignalRun(id=run_id, registry_id=1, registry_version=registry.version,
                mode=registry.mode, provider=registry.provider, status="running", created_at=as_of,
                source_outcomes_json={}, application_progress_json={"sources": {
                    str(s.content_source_id): {
                        "list_id": s.x_list_id,
                        "name": db.get(ContentSource, s.content_source_id).name,
                        "version": s.version,
                    } for s in sources}, "observations": {}},
                feature_run_ids_json={}, exposure_dates_json={}, coverage_json={})
        db.add(run)
        return run

    def create_replay_run(self, run_id, saved_run_id, as_of):
        """Seed a current evaluation without asserting another collection occurred."""
        validate_utc_timestamp(as_of, "as_of")
        with self.session_factory.begin() as db:
            run = self._create_run(db, run_id, as_of)
            # Production sessions disable autoflush. Persist the parent before
            # adding SocialRunWork rows so PostgreSQL can enforce the FK in the
            # intended order once prior extraction successes exist.
            db.flush()
            old = db.get(SocialSignalRun, saved_run_id)
            if old is None or utc(old.created_at) > as_of:
                raise ValueError("saved_generation_unavailable")
            old_payload = old.application_progress_json
            provenance = old_payload.get("historical", {"sources": old_payload["sources"],
                "provider": old.provider, "registry_version": old.registry_version})
            if old.status == "running":
                historical_ids = tuple(db.scalars(select(SocialRunWork.work_id).where(SocialRunWork.run_id == old.id).order_by(SocialRunWork.work_id)))
            else:
                historical_ids = old_payload.get("prepared", {}).get("work_ids")
                if historical_ids is None and old.status == "failed":
                    failure = old_payload.get("failure", {})
                    if failure.get("reason_code") == "analysis_failed":
                        historical_ids = failure.get("work_ids")
                if historical_ids is None:
                    raise ValueError("terminal_run_manifest_missing")
                historical_ids = tuple(historical_ids)
            historical_ids = tuple(sorted(set(historical_ids)
                | set(old_payload.get("replay", {}).get("historical_work_ids", ()))
                | set(old_payload.get("current_manifest", {}).get("audit_work_ids", ()))))
            observations, missing = {}, []
            for source_id, pin in run.application_progress_json["sources"].items():
                reason = None
                if provenance["provider"] != run.provider:
                    reason = "saved_provider_incompatible"
                elif provenance["sources"].get(source_id) != pin:
                    reason = "saved_source_incompatible"
                elif source_id not in old_payload["observations"]:
                    reason = "saved_source_missing"
                if reason:
                    missing.append((source_id, reason))
                else:
                    observations[source_id] = deepcopy(old_payload["observations"][source_id])
            current, older, after = self._partition_inputs(observations, as_of)
            carry, linked = [], set()
            for work_id in historical_ids:
                work = db.get(SocialExtractionWork, work_id)
                if work is None or work.state != "succeeded":
                    continue
                identity = (work.content_item_id, work.input_hash)
                if identity not in current | older or identity in linked:
                    continue
                _decode(work)
                db.add(SocialRunWork(run_id=run_id, work_id=work.id, input_hash=work.input_hash, included_at=as_of))
                linked.add(identity)
                if identity in older:
                    carry.append(work.id)
            manifest = SocialReplayManifest(saved_run_id, historical_ids, tuple(sorted(observations)),
                tuple(missing), tuple(ReplayInput(*identity, "current") for identity in sorted(current)),
                tuple(carry), ("saved_observations_replayed",) +
                (("saved_read_does_not_cover_current_end",) if any(datetime.fromisoformat(v["observed_at"]) < as_of for v in observations.values()) else ()) +
                (("publication_after_evaluation",) if after else ()) +
                (("outside_window_judgments_missing",) if older - linked else ()))
            run.application_progress_json = {**run.application_progress_json,
                "observations": deepcopy(old_payload["observations"]),
                "replay": serialized(manifest), "historical": deepcopy(provenance)}
            run.source_outcomes_json = deepcopy(old.source_outcomes_json)
        return run_id

    def resume_existing_run(self, run_id, expected_mode_version):
        """Finish or collapse a duplicate delivery without repeating provider I/O."""
        with self.session_factory() as db:
            run = db.get(SocialSignalRun, run_id)
            if run is None:
                raise ValueError("run_not_found")
            if run.status == "running":
                return None
            if run.status in {"staged", "published"}:
                pass
            elif run.status == "failed":
                summary = tuple(
                    (key, value.get("history_status", "limited"))
                    for key, value in sorted(run.source_outcomes_json.items())
                )
                return SocialRunResult(
                    run_id, run.mode, "failed", False, summary, ("run_failed",)
                )
            else:
                raise ValueError("unsupported_existing_run_state")
        return self.publish(run_id, expected_mode_version)

    def fail_analysis(self, run_id, reason_code):
        """Close a generation on terminal extraction failure without publishing it."""
        if reason_code != "analysis_failed":
            raise ValueError("unsupported_social_run_failure")
        completed_at = self.clock()
        validate_utc_timestamp(completed_at, "completed_at")
        with self.session_factory.begin() as db:
            _lock_registry(db)
            run = db.get(SocialSignalRun, run_id)
            if run is None or run.status != "running":
                raise ValueError("run_not_collecting")
            work_ids = tuple(db.scalars(
                select(SocialRunWork.work_id)
                .where(SocialRunWork.run_id == run_id)
                .order_by(SocialRunWork.work_id)
            ))
            run.application_progress_json = {
                **deepcopy(run.application_progress_json),
                "failure": {
                    "reason_code": reason_code,
                    "work_ids": list(work_ids),
                },
            }
            run.source_outcomes_json = {
                source_id: {**value, "processing_status": "failed"}
                for source_id, value in run.source_outcomes_json.items()
            }
            run.coverage_json = deepcopy(run.source_outcomes_json)
            run.status, run.completed_at = "failed", completed_at

    @staticmethod
    def _post(value):
        value = dict(value)
        for field in ("created_at", "observed_at"):
            value[field] = datetime.fromisoformat(value[field])
        return SocialPostRecord(**value)

    @classmethod
    def _partition_inputs(cls, observations, as_of):
        current, older, after = set(), set(), set()
        for observation in observations.values():
            for entry in observation["inputs"]:
                post = cls._post(entry["post"])
                identity = (entry["content_item_id"], SocialExtractionService.input_hash((post,)))
                if post.created_at > as_of:
                    after.add(identity)
                else:
                    (current if post.created_at >= as_of - timedelta(days=14) else older).add(identity)
        return current, older, after

    def select_current_inputs(self, run_id):
        """Classify work on a running generation, preserving aged unfinished audit."""
        with self.session_factory.begin() as db:
            _lock_registry(db)
            run = db.get(SocialSignalRun, run_id)
            if run is None or run.status != "running":
                raise ValueError("run_not_collecting")
            payload = deepcopy(run.application_progress_json)
            observations = payload["observations"]
            replay = payload.get("replay")
            if replay:
                observations = {k: v for k, v in observations.items() if k in replay["participating_source_ids"]}
            current, older, after = self._partition_inputs(observations, utc(run.created_at))
            audit = set(payload.get("current_manifest", {}).get("audit_work_ids", ()))
            scoring = set(payload.get("current_manifest", {}).get("scoring_work_ids", ()))
            if replay:
                audit.update(replay["historical_work_ids"])
            carry, kept, succeeded = [], set(), set()
            for link in db.scalars(select(SocialRunWork).where(SocialRunWork.run_id == run_id).order_by(SocialRunWork.work_id)):
                work = db.get(SocialExtractionWork, link.work_id)
                if work is None or work.input_hash != link.input_hash:
                    raise ValueError("pinned_input_mismatch")
                identity = (work.content_item_id, work.input_hash)
                if identity not in current | older | after:
                    published = datetime.fromisoformat(work.input_snapshot_json["created_at"])
                    if utc(run.created_at) - timedelta(days=15) <= published <= utc(run.created_at):
                        scoring.add(work.id)
                        kept.add(work.id)
                        continue
                if identity in after or (identity in older and work.state != "succeeded"):
                    audit.add(work.id)
                    db.delete(link)
                    continue
                kept.add(work.id)
                if identity in older:
                    _decode(work)
                    carry.append(work.id)
                    succeeded.add(identity)
            reasons = (("outside_window_judgments_missing",) if older - succeeded else ()) + (
                ("publication_after_evaluation",) if after else ())
            manifest = SocialCurrentInputManifest(tuple(ReplayInput(*v, "current") for v in sorted(current)),
                tuple(carry), tuple(sorted(audit - kept)), reasons, tuple(sorted(scoring & kept)))
            payload["current_manifest"] = serialized(manifest)
            run.application_progress_json = payload
        return self.read_run_inputs(run_id)

    def current_inputs_ready(self, run_id):
        """Return whether every pinned current input has a valid saved result."""
        with self.session_factory() as db:
            run = db.get(SocialSignalRun, run_id)
            if run is None or run.status != "running":
                return False
            selection = run.application_progress_json.get("current_manifest")
            if selection is None:
                return False
            expected = {(value["content_item_id"], value["input_hash"])
                        for value in selection["required_inputs"]}
            complete = set()
            expected_work_ids = set(selection.get("scoring_work_ids", ()))
            for link in db.scalars(select(SocialRunWork).where(SocialRunWork.run_id == run_id)):
                work = db.get(SocialExtractionWork, link.work_id)
                if (work is not None and work.state == "succeeded"
                        and link.input_hash == work.input_hash):
                    _decode(work)
                    complete.add((work.content_item_id, work.input_hash))
                    expected_work_ids.discard(work.id)
            return expected <= complete and not expected_work_ids

    def persist_observations(self, batch, *, run_id=None):
        if batch.request.intent == "test":
            raise ValueError("diagnostic_observations_forbidden")
        if len(batch.posts) > batch.request.limit or batch.outcome.received_count != len(batch.posts):
            raise ValueError("invalid_bounded_batch")
        ingress_at = self.clock()
        validate_utc_timestamp(ingress_at, "ingress_at")
        validate_utc_timestamp(batch.request.observed_at, "request_observed_at", reference_at=ingress_at)
        for post in batch.posts:
            validate_utc_timestamp(post.observed_at, "post_observed_at", reference_at=ingress_at)
            validate_utc_timestamp(post.created_at, "post_created_at", reference_at=ingress_at)
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
                if "replay" in run.application_progress_json:
                    raise ValueError("replay_collection_forbidden")
                pin = run.application_progress_json["sources"].get(batch.request.source_id)
                source_name = db.get(ContentSource, source.content_source_id).name
                if (pin != {"list_id": source.x_list_id, "name": source_name, "version": source.version}
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
                external_id = twitter_external_id(post.provider_post_id)
                item = db.scalar(select(ContentItem).where(
                    ContentItem.source_type == "twitter",
                    ContentItem.external_id == external_id,
                ))
                if item is None:
                    # Reuse observations written by early Social prereleases,
                    # which stored the provider id before the legacy identity
                    # convention was unified.
                    item = db.scalar(select(ContentItem).where(
                        ContentItem.source_type == "twitter",
                        ContentItem.external_id == post.provider_post_id,
                    ))
                if item is None:
                    item = ContentItem(source_id=source.content_source_id, source_type="twitter", external_id=external_id,
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

    def latest_collection_progress(self, source_id, provider):
        with self.session_factory() as db:
            observations = []
            for run in db.scalars(select(SocialSignalRun).where(SocialSignalRun.provider == provider)):
                if "replay" in run.application_progress_json:
                    continue
                outcome = run.source_outcomes_json.get(str(source_id), {})
                observation = run.application_progress_json.get("observations", {}).get(str(source_id))
                if observation and outcome.get("read_status") == "success" and observation.get("intent") == "initial":
                    observations.append((
                        observation["observed_at"], utc(run.created_at), run.id,
                        observation.get("committed_progress"),
                        outcome.get("history_status"),
                    ))
            if not observations:
                return None
            _, _, _, cursor, history_status = max(
                observations, key=lambda value: value[:3]
            )
            return SocialCollectionProgress(
                initial_complete=(history_status == "observed_window" or cursor is None),
                cursor=cursor,
            )

    def latest_committed_progress(self, source_id, provider):
        """Compatibility reader for diagnostics that only display the cursor."""
        progress = self.latest_collection_progress(source_id, provider)
        return progress.cursor if progress is not None else None

    def read_run_inputs(self, run_id):
        """Frozen historical read evidence; not a claim of a new provider read."""
        with self.session_factory() as db:
            run = db.get(SocialSignalRun, run_id)
            if run is None:
                raise ValueError("run_unavailable")
            batches, content_ids = [], set()
            observations = run.application_progress_json["observations"]
            outcomes = run.source_outcomes_json
            for source_id, observation in sorted(observations.items()):
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
                outcome = dict(outcomes[source_id])
                for field in ("observed_oldest_at", "observed_newest_at", "rate_limit_reset_at"):
                    outcome[field] = datetime.fromisoformat(outcome[field]) if outcome[field] else None
                outcome["coverage_reason_codes"] = tuple(outcome["coverage_reason_codes"])
                outcome["known_gap_intervals"] = tuple(tuple(datetime.fromisoformat(v) for v in gap) for gap in outcome["known_gap_intervals"])
                batches.append(SocialSourceBatch(SocialReadRequest(**request), tuple(posts), SocialSourceOutcome(**outcome)))
            if run.status == "running":
                work_ids = tuple(db.scalars(select(SocialRunWork.work_id).where(SocialRunWork.run_id == run_id).order_by(SocialRunWork.work_id)))
            else:
                manifest = run.application_progress_json.get("prepared", {}).get("work_ids")
                if manifest is None and run.status == "failed":
                    failure = run.application_progress_json.get("failure", {})
                    if failure.get("reason_code") == "analysis_failed":
                        manifest = failure.get("work_ids")
                if manifest is None:
                    raise ValueError("terminal_run_manifest_missing")
                work_ids = tuple(manifest)
            replay = run.application_progress_json.get("replay")
            manifest = SocialReplayManifest(replay["saved_run_id"], tuple(replay["historical_work_ids"]),
                tuple(replay["participating_source_ids"]), tuple(tuple(v) for v in replay["missing_sources"]),
                tuple(ReplayInput(**v) for v in replay["required_inputs"]), tuple(replay["carry_in_work_ids"]),
                tuple(replay["coverage_reasons"])) if replay else None
            current_manifest = _restore_record(SocialCurrentInputManifest, run.application_progress_json.get("current_manifest"))
            return SavedSocialRunInputs(run_id, utc(run.created_at), run.registry_version, tuple(batches), tuple(sorted(content_ids)), work_ids, manifest, current_manifest)

    def _validate_inputs(self, db, run):
        pins = run.application_progress_json["sources"]
        observations = run.application_progress_json["observations"]
        outcomes = run.source_outcomes_json
        replay = run.application_progress_json.get("replay")
        if replay:
            participants = set(replay["participating_source_ids"])
            observations = {k: v for k, v in observations.items() if k in participants}
            outcomes = {k: v for k, v in outcomes.items() if k in participants}
        if len(pins) < 2 or set(outcomes) != set(pins) or set(observations) != set(pins):
            raise ValueError("source_participation_incomplete")
        if any(outcome["read_status"] != "success" for outcome in outcomes.values()):
            raise ValueError("source_participation_failed")
        required = set()
        for source in observations.values():
            for observation in source["inputs"]:
                value = dict(observation["post"])
                value.update(created_at=datetime.fromisoformat(value["created_at"]), observed_at=datetime.fromisoformat(value["observed_at"]))
                required.add((observation["content_item_id"], SocialExtractionService.input_hash((SocialPostRecord(**value),))))
        selection = run.application_progress_json.get("current_manifest") or replay
        if selection:
            required = {(v["content_item_id"], v["input_hash"]) for v in selection["required_inputs"]}
        actual, ids = set(), []
        scoring_ids = set(selection.get("scoring_work_ids", ())) if selection else set()
        for link in db.scalars(select(SocialRunWork).where(SocialRunWork.run_id == run.id).order_by(SocialRunWork.work_id)):
            work = db.get(SocialExtractionWork, link.work_id)
            if work is None or link.input_hash != work.input_hash:
                raise ValueError("pinned_input_mismatch")
            _decode(work)
            if not selection or work.id not in set(selection["carry_in_work_ids"]) | scoring_ids:
                actual.add((work.content_item_id, work.input_hash))
            ids.append(work.id)
        if actual != required:
            raise ValueError("pinned_inputs_incomplete")
        return tuple(ids)

    def read_publication_context(self, run_id):
        """Read only persisted evidence, never mutable latest Market fields."""
        with self.session_factory() as db:
            run = db.get(SocialSignalRun, run_id)
            if run is None:
                raise ValueError("run_unavailable")
            value = run.application_progress_json.get("prepared", {}).get("context")
            return _restore_record(SocialPublicationContext, value) if value else None

    @staticmethod
    def _validate_candidates(context, rows):
        from app.domain.social_signals.scoring import queue_score, score_confirmation
        from app.domain.social_signals.states import classify_signal_state
        candidates = {(v.candidate_key, v.window_days): v for v in context.candidates}
        if set(candidates) != {(v.candidate_key, v.window_days) for v in rows}:
            raise ValueError("candidate_context_mismatch:identity")
        markets = {batch.pinned_run.market: batch for batch in context.market_batches}
        for row in rows:
            candidate = candidates[(row.candidate_key, row.window_days)]
            decision = classify_signal_state(candidate.state_input)
            if (candidate.state_decision != decision
                    or (candidate.social_result and candidate.social_result.state != decision)):
                raise ValueError("candidate_context_mismatch:state_decision")
            if row.market:
                batch = markets.get(row.market)
                fact = dict(batch.facts).get(row.canonical_symbol) if batch else None
                confirmation_input = next((value for value in batch.inputs
                    if value.candidate_key == f"{row.market}:{row.canonical_symbol}"), None) if batch else None
                if fact is None or confirmation_input is None:
                    raise ValueError("candidate_context_mismatch:market_inputs")
                state = candidate.state_input
                if state != fact.to_signal_state_input(resolved=state.resolved, active=state.active,
                        market=row.market, security_kind=state.security_kind):
                    raise ValueError("candidate_context_mismatch:frozen_checks")
                if candidate.confirmation != score_confirmation(confirmation_input):
                    raise ValueError("candidate_context_mismatch:confirmation_input")
            elif candidate.confirmation is not None:
                raise ValueError("candidate_context_mismatch:confirmation_without_market")
            social = candidate.social_result.social_score if candidate.social_result else None
            confirmation = candidate.confirmation.value if candidate.confirmation else None
            if (row.candidate_state != candidate.state_decision.state or row.market != candidate.state_input.market
                    or row.social_score != social or row.confirmation_score != confirmation
                    or row.queue_score != queue_score(social=social, confirmation=confirmation)
                    or (candidate.social_result and (candidate.social_result.canonical_symbol != row.canonical_symbol
                        or candidate.social_result.formula_version != context.formula_version))):
                raise ValueError("candidate_context_mismatch:state_or_score")

    @staticmethod
    def _validate_social_results(db, run_id, context, as_of):
        if context is None or context.scoring_input_version is None:
            return
        if context.scoring_input_version != "rolling-observations-v1":
            raise ValueError("candidate_context_mismatch:scoring_input_version")
        from app.domain.social_signals.scoring import score_social_candidates
        from app.infra.db.repositories.social_refresh_support import SocialScoringEvidenceReader
        evidence = SocialScoringEvidenceReader.read_in_session(db, run_id, as_of)
        expected = {(item.candidate_key, item.window_days): item
                    for item in context.candidates}
        actual = {}
        for window_days in (1, 7, 14):
            for result in score_social_candidates(evidence, window_days, as_of):
                key = (result.candidate_key, window_days)
                candidate = expected.get(key)
                if candidate is not None:
                    result = replace(result, state=candidate.state_decision)
                actual[key] = result
        supplied = {key: value.social_result for key, value in expected.items()
                    if value.social_result is not None}
        if actual != supplied:
            raise ValueError("candidate_context_mismatch:social_inputs")

    def _validate_context(self, db, context, as_of, *, run_id=None):
        if context is None:
            return
        if run_id is not None:
            self._validate_social_results(db, run_id, context, as_of)
        now = self.clock()
        validate_utc_timestamp(now, "publication_clock")
        if now < as_of:
            raise ValueError("market_context_changed:clock_before_generation")
        reader = self.confirmation_reader_factory(db)
        for batch in context.market_batches:
            market = batch.pinned_run.market
            if batch.market_context.market != market or batch.market_context.observed_at != as_of:
                raise ValueError("market_context_changed:generation")
            if reader.pin_feature_run(market) != batch.pinned_run:
                raise ValueError("market_context_changed:feature_pointer")
            for evidence in batch.theme_evidence:
                if (evidence.market != market or evidence.benchmark_symbol != batch.market_context.benchmark_symbol
                        or evidence.session_date != batch.market_context.freshness.required_session
                        or evidence.benchmark_candidates != batch.market_context.benchmark_candidates
                        or evidence.benchmark_registry_version != batch.market_context.benchmark_registry_version
                        or any(run_id != batch.pinned_run.run_id for _, run_id in evidence.feature_run_ids)
                        or set(dict(evidence.feature_run_ids)) != {m.canonical_symbol for m in evidence.membership}):
                    raise ValueError("market_context_changed:theme_pin")
            for value in batch.inputs:
                linked = tuple(e for e in batch.theme_evidence if any(
                    f"{m.market}:{m.canonical_symbol}" == value.candidate_key for m in e.membership))
                if value.theme_confirmations != linked:
                    raise ValueError("market_context_changed:theme_membership")
            expected = replace(batch, inputs=tuple(replace(v, theme_confirmations=()) for v in batch.inputs),
                theme_evidence=(), theme_reasons=())
            for at in dict.fromkeys((as_of, now)):
                actual = reader.read_market(market, tuple(symbol for symbol, _ in batch.facts), at,
                    pinned_run=batch.pinned_run, theme_keys=())
                actual = replace(actual, market_context=replace(actual.market_context, observed_at=as_of),
                    inputs=tuple(replace(v, observed_at=as_of) for v in actual.inputs))
                if actual != expected:
                    raise ValueError("market_context_changed:confirmation_inputs")

    def prepare_run(self, run_id, rows, as_of, *, theme_evidence=(), context=None):
        validate_utc_timestamp(as_of, "as_of")
        self.select_current_inputs(run_id)
        # Read/validate expensive saved input and measurement work before locking.
        with self.session_factory() as db:
            run = db.get(SocialSignalRun, run_id)
            if run is None or run.status != "running":
                raise ValueError("run_not_collecting")
            if utc(run.created_at) != as_of:
                raise ValueError("generation_time_mismatch")
            work_ids = self._validate_inputs(db, run)
            if context is not None:
                if not isinstance(context, SocialPublicationContext):
                    raise TypeError("invalid_publication_context")
                self._validate_candidates(context, rows)
                progress = tuple((key, run.source_outcomes_json[key]["committed_progress"])
                    for key in sorted(run.application_progress_json["sources"]))
                versions = tuple(sorted({(work.selected_model, work.prompt_version, work.schema_version)
                    for work in (db.get(SocialExtractionWork, work_id) for work_id in work_ids)}))
                if context.source_progress and context.source_progress != progress:
                    raise ValueError("context_source_progress_mismatch")
                if context.extraction_versions and context.extraction_versions != versions:
                    raise ValueError("context_extraction_version_mismatch")
                if any(row.formula_version != context.formula_version for row in rows):
                    raise ValueError("context_formula_version_mismatch")
                context = replace(context, source_progress=progress, extraction_versions=versions)
                context_themes = tuple(e for batch in context.market_batches for e in batch.theme_evidence)
                if theme_evidence and theme_evidence != context_themes:
                    raise ValueError("context_theme_evidence_mismatch")
                theme_evidence = context_themes
                # Also preloads local calendar providers before the final lock.
                self._validate_context(db, context, as_of, run_id=run_id)
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
            result = PreparedSocialPublication(run_id, run.registry_version, as_of, work_ids, rows, theme_evidence, context)
            frozen_input = deepcopy(run.application_progress_json)
        with self.session_factory.begin() as db:
            registry = _lock_registry(db)
            run = db.get(SocialSignalRun, run_id)
            if run.status != "running" or run.application_progress_json != frozen_input or registry.version != result.registry_version:
                raise ValueError("publication_preparation_changed")
            if self._validate_inputs(db, run) != work_ids:
                raise ValueError("publication_preparation_changed")
            self._validate_context(db, context, as_of, run_id=run_id)
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
                "basket_fingerprint": application.fingerprint, "baskets": [serialized(b) for b in application.baskets],
                "context": serialized(context) if context else None}}
            if context is not None:
                run.feature_run_ids_json = {b.pinned_run.market: b.pinned_run.run_id for b in context.market_batches}
                run.exposure_dates_json = {b.pinned_run.market: b.market_context.freshness.actual_session.isoformat()
                    if b.market_context.freshness.actual_session else None for b in context.market_batches}
            run.completed_at, run.status = as_of, "staged"
            run.source_outcomes_json = {
                source_id: {**value, "processing_status": "complete"}
                for source_id, value in run.source_outcomes_json.items()
            }
            run.coverage_json = {key: deepcopy(run.source_outcomes_json[key]) for key in frozen_input["sources"]}
        return result

    def publish(self, run_id, expected_mode_version):
        with self.session_factory() as db:
            run = db.get(SocialSignalRun, run_id)
            if run is None or run.status not in {"staged", "published"}:
                raise ValueError("run_not_prepared")
            summary = tuple((key, run.source_outcomes_json[key]["history_status"])
                for key in sorted(run.application_progress_json["sources"]))
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
            context = _restore_record(SocialPublicationContext, prepared_data.get("context"))
            self._validate_context(db, context, utc(run.created_at), run_id=run_id)
            application = service.prepare_application(projection, theme_keys=tuple(e["theme_key"] for e in prepared_data["theme_evidence"]))
            if (application.fingerprint != prepared_data["basket_fingerprint"]
                    or [serialized(b) for b in application.baskets] != prepared_data["baskets"]):
                raise ValueError("publication_basket_changed")
        with self.session_factory.begin() as db:
            registry = _lock_registry(db)
            run = db.get(SocialSignalRun, run_id)
            if registry.mode != "live" or registry.version != expected_mode_version or run.registry_version != expected_mode_version:
                raise ValueError("publication_live_version_changed")
            sources = {str(s.content_source_id): {
                    "list_id": s.x_list_id,
                    "name": db.get(ContentSource, s.content_source_id).name,
                    "version": s.version,
                }
                for s in db.scalars(select(SocialSourceConfiguration).where(SocialSourceConfiguration.lifecycle_state == "enabled"))}
            if sources != run.application_progress_json["sources"] or registry.provider != run.provider:
                raise ValueError("publication_source_version_changed")
            self._validate_context(db, context, utc(run.created_at), run_id=run_id)
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
