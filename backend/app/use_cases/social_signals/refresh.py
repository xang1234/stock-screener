"""Build one immutable Social Signal generation from bounded saved evidence."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Callable, Mapping
from uuid import uuid4

from app.domain.social_signals.records import (
    CandidatePublicationContext,
    SocialPublicationContext,
    SocialReadRequest,
    SocialRunResult,
    SocialSnapshotRecord,
    validate_utc_timestamp,
)
from app.domain.social_signals.scoring import (
    queue_score,
    score_confirmation,
    score_social_candidates,
)
from app.domain.social_signals.states import classify_signal_state


class RefreshSocialSignals:
    """Coordinate collection, metered analysis, scoring, and publication.

    Dependencies deliberately expose small public contracts.  Provider and LLM
    calls happen outside database publication transactions, and the provider
    lease is always released before metered model work starts.
    """

    def __init__(
        self,
        *,
        catalog,
        providers: Mapping[str, object],
        writer,
        backlog,
        evidence_reader,
        theme_service,
        confirmation_reader,
        provider_lease,
        run_id_factory: Callable[[str, datetime], str],
        input_hash: Callable[[object], str],
        lease_owner_factory: Callable[[], str] | None = None,
        initial_days: int = 14,
        initial_limit: int = 1000,
        incremental_limit: int = 200,
        lease_ttl_seconds: int = 1800,
    ) -> None:
        if initial_days <= 0 or initial_limit <= 0 or incremental_limit <= 0:
            raise ValueError("invalid_social_refresh_limits")
        self.catalog = catalog
        self.providers = dict(providers)
        self.writer = writer
        self.backlog = backlog
        self.evidence_reader = evidence_reader
        self.theme_service = theme_service
        self.confirmation_reader = confirmation_reader
        self.provider_lease = provider_lease
        self.run_id_factory = run_id_factory
        self.input_hash = input_hash
        self.lease_owner_factory = lease_owner_factory or (lambda: uuid4().hex)
        self.initial_days = initial_days
        self.initial_limit = initial_limit
        self.incremental_limit = incremental_limit
        self.lease_ttl_seconds = lease_ttl_seconds

    @staticmethod
    def _result(run_id, mode, status, reason, *, sources=()):
        return SocialRunResult(
            run_id,
            mode,
            status,
            False,
            tuple((source.source_id, "unavailable") for source in sources),
            (reason,),
        )

    def _provider(self, provider_name):
        selected = self.providers.get(provider_name)
        if selected is None:
            raise ValueError("social_provider_not_wired")
        return selected() if callable(selected) and not hasattr(selected, "read_source") else selected

    def _request(self, run_id, source, provider, now):
        progress = self.writer.latest_committed_progress(source.source_id, provider)
        initial = progress is None
        return SocialReadRequest(
            request_id=f"{run_id}:{source.source_id}",
            source_id=source.source_id,
            list_id=source.list_id,
            intent="initial" if initial else "incremental",
            observed_at=now,
            limit=self.initial_limit if initial else self.incremental_limit,
            target_published_after=now - timedelta(days=self.initial_days),
            application_progress=progress,
        )

    async def execute(self, origin: str, now: datetime, *, saved_run_id: str | None = None) -> SocialRunResult:
        validate_utc_timestamp(now, "now")
        if not isinstance(origin, str) or not origin.strip():
            raise ValueError("invalid_refresh_origin")
        runtime, sources, selected_model = self.catalog.read()
        enabled = tuple(source for source in sources if source.lifecycle == "enabled")
        if runtime.mode == "off" or runtime.provider == "disabled":
            return self._result("", "off", "skipped", "social_runtime_unavailable")
        if len(enabled) < 2:
            return self._result("", runtime.mode, "blocked", "minimum_two_enabled", sources=enabled)
        if not isinstance(selected_model, str) or not selected_model.strip():
            return self._result("", runtime.mode, "blocked", "extraction_model_not_configured", sources=enabled)

        run_id = self.run_id_factory(origin, now)
        if saved_run_id is not None:
            self.writer.create_replay_run(run_id, saved_run_id, now)
        else:
            existing = None
            try:
                self.writer.create_run(run_id, now)
            except ValueError as exc:
                if str(exc) != "run_exists":
                    raise
                resumed = self.writer.resume_existing_run(run_id, runtime.version)
                if resumed is not None:
                    return resumed
                existing = self.writer.read_run_inputs(run_id)
                if existing.registry_version != runtime.version:
                    return self._result(
                        run_id, runtime.mode, "blocked", "social_runtime_changed", sources=enabled
                    )
            collected = {batch.request.source_id: batch for batch in existing.batches} if existing else {}
            lease_owner = f"{run_id}:{self.lease_owner_factory()}"
            if not self.provider_lease.acquire(lease_owner, self.lease_ttl_seconds):
                return self._result(run_id, runtime.mode, "deferred", "provider_lease_unavailable", sources=enabled)
            try:
                provider = self._provider(runtime.provider)
                failed = False
                for source in enabled:
                    if source.source_id in collected:
                        failed = failed or collected[source.source_id].outcome.read_status != "success"
                        continue
                    if not self.catalog.admitted(
                        version=runtime.version, mode=runtime.mode, provider=runtime.provider
                    ):
                        return self._result(
                            run_id, runtime.mode, "blocked", "social_runtime_changed", sources=enabled
                        )
                    batch = provider.read_source(self._request(run_id, source, runtime.provider, now))
                    saved = self.writer.persist_observations(batch, run_id=run_id)
                    failed = failed or saved.outcome.read_status != "success"
                if failed:
                    return self._result(
                        run_id, runtime.mode, "blocked", "source_participation_failed", sources=enabled
                    )
            finally:
                self.provider_lease.release(lease_owner)

        selected = self.writer.select_current_inputs(run_id)
        manifest = selected.current_manifest
        if manifest is None:
            raise ValueError("current_input_manifest_missing")
        posts_by_identity = {}
        content_ids = dict(selected.content_ids)
        for batch in selected.batches:
            for post in batch.posts:
                content_id = content_ids.get(post.provider_post_id)
                if content_id is not None:
                    posts_by_identity[(content_id, self.input_hash(post))] = post
        directly_collected_content_ids = {content_id for content_id, _ in posts_by_identity}
        if hasattr(self.evidence_reader, "retained_posts"):
            for content_id, post in self.evidence_reader.retained_posts(run_id, now):
                # The generation already froze a direct observation for this parent.
                # A late attachment belongs to the next generation, never a second
                # same-parent input that would add weight to this one.
                if content_id in directly_collected_content_ids:
                    continue
                posts_by_identity[(content_id, self.input_hash(post))] = post
        run_work_ids = []
        required_identities = {(required.content_item_id, required.input_hash)
                               for required in manifest.required_inputs}
        for identity in sorted(set(posts_by_identity) | required_identities):
            post = posts_by_identity.get(identity)
            if post is None:
                # Replays can refer to an already linked success that is not part
                # of their copied read batch. The writer validates that linkage.
                continue
            run_work_ids.append(self.backlog.enqueue(
                identity[0],
                post,
                selected_model=selected_model,
                now=now,
                run_id=run_id,
            ))
        if run_work_ids:
            processed = await self.backlog.execute(
                now, len(run_work_ids), work_ids=tuple(sorted(set(run_work_ids)))
            )
            if processed.deferred or processed.failed:
                return self._result(run_id, runtime.mode, "deferred", "analysis_incomplete", sources=enabled)
        self.writer.select_current_inputs(run_id)
        if hasattr(self.writer, "current_inputs_ready") and not self.writer.current_inputs_ready(run_id):
            return self._result(run_id, runtime.mode, "deferred", "analysis_incomplete", sources=enabled)

        evidence = self.evidence_reader.read(run_id, now)
        projection = self.theme_service.prepare(run_id, now)
        theme_keys = (
            self.theme_service.theme_keys(projection)
            if hasattr(self.theme_service, "theme_keys")
            else tuple(sorted({claim.theme_key for claim in projection.proposals}))
        )
        application = self.theme_service.prepare_application(projection, theme_keys=theme_keys)

        markets = {}
        for market in sorted({item.market for item in evidence if item.resolved and item.market}):
            symbols = tuple(sorted({item.canonical_symbol for item in evidence
                                    if item.resolved and item.market == market}))
            markets[market] = self.confirmation_reader.read_market(
                market,
                symbols,
                now,
                theme_keys=theme_keys,
                membership_reader=application,
            )

        rows = []
        candidates = []
        for window_days in (1, 7, 14):
            scored = score_social_candidates(evidence, window_days, now)
            for social in scored:
                batch = markets.get(social.market)
                facts = dict(batch.facts) if batch else {}
                fact = facts.get(social.canonical_symbol)
                confirmation_input = next(
                    (value for value in batch.inputs if value.candidate_key == social.candidate_key),
                    None,
                ) if batch else None
                if fact is None:
                    # Resolved rows without coherent Market facts remain visible,
                    # but cannot be ranked as actionable.
                    from app.domain.social_signals.records import SignalStateInput
                    state_input = SignalStateInput(
                        social.candidate_key != "", False, social.market,
                        next((item.security_kind for item in evidence
                              if item.candidate_key == social.candidate_key), "stock"),
                    )
                    confirmation = None
                else:
                    item = next(value for value in evidence if value.candidate_key == social.candidate_key)
                    state_input = fact.to_signal_state_input(
                        resolved=item.resolved,
                        active=item.resolved,
                        market=item.market,
                        security_kind=item.security_kind,
                    )
                    confirmation = score_confirmation(confirmation_input) if confirmation_input else None
                decision = classify_signal_state(state_input)
                social = replace(social, state=decision)
                confirmation_value = confirmation.value if confirmation else None
                blended = queue_score(social=social.social_score, confirmation=confirmation_value)
                coverage = tuple(sorted(set(
                    next(item.coverage_reasons for item in evidence
                         if item.candidate_key == social.candidate_key) + decision.reasons
                )))
                rows.append(SocialSnapshotRecord(
                    run_id, social.candidate_key, social.canonical_symbol or None, social.market,
                    decision.state, social.social_score, confirmation_value, blended,
                    (("state_reasons", ",".join(decision.reasons)),), coverage,
                    social.latest_mention, social.canonical_symbol, social.candidate_key,
                    window_days, social.mention_count, social.observed_list_count,
                    social.enabled_list_count, social.normalization_scope, social.formula_version,
                ))
                candidates.append(CandidatePublicationContext(
                    social.candidate_key, window_days, state_input, decision, social, confirmation
                ))
            scored_keys = {social.candidate_key for social in scored}
            for item in evidence:
                if item.candidate_key in scored_keys:
                    continue
                batch = markets.get(item.market)
                fact = dict(batch.facts).get(item.canonical_symbol) if batch else None
                confirmation_input = next((value for value in batch.inputs
                    if value.candidate_key == item.candidate_key), None) if batch else None
                if fact is not None:
                    state_input = fact.to_signal_state_input(
                        resolved=item.resolved, active=item.resolved, market=item.market,
                        security_kind=item.security_kind,
                    )
                    confirmation = score_confirmation(confirmation_input) if confirmation_input else None
                else:
                    from app.domain.social_signals.records import SignalStateInput
                    state_input = SignalStateInput(
                        item.resolved, item.resolved, item.market, item.security_kind
                    )
                    confirmation = None
                decision = classify_signal_state(state_input)
                current_posts = tuple(post for post in item.posts
                    if now - timedelta(days=window_days) <= post.created_at <= now)
                latest = max((post.created_at for post in current_posts), default=None)
                coverage = tuple(sorted(set(item.coverage_reasons + decision.reasons)))
                confirmation_value = confirmation.value if confirmation else None
                rows.append(SocialSnapshotRecord(
                    run_id, item.candidate_key,
                    item.canonical_symbol if item.resolved else None, item.market,
                    decision.state, None, confirmation_value, None,
                    (("state_reasons", ",".join(decision.reasons)),), coverage,
                    latest, item.canonical_symbol, item.candidate_key, window_days,
                    len(current_posts), len({post.source_id for post in current_posts}),
                    len(item.enabled_source_ids), "not_ranked",
                ))
                candidates.append(CandidatePublicationContext(
                    item.candidate_key, window_days, state_input, decision, None, confirmation
                ))

        market_batches = tuple(markets[key] for key in sorted(markets))
        theme_evidence = tuple(item for batch in market_batches for item in batch.theme_evidence)
        context = SocialPublicationContext(
            market_batches,
            candidates=tuple(candidates),
            scoring_input_version="rolling-observations-v1",
        )
        self.writer.prepare_run(
            run_id, tuple(rows), now, theme_evidence=theme_evidence, context=context
        )
        return self.writer.publish(run_id, runtime.version)


__all__ = ["RefreshSocialSignals"]
