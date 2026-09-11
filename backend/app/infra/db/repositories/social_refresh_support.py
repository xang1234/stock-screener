"""Durable read models used by the Social refresh orchestrator."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import re

from sqlalchemy import select

from app.domain.social_signals.records import SocialEvidenceInput, SocialPostRecord, SocialSourceView
from app.infra.db.models.social_analysis import SocialExtractionWork, SocialRunWork
from app.infra.db.models.social_signals import (
    SocialSignalRun,
    SocialSourceConfiguration,
    SocialSourceRegistry,
)
from app.models.app_settings import AppSetting
from app.models.theme import ContentSource
from app.services.social_company_identity_service import SocialCompanyIdentityService
from app.services.social_source_admin_service import SocialRuntimeState
from app.services.social_ticker_resolver import SocialTickerResolver


def _utc(value):
    return value.replace(tzinfo=timezone.utc) if value is not None and value.tzinfo is None else value


def _with_latest_prepared_evidence(db, content_item_id, post, *, as_of):
    """Make a new generation see durable prepared evidence without mutating its parent post."""
    from app.domain.social_signals.records import SocialPreparedEvidence
    from app.services.live_attachment_service import attachment_snapshot
    import json

    snapshot = attachment_snapshot(db, content_item_id, as_of=as_of)
    if not snapshot["evidence"]:
        return post
    evidence = tuple(SocialPreparedEvidence(
        id=value["id"], kind=value["kind"], url=value["url"], text=value["text"],
        original_text_sha256=value["original_text_sha256"], text_sha256=value["text_sha256"],
        available_at=datetime.fromisoformat(value["available_at"]),
        provenance_json=json.dumps(value["provenance"], sort_keys=True, ensure_ascii=False,
            separators=(",", ":")),
    ) for value in snapshot["evidence"])
    return replace(post, prepared_evidence=evidence, evidence_digest=snapshot["revision"])


class SqlSocialRefreshCatalog:
    """Read the DB-authoritative runtime and its exact enabled source generation."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def read(self):
        with self.session_factory() as db:
            registry = db.get(SocialSourceRegistry, 1, populate_existing=True)
            if registry is None:
                return SocialRuntimeState("off", "disabled", 0), (), None
            rows = db.scalars(select(SocialSourceConfiguration).order_by(
                SocialSourceConfiguration.content_source_id
            )).all()
            sources = []
            for row in rows:
                source = db.get(ContentSource, row.content_source_id)
                sources.append(SocialSourceView(
                    str(source.id), source.name, source.url, row.x_list_id,
                    row.lifecycle_state, row.provenance, None,
                    _utc(row.last_successful_collection_at), row.version,
                    _utc(row.created_at), _utc(row.updated_at),
                ))
            setting = db.scalar(select(AppSetting).where(AppSetting.key == "llm_extraction_model"))
            model = setting.value if setting and setting.value else None
            return SocialRuntimeState(registry.mode, registry.provider, registry.version), tuple(sources), model

    def admitted(self, *, version, mode, provider):
        with self.session_factory() as db:
            registry = db.get(SocialSourceRegistry, 1, populate_existing=True)
            return registry is not None and (
                registry.version, registry.mode, registry.provider
            ) == (version, mode, provider)


class SocialScoringEvidenceReader:
    """Rebuild rolling evidence from immutable successful collection records.

    Current source participation comes from the requested run.  Posts and their
    latest-at-as-of non-null metrics may come from earlier successful generations,
    so a capped incremental response cannot erase a valid 7/14-day mention.
    """

    _CASHTAG = re.compile(r"\$[A-Za-z0-9][A-Za-z0-9.\-]{0,19}")
    _METRICS = ("likes", "reposts", "replies", "quotes", "bookmarks", "views")

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def retained_posts(self, run_id, as_of):
        """Return canonical retained post revisions to pin to this generation."""
        with self.session_factory() as db:
            current = db.get(SocialSignalRun, run_id)
            if current is None:
                raise ValueError("run_unavailable")
            source_ids = set(current.application_progress_json["sources"])
            canonical = {}
            content_ids = {}
            runs = db.scalars(select(SocialSignalRun).where(
                SocialSignalRun.provider == current.provider,
                SocialSignalRun.created_at <= as_of,
            ).order_by(SocialSignalRun.created_at, SocialSignalRun.id)).all()
            for generation in runs:
                if ("replay" in generation.application_progress_json
                        or generation.id == current.id
                        or _utc(generation.created_at) >= as_of):
                    continue
                for source_id, observation in generation.application_progress_json.get("observations", {}).items():
                    if source_id not in source_ids or generation.source_outcomes_json.get(source_id, {}).get("read_status") != "success":
                        continue
                    for entry in observation.get("inputs", ()):
                        data = dict(entry["post"])
                        data["created_at"] = datetime.fromisoformat(data["created_at"])
                        data["observed_at"] = datetime.fromisoformat(data["observed_at"])
                        post = SocialPostRecord(**data)
                        if (post.created_at < as_of - timedelta(days=15)
                                or post.created_at > as_of or post.observed_at > as_of):
                            continue
                        key = (post.provider_post_id, source_id)
                        previous = canonical.get(key)
                        if previous is None or previous.observed_at <= post.observed_at:
                            if previous is not None:
                                post = replace(post, **{
                                    field: getattr(post, field) if getattr(post, field) is not None else getattr(previous, field)
                                    for field in self._METRICS
                                })
                            canonical[key] = post
                            content_ids[post.provider_post_id] = entry["content_item_id"]
            unique = {}
            for (post_id, _), post in canonical.items():
                content_id = content_ids[post_id]
                post = _with_latest_prepared_evidence(db, content_id, post, as_of=as_of)
                unique[(content_id, self._content_revision(post))] = post
            return tuple((content_id, post) for (content_id, _), post in sorted(unique.items()))

    @staticmethod
    def _content_revision(post):
        from app.services.social_extraction_service import SocialExtractionService
        return SocialExtractionService.input_hash((post,))

    def read(self, run_id, as_of):
        with self.session_factory() as db:
            current = db.get(SocialSignalRun, run_id)
            if current is None:
                raise ValueError("run_unavailable")
            source_ids = tuple(sorted(current.application_progress_json["sources"]))
            outcomes = current.source_outcomes_json
            reasons = set(current.application_progress_json.get("current_manifest", {}).get("coverage_reasons", ()))
            for source_id in source_ids:
                reasons.update(outcomes.get(source_id, {}).get("coverage_reason_codes", ()))
            # One canonical observation per post/source. Apply non-null metrics in
            # observation order, retaining an older known value when a refresh omits it.
            canonical = {}
            content_ids = {}
            runs = db.scalars(select(SocialSignalRun).where(
                SocialSignalRun.provider == current.provider,
                SocialSignalRun.created_at <= as_of,
            ).order_by(SocialSignalRun.created_at, SocialSignalRun.id)).all()
            complete_sources = set()
            for generation in runs:
                if ("replay" in generation.application_progress_json
                        or (generation.id != current.id and _utc(generation.created_at) >= as_of)):
                    continue
                for source_id, observation in generation.application_progress_json.get("observations", {}).items():
                    if source_id not in source_ids or generation.source_outcomes_json.get(source_id, {}).get("read_status") != "success":
                        continue
                    outcome = generation.source_outcomes_json[source_id]
                    if (outcome.get("history_status") == "observed_window"
                            and not outcome.get("known_gap_intervals")):
                        complete_sources.add(source_id)
                    for entry in observation.get("inputs", ()):
                        data = dict(entry["post"])
                        data["created_at"] = datetime.fromisoformat(data["created_at"])
                        data["observed_at"] = datetime.fromisoformat(data["observed_at"])
                        post = SocialPostRecord(**data)
                        if (post.created_at < as_of - timedelta(days=15)
                                or post.created_at > as_of or post.observed_at > as_of):
                            continue
                        key = (post.provider_post_id, source_id)
                        previous = canonical.get(key)
                        if previous is None or previous.observed_at <= post.observed_at:
                            if previous is not None:
                                post = replace(post, **{
                                    field: getattr(post, field) if getattr(post, field) is not None else getattr(previous, field)
                                    for field in self._METRICS
                                })
                            canonical[key] = post
                            content_ids[post.provider_post_id] = entry["content_item_id"]

            history_complete = set(source_ids) <= complete_sources

            identity = SocialCompanyIdentityService(db).read()
            resolver = SocialTickerResolver(db, verified_company_ids=identity.verified_company_ids)
            candidates = {}
            judgments = {}
            claims = {}
            by_content = {}
            for (post_id, _), post in canonical.items():
                by_content.setdefault(content_ids[post_id], post_id)
            work_ids = tuple(db.scalars(select(SocialRunWork.work_id).where(
                SocialRunWork.run_id == run_id
            ).order_by(SocialRunWork.work_id)))
            for work_id in work_ids:
                work = db.get(SocialExtractionWork, work_id)
                if work is None or work.state != "succeeded" or not isinstance(work.result_json, dict):
                    continue
                post_id = by_content.get(work.content_item_id)
                if post_id is None:
                    continue
                for value in work.result_json.get("judgments", ()):
                    if value.get("post_id") == post_id:
                        judgments[post_id] = (
                            value.get("has_new_thesis"), value.get("canonical_claim_key")
                        )
                claims.setdefault(post_id, []).extend(work.result_json.get("claims", ()))

            for (post_id, source_id), post in canonical.items():
                judgment = judgments.get(post_id)
                if judgment is not None:
                    post = replace(post, has_new_thesis=judgment[0], canonical_claim_key=judgment[1])
                    canonical[(post_id, source_id)] = post
                resolutions = [resolver.resolve(token) for token in sorted(set(self._CASHTAG.findall(post.text)))]
                resolutions.extend(resolver.resolve(value.get("company_token", ""))
                                    for value in claims.get(post_id, ()))
                for resolution in resolutions:
                    resolved = resolution.status == "resolved"
                    key = (f"{resolution.market}:{resolution.symbol}" if resolved
                           else f"unresolved:{resolution.raw_token.casefold()}")
                    candidates.setdefault(key, {
                        "symbol": resolution.symbol or resolution.raw_token,
                        "market": resolution.market,
                        "kind": resolution.security_kind,
                        "resolved": resolved,
                        "posts": {},
                    })["posts"][(post_id, source_id)] = post

            return tuple(SocialEvidenceInput(
                key, value["symbol"], value["market"],
                tuple(value["posts"][post_key] for post_key in sorted(value["posts"])),
                source_ids, history_complete, tuple(sorted(reasons)), value["resolved"], value["kind"],
            ) for key, value in sorted(candidates.items()))

    @classmethod
    def read_in_session(cls, db, run_id, as_of):
        class BorrowedSession:
            def __enter__(self):
                return db

            def __exit__(self, exc_type, exc, traceback):
                return False

        return cls(lambda: BorrowedSession()).read(run_id, as_of)


class SqlThemeProjectionFacade:
    """Give orchestration a transaction-free facade over session-bound services."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def prepare(self, run_id, now):
        from app.services.social_theme_projection_service import SocialThemeProjectionService
        with self.session_factory() as db:
            return SocialThemeProjectionService(db).prepare(run_id, now)

    def prepare_application(self, projection, *, theme_keys=()):
        from app.services.social_theme_projection_service import SocialThemeProjectionService
        with self.session_factory() as db:
            return SocialThemeProjectionService(db).prepare_application(
                projection, theme_keys=theme_keys
            )

    def theme_keys(self, projection):
        from app.models.theme import ThemeCluster
        with self.session_factory() as db:
            existing = db.scalars(select(ThemeCluster.canonical_key).where(
                ThemeCluster.pipeline == "technical",
                ThemeCluster.is_active.is_(True),
                ThemeCluster.lifecycle_state != "retired",
            )).all()
        return tuple(sorted(set(existing) | {claim.theme_key for claim in projection.proposals}))


class SqlConfirmationReaderFacade:
    def __init__(self, session_factory, *, grace_minutes=120):
        self.session_factory = session_factory
        self.grace_minutes = grace_minutes

    def read_market(self, market, symbols, now, **kwargs):
        from app.services.social_confirmation_reader import SocialConfirmationReader
        with self.session_factory() as db:
            return SocialConfirmationReader(
                db, grace_minutes=self.grace_minutes
            ).read_market(market, symbols, now, **kwargs)


__all__ = [
    "SocialScoringEvidenceReader",
    "SqlConfirmationReaderFacade",
    "SqlSocialRefreshCatalog",
    "SqlThemeProjectionFacade",
]
