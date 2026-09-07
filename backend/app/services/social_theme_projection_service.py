"""Stage saved evidence, then project shared Themes in the publisher transaction.

ThemeConstituent remains independently supported legacy membership. Live callers
use effective_live_membership to include accepted Social membership. No provider
calls, commits, Social pointer changes, or legacy attention writes occur here.
"""
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from hashlib import sha256
import json

from sqlalchemy import select, update

from app.domain.social_signals.records import (
    EffectiveThemeMembership, ExtractionResult, SocialPostRecord, ThemeProjection,
    validate_utc_timestamp,
)
from app.infra.db.models.social_analysis import SocialExtractionWork, SocialRunWork, SocialThemeAssociation, SocialThemeDecision
from app.infra.db.models.social_signals import SocialSignalRun, SocialSourceRegistry
from app.models.theme import ThemeCluster, ThemeConstituent, ThemeMention
from app.services.social_company_identity_service import SocialCompanyIdentityService
from app.services.social_extraction_service import SocialExtractionParser, SocialExtractionService
from app.services.social_ticker_resolver import SocialTickerResolver
from app.services.theme_extraction_service import find_read_only_theme_match
from app.services.theme_identity_normalization import canonical_theme_key, display_theme_name, UNKNOWN_THEME_KEY
from app.services.theme_lifecycle_service import apply_lifecycle_transition, set_initial_lifecycle_defaults

POLICY = "social-theme-v1"


@dataclass(frozen=True)
class PreparedThemeApplication:
    projection: ThemeProjection
    baskets: tuple
    decoded_work: tuple
    fingerprint: str

    def read(self, theme_key, market):
        from app.services.social_theme_market_service import MeasurementUnavailable
        for basket in self.baskets:
            if basket.theme_key == theme_key and basket.market == market:
                return basket
        raise MeasurementUnavailable("theme_unavailable")


def _lock_registry(db):
    with db.no_autoflush:
        if db.get_bind().dialect.name == "sqlite":
            db.execute(update(SocialSourceRegistry).where(SocialSourceRegistry.id == 1).values(version=SocialSourceRegistry.version))
        return db.scalar(select(SocialSourceRegistry).where(SocialSourceRegistry.id == 1)
                         .with_for_update().execution_options(populate_existing=True))


def guard_social_theme_merge(db, source_id, target_id):
    """Serialize before Theme locks and refuse unsupported Social-aware merges.

    Migration-only legacy provenance does not block ordinary merges. Actual
    Social evidence/decisions require a future explicit reconciliation policy.
    """
    from app.services.errors import ThemeMergeConflictError
    _lock_registry(db)
    ids = (source_id, target_id)
    social_mention = db.scalar(select(ThemeMention.id).where(ThemeMention.theme_cluster_id.in_(ids), ThemeMention.social_work_id.is_not(None)).limit(1))
    social_association = db.scalar(select(SocialThemeAssociation.id).where(SocialThemeAssociation.theme_cluster_id.in_(ids), SocialThemeAssociation.origin == "social").limit(1))
    social_decision = db.scalar(select(SocialThemeDecision.id).join(SocialThemeAssociation,
        SocialThemeAssociation.id == SocialThemeDecision.association_id).where(SocialThemeAssociation.theme_cluster_id.in_(ids)).limit(1))
    if social_mention is not None or social_association is not None or social_decision is not None:
        raise ThemeMergeConflictError("Social-aware theme merging is not supported; evidence and administrator decisions must be preserved.",
                                      error_code="theme_merge_social_evidence_unsupported")


def _decode(work):
    """Revalidate cached serialized records against their actual saved source."""
    if work.state != "succeeded" or not isinstance(work.result_json, dict):
        raise ValueError("social_work_incomplete")
    try:
        snapshot = dict(work.input_snapshot_json)
        for field in ("created_at", "observed_at"):
            snapshot[field] = datetime.fromisoformat(snapshot[field])
        post = SocialPostRecord(**snapshot)
        data = work.result_json
        if (data["input_hash"] != work.input_hash or SocialExtractionService.input_hash((post,)) != work.input_hash
                or data["provider"] != work.actual_provider or data["model"] != work.actual_model
                or data["prompt_version"] != work.prompt_version or data["schema_version"] != work.schema_version):
            raise ValueError("social_work_identity_mismatch")
        judgments = data["judgments"]
        if len(judgments) != 1 or judgments[0]["post_id"] != post.provider_post_id:
            raise ValueError("social_work_attribution_mismatch")
        claim_values = []
        for claim in data["claims"]:
            if claim["post_id"] != post.provider_post_id:
                raise ValueError("social_work_attribution_mismatch")
            claim_values.append({key: value for key, value in claim.items() if key != "post_id"})
        claims, parsed_judgments = SocialExtractionParser().parse_batch(
            SocialExtractionService.source_inputs((post,)), json.dumps({"posts": [{**judgments[0], "claims": claim_values}]}))
        result = ExtractionResult(**{**data, "claims": claims, "judgments": parsed_judgments})
        return post, result
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("invalid_saved_social_result") from None


class SocialThemeProjectionService:
    def __init__(self, db, *, pipeline="technical", admin_authorized=False):
        if pipeline not in {"technical", "fundamental"}:
            raise ValueError("invalid_pipeline")
        self.db, self.pipeline, self.admin_authorized = db, pipeline, admin_authorized
        self._prepared_decoded = {}

    def _decode_work(self, work):
        return self._prepared_decoded.get(work.id) or _decode(work)

    def _fingerprint(self, projection, theme_keys):
        """Fence catalog, manual/legacy basket edits and saved-work changes.

        No semantic decoding, feature reads or price calculations under the lock.
        Includes shared identity tables because legacy writers do not bump registry.
        """
        from app.models.stock_universe import StockUniverse
        catalog = self.db.execute(select(ThemeCluster.id, ThemeCluster.canonical_key, ThemeCluster.aliases,
            ThemeCluster.is_active, ThemeCluster.lifecycle_state).where(ThemeCluster.pipeline == self.pipeline).order_by(ThemeCluster.id)).all()
        ids = [row.id for row in catalog if row.canonical_key in theme_keys]
        work_ids = set(projection.work_ids)
        work_ids.update(self.db.scalars(select(ThemeMention.social_work_id).where(
            ThemeMention.theme_cluster_id.in_(ids), ThemeMention.social_work_id.is_not(None))))
        rows = [("catalog_identity", [tuple(row) for row in catalog])]
        scopes = ((ThemeConstituent, ThemeConstituent.theme_cluster_id.in_(ids)),
            (SocialThemeAssociation, SocialThemeAssociation.theme_cluster_id.in_(ids)),
            (ThemeMention, ThemeMention.theme_cluster_id.in_(ids)),
            (SocialExtractionWork, SocialExtractionWork.id.in_(work_ids)),
            (SocialRunWork, SocialRunWork.run_id == projection.run_id))
        for model, condition in scopes:
            values = self.db.execute(select(*model.__table__.columns).where(condition).order_by(*model.__table__.primary_key.columns)).all()
            rows.append((model.__tablename__, [tuple(row) for row in values]))
        # Security resolution can change without the Social registry. Pin the
        # small identity columns, not every stored stock/feature payload.
        values = self.db.execute(select(StockUniverse.id, StockUniverse.symbol, StockUniverse.market,
            StockUniverse.is_active, StockUniverse.is_common_stock, StockUniverse.exchange).order_by(StockUniverse.id)).all()
        rows.append(("security_identity", [tuple(row) for row in values]))
        return sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()

    @staticmethod
    def _automatic_accepts(state, owner, resolution, qualifying):
        return (owner == "system" and state == "proposed" and resolution.company_count_eligible
            and len({row[4] for row in qualifying if row[3] == resolution.company_id}) >= 2)

    def prepare_application(self, projection, *, theme_keys=()):
        """Read-only adjudicated effective baskets, including newly found Themes."""
        from app.services.social_theme_market_service import AcceptedBasketSnapshot
        with self.db.no_autoflush:
            if self.prepare(projection.run_id, projection.prepared_at) != projection:
                raise ValueError("social_projection_version_conflict")
            identity = SocialCompanyIdentityService(self.db).read()
            resolver = SocialTickerResolver(self.db, verified_company_ids=identity.verified_company_ids)
            themes = {t.canonical_key: t for t in self.db.scalars(select(ThemeCluster).where(
                ThemeCluster.pipeline == self.pipeline, ThemeCluster.canonical_key.in_(theme_keys),
                ThemeCluster.is_active.is_(True), ThemeCluster.lifecycle_state != "retired"))}
            proposed = {}
            for claim in projection.proposals:
                match = find_read_only_theme_match(self.db, claim.raw_theme, self.pipeline)
                key = match.canonical_key if match else canonical_theme_key(claim.raw_theme)
                if key == UNKNOWN_THEME_KEY or (match and match.lifecycle_state == "retired"):
                    continue
                themes.setdefault(key, match)
                proposed.setdefault(key, []).append(resolver.resolve(claim.company_token))
            ids = [theme.id for theme in themes.values() if theme is not None]
            work_ids = set(projection.work_ids)
            work_ids.update(self.db.scalars(select(ThemeMention.social_work_id).where(
                ThemeMention.theme_cluster_id.in_(ids), ThemeMention.social_work_id.is_not(None))))
            decoded = tuple((wid, *_decode(self.db.get(SocialExtractionWork, wid))) for wid in sorted(work_ids))
            self._prepared_decoded = {wid: (post, result) for wid, post, result in decoded}
            baskets = []
            for key, theme in sorted(themes.items()):
                members = {(m.market, m.canonical_symbol): m for m in self.effective_live_membership(theme.id)} if theme else {}
                associations = self.db.scalars(select(SocialThemeAssociation).where(SocialThemeAssociation.theme_cluster_id == theme.id)).all() if theme else []
                candidates = {(a.market, a.canonical_symbol): (resolver.resolve(a.canonical_symbol, a.market),
                    "proposed" if a.origin == "legacy" else a.state, a.decision_owner) for a in associations}
                for r in proposed.get(key, ()):
                    if r.status == "resolved":
                        candidates.setdefault((r.market, r.symbol), (r, "proposed", "system"))
                ids = set(projection.work_ids)
                if theme:
                    ids.update(self.db.scalars(select(ThemeMention.social_work_id).where(
                        ThemeMention.theme_cluster_id == theme.id, ThemeMention.social_work_id.is_not(None))))
                qualifying = self._qualifying_inputs(ids, key, projection.prepared_at, resolver)
                for pair, (r, state, owner) in candidates.items():
                    if self._automatic_accepts(state, owner, r, qualifying):
                        origins = set(members[pair].origins) if pair in members else set()
                        origins.add("social")
                        members[pair] = EffectiveThemeMembership(r.symbol, r.market, r.company_id, r.company_count_eligible, tuple(sorted(origins)))
                for market in ("US", "HK", "CN", "JP", "TW"):
                    selected = tuple(m for pair, m in sorted(members.items()) if m.market == market)
                    stocks = tuple(m.canonical_symbol for m in selected if resolver.resolve(m.canonical_symbol, market).security_kind == "stock")
                    baskets.append(AcceptedBasketSnapshot(key, market, selected, stocks, identity.version, identity.policy_version, identity.registry_version))
            return PreparedThemeApplication(projection, tuple(baskets), decoded, self._fingerprint(projection, tuple(themes)))

    def prepare(self, run_id: str, now: datetime) -> ThemeProjection:
        validate_utc_timestamp(now, "now")
        with self.db.no_autoflush:
            run = self.db.get(SocialSignalRun, run_id, populate_existing=True)
            if run is None or run.status == "failed":
                raise ValueError("invalid_social_run")
            identity = SocialCompanyIdentityService(self.db).read()
            resolver = SocialTickerResolver(self.db, verified_company_ids=identity.verified_company_ids)
            proposals, work_ids = [], []
            links = self.db.scalars(select(SocialRunWork).where(SocialRunWork.run_id == run_id).order_by(SocialRunWork.work_id)).all()
            for link in links:
                work = self.db.get(SocialExtractionWork, link.work_id, populate_existing=True)
                if work is None or work.input_hash != link.input_hash:
                    raise ValueError("social_pinned_input_mismatch")
                post, result = _decode(work)
                work_ids.append(work.id)
                if now - timedelta(days=14) <= post.created_at <= now:
                    proposals.extend(result.claims)
                    for claim in result.claims:
                        find_read_only_theme_match(self.db, claim.raw_theme, self.pipeline)
            return ThemeProjection(run_id, POLICY, tuple(proposals), identity.registry_version,
                                   identity.version, identity.policy_version, now, tuple(work_ids),
                                   tuple(resolver.resolve(claim.company_token) for claim in proposals), self.pipeline)

    def _lock_live(self, expected_version=None):
        if not self.db.in_transaction():
            raise ValueError("caller_transaction_required")
        registry = _lock_registry(self.db)
        if registry is None or registry.mode != "live":
            raise ValueError("social_live_required")
        if expected_version is not None and registry.version != expected_version:
            raise ValueError("social_projection_version_conflict")
        return registry

    def apply_live(self, projection: ThemeProjection, expected_mode_version: int, *, prepared=None) -> None:
        self._lock_live(expected_mode_version)
        run = self.db.get(SocialSignalRun, projection.run_id)
        if run.mode != "live" or run.registry_version != expected_mode_version:
            raise ValueError("social_live_run_required")
        if prepared is not None:
            if prepared.projection != projection or prepared.fingerprint != self._fingerprint(projection, tuple(b.theme_key for b in prepared.baskets)):
                raise ValueError("social_projection_version_conflict")
            self._prepared_decoded = {wid: (post, result) for wid, post, result in prepared.decoded_work}
            current = projection
        else:
            self._prepared_decoded = {}
            current = self.prepare(projection.run_id, projection.prepared_at)
        if current != projection or projection.registry_version != expected_mode_version:
            raise ValueError("social_projection_version_conflict")
        identity = SocialCompanyIdentityService(self.db).read()
        resolver = SocialTickerResolver(self.db, verified_company_ids=identity.verified_company_ids)
        touched = set()
        for work_id in projection.work_ids:
            work = self.db.get(SocialExtractionWork, work_id)
            post, result = self._decode_work(work)
            if not projection.prepared_at - timedelta(days=14) <= post.created_at <= projection.prepared_at:
                continue
            by_theme = {}
            for claim in result.claims:
                key = canonical_theme_key(claim.raw_theme)
                if key == UNKNOWN_THEME_KEY:
                    continue
                theme = find_read_only_theme_match(self.db, claim.raw_theme, self.pipeline)
                if theme is None:
                    theme = self.db.scalar(select(ThemeCluster).where(ThemeCluster.pipeline == self.pipeline, ThemeCluster.canonical_key == key))
                if theme is None:
                    name = display_theme_name(claim.raw_theme)
                    theme = ThemeCluster(name=name, display_name=name, canonical_key=key, pipeline=self.pipeline,
                        aliases=[], discovery_source="social", first_seen_at=post.created_at, last_seen_at=post.created_at,
                        lifecycle_state="candidate", lifecycle_state_metadata={"social_policy_version": POLICY}, is_active=True)
                    set_initial_lifecycle_defaults(theme, now=projection.prepared_at)
                    self.db.add(theme)
                    self.db.flush()
                # Retired catalog identities are retained, never silently revived.
                if theme.lifecycle_state == "retired":
                    continue
                touched.add(theme.id)
                by_theme.setdefault(theme.id, []).append(claim)
                resolution = resolver.resolve(claim.company_token)
                if resolution.status != "resolved":
                    continue
                association = self.db.scalar(select(SocialThemeAssociation).where(
                    SocialThemeAssociation.theme_cluster_id == theme.id,
                    SocialThemeAssociation.market == resolution.market,
                    SocialThemeAssociation.canonical_symbol == resolution.symbol))
                if association is None:
                    association = SocialThemeAssociation(theme_cluster_id=theme.id, company_key=resolution.company_id,
                        market=resolution.market, canonical_symbol=resolution.symbol, state="proposed", origin="social",
                        decision_owner="system", evidence_work_ids=[], policy_version=POLICY, version=1,
                        first_seen_at=post.created_at, updated_at=projection.prepared_at)
                    self.db.add(association)
                    self.db.flush()
                elif association.origin == "legacy":
                    # The original constituent remains the independent legacy
                    # proof. This unique listing row now tracks Social decisions.
                    self._decision(association, "proposed", "social_proposal_legacy_membership_preserved",
                                   "system", projection.prepared_at, projection.run_id)
                    association.origin = "social"
                    association.policy_version = POLICY
                    association.accepted_at = None
                if work.id not in association.evidence_work_ids:
                    association.evidence_work_ids = sorted(set(association.evidence_work_ids) | {work.id})
                    association.version += 1
                    association.updated_at = projection.prepared_at
            for theme_id, claims in by_theme.items():
                existing = self.db.scalar(select(ThemeMention).where(ThemeMention.social_work_id == work.id, ThemeMention.theme_cluster_id == theme_id))
                if existing is None:
                    symbols = sorted({r.symbol for c in claims if (r := resolver.resolve(c.company_token)).status == "resolved"})
                    self.db.add(ThemeMention(content_item_id=work.content_item_id, source_type="twitter", source_name=post.source_id,
                        social_work_id=work.id, social_run_id=projection.run_id, theme_cluster_id=theme_id, pipeline=self.pipeline,
                        raw_theme=claims[0].raw_theme, canonical_theme=self.db.get(ThemeCluster, theme_id).canonical_key,
                        excerpt=claims[0].excerpt, tickers=symbols, ticker_count=len(symbols), mentioned_at=post.created_at,
                        extracted_at=projection.prepared_at, match_method="social_projection", threshold_version=POLICY))
        self.db.flush()
        from app.services.theme_taxonomy_service import ThemeTaxonomyService
        taxonomy = ThemeTaxonomyService(self.db, pipeline=self.pipeline)
        for theme_id in sorted(touched):
            self._assess(theme_id, projection, resolver)
            # Existing embeddings only: this utility never generates embeddings
            # or commits, and leaves an unembedded new theme unclassified.
            taxonomy.classify_new_l2_to_l1(self.db.get(ThemeCluster, theme_id))
        self.db.flush()

    def _qualifying(self, theme_id, now, resolver):
        work_ids = self.db.scalars(select(ThemeMention.social_work_id).where(
            ThemeMention.theme_cluster_id == theme_id, ThemeMention.social_work_id.is_not(None))).all()
        return self._qualifying_inputs(work_ids, self.db.get(ThemeCluster, theme_id).canonical_key, now, resolver)

    def _qualifying_inputs(self, work_ids, theme_key, now, resolver):
        evidence = []
        for work_id in set(work_ids):
            work = self.db.get(SocialExtractionWork, work_id)
            post, result = self._decode_work(work)
            judgment = result.judgments[0]
            if (not now - timedelta(days=14) <= post.created_at <= now or post.is_repost
                    or not judgment.has_new_thesis or not judgment.canonical_claim_key):
                continue
            for claim in result.claims:
                match = find_read_only_theme_match(self.db, claim.raw_theme, self.pipeline)
                key = match.canonical_key if match else canonical_theme_key(claim.raw_theme)
                if key != theme_key or claim.support != "supported" or claim.duplicate_of_post_ids:
                    continue
                resolution = resolver.resolve(claim.company_token)
                if resolution.company_count_eligible:
                    evidence.append((post.created_at, work.id, work.content_item_id, resolution.company_id,
                                     post.author_handle.casefold().lstrip("@"), judgment.canonical_claim_key,
                                     post.canonical_url or post.url))
        # One canonical post, URL or copied thesis contributes at most once/company,
        # even after model/content revisions or alternate listing extraction.
        seen_posts, seen_urls, seen_claims, qualifying = set(), set(), set(), []
        for row in sorted(evidence):
            date, work_id, item_id, company, author, key, url = row
            if (company, item_id) in seen_posts or (company, url) in seen_urls or (company, key) in seen_claims:
                continue
            seen_posts.add((company, item_id)); seen_urls.add((company, url)); seen_claims.add((company, key))
            qualifying.append(row)
        return qualifying

    def _assess(self, theme_id, projection, resolver):
        qualifying = self._qualifying(theme_id, projection.prepared_at, resolver)
        associations = self.db.scalars(select(SocialThemeAssociation).where(SocialThemeAssociation.theme_cluster_id == theme_id)).all()
        for association in associations:
            resolution = resolver.resolve(association.canonical_symbol, association.market)
            if association.company_key != resolution.company_id:
                association.company_key = resolution.company_id
                association.version += 1
            company_evidence = [row for row in qualifying if row[3] == resolution.company_id]
            authors = {row[4] for row in company_evidence}
            if self._automatic_accepts(association.state, association.decision_owner, resolution, qualifying):
                self._decision(association, "accepted", "two_independent_authors_14d", "system", projection.prepared_at,
                               projection.run_id, evidence_work_ids=sorted({row[1] for row in company_evidence}))
        companies = {item.company_key for item in self.effective_live_membership(theme_id) if item.company_count_eligible}
        dates = {row[0].date() for row in qualifying if row[3] in companies}
        theme = self.db.get(ThemeCluster, theme_id)
        if len(companies) >= 3 and len(dates) >= 3:
            third_date = sorted(dates, reverse=True)[2]
            expires = max(row[0] for row in qualifying if row[3] in companies and row[0].date() == third_date) + timedelta(days=14)
            metadata = {**(theme.lifecycle_state_metadata or {}), "social_policy_version": POLICY,
                "social_qualified_at": projection.prepared_at.isoformat(), "accepted_companies": len(companies),
                "social_valid_until": expires.isoformat(),
                "discussion_dates": sorted(str(day) for day in dates), "identity_policy_version": projection.identity_policy_version,
                "identity_version": projection.identity_version, "run_id": projection.run_id}
            if theme.lifecycle_state in {"candidate", "dormant"}:
                apply_lifecycle_transition(db=self.db, theme=theme, to_state="active" if theme.lifecycle_state == "candidate" else "reactivated",
                    actor="system", job_name="social_theme_projection", rule_version=POLICY, reason="social_company_discussion_gate_met",
                    metadata=metadata, transitioned_at=projection.prepared_at)
            else:
                theme.lifecycle_state_metadata = metadata

    def _decision(self, association, target, reason, actor, now, run_id=None, *, evidence_work_ids=None):
        self.db.add(SocialThemeDecision(association_id=association.id, run_id=run_id, actor=actor, reason=reason,
            before_state=association.state, after_state=target, policy_version=POLICY,
            evidence_work_ids=list(association.evidence_work_ids if evidence_work_ids is None else evidence_work_ids), created_at=now))
        association.state = target
        association.version += 1
        association.updated_at = now
        if target == "accepted":
            association.accepted_at = association.accepted_at or now

    def decide(self, association_id, target, reason, actor, expected_version):
        if self.admin_authorized is not True:
            raise PermissionError("admin_required")
        if target not in {"accepted", "rejected"} or not isinstance(reason, str) or not reason.strip() or not isinstance(actor, str) or not actor.strip():
            raise ValueError("decision_reason_and_actor_required")
        registry = self._lock_live()
        association = self.db.get(SocialThemeAssociation, association_id, populate_existing=True)
        if association is None or association.version != expected_version:
            raise ValueError("association_version_conflict")
        self._decision(association, target, reason.strip(), actor.strip(), datetime.now(timezone.utc))
        association.origin = "social"
        association.policy_version = POLICY
        association.decision_owner = "admin"
        registry.version += 1
        self.db.flush()

    def effective_live_membership(self, theme_cluster_id):
        """Active legacy UNION accepted Social listings, current trusted grouping.

        A dual-origin listing stays a member after Social rejection; callers may
        count a verified company only once within a Market. Unknown identities
        remain visible, with company_count_eligible false.
        """
        identity = SocialCompanyIdentityService(self.db).read()
        resolver = SocialTickerResolver(self.db, verified_company_ids=identity.verified_company_ids)
        members = {}
        for row in self.db.scalars(select(ThemeConstituent).where(ThemeConstituent.theme_cluster_id == theme_cluster_id, ThemeConstituent.is_active.is_(True))):
            r = resolver.resolve(row.symbol)
            if r.status == "resolved":
                members[(r.market, r.symbol)] = (r, {"legacy"})
        for row in self.db.scalars(select(SocialThemeAssociation).where(SocialThemeAssociation.theme_cluster_id == theme_cluster_id,
                                                                       SocialThemeAssociation.state == "accepted", SocialThemeAssociation.origin == "social")):
            r = resolver.resolve(row.canonical_symbol, row.market)
            if r.status == "resolved":
                members.setdefault((r.market, r.symbol), (r, set()))[1].add("social")
        return tuple(EffectiveThemeMembership(r.symbol, r.market, r.company_id, r.company_count_eligible, tuple(sorted(origins)))
                     for _, (r, origins) in sorted(members.items()))
