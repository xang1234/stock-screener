"""Stage saved evidence, then project shared Themes in the publisher transaction.

ThemeConstituent remains independently supported legacy membership. Live callers
use effective_live_membership to include accepted Social membership. No provider
calls, commits, Social pointer changes, or legacy attention writes occur here.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256

from sqlalchemy import select, update

from app.domain.social_signals.records import (
    EffectiveThemeMembership,
    ExtractionResult,
    SocialPostRecord,
    ThemeProjection,
    validate_utc_timestamp,
)
from app.infra.db.models.social_analysis import (
    EconomicSocialAssociation,
    SocialExtractionWork,
    SocialRunWork,
    SocialThemeAssociation,
    SocialThemeDecision,
)
from app.infra.db.models.social_signals import SocialSignalRun, SocialSourceRegistry
from app.models.economic_taxonomy import (
    LegacyClaimAllocation,
    LegacyDestinationMapping,
)
from app.models.economic_taxonomy_runtime import TaxonomyAuthority
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeAlias, ThemeCluster, ThemeConstituent, ThemeMention
from app.services.economic_social_taxonomy_adapter import EconomicSocialTaxonomyAdapter
from app.services.economic_source_admission import EvidenceAdmission
from app.services.economic_taxonomy_fence import producer_write
from app.services.economic_taxonomy_runtime import EconomicTaxonomyRuntimeService
from app.services.social_company_identity_service import SocialCompanyIdentityService
from app.services.social_extraction_service import (
    SocialExtractionParser,
    SocialExtractionService,
)
from app.services.social_ticker_resolver import SocialTickerResolver
from app.services.theme_development_preparation import economic_authority
from app.services.theme_extraction_service import find_read_only_theme_match
from app.services.theme_identity_normalization import (
    UNKNOWN_THEME_KEY,
    canonical_theme_key,
    display_theme_name,
    social_membership_key,
)
from app.services.theme_lifecycle_service import (
    apply_lifecycle_transition,
    set_initial_lifecycle_defaults,
)
from app.utils.file_hashing import canonical_json_sha256 as _semantic_hash

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
            SocialExtractionService.source_inputs((post,)),
            json.dumps({"posts": [{**judgments[0], "claims": claim_values}]}),
            strict_claims=True,
        )
        result = ExtractionResult(**{**data, "claims": claims, "judgments": parsed_judgments})
        return post, result
    except (KeyError, TypeError, AttributeError):
        raise ValueError("invalid_saved_social_result") from None


def qualifying_social_evidence(decoded, now, resolver, claim_matches):
    """Rows automatic acceptance counts: one per company and independent post.

    ``decoded`` yields ``(work, post, result)``; ``claim_matches`` says whether a
    claim is about the theme in question. A row is ``(created_at, work_id,
    content_item_id, company_id, author, claim_key, url)``.
    """
    evidence = []
    for work, post, result in decoded:
        judgment = result.judgments[0]
        if (not now - timedelta(days=14) <= post.created_at <= now or post.is_repost
                or not judgment.has_new_thesis or not judgment.canonical_claim_key):
            continue
        for claim in result.claims:
            if not claim_matches(claim) or claim.support != "supported" or claim.duplicate_of_post_ids:
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


class SocialThemeProjectionService:
    def __init__(self, db, *, pipeline="technical", admin_authorized=False):
        if pipeline not in {"technical", "fundamental"}:
            raise ValueError("invalid_pipeline")
        self.db, self.pipeline, self.admin_authorized = db, pipeline, admin_authorized
        self._prepared_decoded = {}

    def _decode_work(self, work):
        return self._prepared_decoded.get(work.id) or _decode(work)

    def _economic_catalog(self):
        """``{normalized key: (economic theme id, theme key)}`` (#515).

        The processing version's display names and aliases; a theme's key is
        its display name's, so an alias lands on the same basket.
        """
        from app.models.economic_taxonomy import EconomicThemeAlias, EconomicThemeRevision
        version_id = self.db.get(TaxonomyAuthority, 1).processing_taxonomy_version_id
        names = self.db.execute(select(EconomicThemeRevision.theme_id, EconomicThemeRevision.display_name).where(
            EconomicThemeRevision.taxonomy_version_id == version_id,
            EconomicThemeRevision.lifecycle != "retired").order_by(EconomicThemeRevision.theme_id)).all()
        keys = {theme_id: canonical_theme_key(name) for theme_id, name in names}
        catalog = {}
        for theme_id, key in keys.items():
            catalog.setdefault(key, (theme_id, key))
        for theme_id, alias in self.db.execute(select(EconomicThemeAlias.theme_id, EconomicThemeAlias.alias).where(
                EconomicThemeAlias.taxonomy_version_id == version_id).order_by(
                EconomicThemeAlias.theme_id, EconomicThemeAlias.alias)):
            if theme_id in keys:
                catalog.setdefault(canonical_theme_key(alias), (theme_id, keys[theme_id]))
        catalog.pop(UNKNOWN_THEME_KEY, None)
        return catalog

    @staticmethod
    def _catalog_theme_key(raw_theme, catalog):
        """A claim's theme key under economic authority: the catalog theme's, else its own."""
        key = canonical_theme_key(raw_theme)
        return catalog.get(key, (None, key))[1]

    def _legacy_theme_key(self, raw_theme):
        match = find_read_only_theme_match(self.db, raw_theme, self.pipeline)
        return match.canonical_key if match else canonical_theme_key(raw_theme)

    def theme_keys(self, projection):
        """Themes a run measures: every current theme plus the run's own keys."""
        if economic_authority(self.db):
            catalog = self._economic_catalog()
            return tuple(sorted({key for _, key in catalog.values()}
                                | {self._catalog_theme_key(claim.raw_theme, catalog) for claim in projection.proposals}))
        existing = self.db.scalars(select(ThemeCluster.canonical_key).where(
            ThemeCluster.pipeline == self.pipeline, ThemeCluster.is_active.is_(True),
            ThemeCluster.lifecycle_state != "retired")).all()
        return tuple(sorted(set(existing) | {claim.theme_key for claim in projection.proposals}))

    def _economic_fingerprint(self, projection, themes):
        """Fence the catalog mapping, the themes' Social associations and saved work (#515)."""
        from app.infra.db.models.social_analysis import EconomicSocialAssociationRevision
        from app.models.stock_universe import StockUniverse
        rows = [("catalog", sorted((key, str(theme_id)) for key, theme_id in themes.items()))]
        revisions = self.db.execute(select(EconomicSocialAssociation.id, EconomicSocialAssociation.security_id,
            EconomicSocialAssociationRevision.revision_number, EconomicSocialAssociationRevision.state,
            EconomicSocialAssociationRevision.live).join(EconomicSocialAssociationRevision,
            EconomicSocialAssociationRevision.association_id == EconomicSocialAssociation.id).where(
            EconomicSocialAssociation.economic_theme_id.in_(list(themes.values()))).order_by(
            EconomicSocialAssociation.id, EconomicSocialAssociationRevision.revision_number)).all()
        # Only what a basket holds: an association whose latest revision is a
        # live acceptance. A proposed or pending revision changes no basket.
        latest = {}
        for association_id, security_id, number, state, live in revisions:
            latest[association_id] = (security_id, number, state, live)
        rows.append(("economic_social_associations", sorted(
            (str(association_id), str(security_id), number)
            for association_id, (security_id, number, state, live) in latest.items()
            if live and state == "accepted")))
        for model, condition in ((SocialExtractionWork, SocialExtractionWork.id.in_(projection.work_ids)),
                                 (SocialRunWork, SocialRunWork.run_id == projection.run_id)):
            values = self.db.execute(select(*model.__table__.columns).where(condition).order_by(
                *model.__table__.primary_key.columns)).all()
            rows.append((model.__tablename__, [tuple(row) for row in values]))
        values = self.db.execute(select(StockUniverse.id, StockUniverse.symbol, StockUniverse.market,
            StockUniverse.is_active, StockUniverse.is_common_stock, StockUniverse.exchange).order_by(StockUniverse.id)).all()
        rows.append(("security_identity", [tuple(row) for row in values]))
        return sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()

    def _fingerprint(self, projection, theme_keys):
        """Fence catalog, manual/legacy basket edits and saved-work changes.

        No semantic decoding, feature reads or price calculations under the lock.
        Includes shared identity tables because legacy writers do not bump registry.
        """
        if economic_authority(self.db):
            catalog = self._economic_catalog()
            return self._economic_fingerprint(
                projection, {key: catalog[key][0] for key in theme_keys if key in catalog})
        from app.models.stock_universe import StockUniverse
        catalog = self.db.execute(select(ThemeCluster.id, ThemeCluster.canonical_key, ThemeCluster.aliases,
            ThemeCluster.is_active, ThemeCluster.lifecycle_state).where(ThemeCluster.pipeline == self.pipeline).order_by(ThemeCluster.id)).all()
        ids = [row.id for row in catalog if row.canonical_key in theme_keys]
        work_ids = set(projection.work_ids)
        work_ids.update(self.db.scalars(select(ThemeMention.social_work_id).where(
            ThemeMention.theme_cluster_id.in_(ids), ThemeMention.social_work_id.is_not(None))))
        rows = [("catalog_identity", [tuple(row) for row in catalog])]
        alias_keys = {canonical_theme_key(claim.raw_theme) for claim in projection.proposals}
        scopes = ((ThemeConstituent, ThemeConstituent.theme_cluster_id.in_(ids)),
            (SocialThemeAssociation, SocialThemeAssociation.theme_cluster_id.in_(ids)),
            (ThemeMention, ThemeMention.theme_cluster_id.in_(ids)),
            (SocialExtractionWork, SocialExtractionWork.id.in_(work_ids)),
            (SocialRunWork, SocialRunWork.run_id == projection.run_id))
        for model, condition in scopes:
            values = self.db.execute(select(*model.__table__.columns).where(condition).order_by(*model.__table__.primary_key.columns)).all()
            rows.append((model.__tablename__, [tuple(row) for row in values]))
            if model is SocialExtractionWork:
                for row in values:
                    alias_keys.update(canonical_theme_key(claim["raw_theme"])
                        for claim in (row.result_json or {}).get("claims", ()))
        aliases = self.db.execute(select(ThemeAlias.id, ThemeAlias.alias_key, ThemeAlias.theme_cluster_id,
            ThemeAlias.is_active, ThemeAlias.confidence, ThemeAlias.source, ThemeAlias.evidence_count).where(
                ThemeAlias.pipeline == self.pipeline, ThemeAlias.alias_key.in_(alias_keys)).order_by(ThemeAlias.id)).all()
        rows.append(("qualified_aliases", [tuple(row) for row in aliases]))
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
            if economic_authority(self.db):
                return self._prepare_economic_application(projection, theme_keys)
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

    def _prepare_economic_application(self, projection, theme_keys):
        """Baskets of the economic themes the run's keys map to (#515).

        Accepted economic Social memberships only: acceptance is the
        processor's, and a key the catalog does not know has no basket.
        """
        from app.services.social_theme_market_service import EconomicAcceptedBasketReader
        catalog = self._economic_catalog()
        themes = {catalog[key][1]: catalog[key][0] for key in theme_keys if key in catalog}
        for claim in projection.proposals:
            mapped = catalog.get(canonical_theme_key(claim.raw_theme))
            if mapped is not None:
                themes.setdefault(mapped[1], mapped[0])
        decoded = tuple((wid, *_decode(self.db.get(SocialExtractionWork, wid))) for wid in sorted(projection.work_ids))
        self._prepared_decoded = {wid: (post, result) for wid, post, result in decoded}
        baskets = []
        for key, theme_id in sorted(themes.items()):
            reader = EconomicAcceptedBasketReader(self.db, economic_theme_id=theme_id)
            baskets.extend(reader.read(key, market) for market in ("US", "HK", "CN", "JP", "TW"))
        return PreparedThemeApplication(projection, tuple(baskets), decoded,
                                        self._economic_fingerprint(projection, themes))

    def prepare(self, run_id: str, now: datetime) -> ThemeProjection:
        validate_utc_timestamp(now, "now")
        with self.db.no_autoflush:
            run = self.db.get(SocialSignalRun, run_id, populate_existing=True)
            if run is None or run.status == "failed":
                raise ValueError("invalid_social_run")
            identity = SocialCompanyIdentityService(self.db).read()
            resolver = SocialTickerResolver(self.db, verified_company_ids=identity.verified_company_ids)
            economic = economic_authority(self.db)
            proposals, work_ids = [], []
            links = self.db.scalars(select(SocialRunWork).where(SocialRunWork.run_id == run_id).order_by(SocialRunWork.work_id)).all()
            for link in links:
                work = self.db.get(SocialExtractionWork, link.work_id, populate_existing=True)
                if work is None or work.input_hash != link.input_hash:
                    raise ValueError("social_pinned_input_mismatch")
                post, result = _decode(work)
                work_ids.append(work.id)
                if now - timedelta(days=14) <= post.created_at <= now:
                    supported_claims = tuple(
                        claim for claim in result.claims
                        if claim.support != "unsupported"
                    )
                    proposals.extend(supported_claims)
                    if not economic:
                        for claim in supported_claims:
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

    def admit_economic_evidence(
        self,
        projection: ThemeProjection,
        *,
        prepared: PreparedThemeApplication | None = None,
    ) -> None:
        """Admit the exact saved inputs after their Social run becomes published."""

        run = self.db.get(SocialSignalRun, projection.run_id)
        if run is None or run.mode != "live" or run.status != "published":
            raise ValueError("published_social_run_required")
        identity = SocialCompanyIdentityService(self.db).read()
        resolver = SocialTickerResolver(
            self.db, verified_company_ids=identity.verified_company_ids
        )
        adapter = EconomicSocialTaxonomyAdapter(self.db)
        # Under economic authority acceptance is the processor's (#515).
        economic = economic_authority(self.db)
        catalog = self._economic_catalog() if economic else None
        for work_id in projection.work_ids:
            work = self.db.get(SocialExtractionWork, work_id)
            post, result = self._decode_work(work)
            if (
                work.requested_by_admin
                or post.created_at < projection.prepared_at - timedelta(days=14)
                or post.created_at > projection.prepared_at
            ):
                continue
            resolutions = {
                claim.company_token: resolver.resolve(claim.company_token)
                for claim in result.claims
            }
            accepted_pairs = {
                (basket.theme_key, basket.market, member.canonical_symbol)
                for basket in (prepared.baskets if prepared is not None else ())
                for member in basket.membership
            }
            social_memberships = []
            for claim in result.claims:
                resolution = resolutions[claim.company_token]
                if claim.support == "unsupported" or resolution.status != "resolved":
                    continue
                if economic:
                    theme_key = self._catalog_theme_key(claim.raw_theme, catalog)
                    state = "proposed"
                else:
                    matched = find_read_only_theme_match(
                        self.db, claim.raw_theme, self.pipeline
                    )
                    theme_key = (
                        matched.canonical_key
                        if matched is not None
                        else canonical_theme_key(claim.raw_theme)
                    )
                    state = (
                        "accepted"
                        if (theme_key, resolution.market, resolution.symbol)
                        in accepted_pairs
                        else "proposed"
                    )
                    if matched is not None and state != "accepted":
                        legacy = self.db.scalar(
                            select(SocialThemeAssociation).where(
                                SocialThemeAssociation.theme_cluster_id == matched.id,
                                SocialThemeAssociation.market == resolution.market,
                                SocialThemeAssociation.canonical_symbol
                                == resolution.symbol,
                            )
                        )
                        if legacy is not None and legacy.state == "rejected":
                            state = "rejected"
                social_memberships.append(
                    {
                        "membership_key": social_membership_key(
                            theme_key, int(resolution.security_id)
                        ),
                        "theme_key": theme_key,
                        "security_id": int(resolution.security_id),
                        "state": state,
                    }
                )
            prepared_evidence = tuple(post.prepared_evidence)
            admission = adapter.admit_saved_work(
                work.id,
                EvidenceAdmission(
                    provider="x",
                    canonical_item_id=post.provider_post_id,
                    canonical_source_family=f"x:post:{post.provider_post_id}",
                    capture_route="social",
                    route_record_id=str(work.id),
                    original_text=post.text,
                    attachment_hashes=tuple(
                        item.original_text_sha256 for item in prepared_evidence
                    ),
                    extracted_text_hashes=tuple(
                        item.text_sha256 for item in prepared_evidence
                    ),
                    grounding_snapshot={
                        "company_resolutions": [
                            asdict(resolutions[token]) for token in sorted(resolutions)
                        ]
                    },
                    preparation_version="social-saved-work-v1",
                    source_metadata={
                        "content_item_id": work.content_item_id,
                        "source_provider": post.provider,
                        "source_id": post.source_id,
                        "url": post.url,
                        "canonical_url": post.canonical_url,
                        "author_handle": post.author_handle,
                        "quoted_text": post.quoted_text,
                        "input_snapshot": dict(work.input_snapshot_json),
                        "input_hash": work.input_hash,
                        "prompt_version": work.prompt_version,
                        "schema_version": work.schema_version,
                        "actual_provider": work.actual_provider,
                        "actual_model": work.actual_model,
                        "social_admission_state": "live",
                        "social_memberships": sorted(
                            social_memberships,
                            key=lambda row: (
                                row["theme_key"], row["security_id"], row["state"]
                            ),
                        ),
                    },
                    captured_at=post.observed_at,
                    observed_at=post.created_at,
                    available_at=max(
                        (
                            post.observed_at,
                            *(item.available_at for item in prepared_evidence),
                        )
                    ),
                    evidence_channels=("narrative",),
                ),
            )
            if (
                not admission.live
                and admission.precedence_state != "equivalent"
            ):
                raise ValueError("social_evidence_live_admission_required")
            authority = self.db.get(TaxonomyAuthority, 1)
            if (
                admission.precedence_state == "equivalent"
                and admission.effective_packet_id is not None
                and authority is not None
                and authority.mode == "economic"
            ):
                adapter.project_equivalent_social_packet(
                    evidence_packet_id=admission.packet_id,
                    effective_packet_id=admission.effective_packet_id,
                    authority_epoch=authority.authority_epoch,
                )

    def apply_live(
        self,
        projection: ThemeProjection,
        expected_mode_version: int,
        *,
        prepared=None,
    ) -> bool:
        authority = self.db.get(TaxonomyAuthority, 1)
        expected_epoch = authority.authority_epoch if authority is not None else 1
        if authority is not None and authority.mode == "economic":
            with producer_write(
                self.db,
                expected_epoch=expected_epoch,
                allowed_modes={"economic"},
            ):
                self._validate_application(
                    projection,
                    expected_mode_version,
                    prepared=prepared,
                )
            return False
        payload = {
            "run_id": projection.run_id,
            "pipeline": self.pipeline,
            "work_ids": list(projection.work_ids),
        }
        runtime = EconomicTaxonomyRuntimeService(self.db)
        with runtime.legacy_producer_write(
            expected_epoch=expected_epoch,
            logical_source_key=f"social-run:{projection.run_id}:{self.pipeline}",
            revision_kind="social_theme_projection",
            content_hash=lambda: _semantic_hash(payload),
            auto_commit=False,
        ) as write:
            projected_count = self._apply_live(
                projection,
                expected_mode_version,
                prepared=prepared,
            )
            payload["legacy_association_count"] = projected_count
            write.stage_next_compatibility_projection(
                source_lineage=f"social-run:{projection.run_id}:{self.pipeline}",
                projection_kind="social_theme_projection",
                projection_version=1,
                target="economic",
                payload=payload,
            )
        return True

    def _validate_application(
        self,
        projection: ThemeProjection,
        expected_mode_version: int,
        *,
        prepared=None,
    ) -> ThemeProjection:
        self._lock_live(expected_mode_version)
        run = self.db.get(SocialSignalRun, projection.run_id)
        if (
            run is None
            or run.mode != "live"
            or run.registry_version != expected_mode_version
        ):
            raise ValueError("social_live_run_required")
        if prepared is not None:
            if (
                prepared.projection != projection
                or prepared.fingerprint
                != self._fingerprint(
                    projection, tuple(b.theme_key for b in prepared.baskets)
                )
            ):
                raise ValueError("social_projection_version_conflict")
            self._prepared_decoded = {
                wid: (post, result) for wid, post, result in prepared.decoded_work
            }
            current = projection
        else:
            self._prepared_decoded = {}
            current = self.prepare(projection.run_id, projection.prepared_at)
        if current != projection or projection.registry_version != expected_mode_version:
            raise ValueError("social_projection_version_conflict")
        return current

    def _apply_live(
        self,
        projection: ThemeProjection,
        expected_mode_version: int,
        *,
        prepared=None,
    ) -> int:
        self._validate_application(
            projection,
            expected_mode_version,
            prepared=prepared,
        )
        identity = SocialCompanyIdentityService(self.db).read()
        resolver = SocialTickerResolver(self.db, verified_company_ids=identity.verified_company_ids)
        touched = set()
        projected_association_ids = set()
        for work_id in projection.work_ids:
            work = self.db.get(SocialExtractionWork, work_id)
            post, result = self._decode_work(work)
            if not projection.prepared_at - timedelta(days=14) <= post.created_at <= projection.prepared_at:
                continue
            by_theme = {}
            for claim in result.claims:
                if claim.support == "unsupported":
                    continue
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
                projected_association_ids.add(association.id)
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
        for association_id in sorted(projected_association_ids):
            self._project_legacy_association_to_economic(association_id)
        self.db.flush()
        return len(projected_association_ids)

    def _project_legacy_association_to_economic(self, association_id: int) -> None:
        authority = self.db.get(TaxonomyAuthority, 1)
        if authority is None or authority.processing_taxonomy_version_id is None:
            return
        legacy = self.db.get(SocialThemeAssociation, association_id)
        allocation = self.db.scalar(
            select(LegacyClaimAllocation).where(
                LegacyClaimAllocation.taxonomy_version_id
                == authority.processing_taxonomy_version_id,
                LegacyClaimAllocation.legacy_theme_cluster_id
                == legacy.theme_cluster_id,
                LegacyClaimAllocation.allocation_kind == "social_association",
                LegacyClaimAllocation.allocation_key
                == f"social_theme_association:{association_id}",
                LegacyClaimAllocation.destination_theme_id.is_not(None),
            )
        )
        if allocation is not None:
            economic_theme_id = allocation.destination_theme_id
        else:
            destinations = self.db.scalars(
                select(LegacyDestinationMapping).where(
                    LegacyDestinationMapping.taxonomy_version_id
                    == authority.processing_taxonomy_version_id,
                    LegacyDestinationMapping.legacy_theme_cluster_id
                    == legacy.theme_cluster_id,
                )
            ).all()
            if len(destinations) != 1:
                return
            economic_theme_id = destinations[0].destination_theme_id
        security = self.db.scalar(
            select(StockUniverse).where(
                StockUniverse.symbol == legacy.canonical_symbol,
                StockUniverse.market == legacy.market,
            )
        )
        if security is None:
            return
        EconomicSocialTaxonomyAdapter(self.db).project_legacy_associations(
            economic_theme_id=economic_theme_id,
            security_id=security.id,
            legacy_association_ids=(legacy.id,),
        )

    def _qualifying(self, theme_id, now, resolver):
        work_ids = self.db.scalars(select(ThemeMention.social_work_id).where(
            ThemeMention.theme_cluster_id == theme_id, ThemeMention.social_work_id.is_not(None))).all()
        return self._qualifying_inputs(work_ids, self.db.get(ThemeCluster, theme_id).canonical_key, now, resolver)

    def _qualifying_inputs(self, work_ids, theme_key, now, resolver):
        works = (self.db.get(SocialExtractionWork, work_id) for work_id in set(work_ids))
        return qualifying_social_evidence(
            ((work, *self._decode_work(work)) for work in works), now, resolver,
            lambda claim: self._legacy_theme_key(claim.raw_theme) == theme_key,
        )

    def _assess(self, theme_id, projection, resolver):
        qualifying = self._qualifying(theme_id, projection.prepared_at, resolver)
        associations = self.db.scalars(select(SocialThemeAssociation).where(SocialThemeAssociation.theme_cluster_id == theme_id)).all()
        for association in associations:
            resolution = resolver.resolve(association.canonical_symbol, association.market)
            if association.company_key != resolution.company_id:
                association.company_key = resolution.company_id
                association.version += 1
            company_evidence = [row for row in qualifying if row[3] == resolution.company_id]
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
        authority = self.db.get(TaxonomyAuthority, 1)
        if authority is not None and authority.mode == "economic":
            # Decisions address economic associations once economic is
            # authoritative; the legacy row is only a mirror (#515).
            raise ValueError("economic_association_decision_required")
        expected_epoch = authority.authority_epoch if authority is not None else 1
        payload = {
            "association_id": association_id,
            "target": target,
            "expected_version": expected_version,
        }
        runtime = EconomicTaxonomyRuntimeService(self.db)
        with runtime.legacy_producer_write(
            expected_epoch=expected_epoch,
            logical_source_key=f"social-association:{association_id}",
            revision_kind="administrator_decision",
            content_hash=lambda: _semantic_hash(payload),
            auto_commit=False,
        ) as write:
            registry = self._lock_live()
            association = self.db.get(
                SocialThemeAssociation, association_id, populate_existing=True
            )
            if association is None or association.version != expected_version:
                raise ValueError("association_version_conflict")
            self._decision(
                association,
                target,
                reason.strip(),
                actor.strip(),
                datetime.now(timezone.utc),
            )
            association.origin = "social"
            association.policy_version = POLICY
            association.decision_owner = "admin"
            registry.version += 1
            self.db.flush()
            self._project_legacy_association_to_economic(association_id)
            payload["new_version"] = association.version
            write.stage_next_compatibility_projection(
                source_lineage=f"social-association:{association_id}",
                projection_kind="social_administrator_decision",
                projection_version=1,
                target="economic",
                payload=payload,
            )

    def decide_economic_association(self, association_id, target, reason, actor, expected_revision):
        """Revise an economic Social association by its own id (#515).

        Native and legacy-bridged associations alike; a bridged legacy row is a
        compatibility mirror that changes through ordered delivery.
        """
        if self.admin_authorized is not True:
            raise PermissionError("admin_required")
        if target not in {"accepted", "rejected"} or not isinstance(reason, str) or not reason.strip() or not isinstance(actor, str) or not actor.strip():
            raise ValueError("decision_reason_and_actor_required")
        authority = self.db.get(TaxonomyAuthority, 1)
        if authority is None or authority.mode != "economic":
            raise ValueError("economic_authority_required")
        if self.db.get(EconomicSocialAssociation, association_id) is None:
            raise ValueError("association_not_found")
        reason, actor = reason.strip(), actor.strip()
        return EconomicSocialTaxonomyAdapter(self.db).revise(
            association_id,
            state=target,
            # The revision tells a repeated decision apart from a retry.
            idempotency_key=(
                f"economic-admin:{association_id}:r{expected_revision}:{target}:"
                f"{_semantic_hash({'reason': reason, 'actor': actor})}"
            ),
            actor=actor,
            reason=reason,
            mirror_acknowledged=False,
            expected_revision=expected_revision,
        )

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
