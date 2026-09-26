"""Leased verify/refresh stages for one issuer–theme pair (plan Task 17A).

Each stage is one leased work item with a finite unit of work; progress is
an append-only event stream:

``resolve_issuer``
    Use the accepted issuer link. Without one, the market's official
    registry route proposes a match that is accepted only through
    ``accept_registry_match``; an ambiguous result pauses the job as
    ``review_required`` with the failed condition visible.
``acquire``
    The market's official enumerator, then retained originals. Every gap is
    typed coverage; nothing here decides exposure truth.
``verify``
    Prepare passages, run one bounded claim-extraction call on the
    subscription route (cached per input), then append a dossier revision.

The runner never calls a publication pointer, never mutates membership and
never schedules discovery. Operational job state is not product state.
Requests are accepted by ``research_requests.ResearchRequests``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import (
    RESEARCH_STAGES,
    SERVICE_PRINCIPAL,
    CoverageItem,
    CoverageOutcome,
    RegistryMatch,
    ResearchJobState,
    ResearchMode,
    content_hash,
    utc_now,
)
from app.infra.db.repositories.company_exposure_work_repo import (
    CompanyExposureWorkRepository,
    WorkLeaseError,
)
from app.models.company_exposure import (
    ExposureDocument,
    ExposureDocumentRevision,
    ExposureResearchRequest,
    ResearchEvent,
)
from app.models.stock_universe import StockUniverse
from app.services.company_exposure.acquisition import JobBudgetRef
from app.services.company_exposure.assessments import (
    AssessmentAttemptInput,
    ExposureAssessmentService,
)
from app.services.company_exposure.claims import (
    RETRIEVAL_AID_KINDS,
    AssessmentScope,
    ClaimReviewBatch,
    ClaimVerifier,
    EvidenceItem,
    evidence_item_from_rows,
)
from app.services.company_exposure.config import ExposureRuntimeConfig
from app.services.company_exposure.issuer_identity import (
    IssuerIdentityAdapter,
    IssuerLinkRef,
)
from app.services.company_exposure.markets.base import (
    AcquisitionLimits,
    DocumentQuery,
    MarketDocumentAdapter,
)
from app.services.company_exposure.preparation import (
    ExposureEvidencePreparer,
    PreparationFailed,
    QuestionSet,
    persist_passages,
    select_passages,
)
from app.services.company_exposure.research_requests import enqueue_stage

ANNUAL_REPORTS = DocumentQuery(document_kinds=("annual_report",), max_documents=2)
MAX_STAGE_ATTEMPTS = 4
MAX_RETAINED_DOCUMENTS = 4
MAX_PASSAGES = 24
RETRY_BASE = timedelta(minutes=2)

_UNAVAILABLE_REASONS = frozenset(
    {"route_not_approved", "subscription_credentials_missing"}
)
_RETRYABLE_COVERAGE = frozenset(
    {CoverageOutcome.RATE_LIMITED, CoverageOutcome.FETCH_FAILED}
)


class StepStatus(StrEnum):
    COMPLETED = "completed"
    PAUSED = "paused"
    RETRYABLE = "retryable"
    FAILED = "failed"
    LEASE_LOST = "lease_lost"


@dataclass(frozen=True, slots=True)
class StageOutcome:
    status: StepStatus
    state: ResearchJobState
    detail: dict
    next_stage: str | None = None

    @classmethod
    def complete(cls, state: ResearchJobState, next_stage: str | None = None, **detail):
        return cls(StepStatus.COMPLETED, state, detail, next_stage)

    @classmethod
    def pause(cls, state: ResearchJobState, condition: str, **detail):
        return cls(StepStatus.PAUSED, state, {"condition": condition, **detail})

    @classmethod
    def retry(cls, condition: str, **detail):
        return cls(
            StepStatus.RETRYABLE,
            ResearchJobState.RETRYABLE_FAILURE,
            {"condition": condition, **detail},
        )

    @classmethod
    def fail(cls, condition: str, **detail):
        return cls(
            StepStatus.FAILED,
            ResearchJobState.TERMINAL_FAILURE,
            {"condition": condition, **detail},
        )


def _issuer_link_required(**detail) -> StageOutcome:
    return StageOutcome.pause(
        ResearchJobState.REVIEW_REQUIRED, "issuer_link_required", **detail
    )


@dataclass(frozen=True, slots=True)
class ResearchStepResult:
    work_id: UUID
    request_id: UUID
    stage: str
    status: StepStatus
    state: str
    next_stage: str | None = None
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ThemeContext:
    theme_id: UUID
    label: str
    terms: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class MarketRoute:
    """A market's official document enumerator and issuer registry."""

    adapter: MarketDocumentAdapter
    resolve_registry: Callable[[int, JobBudgetRef], RegistryMatch | CoverageItem]
    query: DocumentQuery = ANNUAL_REPORTS


@dataclass(frozen=True, slots=True)
class _Issuer:
    issuer_id: UUID
    identifiers: dict
    link_revision_id: UUID | None


@dataclass(frozen=True, slots=True)
class _IssuerRef:
    """What a market adapter needs to know about the issuer."""

    identifiers: dict


@dataclass(frozen=True, slots=True)
class AcquiredEvidence:
    """The acquire stage's result, carried to verify in its event detail."""

    issuer_id: UUID
    document_revision_ids: tuple[UUID, ...] = ()
    coverage: tuple[CoverageItem, ...] = ()

    def to_detail(self) -> dict:
        return {
            "issuer_id": str(self.issuer_id),
            "document_revision_ids": [str(r) for r in self.document_revision_ids],
            "coverage": [c.to_dict() for c in self.coverage],
        }

    @classmethod
    def from_detail(cls, detail: dict) -> AcquiredEvidence:
        return cls(
            issuer_id=UUID(detail["issuer_id"]),
            document_revision_ids=tuple(
                UUID(r) for r in detail["document_revision_ids"]
            ),
            coverage=tuple(CoverageItem.from_dict(c) for c in detail["coverage"]),
        )


def load_theme_context(session: Session, theme_id: UUID) -> ThemeContext | None:
    """The latest sealed definition and aliases of an Economic Theme."""

    from app.models.economic_taxonomy import (
        EconomicThemeAlias,
        EconomicThemeRevision,
        TaxonomyVersion,
    )

    row = session.execute(
        select(EconomicThemeRevision, TaxonomyVersion)
        .join(
            TaxonomyVersion,
            TaxonomyVersion.id == EconomicThemeRevision.taxonomy_version_id,
        )
        .where(
            EconomicThemeRevision.theme_id == theme_id,
            TaxonomyVersion.status == "sealed",
        )
        .order_by(TaxonomyVersion.sealed_at.desc(), TaxonomyVersion.created_at.desc())
        .limit(1)
    ).first()
    if row is None:
        return None
    revision, version = row
    aliases = tuple(
        sorted(
            session.execute(
                select(EconomicThemeAlias.alias).where(
                    EconomicThemeAlias.taxonomy_version_id == version.id,
                    EconomicThemeAlias.theme_id == theme_id,
                )
            ).scalars()
        )
    )
    return ThemeContext(
        theme_id=theme_id,
        label=revision.display_name,
        terms=tuple(dict.fromkeys((revision.display_name, *aliases))),
        fingerprint=content_hash(
            {
                "name": revision.display_name,
                "definition": revision.definition,
                "mechanism": revision.mechanism,
                "aliases": list(aliases),
            }
        ),
    )


class ResearchStageRunner:
    def __init__(
        self,
        session: Session,
        config: ExposureRuntimeConfig,
        *,
        markets: Mapping[str, MarketRoute],
        verifier: ClaimVerifier,
        store,
        theme_loader: Callable[
            [Session, UUID], ThemeContext | None
        ] = load_theme_context,
        clock: Callable[[], datetime] = utc_now,
        commit: Callable[[], None] | None = None,
    ):
        self.session = session
        self.config = config
        self.markets = markets
        self.verifier = verifier
        self.preparer = ExposureEvidencePreparer(session, store)
        self.theme_loader = theme_loader
        self.clock = clock
        self.commit = commit or session.commit
        self.repo = CompanyExposureWorkRepository(session, clock=clock)
        self.identity = IssuerIdentityAdapter(session, clock=clock)
        self._stages = dict(
            zip(
                RESEARCH_STAGES,
                (self._resolve_issuer, self._acquire, self._verify),
                strict=True,
            )
        )

    # ---------------------------------------------------------------- steps
    def run_step(self, work_id: UUID, lease_token: UUID) -> ResearchStepResult:
        item = self.repo.heartbeat(work_id, lease_token)
        request = self.session.get(ExposureResearchRequest, item.request_id)
        stage, attempts = item.stage, int(item.claim_count or 1)
        # Release row locks before any pacing wait, network or provider call.
        self.commit()
        if self.config.research_mode == ResearchMode.SHADOW:
            outcome = self._stages[stage](request)
        else:
            outcome = StageOutcome.pause(
                ResearchJobState.UNAVAILABLE_CAPABILITY,
                "research_disabled"
                if self.config.research_mode == ResearchMode.DISABLED
                else "live_mode_not_installed",
            )
        if outcome.status == StepStatus.RETRYABLE and attempts >= MAX_STAGE_ATTEMPTS:
            outcome = StageOutcome(
                StepStatus.FAILED,
                ResearchJobState.TERMINAL_FAILURE,
                {**outcome.detail, "attempts": attempts},
            )
        return self._finish(work_id, lease_token, request, stage, outcome, attempts)

    def _finish(self, work_id, lease_token, request, stage, outcome, attempts):
        detail = {"stage": stage, **outcome.detail}
        try:
            self.repo.complete_step(
                work_id,
                lease_token,
                status=outcome.status.value,
                retry_at=self.clock() + RETRY_BASE * (2 ** max(0, attempts - 1))
                if outcome.status == StepStatus.RETRYABLE
                else None,
                pause_reason=outcome.state.value
                if outcome.status == StepStatus.PAUSED
                else None,
            )
        except WorkLeaseError:
            # The lease expired during I/O: another worker owns the stage now.
            self.session.rollback()
            return ResearchStepResult(
                work_id,
                request.id,
                stage,
                StepStatus.LEASE_LOST,
                StepStatus.LEASE_LOST.value,
            )
        self.repo.append_event(request.id, outcome.state, detail)
        if outcome.next_stage is not None:
            enqueue_stage(self.repo, request, outcome.next_stage)
        self.commit()
        return ResearchStepResult(
            work_id,
            request.id,
            stage,
            outcome.status,
            outcome.state.value,
            outcome.next_stage,
            detail,
        )

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _budget(request) -> JobBudgetRef:
        return JobBudgetRef(root_request_id=request.effective_root_id)

    def _issuer(self, request) -> _Issuer | None:
        if request.issuer_id is not None:
            return _Issuer(
                request.issuer_id,
                self.identity.identifiers_for(request.issuer_id),
                None,
            )
        resolution = self.identity.resolve_security(request.security_id)
        if not resolution.resolved:
            return None
        return _Issuer(
            resolution.issuer_id, resolution.identifiers, resolution.link_revision_id
        )

    @staticmethod
    def _coverage_outcome(item: CoverageItem) -> StageOutcome:
        detail = {"coverage": [item.to_dict()]}
        if item.outcome in _RETRYABLE_COVERAGE:
            return StageOutcome.retry(item.reason, **detail)
        state = (
            ResearchJobState.PAUSED_STORAGE
            if item.reason == "paused_storage"
            else ResearchJobState.UNAVAILABLE_CAPABILITY
        )
        return StageOutcome.pause(state, item.reason, **detail)

    # --------------------------------------------------------------- stages
    def _resolve_issuer(self, request) -> StageOutcome:
        issuer = self._issuer(request)
        if issuer is not None:
            return StageOutcome.complete(
                ResearchJobState.RESEARCHING,
                "acquire",
                issuer_id=str(issuer.issuer_id),
                link_revision_id=None
                if issuer.link_revision_id is None
                else str(issuer.link_revision_id),
                source="request" if request.issuer_id else "accepted_link",
            )
        route = self.markets.get(request.market)
        if route is None:
            return _issuer_link_required(market=request.market)
        match = route.resolve_registry(request.security_id, self._budget(request))
        self.commit()
        if isinstance(match, CoverageItem):
            return self._coverage_outcome(match)
        ref = self.identity.accept_registry_match(match, SERVICE_PRINCIPAL)
        if isinstance(ref, IssuerLinkRef):
            return StageOutcome.complete(
                ResearchJobState.RESEARCHING,
                "acquire",
                issuer_id=str(ref.issuer_id),
                link_revision_id=str(ref.link_revision_id),
                acceptance_policy=ref.acceptance_policy,
                source="official_registry",
            )
        return StageOutcome.pause(
            ResearchJobState.REVIEW_REQUIRED,
            ref.reason,
            link_revision_id=None
            if ref.link_revision_id is None
            else str(ref.link_revision_id),
            candidate_count=match.candidate_count,
        )

    def _retained(self, issuer_id: UUID) -> list[ExposureDocumentRevision]:
        rows = self.session.execute(
            select(ExposureDocumentRevision)
            .join(
                ExposureDocument,
                ExposureDocument.id == ExposureDocumentRevision.document_id,
            )
            .where(
                ExposureDocument.issuer_id == issuer_id,
                ExposureDocument.source_kind.not_in(tuple(RETRIEVAL_AID_KINDS)),
            )
            .order_by(
                ExposureDocumentRevision.published_at.desc().nulls_last(),
                ExposureDocumentRevision.first_available_at.desc(),
            )
        ).scalars()
        latest: dict[UUID, ExposureDocumentRevision] = {}
        for row in rows:
            latest.setdefault(row.document_id, row)
            if len(latest) >= MAX_RETAINED_DOCUMENTS:
                break
        return list(latest.values())

    def _acquire(self, request) -> StageOutcome:
        issuer = self._issuer(request)
        if issuer is None:
            return _issuer_link_required()
        coverage: list[CoverageItem] = []
        revision_ids: list[UUID] = []
        route = self.markets.get(request.market)
        if route is None:
            coverage.append(
                CoverageItem(
                    route=f"{(request.market or 'unknown').lower()}_official",
                    outcome=CoverageOutcome.UNAVAILABLE_CAPABILITY,
                    reason="market_adapter_not_installed",
                )
            )
        else:
            budget = self._budget(request)
            discovery = route.adapter.discover(
                _IssuerRef(issuer.identifiers), route.query, AcquisitionLimits(), budget
            )
            coverage.extend(discovery.coverage)
            for target in discovery.targets:
                capture = route.adapter.fetch(target, budget)
                coverage.append(capture.coverage)
                if capture.coverage.reason == "paused_storage":
                    self.commit()
                    return self._coverage_outcome(capture.coverage)
                if capture.revision_id is not None:
                    revision_ids.append(capture.revision_id)
            self.commit()
        coverage.extend(
            CoverageItem(
                route="supplied_link",
                outcome=CoverageOutcome.UNAVAILABLE_CAPABILITY,
                reason="supplied_link_route_not_installed",
                detail={"link": str(link)[:500]},
            )
            for link in request.supplied_links or []
        )
        revision_ids += [
            r.id for r in self._retained(issuer.issuer_id) if r.id not in revision_ids
        ]
        acquired = AcquiredEvidence(
            issuer.issuer_id,
            tuple(revision_ids[:MAX_RETAINED_DOCUMENTS]),
            tuple(coverage),
        )
        return StageOutcome.complete(
            ResearchJobState.EVIDENCE_READY, "verify", **acquired.to_detail()
        )

    def _acquired(self, request_id: UUID) -> AcquiredEvidence | None:
        event = self.session.execute(
            select(ResearchEvent)
            .where(
                ResearchEvent.request_id == request_id,
                ResearchEvent.state == ResearchJobState.EVIDENCE_READY.value,
            )
            .order_by(ResearchEvent.sequence.desc())
            .limit(1)
        ).scalar_one_or_none()
        return None if event is None else AcquiredEvidence.from_detail(event.detail)

    def _prepare(
        self, revision_ids, questions: QuestionSet
    ) -> tuple[list[EvidenceItem], list[CoverageItem]]:
        evidence: list[EvidenceItem] = []
        coverage: list[CoverageItem] = []

        def partial(reason, revision_id, **detail):
            coverage.append(
                CoverageItem(
                    "preparation",
                    CoverageOutcome.PARTIAL,
                    reason,
                    {"revision": str(revision_id), **detail},
                )
            )

        for revision_id in revision_ids:
            remaining = MAX_PASSAGES - len(evidence)
            if remaining <= 0:
                partial("passage_limit", revision_id)
                continue
            revision = self.session.get(ExposureDocumentRevision, revision_id)
            document = self.session.get(ExposureDocument, revision.document_id)
            try:
                prepared = self.preparer.prepare(revision, questions)
            except PreparationFailed as exc:
                partial(exc.code, revision_id)
                continue
            selection = select_passages(prepared, questions, limit=remaining)
            if selection.omitted_matches:
                partial("passage_limit", revision_id, omitted=selection.omitted_matches)
            for passage in persist_passages(self.session, prepared, selection):
                evidence.append(
                    evidence_item_from_rows(
                        f"P{len(evidence) + 1}", passage, revision, document
                    )
                )
        return evidence, coverage

    @staticmethod
    def _batch_outcome(batch: ClaimReviewBatch) -> StageOutcome | None:
        if batch.pause_reason:
            state = (
                ResearchJobState.UNAVAILABLE_CAPABILITY
                if batch.pause_reason in _UNAVAILABLE_REASONS
                else ResearchJobState.PAUSED_ALLOWANCE
            )
            return StageOutcome.pause(state, batch.pause_reason)
        if batch.failure_code:
            factory = StageOutcome.retry if batch.retryable else StageOutcome.fail
            return factory(batch.failure_code)
        return None

    def _scope(self, request, issuer: _Issuer, theme: ThemeContext) -> AssessmentScope:
        security = (
            None
            if request.security_id is None
            else self.session.get(StockUniverse, request.security_id)
        )
        return AssessmentScope(
            issuer_id=issuer.issuer_id,
            economic_theme_id=request.economic_theme_id,
            theme_fingerprint=theme.fingerprint,
            theme_label=theme.label,
            theme_terms=theme.terms,
            issuer_names=tuple(n for n in (getattr(security, "name", None),) if n),
            link_revision_ids=()
            if issuer.link_revision_id is None
            else (str(issuer.link_revision_id),),
        )

    def _verify(self, request) -> StageOutcome:
        theme = self.theme_loader(self.session, request.economic_theme_id)
        if theme is None:
            return StageOutcome.pause(
                ResearchJobState.UNAVAILABLE_CAPABILITY, "theme_definition_unavailable"
            )
        issuer = self._issuer(request)
        if issuer is None:
            return _issuer_link_required()
        acquired = self._acquired(request.id) or AcquiredEvidence(issuer.issuer_id)
        evidence, preparation_gaps = self._prepare(
            acquired.document_revision_ids, QuestionSet(terms=theme.terms)
        )
        coverage = [*acquired.coverage, *preparation_gaps]
        self.commit()

        scope = self._scope(request, issuer, theme)
        claims, rejected, artifacts = (), (), ()
        if evidence:
            batch = self.verifier.verify_claims(
                evidence,
                scope,
                root_request_id=request.effective_root_id,
                request_id=request.id,
            )
            blocked = self._batch_outcome(batch)
            if blocked is not None:
                return blocked
            claims, rejected = batch.claims, batch.rejected
            artifacts = () if batch.artifact_id is None else (str(batch.artifact_id),)
        else:
            coverage.append(
                CoverageItem(
                    "passages",
                    CoverageOutcome.NO_MATCHING_DOCUMENT,
                    "no_relevant_passages",
                )
            )

        service = ExposureAssessmentService(self.session, clock=self.clock)
        result = service.assess(
            AssessmentAttemptInput(
                scope=scope,
                claims=tuple(claims),
                document_revision_ids=acquired.document_revision_ids,
                coverage=tuple(coverage),
                unresolved_questions=tuple(f"rejected:{r}" for r in rejected),
                model_attempt_refs=artifacts,
                request_id=request.id,
                assessed_at=self.clock(),
            )
        )
        ref = service.persist_assessment(result, principal=SERVICE_PRINCIPAL)
        self.commit()
        complete = all(c.complete for c in coverage)
        return StageOutcome.complete(
            ResearchJobState.READY_FOR_PUBLICATION
            if complete
            else ResearchJobState.PARTIAL,
            assessment_id=str(ref.assessment_id),
            assessment_revision_id=str(ref.id),
            revision_number=ref.revision_number,
            unchanged=ref.unchanged,
            claims=len(claims),
            rejected=list(rejected),
            passages=len(evidence),
            coverage=[c.to_dict() for c in coverage],
        )


def build_runner(
    session: Session, config: ExposureRuntimeConfig
) -> ResearchStageRunner:
    """Production wiring. Secrets are read from settings, never logged."""

    from app.config import settings
    from app.services.company_exposure.acquisition import DocumentAcquisitionRegistry
    from app.services.company_exposure.markets.us import (
        USDocumentAdapter,
        USIssuerResolver,
    )
    from app.services.company_exposure.network import PublicDocumentTransport
    from app.services.company_exposure.pacing import ResearchRateGate
    from app.services.company_exposure.providers import (
        SubscriptionArtifactRunner,
        SubscriptionProvider,
        default_client_factory,
    )
    from app.services.company_exposure.resources import ResearchResources
    from app.services.company_exposure.storage import OriginalStore

    store = OriginalStore(
        session,
        config.document_store,
        max_bytes=config.storage_max_bytes,
        min_free_bytes=config.storage_min_free_bytes,
    )
    acquisition = DocumentAcquisitionRegistry(
        session,
        transport=PublicDocumentTransport(),
        store=store,
        rate_gate=ResearchRateGate(),
        user_agent=config.sec_user_agent,
        limits=config.limits,
    )
    us_adapter = USDocumentAdapter(
        session, acquisition, user_agent=config.sec_user_agent
    )
    runner = SubscriptionArtifactRunner(
        session,
        ResearchResources(session, config),
        SubscriptionProvider(
            api_key=settings.opencode_go_api_key or "",
            client_factory=default_client_factory(),
        ),
    )
    return ResearchStageRunner(
        session,
        config,
        markets={
            "US": MarketRoute(
                us_adapter, USIssuerResolver(session, us_adapter).resolve_cik
            )
        },
        verifier=ClaimVerifier(runner),
        store=store,
    )


__all__ = (
    "AcquiredEvidence",
    "MarketRoute",
    "ResearchStageRunner",
    "ResearchStepResult",
    "StageOutcome",
    "StepStatus",
    "ThemeContext",
    "build_runner",
    "load_theme_context",
)
