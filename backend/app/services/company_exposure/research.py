"""Bounded verify/refresh research for one issuer–theme pair (plan Task 17A).

A request is an immutable envelope; progress is an append-only event stream
and each stage is one leased work item with a finite unit of work:

``resolve_issuer``
    Use the accepted issuer link. A US listing without one resolves its CIK
    from SEC's official files and accepts it only through
    ``accept_registry_match``; an ambiguous result pauses the job as
    ``review_required`` with the failed condition visible, and ``resume``
    re-runs this stage after an administrator applies a link.
``acquire``
    Retained originals first, then the approved official enumerator (SEC for
    US). Every gap is typed coverage; nothing here decides exposure truth.
``verify``
    Prepare passages, run one bounded claim-extraction call on the
    subscription route (cached per input), then append a dossier revision.

The coordinator never calls a publication pointer, never mutates membership
and never schedules discovery; ``discover`` requests are refused in this
slice. Operational job state is not product state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import (
    SERVICE_PRINCIPAL,
    CoverageItem,
    CoverageOutcome,
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
    ResearchWorkItem,
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
    ClaimVerifier,
    evidence_item_from_rows,
)
from app.services.company_exposure.config import ExposureRuntimeConfig
from app.services.company_exposure.issuer_identity import (
    IssuerIdentityAdapter,
    IssuerLinkRef,
)
from app.services.company_exposure.markets.base import AcquisitionLimits, DocumentQuery
from app.services.company_exposure.markets.us import USDocumentAdapter, USIssuerResolver
from app.services.company_exposure.preparation import (
    ExposureEvidencePreparer,
    PreparationFailed,
    QuestionSet,
    persist_passages,
    select_passages,
)

POLICY_BUNDLE = "exposure-verify-v1"
STAGES = ("resolve_issuer", "acquire", "verify")
VERIFY_KINDS = frozenset({"verify", "refresh"})
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


class ResearchUnavailable(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ResearchRequestInput:
    economic_theme_id: UUID
    kind: str = "verify"
    security_id: int | None = None
    issuer_id: UUID | None = None
    market: str | None = None
    supplied_links: tuple[str, ...] = ()
    trigger_origin: str = "requested"


@dataclass(frozen=True, slots=True)
class ResearchRequestRef:
    id: UUID
    created: bool
    state: str | None


@dataclass(frozen=True, slots=True)
class ResearchStepResult:
    work_id: UUID
    request_id: UUID
    stage: str
    status: str  # completed | paused | retryable | failed
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
class _StageOutcome:
    status: str
    state: ResearchJobState
    detail: dict
    next_stage: str | None = None


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
    terms = tuple(dict.fromkeys((revision.display_name, *aliases)))
    return ThemeContext(
        theme_id=theme_id,
        label=revision.display_name,
        terms=terms,
        fingerprint=content_hash(
            {
                "name": revision.display_name,
                "definition": revision.definition,
                "mechanism": revision.mechanism,
                "aliases": list(aliases),
            }
        ),
    )


def _coverage_dict(item: CoverageItem) -> dict:
    return {
        "route": item.route,
        "outcome": item.outcome.value,
        "reason": item.reason,
        "detail": item.detail,
    }


def _coverage_item(data: dict) -> CoverageItem:
    return CoverageItem(
        data["route"],
        CoverageOutcome(data["outcome"]),
        data.get("reason"),
        data.get("detail") or {},
    )


@dataclass(frozen=True, slots=True)
class _IssuerView:
    identifiers: dict


class ExposureResearchCoordinator:
    def __init__(
        self,
        session: Session,
        config: ExposureRuntimeConfig,
        *,
        us_adapter: USDocumentAdapter | None = None,
        verifier: ClaimVerifier | None = None,
        store=None,
        theme_loader: Callable[
            [Session, UUID], ThemeContext | None
        ] = load_theme_context,
        clock: Callable[[], datetime] = utc_now,
        commit: Callable[[], None] | None = None,
    ):
        self.session = session
        self.config = config
        self.us_adapter = us_adapter
        self.verifier = verifier
        self.store = store
        self.theme_loader = theme_loader
        self.clock = clock
        self.commit = commit or session.commit
        self.repo = CompanyExposureWorkRepository(session, clock=clock)
        self.identity = IssuerIdentityAdapter(session, clock=clock)

    # ------------------------------------------------------------- requests
    def request(
        self, request: ResearchRequestInput, principal: str, idempotency_key: str
    ) -> ResearchRequestRef:
        if self.config.research_mode == ResearchMode.DISABLED:
            raise ResearchUnavailable("research_disabled")
        if request.kind not in VERIFY_KINDS:
            raise ResearchUnavailable("discovery_not_installed")
        if request.security_id is None and request.issuer_id is None:
            raise ValueError("security_or_issuer_required")
        market = request.market
        if request.security_id is not None:
            security = self.session.get(StockUniverse, request.security_id)
            if security is None:
                raise ValueError("unknown_security")
            market = market or security.market
        row, created = self.repo.create_request(
            kind=request.kind,
            requester_principal=principal,
            idempotency_namespace=f"company-exposure:{principal}",
            idempotency_key=idempotency_key,
            economic_theme_id=request.economic_theme_id,
            security_id=request.security_id,
            issuer_id=request.issuer_id,
            market=market,
            supplied_links=list(request.supplied_links),
            limits=self.config.limits,
            trigger_origin=request.trigger_origin,
        )
        if created:
            self._enqueue(row, STAGES[0])
        return ResearchRequestRef(row.id, created, self.repo.latest_state(row.id))

    def _enqueue(
        self, request: ExposureResearchRequest, stage: str, *, attempt: int = 0
    ) -> ResearchWorkItem:
        return self.repo.enqueue(
            request=request,
            stage=stage,
            input_hash=content_hash(
                {"request": request.id, "stage": stage, "attempt": attempt}
            ),
            policy_bundle_version=POLICY_BUNDLE,
            priority=0,
        )

    def status(self, request_id: UUID) -> dict | None:
        request = self.session.get(ExposureResearchRequest, request_id)
        if request is None:
            return None
        events = self.repo.events(request_id)
        items = sorted(
            self.session.execute(
                select(ResearchWorkItem).where(
                    ResearchWorkItem.request_id == request_id
                )
            ).scalars(),
            key=lambda i: STAGES.index(i.stage) if i.stage in STAGES else len(STAGES),
        )
        return {
            "id": str(request.id),
            "kind": request.kind,
            "security_id": request.security_id,
            "issuer_id": None if request.issuer_id is None else str(request.issuer_id),
            "economic_theme_id": str(request.economic_theme_id),
            "market": request.market,
            "state": events[-1].state if events else None,
            "events": [
                {"sequence": e.sequence, "state": e.state, "detail": e.detail}
                for e in events
            ],
            "stages": [
                {"stage": i.stage, "status": i.status, "pause_reason": i.pause_reason}
                for i in items
            ],
        }

    # ---------------------------------------------------------------- steps
    def run_step(self, work_id: UUID, lease_token: UUID) -> ResearchStepResult:
        item = self.repo.heartbeat(work_id, lease_token)
        request = self.session.get(ExposureResearchRequest, item.request_id)
        stage = item.stage
        attempts = int(item.claim_count or 1)
        # Release row locks before any pacing wait, network or provider call.
        self.commit()
        if self.config.research_mode == ResearchMode.DISABLED:
            outcome = _StageOutcome(
                "paused",
                ResearchJobState.UNAVAILABLE_CAPABILITY,
                {"condition": "research_disabled"},
            )
        else:
            handler = {
                "resolve_issuer": self._resolve_issuer,
                "acquire": self._acquire,
                "verify": self._verify,
            }[stage]
            outcome = handler(request)
        if outcome.status == "retryable" and attempts >= MAX_STAGE_ATTEMPTS:
            outcome = _StageOutcome(
                "failed",
                ResearchJobState.TERMINAL_FAILURE,
                {**outcome.detail, "attempts": attempts},
            )
        return self._finish(item.id, lease_token, request, stage, outcome, attempts)

    def _finish(
        self, work_id, lease_token, request, stage, outcome, attempts
    ) -> ResearchStepResult:
        detail = {"stage": stage, **outcome.detail}
        try:
            self.repo.complete_step(
                work_id,
                lease_token,
                status=outcome.status,
                retry_at=self.clock() + RETRY_BASE * (2 ** max(0, attempts - 1))
                if outcome.status == "retryable"
                else None,
                pause_reason=outcome.state.value
                if outcome.status == "paused"
                else None,
            )
        except WorkLeaseError:
            # The lease expired during I/O: another worker owns the stage now.
            self.session.rollback()
            return ResearchStepResult(
                work_id, request.id, stage, "lease_lost", "lease_lost"
            )
        state = outcome.state
        if outcome.status == "retryable":
            state = ResearchJobState.RETRYABLE_FAILURE
        self.repo.append_event(request.id, state, detail)
        if outcome.next_stage is not None:
            self._enqueue(request, outcome.next_stage)
        self.commit()
        return ResearchStepResult(
            work_id,
            request.id,
            stage,
            outcome.status,
            state.value,
            outcome.next_stage,
            detail,
        )

    def resume(self, request_id: UUID) -> None:
        self.repo.resume(request_id)

    # --------------------------------------------------------------- stages
    def _budget(self, request) -> JobBudgetRef:
        return JobBudgetRef(root_request_id=request.effective_root_id)

    def _resolve_issuer(self, request) -> _StageOutcome:
        if request.issuer_id is not None:
            return _StageOutcome(
                "completed",
                ResearchJobState.RESEARCHING,
                {"issuer_id": str(request.issuer_id), "source": "request"},
                "acquire",
            )
        resolution = self.identity.resolve_security(request.security_id)
        if resolution.resolved:
            return _StageOutcome(
                "completed",
                ResearchJobState.RESEARCHING,
                {
                    "issuer_id": str(resolution.issuer_id),
                    "link_revision_id": str(resolution.link_revision_id),
                    "acceptance_policy": resolution.acceptance_policy,
                    "source": "accepted_link",
                },
                "acquire",
            )
        if request.market != "US" or self.us_adapter is None:
            return _StageOutcome(
                "paused",
                ResearchJobState.REVIEW_REQUIRED,
                {"condition": "issuer_link_required", "market": request.market},
            )
        match = USIssuerResolver(self.session, self.us_adapter).resolve_cik(
            request.security_id, self._budget(request)
        )
        self.commit()
        if isinstance(match, CoverageItem):
            return self._coverage_outcome(match)
        ref = self.identity.accept_registry_match(match, SERVICE_PRINCIPAL)
        if isinstance(ref, IssuerLinkRef):
            return _StageOutcome(
                "completed",
                ResearchJobState.RESEARCHING,
                {
                    "issuer_id": str(ref.issuer_id),
                    "link_revision_id": str(ref.link_revision_id),
                    "acceptance_policy": ref.acceptance_policy,
                    "source": "official_registry",
                },
                "acquire",
            )
        return _StageOutcome(
            "paused",
            ResearchJobState.REVIEW_REQUIRED,
            {
                "condition": ref.reason,
                "link_revision_id": None
                if ref.link_revision_id is None
                else str(ref.link_revision_id),
                "candidate_count": match.candidate_count,
            },
        )

    def _coverage_outcome(self, item: CoverageItem) -> _StageOutcome:
        detail = {"condition": item.reason, "coverage": [_coverage_dict(item)]}
        if item.outcome in _RETRYABLE_COVERAGE:
            return _StageOutcome(
                "retryable", ResearchJobState.RETRYABLE_FAILURE, detail
            )
        if item.reason == "paused_storage":
            return _StageOutcome("paused", ResearchJobState.PAUSED_STORAGE, detail)
        return _StageOutcome("paused", ResearchJobState.UNAVAILABLE_CAPABILITY, detail)

    def _issuer(self, request):
        if request.issuer_id is not None:
            return (
                request.issuer_id,
                self.identity.identifiers_for(request.issuer_id),
                None,
            )
        resolution = self.identity.resolve_security(request.security_id)
        return resolution.issuer_id, resolution.identifiers, resolution.link_revision_id

    def _retained(self, issuer_id: UUID, limit: int) -> list[ExposureDocumentRevision]:
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
            if len(latest) >= limit:
                break
        return list(latest.values())

    def _acquire(self, request) -> _StageOutcome:
        issuer_id, identifiers, _ = self._issuer(request)
        if issuer_id is None:
            return _StageOutcome(
                "paused",
                ResearchJobState.REVIEW_REQUIRED,
                {"condition": "issuer_link_required"},
            )
        coverage: list[CoverageItem] = []
        revision_ids: list[UUID] = []
        if request.market == "US" and self.us_adapter is not None:
            discovery = self.us_adapter.discover(
                _IssuerView(identifiers),
                DocumentQuery(document_kinds=("annual_report",), max_documents=2),
                AcquisitionLimits(),
                self._budget(request),
            )
            coverage.extend(discovery.coverage)
            for target in discovery.targets:
                capture = self.us_adapter.fetch(target, self._budget(request))
                coverage.append(capture.coverage)
                if capture.coverage.reason == "paused_storage":
                    self.commit()
                    return self._coverage_outcome(capture.coverage)
                if capture.revision_id is not None:
                    revision_ids.append(capture.revision_id)
            self.commit()
        else:
            coverage.append(
                CoverageItem(
                    route=f"{(request.market or 'unknown').lower()}_official",
                    outcome=CoverageOutcome.UNAVAILABLE_CAPABILITY,
                    reason="market_adapter_not_installed",
                )
            )
        for link in request.supplied_links or []:
            coverage.append(
                CoverageItem(
                    route="supplied_link",
                    outcome=CoverageOutcome.UNAVAILABLE_CAPABILITY,
                    reason="supplied_link_route_not_installed",
                    detail={"link": str(link)[:500]},
                )
            )
        for revision in self._retained(issuer_id, MAX_RETAINED_DOCUMENTS):
            if revision.id not in revision_ids:
                revision_ids.append(revision.id)
        return _StageOutcome(
            "completed",
            ResearchJobState.EVIDENCE_READY,
            {
                "issuer_id": str(issuer_id),
                "document_revision_ids": [
                    str(r) for r in revision_ids[:MAX_RETAINED_DOCUMENTS]
                ],
                "coverage": [_coverage_dict(c) for c in coverage],
            },
            "verify",
        )

    def _stage_detail(self, request_id: UUID, stage: str) -> dict:
        events = self.session.execute(
            select(ResearchEvent)
            .where(ResearchEvent.request_id == request_id)
            .order_by(ResearchEvent.sequence.desc())
        ).scalars()
        for event in events:
            if (event.detail or {}).get(
                "stage"
            ) == stage and "coverage" in event.detail:
                return event.detail
        return {}

    def _verify(self, request) -> _StageOutcome:
        theme = self.theme_loader(self.session, request.economic_theme_id)
        if theme is None:
            return _StageOutcome(
                "paused",
                ResearchJobState.UNAVAILABLE_CAPABILITY,
                {"condition": "theme_definition_unavailable"},
            )
        issuer_id, _, link_revision_id = self._issuer(request)
        if issuer_id is None:
            return _StageOutcome(
                "paused",
                ResearchJobState.REVIEW_REQUIRED,
                {"condition": "issuer_link_required"},
            )
        acquired = self._stage_detail(request.id, "acquire")
        coverage = [_coverage_item(c) for c in acquired.get("coverage", [])]
        revision_ids = [UUID(r) for r in acquired.get("document_revision_ids", [])]
        evidence = []
        questions = QuestionSet(terms=theme.terms)
        if revision_ids and self.store is None:
            coverage.append(
                CoverageItem(
                    "preparation",
                    CoverageOutcome.NOT_CONFIGURED,
                    "document_store_unavailable",
                )
            )
            revision_ids = []
        preparer = (
            ExposureEvidencePreparer(self.session, self.store) if self.store else None
        )
        for revision_id in revision_ids:
            revision = self.session.get(ExposureDocumentRevision, revision_id)
            document = self.session.get(ExposureDocument, revision.document_id)
            try:
                prepared = preparer.prepare(revision, questions)
            except PreparationFailed as exc:
                coverage.append(
                    CoverageItem(
                        "preparation",
                        CoverageOutcome.PARTIAL,
                        exc.code,
                        {"revision": str(revision_id)},
                    )
                )
                continue
            remaining = MAX_PASSAGES - len(evidence)
            if remaining <= 0:
                coverage.append(
                    CoverageItem(
                        "preparation",
                        CoverageOutcome.PARTIAL,
                        "passage_limit",
                        {"revision": str(revision_id)},
                    )
                )
                continue
            selection = select_passages(prepared, questions, limit=remaining)
            if selection.omitted_matches:
                coverage.append(
                    CoverageItem(
                        "preparation",
                        CoverageOutcome.PARTIAL,
                        "passage_limit",
                        {
                            "revision": str(revision_id),
                            "omitted": selection.omitted_matches,
                        },
                    )
                )
            for passage in persist_passages(self.session, prepared, selection):
                evidence.append(
                    evidence_item_from_rows(
                        f"P{len(evidence) + 1}", passage, revision, document
                    )
                )
        self.commit()

        security = (
            None
            if request.security_id is None
            else self.session.get(StockUniverse, request.security_id)
        )
        scope = AssessmentScope(
            issuer_id=issuer_id,
            economic_theme_id=request.economic_theme_id,
            theme_fingerprint=theme.fingerprint,
            theme_label=theme.label,
            theme_terms=theme.terms,
            issuer_names=tuple(n for n in (getattr(security, "name", None),) if n),
            link_revision_ids=()
            if link_revision_id is None
            else (str(link_revision_id),),
        )
        claims, rejected, artifacts = (), (), ()
        if evidence:
            if self.verifier is None:
                return _StageOutcome(
                    "paused",
                    ResearchJobState.UNAVAILABLE_CAPABILITY,
                    {"condition": "text_route_not_configured"},
                )
            batch = self.verifier.verify_claims(
                evidence,
                scope,
                root_request_id=request.effective_root_id,
                request_id=request.id,
            )
            if batch.pause_reason:
                state = (
                    ResearchJobState.UNAVAILABLE_CAPABILITY
                    if batch.pause_reason in _UNAVAILABLE_REASONS
                    else ResearchJobState.PAUSED_ALLOWANCE
                )
                return _StageOutcome("paused", state, {"condition": batch.pause_reason})
            if batch.failure_code:
                status = "retryable" if batch.retryable else "failed"
                state = (
                    ResearchJobState.RETRYABLE_FAILURE
                    if batch.retryable
                    else ResearchJobState.TERMINAL_FAILURE
                )
                return _StageOutcome(status, state, {"condition": batch.failure_code})
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
                document_revision_ids=tuple(revision_ids),
                coverage=tuple(coverage),
                unresolved_questions=tuple(f"rejected:{r}" for r in rejected),
                model_attempt_refs=artifacts,
                request_id=request.id,
                assessed_at=self.clock(),
            )
        )
        ref = service.persist_assessment(
            result,
            expected_prior_revision_id=result.prior_revision_id,
            principal=SERVICE_PRINCIPAL,
        )
        self.commit()
        gaps = [
            c
            for c in coverage
            if c.outcome != CoverageOutcome.COMPLETE_FOR_REQUESTED_SCOPE
        ]
        return _StageOutcome(
            "completed",
            ResearchJobState.PARTIAL
            if gaps
            else ResearchJobState.READY_FOR_PUBLICATION,
            {
                "assessment_id": str(ref.assessment_id),
                "assessment_revision_id": str(ref.id),
                "revision_number": ref.revision_number,
                "unchanged": ref.unchanged,
                "claims": len(claims),
                "rejected": list(rejected),
                "passages": len(evidence),
                "coverage": [_coverage_dict(c) for c in coverage],
                "shadow": self.config.research_mode == ResearchMode.SHADOW,
            },
        )


def build_coordinator(
    session: Session, config: ExposureRuntimeConfig
) -> ExposureResearchCoordinator:
    """Production wiring. Secrets are read from settings, never logged."""

    from app.config import settings
    from app.services.company_exposure.acquisition import DocumentAcquisitionRegistry
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
            api_key=getattr(settings, "opencode_go_api_key", "") or "",
            client_factory=default_client_factory(),
        ),
    )
    return ExposureResearchCoordinator(
        session,
        config,
        us_adapter=us_adapter,
        verifier=ClaimVerifier(runner),
        store=store,
    )


__all__ = (
    "ExposureResearchCoordinator",
    "ResearchRequestInput",
    "ResearchRequestRef",
    "ResearchStepResult",
    "ResearchUnavailable",
    "ThemeContext",
    "build_coordinator",
    "load_theme_context",
)
