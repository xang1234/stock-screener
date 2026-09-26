"""Issuer identity: reviewed links and registry matches.

Rules (spec §4.2):

* Names never merge issuers; identifiers are scheme- and market-scoped.
* A new shared-issuer/cross-listing link needs administrator acceptance.
* A single, unambiguous, non-conflicting official-registry match (e.g. SEC
  ticker→CIK confirmed by the submissions record) may be accepted by the
  research service principal as a registry-resolved single-listing link.
  Every ambiguity, conflict, ticker change or cross-listing becomes a
  ``review_required`` proposal instead.
* Link history is append-only; a pending proposal never removes an accepted
  link, and research never overwrites an administrator link.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import (
    LAUNCH_MARKETS,
    SERVICE_PRINCIPAL,
    LinkAcceptancePolicy,
    LinkState,
    RegistryMatch,
    content_hash,
    normalized_identifier,
    utc_now,
)
from app.models.company_exposure import (
    ExposureIssuer,
    IssuerIdentifierRevision,
    IssuerSecurityLinkRevision,
)
from app.models.stock_universe import StockUniverse
from app.services.company_exposure.fence import research_write


class IssuerIdentityError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class IssuerResolution:
    security_id: int
    issuer_id: UUID | None
    link_revision_id: UUID | None
    acceptance_policy: str | None
    identifiers: dict = field(default_factory=dict)
    listing_security_ids: tuple[int, ...] = ()
    pending_review: bool = False

    @property
    def resolved(self) -> bool:
        return self.issuer_id is not None


@dataclass(frozen=True, slots=True)
class IssuerLinkRef:
    state: str
    link_revision_id: UUID
    issuer_id: UUID
    acceptance_policy: str
    created: bool = True


@dataclass(frozen=True, slots=True)
class ProposalRef:
    state: str
    link_revision_id: UUID | None
    reason: str
    proposal_hash: str | None = None


@dataclass(frozen=True, slots=True)
class LinkProposal:
    security_id: int
    issuer_id: UUID | None
    identifiers: tuple[tuple[str, str, str], ...]
    evidence: dict
    requested_by: str
    reason: str


def _is_admin(principal) -> bool:
    roles = getattr(principal, "roles", None)
    subject = getattr(principal, "subject", None)
    return bool(subject) and roles is not None and "taxonomy:review" in roles


class IssuerIdentityAdapter:
    def __init__(self, session: Session, *, clock: Callable = utc_now):
        self.session = session
        self.clock = clock

    # -- reads -------------------------------------------------------------

    def _link_revisions(self, security_id: int) -> list[IssuerSecurityLinkRevision]:
        return list(
            self.session.execute(
                select(IssuerSecurityLinkRevision)
                .where(IssuerSecurityLinkRevision.security_id == security_id)
                .order_by(IssuerSecurityLinkRevision.revision_number.desc())
            ).scalars()
        )

    def current_link(self, security_id: int) -> IssuerSecurityLinkRevision | None:
        """Latest accepted link unless a later revision rejected it.

        Proposals and review requests never displace an accepted link.
        """

        for revision in self._link_revisions(security_id):
            if revision.state == LinkState.ACCEPTED:
                return revision
            if revision.state == LinkState.REJECTED:
                return None
        return None

    def _issuer_listings(self, issuer_id: UUID) -> tuple[int, ...]:
        security_ids = self.session.execute(
            select(IssuerSecurityLinkRevision.security_id)
            .where(IssuerSecurityLinkRevision.issuer_id == issuer_id)
            .distinct()
        ).scalars()
        return tuple(
            sorted(
                sid
                for sid in security_ids
                if (link := self.current_link(sid)) is not None
                and link.issuer_id == issuer_id
            )
        )

    def _identifier_owner(self, key: tuple[str, str, str]) -> UUID | None:
        revision = self.session.execute(
            select(IssuerIdentifierRevision)
            .where(
                IssuerIdentifierRevision.market == key[0],
                IssuerIdentifierRevision.scheme == key[1],
                IssuerIdentifierRevision.value == key[2],
            )
            .order_by(IssuerIdentifierRevision.revision_number.desc())
            .limit(1)
        ).scalar_one_or_none()
        if revision is None or revision.state != LinkState.ACCEPTED:
            return None
        return revision.issuer_id

    def identifiers_for(self, issuer_id: UUID) -> dict[tuple[str, str], str]:
        rows = self.session.execute(
            select(IssuerIdentifierRevision)
            .where(IssuerIdentifierRevision.issuer_id == issuer_id)
            .order_by(IssuerIdentifierRevision.revision_number)
        ).scalars()
        identifiers: dict[tuple[str, str], str] = {}
        for row in rows:
            key = (row.market, row.scheme, row.value)
            if self._identifier_owner(key) == issuer_id:
                identifiers[(row.market, row.scheme)] = row.value
        return identifiers

    def resolve_security(self, security_id: int, selection=None) -> IssuerResolution:
        """Current accepted issuer for a listing (``selection`` is reserved
        for generation-pinned reads once issuer links are published)."""

        del selection
        link = self.current_link(security_id)
        revisions = self._link_revisions(security_id)
        pending = bool(revisions) and revisions[0].state in {
            LinkState.PROPOSED,
            LinkState.REVIEW_REQUIRED,
        }
        if link is None:
            return IssuerResolution(
                security_id, None, None, None, pending_review=pending
            )
        return IssuerResolution(
            security_id=security_id,
            issuer_id=link.issuer_id,
            link_revision_id=link.id,
            acceptance_policy=link.acceptance_policy,
            identifiers=self.identifiers_for(link.issuer_id),
            listing_security_ids=self._issuer_listings(link.issuer_id),
            pending_review=pending,
        )

    # -- writes --------------------------------------------------------------

    def _next_link_number(self, security_id: int) -> int:
        current = self.session.execute(
            select(func.max(IssuerSecurityLinkRevision.revision_number)).where(
                IssuerSecurityLinkRevision.security_id == security_id
            )
        ).scalar_one()
        return int(current or 0) + 1

    def _new_issuer(self, provenance: dict, actor: str) -> ExposureIssuer:
        issuer = ExposureIssuer(id=uuid4(), provenance=provenance, created_by=actor)
        self.session.add(issuer)
        self.session.flush()
        return issuer

    def _add_identifier(
        self,
        issuer_id: UUID,
        key: tuple[str, str, str],
        *,
        state: LinkState,
        policy: LinkAcceptancePolicy | None,
        evidence: dict,
        actor: str,
        reason: str,
    ) -> IssuerIdentifierRevision:
        current = self.session.execute(
            select(func.max(IssuerIdentifierRevision.revision_number)).where(
                IssuerIdentifierRevision.market == key[0],
                IssuerIdentifierRevision.scheme == key[1],
                IssuerIdentifierRevision.value == key[2],
            )
        ).scalar_one()
        row = IssuerIdentifierRevision(
            issuer_id=issuer_id,
            market=key[0],
            scheme=key[1],
            value=key[2],
            revision_number=int(current or 0) + 1,
            state=state.value,
            acceptance_policy=None if policy is None else policy.value,
            evidence=evidence,
            actor=actor,
            reason=reason,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def _add_link(
        self,
        security: StockUniverse,
        issuer_id: UUID,
        *,
        state: LinkState,
        policy: LinkAcceptancePolicy,
        actor: str,
        reason: str,
        evidence: dict,
        snapshot_extra: dict | None = None,
    ) -> IssuerSecurityLinkRevision:
        prior = self._link_revisions(security.id)
        row = IssuerSecurityLinkRevision(
            security_id=security.id,
            issuer_id=issuer_id,
            revision_number=self._next_link_number(security.id),
            state=state.value,
            acceptance_policy=policy.value,
            link_scope="listing",
            actor=actor,
            reason=reason,
            evidence=evidence,
            snapshot={
                "ticker": security.symbol,
                "market": security.market,
                "exchange": security.exchange,
                "listing_name": security.name,
                **(snapshot_extra or {}),
            },
            prior_revision_id=prior[0].id if prior else None,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def _security(self, security_id: int, *, lock: bool = False) -> StockUniverse:
        statement = select(StockUniverse).where(StockUniverse.id == security_id)
        if lock:
            # Serializes link decisions and revision numbering per listing.
            statement = statement.with_for_update()
        security = self.session.execute(statement).scalar_one_or_none()
        if security is None:
            raise IssuerIdentityError("security_not_found")
        return security

    def propose_link(self, proposal: LinkProposal) -> ProposalRef:
        """Record a link proposal for administrator review. Never accepts."""

        with research_write(self.session):
            security = self._security(proposal.security_id, lock=True)
            issuer_id = proposal.issuer_id
            if issuer_id is None:
                issuer_id = self._new_issuer(
                    {"origin": "link_proposal", "requested_by": proposal.requested_by},
                    proposal.requested_by,
                ).id
            for key in proposal.identifiers:
                self._add_identifier(
                    issuer_id,
                    normalized_identifier(*key),
                    state=LinkState.REVIEW_REQUIRED,
                    policy=None,
                    evidence=proposal.evidence,
                    actor=proposal.requested_by,
                    reason=proposal.reason,
                )
            link = self._add_link(
                security,
                issuer_id,
                state=LinkState.REVIEW_REQUIRED,
                policy=LinkAcceptancePolicy.ADMINISTRATOR_REVIEWED,
                actor=proposal.requested_by,
                reason=proposal.reason,
                evidence={
                    **proposal.evidence,
                    "proposed_identifiers": [list(k) for k in proposal.identifiers],
                },
            )
        return ProposalRef(
            state=LinkState.REVIEW_REQUIRED.value,
            link_revision_id=link.id,
            reason=proposal.reason,
            proposal_hash=self.proposal_hash(link),
        )

    @staticmethod
    def proposal_hash(link: IssuerSecurityLinkRevision) -> str:
        return content_hash(
            {
                "id": link.id,
                "security_id": link.security_id,
                "issuer_id": link.issuer_id,
                "revision_number": link.revision_number,
                "evidence": link.evidence,
            }
        )

    def apply_link(
        self, preview_id: UUID, principal, expected_hash: str
    ) -> IssuerLinkRef:
        """Administrator acceptance of a pending proposal (trusted principal)."""

        if not _is_admin(principal):
            raise PermissionError("admin_required")
        proposal = self.session.get(IssuerSecurityLinkRevision, preview_id)
        if proposal is None or proposal.state != LinkState.REVIEW_REQUIRED:
            raise IssuerIdentityError("proposal_not_pending")
        if self.proposal_hash(proposal) != expected_hash:
            raise IssuerIdentityError("stale_proposal")
        with research_write(self.session):
            security = self._security(proposal.security_id, lock=True)
            latest = self._link_revisions(proposal.security_id)[0]
            if latest.id != proposal.id:
                raise IssuerIdentityError("stale_proposal")
            for key in proposal.evidence.get("proposed_identifiers", []):
                key = normalized_identifier(*key)
                owner = self._identifier_owner(key)
                if owner is not None and owner != proposal.issuer_id:
                    raise IssuerIdentityError("identifier_owned_by_other_issuer")
                if owner is None:
                    self._add_identifier(
                        proposal.issuer_id,
                        key,
                        state=LinkState.ACCEPTED,
                        policy=LinkAcceptancePolicy.ADMINISTRATOR_REVIEWED,
                        evidence={"proposal_id": str(proposal.id)},
                        actor=principal.subject,
                        reason="administrator accepted link proposal",
                    )
            link = self._add_link(
                security,
                proposal.issuer_id,
                state=LinkState.ACCEPTED,
                policy=LinkAcceptancePolicy.ADMINISTRATOR_REVIEWED,
                actor=principal.subject,
                reason="administrator accepted link proposal",
                evidence={"proposal_id": str(proposal.id), **proposal.evidence},
            )
        return IssuerLinkRef(
            LinkState.ACCEPTED.value,
            link.id,
            link.issuer_id,
            LinkAcceptancePolicy.ADMINISTRATOR_REVIEWED.value,
        )

    def accept_registry_match(
        self, match: RegistryMatch, service_principal: str = SERVICE_PRINCIPAL
    ) -> IssuerLinkRef | ProposalRef:
        """Accept an unambiguous official-registry match for a single listing."""

        if service_principal != SERVICE_PRINCIPAL:
            raise PermissionError("service_principal_required")
        with research_write(self.session):
            return self._accept_registry_match_locked(match)

    def _accept_registry_match_locked(self, match: RegistryMatch):
        security = self._security(match.security_id, lock=True)
        evidence = {
            "registry_capture_revision_id": (
                None
                if match.registry_capture_revision_id is None
                else str(match.registry_capture_revision_id)
            ),
            "official_record_capture_revision_id": (
                None
                if match.official_record_capture_revision_id is None
                else str(match.official_record_capture_revision_id)
            ),
            "entity_title": match.entity_title,
            "matched_ticker": match.matched_ticker,
            "matched_exchange": match.matched_exchange,
            "resolver_policy_version": match.resolver_policy_version,
            "candidates": list(match.candidates),
        }
        reason = self._registry_blocker(security, match)
        key = (
            None
            if match.value is None
            else normalized_identifier(match.market, match.scheme, match.value)
        )
        if reason is None:
            owner = self._identifier_owner(key)
            current = self.current_link(security.id)
            if owner is not None and current is not None and owner == current.issuer_id:
                return IssuerLinkRef(
                    LinkState.ACCEPTED.value,
                    current.id,
                    current.issuer_id,
                    current.acceptance_policy,
                    created=False,
                )
            if owner is not None and (
                current is not None or self._issuer_listings(owner)
            ):
                reason = "cik_linked_to_other_issuer"
            elif current is not None:
                existing = self.identifiers_for(current.issuer_id).get((key[0], key[1]))
                if len(self._issuer_listings(current.issuer_id)) > 1:
                    reason = "cross_listed_issuer"
                elif existing is not None and existing != key[2]:
                    reason = "security_already_linked_elsewhere"
        if reason is not None:
            with research_write(self.session):
                issuer_id = (
                    self.current_link(security.id).issuer_id
                    if self.current_link(security.id) is not None
                    else self._new_issuer(
                        {"origin": "registry_review", "market": match.market},
                        SERVICE_PRINCIPAL,
                    ).id
                )
                link = self._add_link(
                    security,
                    issuer_id,
                    state=LinkState.REVIEW_REQUIRED,
                    policy=LinkAcceptancePolicy.ADMINISTRATOR_REVIEWED,
                    actor=SERVICE_PRINCIPAL,
                    reason=reason,
                    evidence={
                        **evidence,
                        "blocked_by": reason,
                        "proposed_identifiers": [] if key is None else [list(key)],
                    },
                )
            return ProposalRef(
                LinkState.REVIEW_REQUIRED.value,
                link.id,
                reason,
                proposal_hash=self.proposal_hash(link),
            )

        with research_write(self.session):
            current = self.current_link(security.id)
            owner = self._identifier_owner(key)
            if current is not None:
                issuer_id = current.issuer_id
            elif owner is not None:
                issuer_id = owner
            else:
                issuer_id = self._new_issuer(
                    {"origin": "official_registry", "market": match.market},
                    SERVICE_PRINCIPAL,
                ).id
            if owner is None:
                self._add_identifier(
                    issuer_id,
                    key,
                    state=LinkState.ACCEPTED,
                    policy=LinkAcceptancePolicy.OFFICIAL_REGISTRY_SINGLE_LISTING,
                    evidence=evidence,
                    actor=SERVICE_PRINCIPAL,
                    reason="unambiguous official registry match",
                )
            link = self._add_link(
                security,
                issuer_id,
                state=LinkState.ACCEPTED,
                policy=LinkAcceptancePolicy.OFFICIAL_REGISTRY_SINGLE_LISTING,
                actor=SERVICE_PRINCIPAL,
                reason="unambiguous official registry match",
                evidence=evidence,
                snapshot_extra={
                    "entity_title": match.entity_title,
                    "identifier": list(key),
                },
            )
        return IssuerLinkRef(
            LinkState.ACCEPTED.value,
            link.id,
            issuer_id,
            LinkAcceptancePolicy.OFFICIAL_REGISTRY_SINGLE_LISTING.value,
        )

    def _registry_blocker(
        self, security: StockUniverse, match: RegistryMatch
    ) -> str | None:
        if security.market not in LAUNCH_MARKETS or security.market != match.market:
            return "unsupported_market"
        if not security.is_active or security.status != "active":
            return "inactive_listing"
        if match.candidate_count != 1 or match.value is None:
            return "multiple_ciks" if match.candidate_count > 1 else "no_registry_match"
        if not match.ticker_confirmed:
            return "ticker_not_in_submissions"
        for revision in self._link_revisions(security.id):
            ticker = (revision.snapshot or {}).get("ticker")
            if ticker is not None and ticker != match.matched_ticker:
                return "ticker_changed_since_prior_link"
            identifier = (revision.snapshot or {}).get("identifier")
            if identifier is not None:
                prior = tuple(identifier)
                try:
                    wanted = normalized_identifier(
                        match.market, match.scheme, match.value
                    )
                except ValueError:
                    return "invalid_identifier"
                if prior != wanted:
                    return "security_already_linked_elsewhere"
        if match.matched_ticker != security.symbol and (
            match.matched_ticker.replace("-", ".") != security.symbol
        ):
            return "ticker_mismatch"
        return None
