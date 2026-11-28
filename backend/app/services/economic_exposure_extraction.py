"""Schema-constrained extraction of reusable Economic Theme candidates."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import Any, Protocol
from uuid import UUID, uuid4

from sqlalchemy import select

from app.domain.economic_taxonomy.contracts import (
    DevelopmentSupport,
    ExposureSupport,
    ProviderAttemptOutcome,
)
from app.infra.db.repositories.economic_taxonomy_work_repo import (
    EconomicTaxonomyWorkRepository,
)
from app.models.economic_taxonomy_runtime import (
    EvidencePacket,
    ExtractionArtifact,
    ProcessingRequest,
)
from app.services.theme_identity_normalization import social_membership_key


class EvidenceSchemaError(ValueError):
    """A provider response cannot become a valid extraction artifact."""


# The provider refused the call before inference for rate or spend limits.
CAPACITY_UNAVAILABLE = "provider_capacity_unavailable"


class BudgetExhausted(RuntimeError):
    """No provider reservation can be made for this call."""


class ProviderCallError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        provider_request_id: str | None = None,
        actual_cost: Decimal | None = None,
        dispatch_confirmed: bool = True,
    ):
        super().__init__(message)
        self.provider_request_id = provider_request_id
        self.actual_cost = actual_cost
        self.dispatch_confirmed = dispatch_confirmed


class RetryableProviderError(ProviderCallError):
    pass


class UncertainProviderError(ProviderCallError):
    pass


class TerminalProviderError(ProviderCallError):
    pass


class RetryableProviderFailure(RuntimeError):
    pass


class ProviderOutcomeUncertain(RuntimeError):
    pass


class ProviderTerminalFailure(RuntimeError):
    pass


class ProviderReservation(Protocol):
    state: str

    def mark_dispatched(self) -> None: ...

    def release_pre_dispatch(self) -> None: ...

    def reconcile(
        self, *, actual_cost: Decimal | None, provider_request_id: str | None
    ) -> None: ...


class ReservationManager(Protocol):
    def reserve(
        self,
        *,
        attempt_key: str,
        logical_request_id: UUID,
        operation_kind: str,
    ) -> ProviderReservation | None: ...


class ExtractionProvider(Protocol):
    def extract(self, *, evidence: Mapping[str, Any], policy_version: str): ...


@dataclass(frozen=True, slots=True)
class AttemptState:
    state: str


@dataclass(frozen=True, slots=True)
class FailureDisposition:
    attempt: AttemptState
    retry_allowed: bool


class _NoopReservation:
    state = "reserved"

    def mark_dispatched(self) -> None:
        self.state = "dispatched"

    def release_pre_dispatch(self) -> None:
        self.state = "released"

    def reconcile(
        self, *, actual_cost: Decimal | None, provider_request_id: str | None
    ) -> None:
        self.state = "uncertain" if actual_cost is None else "reconciled"


def canonical_response_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def jsonable(payload: Any) -> Any:
    return json.loads(json.dumps(payload, default=str))


def provider_metadata(payload: Mapping[str, Any]) -> tuple[str | None, Decimal | None]:
    provider_request_id = payload.get("provider_request_id")
    raw_cost = payload.get("actual_cost")
    actual_cost = Decimal(str(raw_cost)) if raw_cost is not None else None
    return (
        str(provider_request_id) if provider_request_id is not None else None,
        actual_cost,
    )


class EconomicExposureExtractor:
    """Own short DB transactions and run provider I/O between them."""

    def __init__(
        self,
        session_factory,
        *,
        provider: ExtractionProvider,
        extraction_policy_version: str,
        approved_dimensions: Iterable[str],
        reservations: ReservationManager | None = None,
    ):
        if not extraction_policy_version.strip():
            raise ValueError("extraction_policy_version must be non-empty")
        self.session_factory = session_factory
        self.provider = provider
        self.extraction_policy_version = extraction_policy_version
        self.approved_dimensions = frozenset(
            str(value) for value in approved_dimensions
        )
        self.reservations = reservations

    def extract(self, request: ProcessingRequest | UUID) -> ExtractionArtifact:
        request_id = request if isinstance(request, UUID) else request.id
        with self.session_factory() as session:
            persisted = session.get(ProcessingRequest, request_id)
            if persisted is None:
                raise KeyError(f"processing request {request_id} not found")
            existing = session.execute(
                select(ExtractionArtifact).where(
                    ExtractionArtifact.evidence_packet_id
                    == persisted.evidence_packet_id,
                    ExtractionArtifact.extraction_policy_version
                    == self.extraction_policy_version,
                )
            ).scalar_one_or_none()
            if existing is not None:
                return self._detach(session, existing)
            packet = session.get(EvidencePacket, persisted.evidence_packet_id)
            if packet is None:
                raise KeyError(
                    f"evidence packet {persisted.evidence_packet_id} not found"
                )
            evidence = self._packet_snapshot(packet)
            evidence_packet_id = packet.id

        dispatch_id = str(uuid4())
        reservation = self._reserve(request_id, dispatch_id, "extract")
        with self.session_factory() as session:
            attempt = EconomicTaxonomyWorkRepository(session).begin_provider_attempt(
                request_id,
                operation="extract",
                dispatch_id=dispatch_id,
            )
            attempt_id = attempt.id
            session.commit()

        reservation.mark_dispatched()
        try:
            raw_response = self.provider.extract(
                evidence=evidence,
                policy_version=self.extraction_policy_version,
            )
        except ProviderCallError as exc:
            self._handle_provider_failure(attempt_id, reservation, exc)
        except Exception as exc:
            wrapped = TerminalProviderError(str(exc), dispatch_confirmed=True)
            self._handle_provider_failure(attempt_id, reservation, wrapped)
            raise AssertionError("unreachable") from exc

        if not isinstance(raw_response, Mapping):
            raw_response = {"status": "invalid", "raw_response": raw_response}
        provider_request_id, actual_cost = provider_metadata(raw_response)
        reservation.reconcile(
            actual_cost=actual_cost,
            provider_request_id=provider_request_id,
        )
        response_hash = canonical_response_hash(raw_response)
        try:
            normalized = self._normalize_response(raw_response, evidence=evidence)
        except EvidenceSchemaError:
            with self.session_factory() as session:
                EconomicTaxonomyWorkRepository(session).fail_attempt(
                    attempt_id,
                    outcome=ProviderAttemptOutcome.TERMINAL_FAILURE.value,
                    payload={"failure_code": "invalid_schema"},
                )
                session.commit()
            raise

        with self.session_factory() as session:
            work_repo = EconomicTaxonomyWorkRepository(session)
            artifact = session.execute(
                select(ExtractionArtifact).where(
                    ExtractionArtifact.evidence_packet_id == evidence_packet_id,
                    ExtractionArtifact.extraction_policy_version
                    == self.extraction_policy_version,
                )
            ).scalar_one_or_none()
            if artifact is None:
                artifact = ExtractionArtifact(
                    evidence_packet_id=evidence_packet_id,
                    extraction_policy_version=self.extraction_policy_version,
                    result_status=normalized["status"],
                    result_payload=normalized,
                    provider_response_hash=response_hash,
                )
                session.add(artifact)
                session.flush()
                dimension_proposals: dict[str, dict[str, set[str]]] = {}
                for candidate in normalized["candidates"]:
                    work_repo.record_candidate(
                        request_id,
                        candidate_key=candidate["candidate_key"],
                        payload=candidate,
                    )
                    for dimension in candidate["unknown_dimensions"]:
                        proposal = dimension_proposals.setdefault(
                            dimension, {"raw_values": set(), "candidate_keys": set()}
                        )
                        proposal["raw_values"].add(
                            str(candidate["raw_facets"][dimension])
                        )
                        proposal["candidate_keys"].add(candidate["candidate_key"])
                for dimension, proposal in dimension_proposals.items():
                    raw_values = sorted(proposal["raw_values"])
                    candidate_keys = sorted(proposal["candidate_keys"])
                    work_repo.record_dimension_proposal(
                        request_id,
                        dimension_key=dimension,
                        payload={
                            "dimension_key": dimension,
                            "raw_value": raw_values[0] if len(raw_values) == 1 else None,
                            "raw_values": raw_values,
                            "candidate_keys": candidate_keys,
                        },
                    )
            work_repo.succeed_attempt(
                attempt_id,
                result_artifact_id=artifact.id,
                payload={"provider_response_hash": response_hash},
            )
            session.commit()
            session.refresh(artifact)
            return self._detach(session, artifact)

    def record_timeout(
        self, request_id: UUID, *, provider_dispatch_confirmed: bool
    ) -> FailureDisposition:
        del request_id
        state = "uncertain" if provider_dispatch_confirmed else "released"
        return FailureDisposition(
            attempt=AttemptState(state), retry_allowed=not provider_dispatch_confirmed
        )

    def record_failure(
        self, request_id: UUID, *, provider_dispatch_confirmed: bool
    ) -> FailureDisposition:
        return self.record_timeout(
            request_id, provider_dispatch_confirmed=provider_dispatch_confirmed
        )

    def _reserve(
        self, request_id: UUID, dispatch_id: str, operation: str
    ) -> ProviderReservation:
        if self.reservations is None:
            return _NoopReservation()
        reservation = self.reservations.reserve(
            attempt_key=dispatch_id,
            logical_request_id=request_id,
            operation_kind=operation,
        )
        if reservation is None:
            raise BudgetExhausted("budget_exhausted")
        return reservation

    def _handle_provider_failure(
        self,
        attempt_id: UUID,
        reservation: ProviderReservation,
        error: ProviderCallError,
    ) -> None:
        if not error.dispatch_confirmed:
            reservation.release_pre_dispatch()
            outcome = ProviderAttemptOutcome.RETRYABLE_FAILURE.value
            raised = RetryableProviderFailure(str(error))
        else:
            reservation.reconcile(
                actual_cost=error.actual_cost,
                provider_request_id=error.provider_request_id,
            )
            if isinstance(error, TerminalProviderError):
                outcome = ProviderAttemptOutcome.TERMINAL_FAILURE.value
                raised = ProviderTerminalFailure(str(error))
            elif isinstance(error, UncertainProviderError) or error.actual_cost is None:
                outcome = ProviderAttemptOutcome.UNCERTAIN.value
                raised = ProviderOutcomeUncertain(str(error))
            else:
                outcome = ProviderAttemptOutcome.RETRYABLE_FAILURE.value
                raised = RetryableProviderFailure(str(error))
        with self.session_factory() as session:
            EconomicTaxonomyWorkRepository(session).fail_attempt(
                attempt_id,
                outcome=outcome,
                payload={"provider_request_id": error.provider_request_id},
            )
            session.commit()
        raise raised from error

    def _normalize_response(
        self,
        response: Mapping[str, Any],
        *,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        status = str(response.get("status") or "")
        candidates = response.get("candidates")
        if status == "successful_empty":
            if candidates not in (None, []):
                raise EvidenceSchemaError("successful_empty_requires_no_candidates")
            return {
                "status": "successful_empty",
                "candidates": [],
                "accepted_names": [],
            }
        if status not in {"accepted_candidates", "review_required"}:
            raise EvidenceSchemaError("invalid_result_status")
        if not isinstance(candidates, list) or not candidates:
            raise EvidenceSchemaError("candidate_list_required")

        evidence_text = "\n".join(
            str(value)
            for value in (
                evidence.get("original_text"),
                evidence.get("translated_text"),
            )
            if value
        )
        normalized = [
            self._normalize_candidate(
                candidate,
                evidence=evidence,
                evidence_text=evidence_text,
            )
            for candidate in candidates
        ]
        has_review = status == "review_required" or any(
            row["review_reasons"] for row in normalized
        )
        accepted_names = [
            row["display_name"] for row in normalized if not row["review_reasons"]
        ]
        return {
            "status": "review_required" if has_review else "accepted_candidates",
            "candidates": normalized,
            "accepted_names": accepted_names,
        }

    def _normalize_candidate(
        self,
        candidate: Any,
        *,
        evidence: Mapping[str, Any],
        evidence_text: str,
    ) -> dict[str, Any]:
        if not isinstance(candidate, Mapping):
            raise EvidenceSchemaError("candidate_must_be_object")
        raw_facets = candidate.get("raw_facets")
        if not isinstance(raw_facets, Mapping):
            raise EvidenceSchemaError("candidate_facets_required")
        exposure_support = str(candidate.get("exposure_support") or "")
        if exposure_support not in {value.value for value in ExposureSupport}:
            raise EvidenceSchemaError("invalid_exposure_support")
        development_support = str(candidate.get("development_support") or "")
        if development_support not in {value.value for value in DevelopmentSupport}:
            raise EvidenceSchemaError("invalid_development_support")
        key = str(candidate.get("candidate_key") or "").strip()
        display_name = str(candidate.get("display_name") or "").strip()
        if not key or not display_name:
            raise EvidenceSchemaError("candidate_identity_required")
        spans = candidate.get("evidence_spans")
        if not isinstance(spans, list):
            raise EvidenceSchemaError("evidence_spans_must_be_list")
        spans = [str(span).strip() for span in spans if str(span).strip()]
        relationship_evidence = candidate.get("relationship_evidence") or []
        if not isinstance(relationship_evidence, list):
            raise EvidenceSchemaError("relationship_evidence_must_be_list")
        relationship_evidence = [
            str(span).strip() for span in relationship_evidence if str(span).strip()
        ]

        raw_facets = {str(key): value for key, value in raw_facets.items()}
        unknown = sorted(set(raw_facets) - self.approved_dimensions)
        reasons: list[str] = []
        if unknown:
            reasons.extend(f"unknown_dimension:{dimension}" for dimension in unknown)
        if not spans:
            reasons.append("quoted_evidence_required")
        elif any(span.casefold() not in evidence_text.casefold() for span in spans):
            reasons.append("evidence_span_not_in_packet")
        facet_count = sum(value is not None for value in raw_facets.values())
        if facet_count > 1 and not relationship_evidence:
            reasons.append("compound_relationship_evidence_required")
        elif facet_count > 1 and not self._supports_compound_relationship(
            relationship_evidence,
            raw_facets,
            evidence_text,
        ):
            reasons.append("relationship_evidence_not_in_packet")
        if exposure_support in {
            ExposureSupport.UNSUPPORTED.value,
            ExposureSupport.UNRESOLVED.value,
        }:
            reasons.append("exposure_support_review_required")
        source_memberships = (
            (evidence.get("source_metadata") or {}).get("social_memberships") or []
        )
        if not isinstance(source_memberships, list):
            raise EvidenceSchemaError("social_memberships_must_be_list")
        raw_membership_keys = candidate.get("source_membership_keys") or []
        if not isinstance(raw_membership_keys, list):
            raise EvidenceSchemaError("source_membership_keys_must_be_list")
        source_membership_keys = sorted(
            {str(value).strip() for value in raw_membership_keys if str(value).strip()}
        )
        known_membership_keys = {
            str(row.get("membership_key") or social_membership_key(
                str(row.get("theme_key") or ""), row.get("security_id")
            ))
            for row in source_memberships
            if isinstance(row, Mapping)
            and isinstance(row.get("security_id"), int)
            and not isinstance(row.get("security_id"), bool)
        }
        if source_memberships and not source_membership_keys:
            reasons.append("social_membership_link_required")
        elif set(source_membership_keys) - known_membership_keys:
            reasons.append("unknown_social_membership_link")
        return {
            "candidate_key": key,
            "display_name": display_name,
            "raw_facets": jsonable(raw_facets),
            "mechanism": str(candidate.get("mechanism") or "").strip() or None,
            "evidence_spans": spans,
            "relationship_evidence": relationship_evidence,
            "exposure_support": exposure_support,
            "development_support": development_support,
            "candidate_kind": str(
                candidate.get("candidate_kind") or "economic_exposure"
            ),
            "securities": jsonable(candidate.get("securities") or []),
            "source_membership_keys": source_membership_keys,
            "unknown_dimensions": unknown,
            "review_reasons": reasons,
        }

    @staticmethod
    def _supports_compound_relationship(
        relationship_evidence: list[str],
        raw_facets: Mapping[str, Any],
        evidence_text: str,
    ) -> bool:
        corpus = evidence_text.casefold()
        facet_values = [
            str(value).casefold() for value in raw_facets.values() if value is not None
        ]
        return any(
            quote.casefold() in corpus
            and all(value in quote.casefold() for value in facet_values)
            for quote in relationship_evidence
        )

    @staticmethod
    def _packet_snapshot(packet: EvidencePacket) -> dict[str, Any]:
        return {
            "evidence_packet_id": str(packet.id),
            "original_text": packet.original_text_ref,
            "translated_text": packet.translated_text_ref,
            "translation_version": packet.translation_version,
            "attachment_hashes": list(packet.attachment_hashes or []),
            "extracted_text_hashes": list(packet.extracted_text_hashes or []),
            "grounding_snapshot": dict(packet.grounding_snapshot or {}),
            "preparation_version": packet.preparation_version,
            "source_metadata": dict(packet.source_metadata or {}),
        }

    @staticmethod
    def _detach(session, row):
        if row is not None:
            session.expunge(row)
        return row
