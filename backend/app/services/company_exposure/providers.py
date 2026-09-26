"""One-dispatch subscription provider and reusable-artifact runner.

Research uses the OpenCode Go / Kimi subscription transport
(``theme_evaluation.kimi_client.OpenCodeGoKimi``) only. It never routes
through ``LLMService.completion``, ``EconomicTaxonomyLLMProvider`` or the
Social dollar ledger, never falls back to another model and never retries
inside the transport: each permitted retry is a new accounted attempt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import DispatchPhase, content_hash
from app.models.company_exposure import ResearchArtifact, ResearchProviderResult
from app.services.company_exposure.config import (
    SUBSCRIPTION_MODEL,
    SUBSCRIPTION_MODEL_IDENTITY,
    SUBSCRIPTION_PROVIDER,
    TEXT,
)
from app.services.company_exposure.resources import (
    DispatchOutcome,
    DispatchRequest,
    ResearchResources,
    ReservationTicket,
)
from app.services.theme_evaluation.kimi_client import OpenCodeGoKimi
from app.services.theme_evaluation.preparation_failures import PreparationFailure


@dataclass(frozen=True, slots=True)
class ProviderInput:
    """Evidence-only messages plus exact identity of what is being asked."""

    operation: str
    messages: list[dict]
    input_hash: str
    policy_hash: str
    max_output_tokens: int
    logical_operation_key: str
    capability: str = TEXT
    read_timeout_seconds: float = 45.0
    root_request_id: UUID | None = None
    request_id: UUID | None = None

    @property
    def estimated_input_tokens(self) -> int:
        # Conservative bound for the reservation; reconciled to reported usage.
        characters = sum(len(str(m.get("content", ""))) for m in self.messages)
        return max(1, (characters + 1) // 2)


@dataclass(frozen=True, slots=True)
class ProviderOutput:
    outcome: DispatchOutcome
    data: dict | None = None


@dataclass(frozen=True, slots=True)
class ArtifactRunResult:
    artifact_id: UUID | None
    ticket_id: UUID | None
    retryable: bool
    pause_reason: str | None
    payload: dict | None = field(default=None)
    reused: bool = False
    failure_code: str | None = None
    retry_after_seconds: float | None = None


ClientFactory = Callable[[str, str], OpenCodeGoKimi]


def default_client_factory(
    transport: httpx.BaseTransport | None = None,
) -> ClientFactory:
    def build(api_key: str, session_id: str) -> OpenCodeGoKimi:
        return OpenCodeGoKimi(api_key, session_id=session_id, transport=transport)

    return build


class SubscriptionProvider:
    def __init__(self, *, api_key: str, client_factory: ClientFactory):
        self._api_key = api_key or ""
        self._client_factory = client_factory

    def call_once(
        self, provider_input: ProviderInput, ticket: ReservationTicket
    ) -> ProviderOutput:
        if (
            not ticket.allowed
            or ticket.attempt_id is None
            or (ticket.route, ticket.model)
            != (SUBSCRIPTION_PROVIDER, SUBSCRIPTION_MODEL)
        ):
            raise ValueError("valid_reservation_ticket_required")
        if not self._api_key.strip():
            return ProviderOutput(
                DispatchOutcome(
                    success=False,
                    dispatch_phase=DispatchPhase.PRE_DISPATCH,
                    failure_code="subscription_credentials_missing",
                )
            )
        client = self._client_factory(self._api_key, str(ticket.attempt_id))
        try:
            response = client.complete_json_response(
                provider_input.messages,
                max_tokens=provider_input.max_output_tokens,
                read_timeout=provider_input.read_timeout_seconds,
            )
        except PreparationFailure as failure:
            # An unclassified failure is treated as possibly executed.
            phase = DispatchPhase(failure.dispatch_phase or DispatchPhase.UNCERTAIN)
            return ProviderOutput(
                DispatchOutcome(
                    success=False,
                    dispatch_phase=phase,
                    retryable=failure.retryable,
                    failure_code=failure.code,
                    retry_after_seconds=failure.retry_after_seconds,
                )
            )
        return ProviderOutput(
            DispatchOutcome(
                success=True,
                dispatch_phase=DispatchPhase.DISPATCHED,
                provider_request_id=response.provider_request_id,
                reported_usage=response.reported_usage,
                response_hash=response.response_hash,
            ),
            data=response.data,
        )


class SubscriptionArtifactRunner:
    """Cache lookup → reservation → one dispatch → immutable result/artifact.

    ``commit`` is called after the dispatch is durably marked and again after
    the result, so no database transaction is held during provider I/O.
    """

    def __init__(
        self,
        session: Session,
        resources: ResearchResources,
        provider: SubscriptionProvider,
        *,
        commit: Callable[[], None] | None = None,
    ):
        self.session = session
        self.resources = resources
        self.provider = provider
        self.commit = commit or session.commit

    def cached(self, provider_input: ProviderInput) -> ResearchArtifact | None:
        return self.session.execute(
            select(ResearchArtifact).where(
                ResearchArtifact.operation == provider_input.operation,
                ResearchArtifact.input_hash == provider_input.input_hash,
                ResearchArtifact.policy_hash == provider_input.policy_hash,
                ResearchArtifact.model_identity == SUBSCRIPTION_MODEL_IDENTITY,
            )
        ).scalar_one_or_none()

    def run(self, provider_input: ProviderInput) -> ArtifactRunResult:
        existing = self.cached(provider_input)
        if existing is not None:
            return ArtifactRunResult(
                artifact_id=existing.id,
                ticket_id=None,
                retryable=False,
                pause_reason=None,
                payload=existing.payload,
                reused=True,
            )
        ticket = self.resources.reserve(
            DispatchRequest(
                logical_operation_key=provider_input.logical_operation_key,
                operation=provider_input.operation,
                capability=provider_input.capability,
                input_hash=provider_input.input_hash,
                policy_hash=provider_input.policy_hash,
                max_output_tokens=provider_input.max_output_tokens,
                estimated_input_tokens=provider_input.estimated_input_tokens,
                root_request_id=provider_input.root_request_id,
                request_id=provider_input.request_id,
            )
        )
        if not ticket.allowed:
            self.commit()
            return ArtifactRunResult(
                artifact_id=None,
                ticket_id=None,
                retryable=False,
                pause_reason=ticket.reason or ticket.state,
            )
        self.resources.mark_dispatched(ticket)
        self.commit()

        output = self.provider.call_once(provider_input, ticket)

        self.resources.finish(ticket, output.outcome)
        if not output.outcome.success:
            self.commit()
            return ArtifactRunResult(
                artifact_id=None,
                ticket_id=ticket.id,
                retryable=output.outcome.retryable,
                pause_reason=None,
                failure_code=output.outcome.failure_code,
                retry_after_seconds=output.outcome.retry_after_seconds,
            )
        result = self.session.execute(
            select(ResearchProviderResult).where(
                ResearchProviderResult.attempt_id == ticket.attempt_id
            )
        ).scalar_one()
        artifact = ResearchArtifact(
            operation=provider_input.operation,
            input_hash=provider_input.input_hash,
            policy_hash=provider_input.policy_hash,
            model_identity=SUBSCRIPTION_MODEL_IDENTITY,
            result_id=result.id,
            payload=output.data,
            payload_hash=content_hash(output.data),
        )
        self.session.add(artifact)
        self.session.flush()
        self.commit()
        return ArtifactRunResult(
            artifact_id=artifact.id,
            ticket_id=ticket.id,
            retryable=False,
            pause_reason=None,
            payload=output.data,
        )
