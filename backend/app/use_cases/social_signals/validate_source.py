"""Explicit, non-persisting source diagnostics for administrators."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

from app.domain.social_signals.records import SocialReadRequest, SourceTestOutcome


class SocialSourceValidationDeferred(RuntimeError):
    pass


class SqlSourceTestRegistry:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def claim(self, source_id, actor):
        from app.services.social_source_admin_service import SocialSourceAdminService
        with self.session_factory() as db:
            return SocialSourceAdminService(db).claim_test(source_id, actor)

    def complete(self, request, outcome, actor):
        from app.services.social_source_admin_service import SocialSourceAdminService
        with self.session_factory() as db:
            return SocialSourceAdminService(db).record_test_result(
                request.source_id, request.provider, outcome, actor,
                request_id=request.request_id, expected_version=request.version,
            )


class ValidateSocialSource:
    def __init__(self, *, registry, providers, provider_lease, clock,
                 lease_owner_factory=None, lease_ttl_seconds=300):
        self.registry = registry
        self.providers = dict(providers)
        self.provider_lease = provider_lease
        self.clock = clock
        self.lease_owner_factory = lease_owner_factory or (lambda: uuid4().hex)
        self.lease_ttl_seconds = lease_ttl_seconds

    def _provider(self, name):
        selected = self.providers.get(name)
        if selected is None:
            raise ValueError("social_provider_not_wired")
        return selected() if callable(selected) and not hasattr(selected, "read_source") else selected

    def execute(self, source_id: int, actor: str) -> SourceTestOutcome:
        owner = f"social-source-test:{source_id}:{self.lease_owner_factory()}"
        if not self.provider_lease.acquire(owner, self.lease_ttl_seconds):
            raise SocialSourceValidationDeferred("provider_read_busy")
        try:
            pinned = self.registry.claim(source_id, actor)
            now = self.clock()
            request = SocialReadRequest(
                pinned.request_id, pinned.source_id, pinned.list_id, "test", now, 5,
                now - timedelta(days=14),
            )
            batch = self._provider(pinned.provider).read_source(request)
            if batch.outcome.read_status == "success":
                outcome = SourceTestOutcome(pinned.provider, "passed", len(batch.posts), now)
            else:
                code = batch.outcome.error_code or "provider_error"
                status = code if code in {"rate_limited", "reauthentication_required"} else "provider_error"
                outcome = SourceTestOutcome(pinned.provider, status, 0, now, code)
            self.registry.complete(pinned, outcome, actor)
            return outcome
        finally:
            self.provider_lease.release(owner)


__all__ = ["SocialSourceValidationDeferred", "SqlSourceTestRegistry", "ValidateSocialSource"]
