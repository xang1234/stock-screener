from datetime import datetime, timedelta, timezone

import pytest

from app.domain.social_signals.records import (
    SocialPostRecord, SocialSourceBatch, SocialSourceOutcome, SourceTestOutcome,
)
from app.services.social_source_admin_service import SocialSourceTestRequest


NOW = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)


class Registry:
    def __init__(self, request=None):
        self.request = request or SocialSourceTestRequest("test-1", "9", "123", "official", 4, 8)
        self.completed = []

    def claim(self, source_id, actor):
        assert source_id == 9 and actor == "admin"
        return self.request

    def complete(self, request, outcome, actor):
        self.completed.append((request, outcome, actor))
        return outcome


class Lease:
    def __init__(self, acquired=True):
        self.acquired = acquired
        self.released = []

    def acquire(self, owner, ttl_seconds):
        return self.acquired

    def release(self, owner):
        self.released.append(owner)


class Provider:
    def __init__(self, status="success", code=None):
        self.status, self.code = status, code
        self.requests = []

    def read_source(self, request):
        self.requests.append(request)
        posts = ()
        if self.status == "success":
            posts = (SocialPostRecord(
                "official", "1", "9", "sample", "https://x.com/a/1", "a",
                NOW - timedelta(hours=1), NOW,
            ),)
        return SocialSourceBatch(request, posts, SocialSourceOutcome(
            self.status, "pending" if posts else "failed", "limited",
            (() if posts else (self.code,)), (),
            posts[0].created_at if posts else None, posts[0].created_at if posts else None,
            len(posts), None, self.code,
        ))


def service(provider=None, lease=None, registry=None):
    from app.use_cases.social_signals.validate_source import ValidateSocialSource
    return ValidateSocialSource(
        registry=registry or Registry(), providers={"official": provider or Provider()},
        provider_lease=lease or Lease(), clock=lambda: NOW,
        lease_owner_factory=lambda: "attempt",
    )


def test_validates_only_pinned_source_with_diagnostic_cap_and_records_redacted_result():
    provider, registry = Provider(), Registry()
    outcome = service(provider=provider, registry=registry).execute(9, "admin")
    request = provider.requests[0]
    assert (request.intent, request.limit, request.source_id, request.list_id) == ("test", 5, "9", "123")
    assert outcome == SourceTestOutcome("official", "passed", 1, NOW)
    assert registry.completed == [(registry.request, outcome, "admin")]


@pytest.mark.parametrize("code,status", [
    ("rate_limited", "rate_limited"),
    ("reauthentication_required", "reauthentication_required"),
    ("invalid_provider_schema", "provider_error"),
    ("provider_network_error", "provider_error"),
])
def test_maps_only_stable_failure_codes(code, status):
    provider, registry = Provider("failed", code), Registry()
    outcome = service(provider=provider, registry=registry).execute(9, "admin")
    assert outcome.status == status and outcome.reason_code == code
    assert outcome.sample_count == 0


def test_busy_lease_does_not_claim_or_complete_test():
    from app.use_cases.social_signals.validate_source import SocialSourceValidationDeferred
    registry = Registry()
    with pytest.raises(SocialSourceValidationDeferred, match="provider_read_busy"):
        service(lease=Lease(False), registry=registry).execute(9, "admin")
    assert not registry.completed


def test_unknown_provider_has_no_fallback_and_lease_is_released():
    request = SocialSourceTestRequest("test-1", "9", "123", "xui", 4, 8)
    lease = Lease()
    with pytest.raises(ValueError, match="not_wired"):
        service(lease=lease, registry=Registry(request)).execute(9, "admin")
    assert lease.released


def test_real_registry_diagnostic_persists_only_redacted_test_state(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database import Base
    from app.infra.db.models.social_signals import (
        SocialContentMetrics, SocialPostSource, SocialSignalRun, SocialSignalRunPointer,
    )
    from app.models.theme import ContentItem
    from app.services.social_source_admin_service import SocialSourceAdminService
    from app.use_cases.social_signals.validate_source import SqlSourceTestRegistry, ValidateSocialSource

    engine = create_engine(f"sqlite:///{tmp_path / 'source-test.sqlite'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    with sessions() as db:
        admin = SocialSourceAdminService(db)
        admin.ensure_seed_sources()
        runtime = admin.read_runtime()
        admin.apply_runtime("validation", "official", runtime.version, "admin")
        created = admin.create_source("New list", "333", "admin")
        request = admin.request_test(created.source_id, created.version, "admin")
        assert admin.list_sources()[-1].test_progress == "queued"

    provider = Provider()
    outcome = ValidateSocialSource(
        registry=SqlSourceTestRegistry(sessions), providers={"official": provider},
        provider_lease=Lease(), clock=lambda: NOW, lease_owner_factory=lambda: "attempt",
    ).execute(int(request.source_id), "admin")
    assert outcome.status == "passed"
    with sessions() as db:
        admin = SocialSourceAdminService(db)
        tested = admin.list_sources()[-1]
        assert tested.test_progress is None and tested.test_outcome.status == "passed"
        assert [event.action for event in admin.audit_events(tested.source_id)] == [
            "created", "test_requested", "test_completed",
        ]
        assert db.query(ContentItem).count() == 0
        assert db.query(SocialPostSource).count() == 0
        assert db.query(SocialContentMetrics).count() == 0
        assert db.query(SocialSignalRun).count() == 0
        assert db.query(SocialSignalRunPointer).count() == 0
    engine.dispose()
