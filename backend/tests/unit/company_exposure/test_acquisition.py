from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from app.domain.company_exposure.contracts import DocumentTarget
from app.infra.db.repositories.company_exposure_work_repo import (
    CompanyExposureWorkRepository,
)
from app.models.company_exposure import (
    DocumentCaptureEvent,
    ExposureClaimRevision,
    ExposureDocumentRevision,
)
from app.services.company_exposure.acquisition import (
    DocumentAcquisitionRegistry,
    JobBudgetRef,
)
from app.services.company_exposure.network import PublicDocumentTransport
from app.services.company_exposure.storage import OriginalStore
from tests.fixtures.company_exposure.factory import (
    FakeRateGate,
    FixedClock,
    make_security,
    make_theme,
)

FILED = datetime(2025, 2, 14, tzinfo=timezone.utc)


class HttpMock:
    def __init__(self):
        self.responses: list = []
        self.calls = 0

    def respond(self, status=200, body=b"<html><body>Annual report</body></html>"):
        self.responses.append((status, body))

    def handler(self, request):
        self.calls += 1
        status, body = self.responses.pop(0)
        return httpx.Response(status, stream=httpx.ByteStream(body))


@pytest.fixture
def clock():
    return FixedClock()


@pytest.fixture
def http_mock():
    return HttpMock()


@pytest.fixture
def rate_spy():
    return FakeRateGate()


def _registry(db_session, tmp_path, clock, http_mock, rate_gate, *, max_bytes=100 * 1024**2):
    store = OriginalStore(
        db_session,
        tmp_path / "store",
        max_bytes=max_bytes,
        min_free_bytes=0,
        clock=clock.now,
        disk_free=lambda _p: 10**12,
    )
    transport = PublicDocumentTransport(
        transport=httpx.MockTransport(http_mock.handler),
        resolver=lambda host: ["93.184.216.34"],
    )
    return DocumentAcquisitionRegistry(
        db_session,
        transport=transport,
        store=store,
        rate_gate=rate_gate,
        user_agent="Test Research test@example.com",
        clock=clock.now,
    )


@pytest.fixture
def acquisition(db_session, tmp_path, clock, http_mock, rate_spy):
    return _registry(db_session, tmp_path, clock, http_mock, rate_spy)


@pytest.fixture
def document_target():
    return DocumentTarget(
        adapter="us_sec",
        provider="sec",
        identity_key="sec:accession:0000320193-25-000008",
        url="https://www.sec.gov/Archives/edgar/data/320193/doc.htm",
        source_kind="annual_report",
        allowed_hosts=("www.sec.gov",),
        rate_provider="sec_edgar",
        market="US",
        published_at=FILED,
        reporting_period="FY2024",
    )


@pytest.fixture
def root_budget(db_session):
    theme = make_theme(db_session)
    security = make_security(db_session, "ACME")
    request, _ = CompanyExposureWorkRepository(db_session).create_request(
        kind="verify",
        requester_principal="test:admin",
        idempotency_namespace="test",
        idempotency_key="acq",
        economic_theme_id=theme.id,
        security_id=security.id,
    )
    return JobBudgetRef(root_request_id=request.id)


@pytest.mark.case("E12")
@pytest.mark.exposure_layer("unit")
def test_identical_bytes_add_a_capture_not_a_revision(
    acquisition, document_target, root_budget, http_mock, db_session, clock
):
    http_mock.respond()
    first = acquisition.fetch(document_target, root_budget)
    clock.advance(days=30)
    http_mock.respond()
    second = acquisition.fetch(document_target, root_budget)
    assert first.changed is True and second.changed is False
    assert first.revision_id == second.revision_id
    assert db_session.query(ExposureDocumentRevision).count() == 1
    assert db_session.query(DocumentCaptureEvent).count() == 2
    revision = db_session.get(ExposureDocumentRevision, first.revision_id)
    # Business evidence date comes from the filing, not the download time.
    assert revision.published_at.replace(tzinfo=timezone.utc) == FILED
    assert revision.reporting_period == "FY2024"


@pytest.mark.case("I06")
@pytest.mark.exposure_layer("unit")
def test_404_records_gap_not_exposure_end(
    acquisition, document_target, root_budget, http_mock, db_session
):
    http_mock.respond(status=404)
    result = acquisition.fetch(document_target, root_budget)
    assert result.coverage.reason == "http_status_404"
    assert result.coverage.outcome.value == "no_matching_document"
    assert result.revision_id is None
    assert db_session.query(ExposureClaimRevision).count() == 0


def test_every_http_attempt_acquires_shared_sec_pacing(
    acquisition, document_target, root_budget, http_mock, rate_spy
):
    http_mock.respond()
    acquisition.fetch(document_target, root_budget)
    assert rate_spy.provider_names == ["sec_edgar"]
    assert all("exposure_sec" not in key for key in rate_spy.keys)


def test_pacing_outage_fails_closed_before_http(
    db_session, tmp_path, clock, http_mock, document_target, root_budget
):
    registry = _registry(
        db_session, tmp_path, clock, http_mock, FakeRateGate(unavailable=True)
    )
    http_mock.respond()
    result = registry.fetch(document_target, root_budget)
    assert result.coverage.reason == "distributed_pacing_unavailable"
    assert http_mock.calls == 0


def test_full_store_pauses_before_download(
    db_session, tmp_path, clock, http_mock, rate_spy, document_target, root_budget
):
    registry = _registry(db_session, tmp_path, clock, http_mock, rate_spy, max_bytes=10)
    http_mock.respond()
    result = registry.fetch(document_target, root_budget)
    assert result.coverage.reason == "paused_storage"
    assert http_mock.calls == 0


def test_executable_content_is_refused(
    acquisition, document_target, root_budget, http_mock, db_session
):
    http_mock.respond(body=b"MZ\x90\x00binary")
    result = acquisition.fetch(document_target, root_budget)
    assert result.coverage.reason == "media_executable_refused"
    assert db_session.query(ExposureDocumentRevision).count() == 0


def test_document_budget_is_cumulative_per_root(
    acquisition, document_target, root_budget, http_mock
):
    for _ in range(12):
        http_mock.respond()
        acquisition.fetch(document_target, root_budget)
    result = acquisition.fetch(document_target, root_budget)
    assert result.coverage.reason == "root_budget_exhausted"


def test_supplied_original_is_retained_without_network(
    acquisition, document_target, http_mock, db_session
):
    result = acquisition.ingest_supplied(document_target, b"%PDF-1.7 supplied report")
    assert result.changed and http_mock.calls == 0
    revision = db_session.get(ExposureDocumentRevision, result.revision_id)
    assert revision.media_type == "application/pdf"
