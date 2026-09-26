from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.domain.company_exposure.contracts import SERVICE_PRINCIPAL, RegistryMatch
from app.models.company_exposure import ExposureDocumentRevision
from app.services.company_exposure.acquisition import (
    DocumentAcquisitionRegistry,
    JobBudgetRef,
)
from app.services.company_exposure.issuer_identity import IssuerIdentityAdapter
from app.services.company_exposure.markets.base import AcquisitionLimits, DocumentQuery
from app.services.company_exposure.markets.us import (
    USDocumentAdapter,
    USIssuerResolver,
    recent_filings,
    sec_submission_url,
    sec_ticker,
)
from app.services.company_exposure.network import PublicDocumentTransport
from app.services.company_exposure.storage import OriginalStore
from tests.fixtures.company_exposure.factory import (
    FakeRateGate,
    FixedClock,
    SecMock,
    make_security,
)

US_FIXTURES = (
    Path(__file__).resolve().parents[2] / "fixtures/company_exposure/documents/us"
)
TICKERS = json.loads((US_FIXTURES / "company_tickers_exchange.json").read_text())
SUBMISSIONS = json.loads((US_FIXTURES / "submissions_example.json").read_text())


@pytest.fixture
def sec_mock():
    return SecMock()


@pytest.fixture
def rate_spy():
    return FakeRateGate()


def _adapter(
    db_session, tmp_path, sec_mock, rate_spy, user_agent="Research Ops ops@example.com"
):
    clock = FixedClock()
    store = OriginalStore(
        db_session,
        tmp_path / "store",
        max_bytes=10**9,
        min_free_bytes=0,
        clock=clock.now,
        disk_free=lambda _p: 10**12,
    )
    registry = DocumentAcquisitionRegistry(
        db_session,
        transport=PublicDocumentTransport(
            transport=httpx.MockTransport(sec_mock.handler),
            resolver=lambda host: ["93.184.216.34"],
        ),
        store=store,
        rate_gate=rate_spy,
        user_agent=user_agent,
        clock=clock.now,
    )
    return USDocumentAdapter(db_session, registry, user_agent=user_agent)


@pytest.fixture
def us_adapter(db_session, tmp_path, sec_mock, rate_spy):
    return _adapter(db_session, tmp_path, sec_mock, rate_spy)


@pytest.fixture
def us_resolver(db_session, us_adapter):
    return USIssuerResolver(db_session, us_adapter)


@pytest.fixture
def budget():
    return JobBudgetRef(root_request_id=None)


@pytest.fixture
def us_security(db_session):
    return make_security(db_session, "EXMP")


@pytest.fixture
def amended_sec_metadata():
    rows = recent_filings(SUBMISSIONS).rows
    return {**rows[0], "cik": "0001234567"}


@pytest.mark.case("E14")
@pytest.mark.exposure_layer("unit")
def test_us_target_preserves_accession_and_amendment(us_adapter, amended_sec_metadata):
    target = us_adapter.resolve_target(amended_sec_metadata)
    assert target.provider_document_id == amended_sec_metadata["accessionNumber"]
    assert target.metadata["is_amendment"] is True
    assert target.correction_identity["accession"] == "0001234567-26-000011"
    assert target.reporting_period == "2025-12-31"
    assert target.url == (
        "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000011/"
        "exmp-20251231a.htm"
    )


@pytest.mark.case("R14")
@pytest.mark.exposure_layer("unit")
def test_us_fetch_uses_existing_sec_rate_key(
    us_adapter, amended_sec_metadata, budget, rate_spy, sec_mock
):
    target = us_adapter.resolve_target(amended_sec_metadata)
    sec_mock.serve_bytes(
        target.url, (US_FIXTURES / "annual_report_example.htm").read_bytes()
    )
    result = us_adapter.fetch(target, budget)
    assert result.changed
    assert rate_spy.provider_names == ["sec_edgar"]
    assert all("exposure_sec" not in key for key in rate_spy.keys)
    assert sec_mock.requests[0].headers["user-agent"] == "Research Ops ops@example.com"


def test_mismatched_parallel_arrays_are_not_realigned():
    broken = json.loads(json.dumps(SUBMISSIONS))
    broken["filings"]["recent"]["form"].pop()
    rows = recent_filings(broken)
    assert (rows.complete, rows.reason, rows.rows) == (
        False,
        "mismatched_filing_arrays",
        (),
    )


def test_discovery_selects_latest_annual_filings_with_amendments(
    us_adapter, sec_mock, budget
):
    sec_mock.serve_json(sec_submission_url("0001234567"), SUBMISSIONS)
    issuer = type("Issuer", (), {"identifiers": {("US", "cik"): "0001234567"}})()
    result = us_adapter.discover(
        issuer,
        DocumentQuery(document_kinds=("annual_report",), max_documents=2),
        AcquisitionLimits(),
        budget,
    )
    assert [t.provider_document_id for t in result.targets] == [
        "0001234567-26-000011",
        "0001234567-25-000007",
    ]
    assert any(item.reason == "document_limit" for item in result.coverage)


def test_missing_user_agent_is_a_capability_gap_without_network(
    db_session, tmp_path, sec_mock, rate_spy, budget
):
    adapter = _adapter(db_session, tmp_path, sec_mock, rate_spy, user_agent="")
    issuer = type("Issuer", (), {"identifiers": {("US", "cik"): "0001234567"}})()
    result = adapter.discover(issuer, DocumentQuery(), AcquisitionLimits(), budget)
    assert result.coverage[0].reason == "sec_user_agent_not_configured"
    assert sec_mock.requests == []


def test_sec_ticker_normalizes_share_classes():
    assert sec_ticker("BRK.B") == "BRK-B"
    assert sec_ticker("brk-b") == "BRK-B"


@pytest.mark.case("I02")
@pytest.mark.exposure_layer("unit")
def test_cik_resolution_retains_registry_and_submissions_evidence(
    us_resolver, us_security, budget, sec_mock, rate_spy, db_session
):
    sec_mock.serve_company_tickers(
        {
            "0": {
                "cik_str": 1234567,
                "ticker": us_security.symbol,
                "title": "Example Corp",
            }
        }
    )
    sec_mock.serve_submissions(
        "0001234567", tickers=[us_security.symbol], exchanges=["Nasdaq"]
    )
    match = us_resolver.resolve_cik(us_security.id, budget)
    assert isinstance(match, RegistryMatch)
    assert match.value == "0001234567"
    assert (
        match.registry_capture_revision_id and match.official_record_capture_revision_id
    )
    assert match.candidate_count == 1 and match.ticker_confirmed is True
    assert set(rate_spy.provider_names) == {"sec_edgar"}
    kinds = {
        db_session.get(ExposureDocumentRevision, rid).media_type
        for rid in (
            match.registry_capture_revision_id,
            match.official_record_capture_revision_id,
        )
    }
    assert kinds == {"application/json"}


@pytest.mark.case("I02")
@pytest.mark.exposure_layer("unit")
def test_submissions_without_ticker_is_not_confirmed(
    us_resolver, us_security, budget, sec_mock
):
    sec_mock.serve_company_tickers(
        {
            "0": {
                "cik_str": 1234567,
                "ticker": us_security.symbol,
                "title": "Example Corp",
            }
        }
    )
    sec_mock.serve_submissions("0001234567", tickers=["OTHER"], exchanges=["NYSE"])
    assert us_resolver.resolve_cik(us_security.id, budget).ticker_confirmed is False


def test_exchange_file_is_preferred_and_duplicate_tickers_are_ambiguous(
    db_session, us_resolver, budget, sec_mock
):
    security = make_security(db_session, "TWIN")
    sec_mock.serve_json(
        "https://www.sec.gov/files/company_tickers_exchange.json", TICKERS
    )
    sec_mock.serve_submissions("0007654321", tickers=["TWIN"], exchanges=["NYSE"])
    sec_mock.serve_submissions("0007654322", tickers=["TWIN"], exchanges=["NYSE"])
    match = us_resolver.resolve_cik(security.id, budget)
    assert match.candidate_count == 2 and match.value is None
    ref = IssuerIdentityAdapter(db_session).accept_registry_match(
        match, SERVICE_PRINCIPAL
    )
    assert (ref.state, ref.reason) == ("review_required", "multiple_ciks")


def test_resolved_cik_is_accepted_end_to_end(
    db_session, us_resolver, us_security, budget, sec_mock
):
    sec_mock.serve_json(
        "https://www.sec.gov/files/company_tickers_exchange.json", TICKERS
    )
    sec_mock.serve_submissions("0001234567", tickers=["EXMP"], exchanges=["Nasdaq"])
    match = us_resolver.resolve_cik(us_security.id, budget)
    identity = IssuerIdentityAdapter(db_session)
    ref = identity.accept_registry_match(match, SERVICE_PRINCIPAL)
    assert ref.acceptance_policy == "official_registry_single_listing"
    resolution = identity.resolve_security(us_security.id)
    assert resolution.identifiers[("US", "cik")] == "0001234567"
