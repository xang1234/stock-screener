from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.domain.company_exposure.contracts import (
    SERVICE_PRINCIPAL,
    RegistryMatch,
    normalized_identifier,
)
from app.domain.economic_taxonomy.contracts import AdminPrincipal
from app.services.company_exposure.issuer_identity import (
    IssuerIdentityAdapter,
    IssuerIdentityError,
    LinkProposal,
)
from tests.fixtures.company_exposure.factory import fixed_uuid, make_security

ADMIN = AdminPrincipal(
    subject="test:admin",
    auth_method="admin_api_key",
    roles=frozenset({"taxonomy:review"}),
)


@pytest.fixture
def identity_adapter(db_session):
    return IssuerIdentityAdapter(db_session)


def _accept(identity, security_id, *, issuer_id=None, identifiers=()):
    """Administrator-reviewed link: propose, then apply."""

    proposal = identity.propose_link(
        LinkProposal(
            security_id=security_id,
            issuer_id=issuer_id,
            identifiers=identifiers,
            evidence={"reference": "filing cover page"},
            requested_by="test:admin",
            reason="administrator-verified listing",
        )
    )
    return identity.apply_link(proposal.link_revision_id, ADMIN, proposal.proposal_hash)


@pytest.fixture
def verified_cross_listings(db_session, identity_adapter):
    left = make_security(db_session, "TSM", market="US", exchange="NYSE")
    right = make_security(db_session, "2330.TW", market="TW", exchange="TWSE")
    issuer = _accept(identity_adapter, left.id).issuer_id
    _accept(identity_adapter, right.id, issuer_id=issuer)
    return SimpleNamespace(left_security_id=left.id, right_security_id=right.id)


@pytest.mark.case("I01")
@pytest.mark.exposure_layer("unit")
def test_verified_cross_listings_resolve_one_issuer(
    identity_adapter, verified_cross_listings
):
    left = identity_adapter.resolve_security(verified_cross_listings.left_security_id)
    right = identity_adapter.resolve_security(verified_cross_listings.right_security_id)
    assert left.issuer_id == right.issuer_id
    assert left.security_id != right.security_id
    assert left.listing_security_ids == right.listing_security_ids
    assert len(left.listing_security_ids) == 2
    assert left.acceptance_policy == "administrator_reviewed"


@pytest.fixture
def conflicting_link(db_session, identity_adapter, verified_cross_listings):
    return LinkProposal(
        security_id=verified_cross_listings.left_security_id,
        issuer_id=None,
        identifiers=(("US", "cik", "1046179"),),
        evidence={"reference": "similar name"},
        requested_by=SERVICE_PRINCIPAL,
        reason="name similarity suggested a different issuer",
    )


@pytest.fixture
def attestation_reader(identity_adapter, verified_cross_listings):
    return SimpleNamespace(
        read=lambda: (
            identity_adapter.resolve_security(
                verified_cross_listings.left_security_id
            ).issuer_id
        )
    )


@pytest.mark.case("I02")
@pytest.mark.exposure_layer("unit")
def test_conflicting_link_stays_a_proposal(
    identity_adapter, conflicting_link, attestation_reader
):
    before = attestation_reader.read()
    proposal = identity_adapter.propose_link(conflicting_link)
    assert proposal.state == "review_required"
    assert attestation_reader.read() == before


def test_admin_apply_accepts_proposal_and_rejects_stale_hash(
    db_session, identity_adapter
):
    security = make_security(db_session, "NEWCO")
    proposal = identity_adapter.propose_link(
        LinkProposal(
            security_id=security.id,
            issuer_id=None,
            identifiers=(("US", "cik", "42"),),
            evidence={"reference": "10-K cover page"},
            requested_by="test:admin",
            reason="administrator-supplied CIK",
        )
    )
    with pytest.raises(IssuerIdentityError):
        identity_adapter.apply_link(proposal.link_revision_id, ADMIN, "0" * 64)
    with pytest.raises(PermissionError):
        identity_adapter.apply_link(
            proposal.link_revision_id, "forged", proposal.proposal_hash
        )
    ref = identity_adapter.apply_link(
        proposal.link_revision_id, ADMIN, proposal.proposal_hash
    )
    resolution = identity_adapter.resolve_security(security.id)
    assert resolution.issuer_id == ref.issuer_id
    assert resolution.identifiers[("US", "cik")] == "0000000042"


def test_identifier_schemes_never_collide():
    assert normalized_identifier("US", "cik", "0000123456") != normalized_identifier(
        "TW", "company_code", "123456"
    )
    assert normalized_identifier("US", "cik", "123456") == ("US", "cik", "0000123456")


def _match(security, **overrides):
    base = RegistryMatch(
        security_id=security.id,
        market="US",
        scheme="cik",
        value="1234567",
        candidate_count=1,
        ticker_confirmed=True,
        matched_ticker=security.symbol,
        registry_capture_revision_id=fixed_uuid("registry-capture"),
        official_record_capture_revision_id=fixed_uuid("submissions-capture"),
        entity_title="Example Corp",
        matched_exchange="Nasdaq",
    )
    return replace(base, **overrides)


@pytest.fixture
def single_us_match(db_session):
    return _match(make_security(db_session, "EXMP"))


@pytest.mark.case("I02")
@pytest.mark.exposure_layer("unit")
def test_unambiguous_registry_cik_is_accepted_for_single_listing(
    identity_adapter, single_us_match
):
    ref = identity_adapter.accept_registry_match(single_us_match, SERVICE_PRINCIPAL)
    link = identity_adapter.resolve_security(single_us_match.security_id)
    assert ref.state == "accepted"
    assert ref.acceptance_policy == "official_registry_single_listing"
    assert link.identifiers[("US", "cik")] == "0001234567"
    again = identity_adapter.accept_registry_match(single_us_match, SERVICE_PRINCIPAL)
    assert (again.link_revision_id, again.created) == (ref.link_revision_id, False)


def test_registry_acceptance_requires_the_service_principal(
    identity_adapter, single_us_match
):
    with pytest.raises(PermissionError):
        identity_adapter.accept_registry_match(single_us_match, "test:admin")


@pytest.fixture
def registry_match_variant(db_session, identity_adapter):
    def build(variant):
        security = make_security(db_session, "VAR")
        if variant == "multiple_ciks":
            return _match(security, candidate_count=2, candidates=("1", "2"))
        if variant == "ticker_not_in_submissions":
            return _match(security, ticker_confirmed=False)
        if variant == "cik_linked_to_other_issuer":
            other = make_security(db_session, "OTHER")
            identity_adapter.accept_registry_match(_match(other), SERVICE_PRINCIPAL)
            return _match(security)
        if variant == "security_already_linked_elsewhere":
            identity_adapter.accept_registry_match(
                _match(security, value="7654321"), SERVICE_PRINCIPAL
            )
            return _match(security)
        if variant == "cross_listed_issuer":
            twin = make_security(db_session, "VAR.TW", market="TW", exchange="TWSE")
            issuer = _accept(identity_adapter, twin.id).issuer_id
            _accept(identity_adapter, security.id, issuer_id=issuer)
            return _match(security)
        if variant == "ticker_changed_since_prior_link":
            identity_adapter.accept_registry_match(_match(security), SERVICE_PRINCIPAL)
            security.symbol = "VAR2"
            db_session.flush()
            return _match(security, matched_ticker="VAR2", value="7777777")
        raise AssertionError(variant)

    return build


@pytest.mark.case("I02")
@pytest.mark.exposure_layer("unit")
@pytest.mark.parametrize(
    "variant",
    [
        "multiple_ciks",
        "ticker_not_in_submissions",
        "cik_linked_to_other_issuer",
        "security_already_linked_elsewhere",
        "cross_listed_issuer",
        "ticker_changed_since_prior_link",
    ],
)
def test_ambiguous_registry_cik_requires_review(
    identity_adapter, registry_match_variant, variant
):
    match = registry_match_variant(variant)
    before = identity_adapter.resolve_security(match.security_id).issuer_id
    ref = identity_adapter.accept_registry_match(match, SERVICE_PRINCIPAL)
    assert ref.state == "review_required"
    assert ref.reason == variant
    # A pending review never displaces an accepted link.
    assert identity_adapter.resolve_security(match.security_id).issuer_id == before
