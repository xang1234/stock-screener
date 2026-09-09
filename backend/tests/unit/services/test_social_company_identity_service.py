"""Trusted identity configuration is explicit, versioned and administrator audited."""
import pytest


def service(db):
    from app.services.social_source_admin_service import SocialSourceAdminService
    from app.services.social_company_identity_service import SocialCompanyIdentityService
    SocialSourceAdminService(db).ensure_seed_sources()
    return SocialCompanyIdentityService(db, admin_authorized=True)


def entry():
    return {"symbol": "AAA", "company_id": "issuer-a", "verification_reference": "review-record:synthetic-1", "verified_at": "2026-09-07T00:00:00+00:00"}


def test_empty_configuration_and_versioned_replace(db_session):
    from app.infra.db.models.social_signals import SocialSourceAuditEvent, SocialSourceRegistry
    from app.services.social_company_identity_service import SocialCompanyIdentityService
    from app.services.social_source_admin_service import SocialSourceVersionError
    s = service(db_session)
    before = s.read()
    assert before.entries == ()
    db_session.rollback()
    s.replace([entry()], expected_version=before.registry_version, actor="admin:1")
    after = s.read()
    assert after.verified_company_ids == {"AAA": "issuer-a"}
    assert after.registry_version == before.registry_version + 1
    assert after.version == 1
    db_session.rollback()
    with pytest.raises(SocialSourceVersionError):
        s.replace([], expected_version=before.registry_version, actor="admin:1")
    with pytest.raises(PermissionError):
        SocialCompanyIdentityService(db_session).replace([], expected_version=after.registry_version, actor="admin:1")
    events = db_session.query(SocialSourceAuditEvent).filter_by(action="runtime_changed").all()
    assert len(events) == 1
    assert events[0].after_json["entry_count"] == 1
    assert db_session.get(SocialSourceRegistry, 1).version == after.registry_version


@pytest.mark.parametrize("field,value", [("symbol", "$AAA"), ("company_id", ""), ("verification_reference", ""), ("verified_at", "yesterday")])
def test_invalid_identity_never_changes_configuration(db_session, field, value):
    s = service(db_session)
    before = s.read()
    db_session.rollback()
    item = entry()
    item[field] = value
    with pytest.raises(ValueError):
        s.replace([item], expected_version=before.registry_version, actor="admin:1")
    assert s.read() == before


def test_conflicting_listing_mapping_is_rejected(db_session):
    s = service(db_session)
    version = s.read().registry_version
    db_session.rollback()
    with pytest.raises(ValueError):
        s.replace([entry(), {**entry(), "company_id": "other"}], expected_version=version, actor="admin:1")
