from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException

from app.api.v1 import config as config_api
from app.api.v1.config import (
    UNBOUND_ADMIN_SUBJECT,
    authenticate_admin,
    require_admin,
)
from app.domain.economic_taxonomy.contracts import AdminPrincipal


def _settings(*, key="secret", principal="operator:alice"):
    return SimpleNamespace(admin_api_key=key, admin_principal_id=principal)


def test_valid_admin_key_returns_trusted_principal():
    principal = authenticate_admin(_settings(), x_admin_key="secret")

    assert principal == AdminPrincipal(
        subject="operator:alice",
        auth_method="admin_api_key",
        roles=frozenset({"taxonomy:review"}),
    )


@pytest.mark.parametrize("configured_key", [None, ""])
def test_missing_admin_key_fails_closed(configured_key):
    with pytest.raises(HTTPException) as error:
        authenticate_admin(_settings(key=configured_key), x_admin_key="secret")

    assert error.value.status_code == 503


@pytest.mark.parametrize("configured_principal", [None, ""])
def test_key_without_principal_keeps_admin_access_without_taxonomy_review(
    configured_principal,
):
    principal = authenticate_admin(
        _settings(principal=configured_principal), x_admin_key="secret"
    )

    assert principal.subject == UNBOUND_ADMIN_SUBJECT
    assert principal.can_review_taxonomy is False


def test_key_without_principal_still_rejects_wrong_key():
    with pytest.raises(HTTPException) as error:
        authenticate_admin(_settings(principal=""), x_admin_key="wrong")

    assert error.value.status_code == 401


def test_invalid_key_does_not_return_a_principal():
    with pytest.raises(HTTPException) as error:
        authenticate_admin(_settings(), x_admin_key="wrong")

    assert error.value.status_code == 401


def test_bearer_key_authenticates_the_configured_principal():
    principal = authenticate_admin(_settings(), authorization="Bearer secret")

    assert principal.subject == "operator:alice"


def test_dependency_reads_module_settings(monkeypatch):
    monkeypatch.setattr(config_api, "settings", _settings())

    assert require_admin(x_admin_key="secret").subject == "operator:alice"


def test_admin_dependency_exposes_no_settings_in_openapi():
    app = FastAPI()
    app.include_router(config_api.router, prefix="/api/v1")

    spec = app.openapi()

    for path, operations in spec["paths"].items():
        for operation in operations.values():
            names = {
                parameter["name"] for parameter in operation.get("parameters", ())
            }
            assert "settings" not in names, path
    assert "admin_api_key" not in str(spec)


def test_admin_principal_cannot_claim_the_reserved_system_subject():
    from app.config.settings import Settings
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="reserved"):
        Settings(admin_principal_id="system:economic-taxonomy-refresh")
    assert Settings(admin_principal_id="").admin_principal_id == ""
