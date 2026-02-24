"""Tests for theme policy admin config endpoints."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.v1.config import (
    get_theme_policy_config,
    promote_staged_theme_policy,
    revert_theme_policy,
    update_theme_policy,
)
from app.database import Base
from app.models.app_settings import AppSetting
from app.schemas.config import ThemePolicyRevertRequest, ThemePolicyUpdateRequest


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.mark.asyncio
async def test_theme_policy_preview_does_not_persist(db_session):
    response = await update_theme_policy(
        request=ThemePolicyUpdateRequest(
            pipeline="technical",
            matcher={"fuzzy_attach_threshold": 0.92},
            mode="preview",
        ),
        x_admin_actor="tester",
        db=db_session,
        _auth=True,
    )
    assert response.status == "preview"
    assert response.mode == "preview"

    fetched = await get_theme_policy_config(
        pipeline="technical",
        db=db_session,
        _auth=True,
    )
    assert fetched.overrides == {}


@pytest.mark.asyncio
async def test_theme_policy_stage_promote_and_revert_flow(db_session):
    staged = await update_theme_policy(
        request=ThemePolicyUpdateRequest(
            pipeline="technical",
            matcher={"embedding_attach_threshold": 0.88},
            lifecycle={"promotion_min_mentions_7d": 7},
            note="stage new thresholds",
            mode="stage",
        ),
        x_admin_actor="tester",
        db=db_session,
        _auth=True,
    )
    assert staged.status == "staged"
    assert staged.version_id is not None

    promoted = await promote_staged_theme_policy(
        pipeline="technical",
        note="ship staged policy",
        x_admin_actor="tester",
        db=db_session,
        _auth=True,
    )
    assert promoted.status == "applied"
    applied_version = promoted.version_id
    assert applied_version is not None

    fetched = await get_theme_policy_config(
        pipeline="technical",
        db=db_session,
        _auth=True,
    )
    assert fetched.overrides["matcher"]["embedding_attach_threshold"] == 0.88
    assert fetched.overrides["lifecycle"]["promotion_min_mentions_7d"] == 7
    assert fetched.active_version_id == applied_version
    assert len(fetched.history) >= 1

    second = await update_theme_policy(
        request=ThemePolicyUpdateRequest(
            pipeline="technical",
            matcher={"embedding_attach_threshold": 0.84},
            lifecycle={"promotion_min_mentions_7d": 5},
            note="second version",
            mode="apply",
        ),
        x_admin_actor="tester",
        db=db_session,
        _auth=True,
    )
    assert second.status == "applied"

    reverted = await revert_theme_policy(
        request=ThemePolicyRevertRequest(
            pipeline="technical",
            version_id=applied_version,
            note="restore first version",
        ),
        x_admin_actor="tester",
        db=db_session,
        _auth=True,
    )
    assert reverted.status == "applied"

    after_revert = await get_theme_policy_config(
        pipeline="technical",
        db=db_session,
        _auth=True,
    )
    assert after_revert.overrides["matcher"]["embedding_attach_threshold"] == 0.88
    assert after_revert.overrides["lifecycle"]["promotion_min_mentions_7d"] == 7


@pytest.mark.asyncio
async def test_promote_staged_sanitizes_unknown_override_keys(db_session):
    db_session.add(
        AppSetting(
            key="theme_policy_staged",
            value=(
                '{"technical":{"version_id":"stage-1","pipeline":"technical","updated_at":"2026-02-24T00:00:00",'
                '"updated_by":"tester","note":"staged","overrides":{"matcher":{"embedding_attach_threshold":0.87,'
                '"unexpected_key":123},"lifecycle":{"promotion_min_mentions_7d":6,"oops":"x"}}}}'
            ),
            category="theme",
        )
    )
    db_session.commit()

    result = await promote_staged_theme_policy(
        pipeline="technical",
        note="promote",
        x_admin_actor="tester",
        db=db_session,
        _auth=True,
    )
    assert result.status == "applied"

    fetched = await get_theme_policy_config(
        pipeline="technical",
        db=db_session,
        _auth=True,
    )
    assert fetched.overrides["matcher"]["embedding_attach_threshold"] == 0.87
    assert "unexpected_key" not in fetched.overrides["matcher"]
    assert fetched.overrides["lifecycle"]["promotion_min_mentions_7d"] == 6
    assert "oops" not in fetched.overrides["lifecycle"]
