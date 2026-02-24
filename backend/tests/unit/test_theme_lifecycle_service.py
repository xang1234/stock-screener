"""Tests for lifecycle transition invariants and audit persistence."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest

from app.database import Base
from app.models.theme import ThemeCluster, ThemeLifecycleTransition
from app.services.theme_lifecycle_service import apply_lifecycle_transition


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _theme() -> ThemeCluster:
    return ThemeCluster(
        name="AI Infrastructure",
        canonical_key="ai_infrastructure",
        display_name="AI Infrastructure",
        pipeline="technical",
        lifecycle_state="candidate",
        is_active=True,
    )


def test_apply_lifecycle_transition_persists_state_and_audit(db_session):
    theme = _theme()
    db_session.add(theme)
    db_session.commit()

    apply_lifecycle_transition(
        db=db_session,
        theme=theme,
        to_state="active",
        actor="job-runner",
        job_name="promote_candidates",
        rule_version="lifecycle-v1",
        reason="minimum_mentions_met",
        metadata={"mentions_7d": 6},
    )
    db_session.commit()

    db_session.refresh(theme)
    assert theme.lifecycle_state == "active"
    assert theme.activated_at is not None
    assert theme.lifecycle_state_updated_at is not None

    audit = db_session.query(ThemeLifecycleTransition).filter(
        ThemeLifecycleTransition.theme_cluster_id == theme.id
    ).one()
    assert audit.from_state == "candidate"
    assert audit.to_state == "active"
    assert audit.actor == "job-runner"
    assert audit.rule_version == "lifecycle-v1"


def test_apply_lifecycle_transition_rejects_invalid_direct_jump(db_session):
    theme = _theme()
    theme.lifecycle_state = "active"
    db_session.add(theme)
    db_session.commit()

    with pytest.raises(ValueError, match="Invalid lifecycle transition"):
        apply_lifecycle_transition(
            db=db_session,
            theme=theme,
            to_state="candidate",
        )


def test_apply_lifecycle_transition_rejects_noop_transition(db_session):
    theme = _theme()
    db_session.add(theme)
    db_session.commit()

    with pytest.raises(ValueError, match="must change state"):
        apply_lifecycle_transition(
            db=db_session,
            theme=theme,
            to_state="candidate",
        )


def test_apply_lifecycle_transition_rejects_retired_outgoing_transition(db_session):
    theme = _theme()
    theme.lifecycle_state = "retired"
    theme.is_active = False
    db_session.add(theme)
    db_session.commit()

    with pytest.raises(ValueError, match="Invalid lifecycle transition"):
        apply_lifecycle_transition(
            db=db_session,
            theme=theme,
            to_state="active",
        )
