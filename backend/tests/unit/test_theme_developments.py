"""Events retain attributed revisions, without rewarding repeated coverage."""

from datetime import datetime, timezone

import pytest
from app.database import Base
from app.models.theme import ContentItem, ThemeCluster
from app.models.theme_intelligence import ThemeDevelopmentObservation
from app.services.theme_development_service import record_developments
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as session:
        yield session


def event(status="rumored", object="Project A order"):
    return {
        "theme_ids": [1],
        "actor": "Nebius",
        "action": "order",
        "object": object,
        "event_time": "2026-09",
        "status": status,
        "summary": "Nebius order",
        "quantities": [],
        "citations": [{"source_id": "primary", "quote": f"Nebius {object} 2026-09"}],
    }


def add(db, item=None, revision="a", facts=None):
    if item is None:
        item = ContentItem(source_type="news", content="Nebius order", published_at=NOW)
        db.add(item)
        db.flush()
    theme = db.query(ThemeCluster).first()
    if theme is None:
        theme = ThemeCluster(
            name="AI", display_name="AI", canonical_key="ai", pipeline="technical"
        )
        db.add(theme)
        db.flush()
    return item, record_developments(
        db,
        item=item,
        pipeline="technical",
        revision=revision * 64,
        theme_ids=[theme.id],
        sources={"primary": f"Nebius {(facts or event())['object']} 2026-09"},
        observations=[facts or event()],
        available_at=NOW,
    )


def test_repeat_confirmation_and_distinct_order(db):
    _, first = add(db)
    _, repeat = add(db)
    _, confirmation = add(db, facts=event("confirmed"))
    _, separate = add(db, facts=event("confirmed", "Project B order"))
    assert first[0].event_id == repeat[0].event_id == confirmation[0].event_id
    assert repeat[0].classification == "repeated_coverage"
    assert confirmation[0].classification == "material_update"
    assert separate[0].event_id != first[0].event_id


def test_idempotent_revision_and_correction_preserve_history(db):
    item, first = add(db)
    _, again = add(db, item=item)
    assert first[0].id == again[0].id
    _, correction = add(db, item=item, revision="b", facts=event("denied"))
    assert first[0].superseded
    assert correction[0].classification == "contradiction"
    assert db.query(ThemeDevelopmentObservation).count() == 2


def test_invalid_citation_cannot_supersede_existing_observation(db):
    item, first = add(db)
    bad = event("confirmed")
    bad["citations"][0]["quote"] = "invented"
    with pytest.raises(ValueError, match="citation"):
        add(db, item=item, revision="b", facts=bad)
    assert not first[0].superseded


def test_missing_event_anchor_does_not_join_posts(db):
    facts = event()
    facts["event_time"] = None
    _, first = add(db, facts=facts)
    _, second = add(db, facts=facts)
    assert first[0].event_id != second[0].event_id


def test_weaker_followup_does_not_reopen_confirmed_event(db):
    add(db, facts=event("confirmed"))
    _, rows = add(db, facts=event("rumored"))
    assert rows[0].classification == "additional_detail"


def test_reworded_quantities_do_not_create_novelty():
    from types import SimpleNamespace

    from app.services.theme_development_service import DevelopmentFacts, classify

    prior = event("confirmed")
    prior["quantities"] = ["USD 1000000"]
    current = event("confirmed")
    current["quantities"] = ["USD 1 million"]
    assert (
        classify(
            DevelopmentFacts.model_validate(current), [SimpleNamespace(facts=prior)]
        )
        == "repeated_coverage"
    )


def test_unsupported_identity_is_rejected_before_superseding(db):
    item, first = add(db)
    bad = event("confirmed")
    bad["reference"] = "invented-order-id"
    with pytest.raises(ValueError, match="anchor"):
        add(db, item=item, revision="b", facts=bad)
    assert not first[0].superseded
