"""Events retain attributed revisions, without rewarding repeated coverage."""

from datetime import datetime, timezone

import pytest
from app.database import Base
from app.models.theme import ContentItem, ThemeCluster
from app.models.theme_intelligence import (
    ThemeDevelopmentEvent,
    ThemeDevelopmentObservation,
    ThemeDevelopmentTheme,
)
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


def test_repeated_confirmation_after_weaker_coverage_is_not_an_update(db):
    rows = [
        add(db, facts=event(status))[1][0]
        for status in ["rumored", "confirmed", "rumored", "confirmed"]
    ]
    assert [row.classification for row in rows].count("material_update") == 1
    assert rows[-1].classification == "repeated_coverage"


def test_conflicting_batch_is_rejected_before_any_write(db):
    item, _ = add(db)
    before = db.query(ThemeDevelopmentObservation).count()
    with pytest.raises(ValueError, match="conflicting_development_batch"):
        record_developments(
            db,
            item=item,
            pipeline="technical",
            revision="z" * 64,
            theme_ids=[1],
            sources={"primary": "Nebius Project A order 2026-09"},
            observations=[event("rumored"), event("confirmed")],
            available_at=NOW,
        )
    assert db.query(ThemeDevelopmentObservation).count() == before


def test_facts_do_not_duplicate_theme_membership(db):
    _, rows = add(db)
    assert "theme_ids" not in rows[0].facts


def test_correction_reduces_current_event_history(db):
    add(db, facts=event("rumored"))
    item, original = add(db, facts=event("confirmed"))
    _, replacement = add(db, item=item, revision="b", facts=event("confirmed"))
    assert original[0].superseded
    assert replacement[0].classification == "material_update"


def test_quantity_history_is_specific_to_claim_status():
    from types import SimpleNamespace

    from app.services.theme_development_facts import DevelopmentFacts
    from app.services.theme_event_state import classify

    prior = [
        dict(event("rumored"), quantities=["USD 2 million"]),
        dict(event("confirmed"), quantities=["USD 1 million"]),
    ]
    current = DevelopmentFacts.model_validate(
        dict(event("confirmed"), quantities=["USD 2 million"])
    )
    assert (
        classify(current, [SimpleNamespace(facts=facts) for facts in prior])
        == "material_update"
    )


def test_compatible_batch_combines_memberships_and_citations(db):
    item, _ = add(db)
    other = ThemeCluster(
        name="Optics",
        display_name="Optics",
        canonical_key="optics",
        pipeline="technical",
    )
    db.add(other)
    db.flush()
    second = dict(
        event(),
        theme_ids=[other.id],
        citations=[{"source_id": "article", "quote": "Nebius Project A order 2026-09"}],
    )
    rows = record_developments(
        db,
        item=item,
        pipeline="technical",
        revision="b" * 64,
        theme_ids=[1, other.id],
        sources={
            "primary": "Nebius Project A order 2026-09",
            "article": "Nebius Project A order 2026-09",
        },
        observations=[event(), second],
        available_at=NOW,
    )
    assert len(rows) == 1
    assert rows[0].theme_ids == [1, other.id]
    assert {citation["source_id"] for citation in rows[0].citations} == {
        "primary",
        "article",
    }
    assert "theme_ids" not in rows[0].facts


def test_duplicate_theme_ids_create_one_membership_link(db):
    item, _ = add(db)
    duplicate = dict(event(), theme_ids=[1, 1])

    rows = record_developments(
        db,
        item=item,
        pipeline="technical",
        revision="d" * 64,
        theme_ids=[1],
        sources={"primary": "Nebius Project A order 2026-09"},
        observations=[duplicate],
        available_at=NOW,
    )

    assert len(rows) == 1
    assert rows[0].theme_ids == [1]
    assert (
        db.query(ThemeDevelopmentTheme)
        .filter(ThemeDevelopmentTheme.observation_id == rows[0].id)
        .count()
        == 1
    )


def test_combined_batch_revalidates_citation_limit(db):
    item, existing = add(db)
    observations = []
    sources = {}
    for index in range(9):
        source_id = f"source-{index}"
        quote = "Nebius Project A order 2026-09"
        sources[source_id] = quote
        observations.append(
            dict(event(), citations=[{"source_id": source_id, "quote": quote}])
        )

    before = (
        db.query(ThemeDevelopmentEvent).count(),
        db.query(ThemeDevelopmentObservation).count(),
        db.query(ThemeDevelopmentTheme).count(),
        existing[0].facts.copy(),
        list(existing[0].citations),
        existing[0].classification,
        existing[0].superseded,
    )
    with pytest.raises(ValueError):
        record_developments(
            db,
            item=item,
            pipeline="technical",
            revision="c" * 64,
            theme_ids=[1],
            sources=sources,
            observations=observations,
            available_at=NOW,
        )
    db.expire_all()
    unchanged = db.get(ThemeDevelopmentObservation, existing[0].id)
    assert (
        db.query(ThemeDevelopmentEvent).count(),
        db.query(ThemeDevelopmentObservation).count(),
        db.query(ThemeDevelopmentTheme).count(),
        unchanged.facts,
        unchanged.citations,
        unchanged.classification,
        unchanged.superseded,
    ) == before


def test_returning_to_prior_evidence_reactivates_its_observation(db):
    item, first = add(db)
    _, correction = add(db, item=item, revision="b", facts=event("confirmed"))
    _, restored = add(db, item=item)
    assert restored[0].id == first[0].id
    assert not restored[0].superseded
    assert correction[0].superseded
