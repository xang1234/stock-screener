import json

import pytest
from app.services.theme_claim_review import ClaimReviewError, review_claims
from app.services.theme_grounding_context import GroundingContext


def candidate(
    theme="CPO", development="Kyber NVL144 co-packaged optics program delayed."
):
    return {
        "theme": theme,
        "development": development,
        "tickers": [],
        "sentiment": "bearish",
        "confidence": 0.8,
        "excerpt": "Kyber NVL144 delayed to 2028.",
    }


def verdict(status="supported", refs=None):
    return {
        "status": status,
        "reason": "Evidence supports this claim."
        if status != "unsupported"
        else "No supplied mapping from product to this industry.",
        "evidence": refs or [],
    }


def decision(theme, development):
    return {"index": 0, "theme": theme, "development": development}


def run_review(
    mentions, decisions, *, text="Kyber NVL144 delayed to 2028.", context=None
):
    return review_claims(
        mentions,
        primary_text=text,
        grounding_context=context or GroundingContext(),
        generate=lambda prompt, **kwargs: json.dumps(decisions),
    )


def test_rejects_unsupported_theme_and_keeps_candidate_for_review():
    raw = candidate()
    accepted, audit = run_review(
        [raw], [decision(verdict("unsupported"), verdict("unsupported"))]
    )
    assert accepted == []
    assert audit["candidates"] == [raw]
    assert audit["decisions"][0]["action"] == "held_theme"


def test_open_new_theme_passes_and_conditional_language_is_preserved():
    text = "Suppliers may benefit from glass core substrates; orders are not confirmed."
    raw = candidate("Glass Core Substrates", text)
    refs = [{"source_id": "primary", "quote": text}]
    accepted, audit = run_review(
        [raw], [decision(verdict(refs=refs), verdict(refs=refs))], text=text
    )
    assert {k: v for k, v in accepted[0].items() if k != "claim_support"} == raw
    assert audit["status"] == "reviewed"


def test_profile_inference_is_labelled_and_unsupported_development_is_withheld():
    context = GroundingContext(
        companies=[
            {
                "symbol": "NBIS",
                "name": "Nebius",
                "identity_source": "frozen",
                "business_description": "Builds AI infrastructure.",
                "profile_status": "available",
            }
        ]
    )
    refs = [
        {
            "source_id": "company:NBIS:business_description",
            "quote": "Builds AI infrastructure.",
        },
        {"source_id": "primary", "quote": "$NBIS builds in Japan now."},
    ]
    raw = candidate("AI Infrastructure", "Nebius doubled Japanese capacity.")
    accepted, audit = run_review(
        [raw],
        [decision(verdict("inferred", refs), verdict("unsupported"))],
        text="$NBIS builds in Japan now.",
        context=context,
    )
    assert accepted[0]["theme"] == "AI Infrastructure"
    assert accepted[0]["development"] is None
    assert accepted[0]["claim_support"]["theme"] == "inferred"
    assert audit["decisions"][0]["action"] == "held_development"


@pytest.mark.parametrize(
    "refs",
    [
        [],
        [{"source_id": "primary", "quote": "CPO is delayed"}],
        [{"source_id": "invented", "quote": "Kyber NVL144 delayed to 2028."}],
    ],
)
def test_missing_or_fabricated_citations_fail_closed(refs):
    with pytest.raises(ClaimReviewError):
        run_review(
            [candidate()], [decision(verdict(refs=refs), verdict("unsupported"))]
        )


def test_empty_extraction_never_calls_reviewer():
    accepted, audit = review_claims(
        [],
        primary_text="Nothing here.",
        grounding_context=GroundingContext(),
        generate=lambda *args, **kwargs: pytest.fail("No model call needed"),
    )
    assert accepted == [] and audit["status"] == "not_needed"


def test_missing_duplicate_decisions_and_provider_failure_fail_closed():
    with pytest.raises(ClaimReviewError):
        run_review([candidate()], [])
    d = decision(verdict("unsupported"), verdict("unsupported"))
    with pytest.raises(ClaimReviewError):
        run_review([candidate()], [d, d])
    with pytest.raises(ClaimReviewError, match="unavailable"):
        review_claims(
            [candidate()],
            primary_text="source",
            grounding_context=GroundingContext(),
            generate=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("private error")
            ),
        )


def test_extractor_reviews_before_returning_and_keeps_audit(monkeypatch):
    from types import SimpleNamespace

    from app.services.theme_extraction_service import ThemeExtractionService

    service = ThemeExtractionService.__new__(ThemeExtractionService)
    service.provider = "litellm"
    service.db = None
    service._valid_tickers = set()
    service._rate_limit = lambda: None
    calls = []

    def generate(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return (
            json.dumps([decision(verdict("unsupported"), verdict("unsupported"))])
            if kwargs
            else json.dumps([candidate()])
        )

    monkeypatch.setattr(service, "_try_generate_litellm", generate)
    item = SimpleNamespace(
        content="Kyber NVL144 delayed to 2028.",
        source_name="source",
        source_type="post",
        source_language="en",
        title="Post by @author",
        published_at=None,
    )
    assert service.extract_from_content(item) == []
    assert len(calls) == 2
    assert "background knowledge" in calls[1][1]["system_prompt"]
    assert service.last_claim_review["decisions"][0]["action"] == "held_theme"


def test_held_candidate_persists_without_creating_a_theme(
    universe_session, monkeypatch
):
    from app.models.theme import ContentItem, ContentItemPipelineState, ThemeMention
    from app.services.theme_extraction_service import ThemeExtractionService

    service = ThemeExtractionService.__new__(ThemeExtractionService)
    service.db = universe_session
    service.pipeline = "fundamental"
    service.provider = "litellm"
    service._rate_limit = lambda: None
    service._valid_tickers = set()
    monkeypatch.setattr(
        service,
        "_try_generate_litellm",
        lambda prompt, **kw: (
            json.dumps([decision(verdict("unsupported"), verdict("unsupported"))])
            if kw
            else json.dumps([candidate()])
        ),
    )
    monkeypatch.setattr(
        service,
        "_resolve_cluster_match",
        lambda *a, **kw: pytest.fail("Held theme must not cluster"),
    )
    item = ContentItem(
        source_type="post", source_name="test", content="Kyber NVL144 delayed to 2028."
    )
    universe_session.add(item)
    universe_session.flush()
    assert service._extract_and_store_mentions(item) == 0
    universe_session.commit()
    universe_session.expire_all()
    assert universe_session.query(ThemeMention).count() == 0
    assert service.last_claim_review["candidates"][0]["theme"] == "CPO"
    # Pipeline-specific audit survives the rollback used for failures as well.
    audit = {**service.last_claim_review, "status": "unavailable"}
    service.last_claim_review = audit
    service._record_processing_failure(
        item.id, ClaimReviewError("claim_review_unavailable")
    )
    stored = universe_session.query(ContentItemPipelineState).one()
    assert stored.claim_review == audit


def test_profile_alone_cannot_prove_an_event_or_be_called_direct_support():
    context = GroundingContext(
        companies=[
            {
                "symbol": "NBIS",
                "identity_source": "frozen",
                "business_description": "Builds AI infrastructure.",
            }
        ]
    )
    refs = [
        {
            "source_id": "company:NBIS:business_description",
            "quote": "Builds AI infrastructure.",
        }
    ]
    with pytest.raises(ClaimReviewError, match="issuer_anchor"):
        run_review(
            [candidate("AI Infrastructure")],
            [decision(verdict("inferred", refs), verdict("unsupported"))],
            context=context,
        )
    refs.append({"source_id": "primary", "quote": "Kyber NVL144 delayed to 2028."})
    with pytest.raises(ClaimReviewError, match="issuer_anchor"):
        run_review(
            [candidate("AI Infrastructure")],
            [decision(verdict("supported", refs), verdict("unsupported"))],
            context=context,
        )
    accepted, audit = run_review(
        [candidate("AI Infrastructure")],
        [decision(verdict("supported", refs), verdict("unsupported"))],
        context=context,
        text="$NBIS: Kyber NVL144 delayed to 2028.",
    )
    assert accepted[0]["claim_support"]["theme"] == "inferred"
    assert "profile_support_labelled_inference" in audit["decisions"][0]["adjustments"]


def test_inferred_development_is_visibly_labelled():
    refs = [{"source_id": "primary", "quote": "Kyber NVL144 delayed to 2028."}]
    accepted, _ = run_review(
        [candidate("AI Infrastructure", "Deployment may shift later.")],
        [decision(verdict("inferred", refs), verdict("inferred", refs))],
    )
    assert accepted[0]["development"] == "Inference: Deployment may shift later."


def test_profile_anchor_and_misattributed_exact_quote_are_audited():
    context = GroundingContext(
        companies=[
            {
                "symbol": "NBIS",
                "identity_source": "frozen",
                "business_description": "Builds AI infrastructure.",
            }
        ]
    )
    refs = [{"source_id": "primary", "quote": "Builds AI infrastructure."}]
    event = [{"source_id": "primary", "quote": "$NBIS builds in Japan now."}]
    accepted, audit = run_review(
        [candidate("AI Infrastructure", "Building in Japan.")],
        [decision(verdict("supported", refs), verdict("supported", event))],
        text="$NBIS builds in Japan now.",
        context=context,
    )
    assert accepted[0]["development"].startswith("Inference:")
    assert set(audit["decisions"][0]["adjustments"]) == {
        "citation_source_corrected_by_exact_match",
        "profile_support_labelled_inference",
        "profile_issuer_anchored_to_explicit_cashtag",
        "profile_dependent_development_labelled_inference",
    }
    assert audit["reviewer_decisions"][0]["theme"]["status"] == "supported"


def test_transactional_reviewer_quota_failure_retains_audit_and_classification(
    universe_session, monkeypatch
):
    from app.models.theme import ContentItem, ContentItemPipelineState, ThemeMention
    from app.services.theme_extraction_service import (
        ProviderQuotaServiceError,
        ThemeExtractionService,
    )
    from sqlalchemy import true

    service = ThemeExtractionService.__new__(ThemeExtractionService)
    service.db = universe_session
    service.pipeline = "fundamental"
    service.provider = "litellm"
    service._rate_limit = lambda: None
    service._valid_tickers = set()
    monkeypatch.setattr(service, "_get_pipeline_source_ids", list)
    monkeypatch.setattr(
        "app.services.theme_extraction_service.legacy_eligibility_exists",
        lambda *a, **kw: true(),
    )

    def generate(prompt, **kwargs):
        if kwargs:
            raise ProviderQuotaServiceError("quota exhausted")
        return json.dumps([candidate()])

    monkeypatch.setattr(service, "_try_generate_litellm", generate)
    item = ContentItem(
        source_type="post", source_name="test", content="Kyber NVL144 delayed to 2028."
    )
    universe_session.add(item)
    universe_session.commit()
    item_id = item.id
    with pytest.raises(ProviderQuotaServiceError):
        service._process_item_transactional(item_id)
    universe_session.expire_all()
    state = universe_session.query(ContentItemPipelineState).one()
    assert state.status == "failed_terminal"
    assert state.claim_review["status"] == "unavailable"
    assert state.claim_review["candidates"][0]["theme"] == "CPO"
    assert universe_session.query(ThemeMention).count() == 0


def test_claim_review_migration_round_trip():
    import importlib.util
    from pathlib import Path

    import sqlalchemy as sa
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    path = (
        Path(__file__).resolve().parents[2]
        / "alembic/versions/20260911_0040_add_theme_claim_review.py"
    )
    spec = importlib.util.spec_from_file_location("claim_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE content_item_pipeline_state (id INTEGER PRIMARY KEY)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE theme_mentions (id INTEGER PRIMARY KEY)"
        )
        module.op = Operations(MigrationContext.configure(connection))
        module.upgrade()
        assert "claim_review" in {
            c["name"]
            for c in sa.inspect(connection).get_columns("content_item_pipeline_state")
        }
        assert "claim_support" in {
            c["name"] for c in sa.inspect(connection).get_columns("theme_mentions")
        }
        module.downgrade()
        assert "claim_review" not in {
            c["name"]
            for c in sa.inspect(connection).get_columns("content_item_pipeline_state")
        }


def test_company_name_and_cashtag_cannot_prove_exposure():
    context = GroundingContext(
        companies=[{"symbol": "NBIS", "name": "Nebius", "identity_source": "frozen"}]
    )
    refs = [{"source_id": "company:NBIS:name", "quote": "Nebius"}]
    with pytest.raises(ClaimReviewError, match="profile_exposure"):
        run_review(
            [candidate("Nuclear Energy")],
            [decision(verdict("supported", refs), verdict("unsupported"))],
            text="$NBIS builds in Japan now.",
            context=context,
        )


def test_ocr_line_wrap_is_normalized_but_words_and_numbers_cannot_change():
    text = "Citi 2026 Global TMT\nConference"
    raw = candidate("AI Infrastructure", None)
    refs = [{"source_id": "primary", "quote": "Citi 2026 Global TMT Conference"}]
    accepted, audit = run_review(
        [raw], [decision(verdict("inferred", refs), verdict("absent"))], text=text
    )
    assert len(accepted) == 1
    assert audit["decisions"][0]["theme"]["evidence"][0]["quote"] == text
    assert audit["decisions"][0]["adjustments"] == [
        "citation_whitespace_normalized_to_source"
    ]
    refs[0]["quote"] = "Citi 2027 Global TMT Conference"
    with pytest.raises(ClaimReviewError, match="citation_invalid"):
        run_review(
            [raw], [decision(verdict("inferred", refs), verdict("absent"))], text=text
        )


def test_quote_typography_restores_original_span_without_changing_claim():
    text = "OpenAI’s “HBM” orders may rise by 20%."
    refs = [
        {"source_id": "primary", "quote": 'OpenAI\'s "HBM" orders may rise by 20%.'}
    ]
    accepted, audit = run_review(
        [candidate("HBM", None)],
        [decision(verdict(refs=refs), verdict("absent"))],
        text=text,
    )
    assert accepted[0]["theme"] == "HBM"
    assert audit["decisions"][0]["theme"]["evidence"][0]["quote"] == text
    assert (
        "citation_typography_normalized_to_source"
        in audit["decisions"][0]["adjustments"]
    )


@pytest.mark.parametrize(
    "quote",
    [
        "Orders will rise by 20%.",
        "Orders may rise by 25%.",
        "Orders may not rise by 20%.",
        "Orders may rise by $20%.",
    ],
)
def test_typography_matching_never_changes_words_numbers_negation_or_units(quote):
    with pytest.raises(ClaimReviewError, match="citation_invalid"):
        run_review(
            [candidate("Memory", None)],
            [
                decision(
                    verdict(refs=[{"source_id": "primary", "quote": quote}]),
                    verdict("absent"),
                )
            ],
            text="Orders may rise by 20%.",
        )


def test_explicit_theme_not_invalidated_by_ancillary_identity_citation():
    context = GroundingContext(
        companies=[
            {
                "symbol": "META",
                "name": "Meta Platforms Inc",
                "identity_source": "frozen",
            }
        ]
    )
    refs = [
        {"source_id": "primary", "quote": "$META is a Mag 7 laggard."},
        {"source_id": "company:META:name", "quote": "Meta Platforms Inc"},
    ]
    accepted, audit = run_review(
        [candidate("Mag 7", None)],
        [decision(verdict(refs=refs), verdict("absent"))],
        text="$META is a Mag 7 laggard.",
        context=context,
    )
    assert accepted[0]["claim_support"]["theme"] == "supported"
    assert audit["reviewer_decisions"][0]["theme"]["evidence"] == refs


def test_invalid_candidate_does_not_discard_valid_candidate():
    good = decision(
        verdict(refs=[{"source_id": "primary", "quote": "Memory demand rose."}]),
        verdict("absent"),
    )
    bad = {
        "index": 1,
        "theme": verdict(
            refs=[{"source_id": "primary", "quote": "Imaginary optics order."}]
        ),
        "development": verdict("absent"),
    }
    accepted, audit = run_review(
        [candidate("Memory", None), candidate("Optics", None)],
        [good, bad],
        text="Memory demand rose.",
    )
    assert [m["theme"] for m in accepted] == ["Memory"]
    assert audit["status"] == "partial"
    assert audit["decisions"][1]["action"] == "review_unavailable"
    assert audit["decisions"][1]["error_code"] == "claim_review_citation_invalid"


def test_schema_error_is_local_to_its_candidate():
    good = decision(
        verdict(refs=[{"source_id": "primary", "quote": "Memory demand rose."}]),
        verdict("absent"),
    )
    bad = {
        "index": 1,
        "theme": {"unexpected": "malformed"},
        "development": verdict("absent"),
    }
    accepted, audit = run_review(
        [candidate("Memory", None), candidate("Optics", None)],
        [good, bad],
        text="Memory demand rose.",
    )
    assert len(accepted) == 1
    assert audit["decisions"][1]["action"] == "review_unavailable"
    assert audit["reviewer_decisions"][1] == bad


def test_review_batches_bound_response_size_and_keep_global_indices():
    sizes = []

    def generate(prompt, **kwargs):
        entries = json.loads(prompt)["candidates"]
        sizes.append(len(entries))
        return json.dumps(
            [
                {
                    "index": c["index"],
                    "theme": verdict(
                        refs=[{"source_id": "primary", "quote": "Memory demand rose."}]
                    ),
                    "development": verdict("absent"),
                }
                for c in entries
            ]
        )

    accepted, audit = review_claims(
        [candidate("Memory", None) for _ in range(7)],
        primary_text="Memory demand rose.",
        grounding_context=GroundingContext(),
        generate=generate,
    )
    assert max(sizes) <= 3
    assert len(accepted) == 7
    assert [d["index"] for d in audit["decisions"]] == list(range(7))


def test_timed_out_batch_is_held_while_later_batch_can_succeed():
    def generate(prompt, **kwargs):
        entries = json.loads(prompt)["candidates"]
        if entries[0]["index"] == 0:
            raise TimeoutError("private provider detail")
        return json.dumps(
            [
                {
                    "index": c["index"],
                    "theme": verdict(
                        refs=[{"source_id": "primary", "quote": "Memory demand rose."}]
                    ),
                    "development": verdict("absent"),
                }
                for c in entries
            ]
        )

    accepted, audit = review_claims(
        [candidate("Memory", None) for _ in range(4)],
        primary_text="Memory demand rose.",
        grounding_context=GroundingContext(),
        generate=generate,
    )
    assert len(accepted) == 1
    assert audit["status"] == "partial"
    assert (
        len([d for d in audit["decisions"] if d["action"] == "review_unavailable"]) == 3
    )
    assert "private provider detail" not in json.dumps(audit)


def test_extractor_preserves_invalid_review_verdict_for_inspection():
    from app.services.theme_extraction_service import ThemeExtractionService

    service = ThemeExtractionService.__new__(ThemeExtractionService)
    service._rate_limit = lambda: None
    raw = decision(
        verdict(refs=[{"source_id": "primary", "quote": "Invented statement."}]),
        verdict("absent"),
    )
    service._try_generate_litellm = lambda *a, **kw: json.dumps([raw])
    with pytest.raises(ClaimReviewError):
        service._review_claims(
            [candidate("Memory", None)], "Memory demand rose.", GroundingContext()
        )
    assert service.last_claim_review["reviewer_decisions"] == [raw]
    assert service.last_claim_review["decisions"][0]["action"] == "review_unavailable"


def test_quota_failure_stops_later_batches_and_preserves_cause():
    from app.services.theme_extraction_service import ProviderQuotaServiceError

    count = 0

    def generate(*args, **kwargs):
        nonlocal count
        count += 1
        raise ProviderQuotaServiceError("private quota detail")

    with pytest.raises(ClaimReviewError) as result:
        review_claims(
            [candidate("Memory", None) for _ in range(7)],
            primary_text="Memory demand rose.",
            grounding_context=GroundingContext(),
            generate=generate,
        )
    assert count == 1
    assert isinstance(result.value.__cause__, ProviderQuotaServiceError)
    assert len(result.value.audit["decisions"]) == 7
    assert "private quota detail" not in json.dumps(result.value.audit)


def test_normalized_source_span_cannot_exceed_citation_size_bound():
    text = "A" + "\n" * 2500 + "B"
    refs = [{"source_id": "primary", "quote": "A B"}]
    with pytest.raises(ClaimReviewError, match="citation_invalid"):
        run_review(
            [candidate("Memory", None)],
            [decision(verdict(refs=refs), verdict("absent"))],
            text=text,
        )
