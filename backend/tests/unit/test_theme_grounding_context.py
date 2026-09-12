import hashlib

import pytest
from app.services.theme_grounding_context import GroundingContext, GroundingEvidence


def test_context_rejects_tampered_evidence_and_unbounded_profile():
    evidence = {
        "input_id": "a" * 64,
        "source_id": "post:1",
        "source_url": "https://x.com/a/status/1",
        "input_kind": "image_transcription",
        "relation": "attached_image",
        "text": "Nebius",
        "original_text_sha256": hashlib.sha256(b"Nebius").hexdigest(),
        "text_sha256": hashlib.sha256(b"wrong").hexdigest(),
        "available_at": "2026-09-11T00:00:00Z",
        "warnings": [],
        "truncated": False,
    }
    with pytest.raises(ValueError, match="text_hash"):
        GroundingEvidence.model_validate(evidence)
    with pytest.raises(ValueError):
        GroundingContext(
            companies=[{"symbol": "NBIS", "business_description": "x" * 2001}]
        )


def test_context_bounds_total_related_text_and_rejects_duplicate_evidence():
    values = {
        "input_id": "a" * 64,
        "source_id": "post:1",
        "source_url": "https://x.com/a/status/1",
        "input_kind": "image_transcription",
        "relation": "attached_image",
        "text": "Nebius",
        "original_text_sha256": hashlib.sha256(b"Nebius").hexdigest(),
        "text_sha256": hashlib.sha256(b"Nebius").hexdigest(),
        "available_at": "2026-09-11T00:00:00Z",
        "warnings": [],
        "truncated": False,
    }
    evidence = GroundingEvidence(**values)
    with pytest.raises(ValueError, match="duplicate"):
        GroundingContext(evidence=[evidence, evidence])


def test_supplied_context_reaches_provider_without_local_lookup(monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from app.services.theme_extraction_service import ThemeExtractionService

    service = ThemeExtractionService.__new__(ThemeExtractionService)
    service.provider = "litellm"
    service._rate_limit = lambda: None
    sent = []

    def respond(prompt):
        sent.append(prompt)
        return "[]"

    service._try_generate_litellm = respond
    context = GroundingContext(
        companies=[
            {
                "symbol": "NBIS",
                "name": "Nebius Group N.V.",
                "identity_source": "frozen_local_universe",
                "profile_status": "missing",
            }
        ]
    )
    item = SimpleNamespace(
        content="$NBIS building for META. " + "x" * 11000,
        source_name="source",
        source_type="post",
        source_language="en",
        title="Post by @author",
        published_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
    )
    assert service.extract_from_content(item, grounding_context=context) == []
    assert "Nebius Group N.V." in sent[0]
    assert "GROUNDING CONTEXT" in sent[0] and "[truncated]" in sent[0]
    assert "unknown exposure" in sent[0]


def test_legacy_extraction_record_keeps_original_serialization():
    from app.services.theme_evaluation.extraction_records import ExtractionRecord

    value = {
        "input_id": "a" * 64,
        "pipeline": "technical",
        "status": "success",
        "mentions": [],
        "error_code": None,
        "generated_at": "2026-09-11T00:00:00Z",
        "requested_model": "test",
        "reference_sha256": "b" * 64,
        "code_revision": "test",
        "calls": [],
    }
    old = ExtractionRecord(**value)
    assert "grounding_context" not in old.model_dump(mode="json")
    context = GroundingContext(
        primary_input_id="a" * 64, warnings=["business_context_missing"]
    )
    new = ExtractionRecord(**value, grounding_context=context.model_dump(mode="json"))
    assert (
        new.model_dump(mode="json")["grounding_context"]["primary_input_id"] == "a" * 64
    )
    with pytest.raises(ValueError, match="grounding_input"):
        ExtractionRecord(
            **value,
            grounding_context=GroundingContext(primary_input_id="c" * 64).model_dump(
                mode="json"
            ),
        )


def test_company_context_bounds_excessive_warnings_with_an_omission_marker(
    universe_session,
):
    from app.services.theme_company_context import build_company_context

    context = build_company_context(
        universe_session,
        " ".join(f"$UNKNOWN{i}" for i in range(101)),
    )

    assert len(context["warnings"]) == 100
    assert context["warnings"][-1] == "warnings_omitted:2"
