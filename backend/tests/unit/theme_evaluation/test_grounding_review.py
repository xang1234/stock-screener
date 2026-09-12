"""Grounding packets remain bound to frozen extraction records and review output."""

import csv
import json
from datetime import datetime, timezone
from hashlib import sha256

import pytest
from app.services.theme_evaluation.extraction_capture import (
    load_extraction_run,
    save_extraction_run,
)
from app.services.theme_evaluation.extraction_records import (
    ExtractionRecord,
    make_input,
)
from app.services.theme_evaluation.extraction_review import render_extractions
from app.services.theme_grounding_context import GroundingContext, GroundingEvidence

STAMP = datetime(2026, 9, 11, 10, tzinfo=timezone.utc)


def _input():
    return make_input(
        source_id="post:1",
        source_kind="post",
        source_url="https://example.com/posts/1",
        title="NBIS capacity update",
        text="$NBIS announced a capacity update.",
        language="en",
        published_at=STAMP,
        available_at=STAMP,
        original_text_sha256=sha256(b"$NBIS announced a capacity update.").hexdigest(),
        result_ids=[],
        input_kind="original",
        normalization_policy=None,
        warnings=["source_has_no_author"],
    )


def _context(input_id):
    evidence_text = "Attached image says capacity is expanding."
    return GroundingContext(
        primary_input_id=input_id,
        context_available_at=STAMP,
        companies=[
            {
                "symbol": "NBIS",
                "name": "Nebius Group N.V.",
                "identity_source": "local_stock_universe:official:2026-09-10T00:00:00+00:00",
                "sector": "Technology",
                "industry": "Cloud Infrastructure",
                "business_description": "Cloud infrastructure provider.",
                "profile_source": {
                    "business_description": "yfinance",
                    "industry": "yfinance",
                    "sector": "yfinance",
                },
                "profile_as_of": {
                    "business_description": "2026-09-10T00:00:00+00:00",
                    "industry": "2026-09-10T00:00:00+00:00",
                    "sector": "2026-09-10T00:00:00+00:00",
                },
                "profile_status": "available",
            }
        ],
        evidence=[
            GroundingEvidence(
                input_id="d" * 64,
                source_id="post:1",
                source_url="https://example.com/posts/1/image/1",
                input_kind="image_transcription",
                relation="attached_image",
                text=evidence_text,
                original_text_sha256=sha256(evidence_text.encode()).hexdigest(),
                text_sha256=sha256(evidence_text.encode()).hexdigest(),
                available_at=STAMP,
                warnings=["ocr_uncertain"],
            )
        ],
        warnings=["company_context_limited"],
    ).model_dump(mode="json")


def _record(input_id, *, status="success", grounding_context=None):
    return ExtractionRecord(
        input_id=input_id,
        pipeline="technical",
        status=status,
        mentions=[] if status == "failed" else [
            {
                "theme": "Cloud capacity",
                "tickers": ["NBIS"],
                "sentiment": "bullish",
                "confidence": 0.8,
                "excerpt": "Capacity is expanding.",
            }
        ],
        error_code="provider_failed" if status == "failed" else None,
        generated_at=STAMP,
        requested_model="sanctioned-model",
        reference_sha256="a" * 64,
        code_revision="test",
        calls=[],
        **({"grounding_context": grounding_context} if grounding_context else {}),
    )


def _manifest():
    return {
        "bundle_id": "a" * 64,
        "preparation_id": "b" * 64,
        "assessment_id": "c" * 64,
        "exclusions": [],
    }


def test_grounded_records_change_run_hash_and_legacy_records_keep_their_shape(tmp_path):
    source = _input()
    legacy = _record(source.input_id)
    grounded = _record(source.input_id, grounding_context=_context(source.input_id))

    legacy_run = save_extraction_run(
        tmp_path, manifest=_manifest(), inputs=[source], records=[legacy]
    )
    grounded_run = save_extraction_run(
        tmp_path, manifest=_manifest(), inputs=[source], records=[grounded]
    )

    assert legacy_run.name != grounded_run.name
    assert "grounding_context" not in load_extraction_run(legacy_run)["records"][0].model_dump(mode="json")
    assert load_extraction_run(grounded_run)["records"][0].grounding_context == _context(source.input_id)


def test_render_extractions_exports_exact_grounding_provenance_and_warnings(tmp_path):
    source = _input()
    context = _context(source.input_id)
    run = save_extraction_run(
        tmp_path,
        manifest=_manifest(),
        inputs=[source],
        records=[_record(source.input_id, grounding_context=context)],
    )
    output = tmp_path / "review"

    render_extractions(run, output)

    with (output / "grounding.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["input_id"] == source.input_id
    assert row["source_url"] == "https://example.com/posts/1"
    assert row["context_available_at"] == "2026-09-11T10:00:00Z"
    assert json.loads(row["companies"]) == context["companies"]
    assert json.loads(row["evidence"]) == context["evidence"]
    assert json.loads(row["warnings"]) == ["company_context_limited"]


def test_failed_extraction_retains_its_grounding_context_in_saved_run_and_review(tmp_path):
    source = _input()
    context = _context(source.input_id)
    run = save_extraction_run(
        tmp_path,
        manifest=_manifest(),
        inputs=[source],
        records=[_record(source.input_id, status="failed", grounding_context=context)],
    )
    output = tmp_path / "review"

    assert load_extraction_run(run)["records"][0].grounding_context == context
    render_extractions(run, output)
    with (output / "grounding.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["pipeline"] == "technical"
    assert json.loads(row["warnings"]) == ["company_context_limited"]


def test_grounding_context_rejects_related_evidence_over_six_thousand_characters():
    def evidence(input_id, text):
        digest = sha256(text.encode()).hexdigest()
        return GroundingEvidence(
            input_id=input_id,
            source_id="post:1",
            source_url=f"https://example.com/{input_id[:4]}",
            input_kind="article",
            relation="linked_article",
            text=text,
            original_text_sha256=digest,
            text_sha256=digest,
            available_at=STAMP,
        )

    with pytest.raises(ValueError, match="grounding_context_too_long"):
        GroundingContext(
            primary_input_id="a" * 64,
            context_available_at=STAMP,
            evidence=[evidence("b" * 64, "a" * 3_001), evidence("c" * 64, "b" * 3_000)],
        )
