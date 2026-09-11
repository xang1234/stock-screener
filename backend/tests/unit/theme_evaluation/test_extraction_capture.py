"""Frozen, standalone extraction capture artifacts."""

import csv
from datetime import datetime, timezone
from hashlib import sha256
from types import SimpleNamespace

import pytest
from app.services.theme_evaluation.extraction_capture import (
    RecordingLLM,
    load_extraction_run,
    save_extraction_run,
    verify_extraction_run,
)
from app.services.theme_evaluation.extraction_records import (
    ExtractionInput,
    ExtractionRecord,
    make_input,
)
from pydantic import BaseModel

STAMP = datetime(2026, 9, 8, 10, tzinfo=timezone.utc)
HASH = "a" * 64


def input_value(**overrides):
    value = {
        "source_id": "post:1",
        "source_kind": "post",
        "source_url": "https://example.com/post/1",
        "title": "Supplier orders",
        "text": "Orders increased.",
        "language": "en",
        "published_at": STAMP,
        "available_at": STAMP,
        "original_text_sha256": sha256(b"Orders increased.").hexdigest(),
        "result_ids": [],
        "input_kind": "original",
        "normalization_policy": None,
        "warnings": [],
    }
    value.update(overrides)
    return value


def record_value(input_id, **overrides):
    value = {
        "input_id": input_id,
        "pipeline": "technical",
        "status": "success",
        "mentions": [],
        "error_code": None,
        "generated_at": STAMP,
        "requested_model": "sanctioned-model",
        "reference_sha256": HASH,
        "code_revision": "abc123",
        "calls": [],
    }
    value.update(overrides)
    return value


def run_manifest(**overrides):
    value = {"bundle_id": HASH, "preparation_id": "b" * 64, "assessment_id": "c" * 64}
    value.update(overrides)
    return value


def test_make_input_calculates_a_stable_content_id():
    first = make_input(**input_value())
    second = make_input(**input_value())

    assert first.input_id == second.input_id
    assert first.input_id != "b" * 64


def test_input_rejects_a_hash_that_does_not_bind_its_content():
    with pytest.raises(ValueError, match="input_id_mismatch"):
        ExtractionInput.model_validate({**input_value(), "input_id": HASH})


def test_successful_empty_result_is_distinct_from_failure():
    source = make_input(**input_value())
    assert ExtractionRecord.model_validate(record_value(source.input_id)).mentions == []

    with pytest.raises(ValueError, match="failed_extraction"):
        ExtractionRecord.model_validate(
            record_value(
                source.input_id, status="failed", mentions=[{"theme": "chips"}]
            )
        )
    with pytest.raises(ValueError, match="failed_extraction"):
        ExtractionRecord.model_validate(
            record_value(source.input_id, status="failed", error_code=None)
        )


def test_run_is_content_addressed_and_verifies_all_bound_references(tmp_path):
    source = make_input(**input_value())
    record = ExtractionRecord.model_validate(record_value(source.input_id))

    path = save_extraction_run(
        tmp_path, manifest=run_manifest(), inputs=[source], records=[record]
    )

    assert path == save_extraction_run(
        tmp_path, manifest=run_manifest(), inputs=[source], records=[record]
    )
    verified = verify_extraction_run(path)
    assert verified["integrity"] == "verified"
    assert verified["input_count"] == 1
    assert load_extraction_run(path)["records"][0].input_id == source.input_id


def test_run_rejects_unknown_input_and_duplicate_pipeline(tmp_path):
    source = make_input(**input_value())
    unknown = ExtractionRecord.model_validate(record_value("d" * 64))
    with pytest.raises(ValueError, match="unknown_extraction_input"):
        save_extraction_run(
            tmp_path, manifest=run_manifest(), inputs=[source], records=[unknown]
        )

    technical = ExtractionRecord.model_validate(record_value(source.input_id))
    duplicate = technical.model_copy(update={"generated_at": STAMP.replace(minute=1)})
    with pytest.raises(ValueError, match="duplicate_extraction"):
        save_extraction_run(
            tmp_path,
            manifest=run_manifest(),
            inputs=[source],
            records=[technical, duplicate],
        )


def test_run_rejects_manifest_without_all_prior_provenance(tmp_path):
    source = make_input(**input_value())
    record = ExtractionRecord.model_validate(record_value(source.input_id))
    with pytest.raises(ValueError, match="extraction_manifest"):
        save_extraction_run(
            tmp_path, manifest={"bundle_id": HASH}, inputs=[source], records=[record]
        )


class PublicDump(BaseModel):
    value: str


class FakeLLM:
    def __init__(self):
        self.preset = SimpleNamespace(primary=SimpleNamespace(model_id="requested"))
        self.response = SimpleNamespace(
            model="returned-fallback",
            provider="public-provider",
            usage=PublicDump(value="one-token"),
            choices=[PublicDump(value="answer")],
        )

    async def completion(self, **kwargs):
        self.kwargs = kwargs
        return self.response


@pytest.mark.asyncio
async def test_recording_proxy_preserves_response_and_captures_public_provenance():
    fake = FakeLLM()
    proxy = RecordingLLM(fake)

    response = await proxy.completion(
        messages=[{"role": "user", "content": "sample"}],
        temperature=0.2,
        api_key="do-not-store",
    )

    assert response is fake.response
    assert proxy.preset is fake.preset
    assert proxy.calls[0]["actual_model"] == "returned-fallback"
    assert proxy.calls[0]["provider"] == "public-provider"
    assert proxy.calls[0]["choices"] == [{"value": "answer"}]
    assert "do-not-store" not in str(proxy.calls[0])


def test_usage_and_generation_token_counts_are_not_credentials():
    from datetime import datetime, timezone

    from app.services.theme_evaluation.extraction_records import ExtractionRecord

    call = RecordingLLM(FakeLLM())._record(
        {"messages": [], "max_tokens": 8192},
        SimpleNamespace(
            model="actual",
            provider=None,
            usage={"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
            choices=[],
        ),
    )
    record = ExtractionRecord(
        input_id="a" * 64,
        pipeline="technical",
        status="success",
        mentions=[],
        error_code=None,
        generated_at=datetime.now(timezone.utc),
        requested_model="minimax/MiniMax-M2.7",
        reference_sha256="b" * 64,
        code_revision="test",
        calls=[call],
    )
    assert record.calls[0]["usage"]["total_tokens"] == 17


@pytest.mark.parametrize(
    "mention",
    [
        {"theme": "chips"},
        {
            "theme": "chips",
            "tickers": ["NVDA"],
            "sentiment": "bullish",
            "confidence": float("nan"),
            "excerpt": "x",
        },
        {
            "theme": " ",
            "tickers": [],
            "sentiment": "neutral",
            "confidence": 0.5,
            "excerpt": "x",
        },
    ],
)
def test_malformed_mentions_cannot_be_frozen(mention):
    with pytest.raises(ValueError):
        ExtractionRecord.model_validate(record_value(HASH, mentions=[mention]))


def test_normalized_input_retains_verifiable_original_spans():
    from dataclasses import asdict

    from app.services.theme_evaluation.quantity_display import normalize_quantities

    original = "Supplier invested KRW 1억 in capacity."
    view = normalize_quantities(original)
    value = input_value(
        text=view.text,
        original_text_sha256=sha256(original.encode()).hexdigest(),
        normalization_policy=view.policy_version,
        normalization=asdict(view),
    )
    item = make_input(**value)
    assert item.normalization["quantities"][0]["original"] == "KRW 1억"
    assert item.normalization["original"] == original
    value["normalization"]["text"] = "tampered"
    with pytest.raises(ValueError):
        make_input(**value)


def test_malformed_call_provenance_cannot_be_frozen():
    with pytest.raises(ValueError):
        ExtractionRecord.model_validate(
            record_value(HASH, calls=[{"status": "success"}])
        )


@pytest.mark.parametrize("with_development", [False, True])
def test_development_and_legacy_mentions_roundtrip_into_readable_review(tmp_path, with_development):
    from app.services.theme_evaluation.extraction_review import render_extractions

    source = make_input(**input_value())
    mention = {"theme": "HBM", "tickers": [], "sentiment": "bullish",
               "confidence": 0.8, "excerpt": "Orders increased."}
    if with_development:
        mention["development"] = "Suppliers reported increased HBM orders."
    record = ExtractionRecord.model_validate(record_value(source.input_id, mentions=[mention]))
    path = save_extraction_run(tmp_path, manifest=run_manifest(exclusions=[]),
                               inputs=[source], records=[record])
    before = {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}
    assert load_extraction_run(path)["records"][0].mentions == [mention]
    output = tmp_path / "review"
    render_extractions(path, output)
    with (output / "theme-mentions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["theme"] == "HBM"
    assert rows[0]["development"] == mention.get("development", "")
    assert rows[0]["source_text"] == "Orders increased."
    assert rows[0]["source_url"] == "https://example.com/post/1"
    report = (output / "extractions.md").read_text()
    assert "**Theme:** HBM" in report
    expected_development = "Suppliers reported increased HBM orders\\." if with_development else "Not recorded"
    assert "**Development:** " + expected_development in report
    assert before == {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_review_rendering_distinguishes_partial_candidate_failures_from_holds(tmp_path):
    """A mixed review outcome is inspectable without treating it as a deliberate hold."""
    from app.services.theme_evaluation.extraction_review import render_extractions

    source = make_input(**input_value())
    accepted = {
        "theme": "Optics",
        "tickers": [],
        "sentiment": "bullish",
        "confidence": 0.8,
        "excerpt": "Optical orders increased.",
    }
    unavailable = {
        "theme": "Copper",
        "tickers": [],
        "sentiment": "bullish",
        "confidence": 0.7,
        "excerpt": "Copper demand increased.",
    }
    partial_audit = {
        "status": "partial",
        "candidates": [accepted, unavailable],
        "decisions": [
            {
                "index": 0,
                "action": "accepted",
                "theme": {"reason": "Directly stated."},
                "development": {"reason": "No development."},
            },
            {
                "index": 1,
                "action": "review_unavailable",
                "error_code": "claim_review_timeout",
                "adjustments": [],
            },
        ],
    }
    all_missing_audit = {
        "status": "partial",
        "candidates": [accepted, unavailable],
        "decisions": [
            {
                "index": 0,
                "action": "held_theme",
                "theme": {"reason": "Unsupported."},
                "development": {"reason": "Not assessed."},
            },
            {
                "index": 1,
                "action": "review_unavailable",
                "error_code": "claim_review_timeout",
                "adjustments": [],
            },
        ],
    }
    technical = ExtractionRecord.model_validate(
        record_value(
            source.input_id,
            mentions=[accepted],
            claim_review=partial_audit,
        )
    )
    fundamental = ExtractionRecord.model_validate(
        record_value(
            source.input_id,
            pipeline="fundamental",
            claim_review=all_missing_audit,
        )
    )
    run = save_extraction_run(
        tmp_path,
        manifest=run_manifest(exclusions=[]),
        inputs=[source],
        records=[technical, fundamental],
    )

    summary = render_extractions(run, tmp_path / "review")

    assert summary["claim_review_partial_outcomes"] == 2
    assert summary["claim_review_unavailable_candidates"] == 2
    report = (tmp_path / "review" / "extractions.md").read_text()
    assert "Technical: 1 theme mentions · 1 candidate review unavailable" in report
    assert "Fundamental: Candidate review partially unavailable (1 candidate unavailable)" in report
    assert "All candidates held after evidence review" not in report
    with (tmp_path / "review" / "claim-review.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["status"] == "partial"
    assert rows[0]["unavailable_candidates"] == "1"
    assert rows[1]["unavailable_candidates"] == "1"
