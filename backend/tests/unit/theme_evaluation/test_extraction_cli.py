"""Offline commands render missing, empty and failed outputs distinctly."""

import json
from datetime import datetime, timezone

from app.services.theme_evaluation.extraction_capture import save_extraction_run
from app.services.theme_evaluation.extraction_records import (
    ExtractionRecord,
    make_input,
)


def test_review_and_verify_work_without_provider_or_database(
    tmp_path, capsys, monkeypatch
):
    from app.services.theme_evaluation.extraction_cli import main

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("THEME_EVAL_DATABASE_URL", raising=False)
    stamp = datetime.now(timezone.utc)
    from app.services.theme_evaluation.bundle import sha256

    item = make_input(
        source_id="post:1",
        source_kind="post",
        source_url="https://example.com/1",
        title="=untrusted",
        text="Nothing investment related.",
        language="en",
        published_at=stamp,
        available_at=stamp,
        original_text_sha256=sha256(b"Nothing investment related."),
        result_ids=[],
        input_kind="original",
        normalization_policy=None,
        warnings=[],
    )
    records = [
        ExtractionRecord(
            input_id=item.input_id,
            pipeline=p,
            status=s,
            mentions=[],
            error_code="provider_failed" if s == "failed" else None,
            generated_at=stamp,
            requested_model="minimax/MiniMax-M2.7",
            reference_sha256="a" * 64,
            code_revision="test",
            calls=[],
        )
        for p, s in [("technical", "success"), ("fundamental", "failed")]
    ]
    run = save_extraction_run(
        tmp_path,
        manifest={
            "bundle_id": "a" * 64,
            "preparation_id": "b" * 64,
            "assessment_id": "c" * 64,
            "extraction_status": "partial",
            "exclusions": [],
        },
        inputs=[item],
        records=records,
    )
    assert main(["verify", "--run", str(run)]) == 0
    assert json.loads(capsys.readouterr().out)["integrity"] == "verified"
    out = tmp_path / "review"
    assert main(["review", "--run", str(run), "--output", str(out)]) == 0
    report = (out / "extractions.md").read_text()
    assert "No themes" in report and "Failed" in report
    assert "ranking" in report.lower()
    assert "'=untrusted" in (out / "inputs.csv").read_text()


def test_live_generation_requires_explicit_model_flag(tmp_path, capsys):
    from app.services.theme_evaluation.extraction_cli import main

    assert (
        main(
            [
                "generate",
                "--run",
                str(tmp_path),
                "--output-root",
                str(tmp_path / "out"),
                "--code-revision",
                "test",
            ]
        )
        == 4
    )
    assert json.loads(capsys.readouterr().err)["error"] == "model_calls_not_authorized"


def test_import_requires_reviewed_evidence_approval(tmp_path, capsys):
    from app.services.theme_evaluation.extraction_cli import main

    run = save_extraction_run(
        tmp_path,
        manifest={
            "bundle_id": "a" * 64,
            "preparation_id": "b" * 64,
            "assessment_id": "c" * 64,
        },
        inputs=[],
        records=[],
    )
    records = tmp_path / "records.json"
    records.write_text(json.dumps({"input_run_id": run.name, "records": []}))
    assert (
        main(
            [
                "import-extractions",
                "--run",
                str(run),
                "--records",
                str(records),
                "--output-root",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "out").exists()


def test_prepare_grounding_writes_new_canonical_packet_only(
    tmp_path, capsys, monkeypatch
):
    from app.services.theme_evaluation.extraction_cli import main

    run = {"run_id": "a" * 64, "inputs": [], "manifest": {}, "records": []}
    packet = {
        "policy_version": "grounding-v1",
        "run_id": run["run_id"],
        "bundle_id": "b" * 64,
        "preparation_id": "c" * 64,
        "assessment_id": "d" * 64,
        "approval": {},
        "as_of": "2026-09-01T00:00:00+00:00",
        "contexts": {},
        "digest": "e" * 64,
    }
    monkeypatch.setattr(
        "app.services.theme_evaluation.extraction_cli.load_extraction_run",
        lambda path: run,
    )
    captured = {}

    def fake_prepare(actual_run, base, store, contexts, as_of):
        captured.update(run=actual_run, base=base, contexts=contexts, as_of=as_of)
        return packet

    monkeypatch.setattr(
        "app.services.theme_evaluation.extraction_cli.prepare_grounding", fake_prepare
    )
    output = tmp_path / "packets" / "grounding.json"
    company_context = tmp_path / "company-context.json"
    company_context.write_text('{"input":{"companies":[],"warnings":[]}}')
    command = [
        "prepare-grounding",
        "--run",
        str(tmp_path / "run"),
        "--bundle",
        str(tmp_path / "bundle"),
        "--store",
        str(tmp_path / "store"),
        "--company-context",
        str(company_context),
        "--as-of",
        "2026-09-01T00:00:00Z",
        "--output",
        str(output),
    ]

    assert main(command) == 0
    assert json.loads(output.read_text()) == packet
    assert captured["contexts"] == {"input": {"companies": [], "warnings": []}}
    assert captured["as_of"] == "2026-09-01T00:00:00Z"
    original = output.read_bytes()
    assert main(command) == 2
    assert output.read_bytes() == original
    assert (
        json.loads(capsys.readouterr().err)["error"]
        == "invalid_extraction_input_or_output"
    )


def test_generate_validates_and_passes_grounding_packet(tmp_path, capsys, monkeypatch):
    import app.services.theme_evaluation.extraction_runtime as runtime
    from app.services.theme_evaluation.extraction_capture import save_extraction_run
    from app.services.theme_evaluation.extraction_cli import main

    approval = {
        "bundle_id": "a" * 64,
        "preparation_id": "b" * 64,
        "assessment_id": "c" * 64,
        "reviewer": "reviewer",
        "approved_at": "2026-09-01T00:00:00Z",
        "reason": "Reviewed evidence is suitable for the extraction.",
        "accepted_result_ids": [],
    }
    run_path = save_extraction_run(
        tmp_path,
        manifest={
            "bundle_id": approval["bundle_id"],
            "preparation_id": approval["preparation_id"],
            "assessment_id": approval["assessment_id"],
            "approval": approval,
            "admitted_inputs": 0,
            "exclusions": [],
        },
        inputs=[],
        records=[],
    )
    grounding = tmp_path / "grounding.json"
    grounding.write_text('{"packet":"provided"}')
    validated = {}
    monkeypatch.setattr(
        "app.services.theme_evaluation.extraction_cli.validate_grounding",
        lambda packet, inputs, *, run_id, manifest, base: (
            validated.update(
                packet=packet,
                inputs=inputs,
                run_id=run_id,
                manifest=manifest,
                base=base,
            )
            or {}
        ),
    )
    generated = {}
    monkeypatch.setattr(
        runtime,
        "generate_extractions",
        lambda *args, **kwargs: generated.update(kwargs) or [],
    )

    assert (
        main(
            [
                "generate",
                "--run",
                str(run_path),
                "--output-root",
                str(tmp_path / "out"),
                "--code-revision",
                "test",
                "--allow-model-calls",
                "--grounding",
                str(grounding),
            ]
        )
        == 0
    )
    assert validated["packet"] == {"packet": "provided"}
    assert validated["run_id"] == run_path.name
    assert generated["grounding_packet"] == {"packet": "provided"}
    assert generated["grounding_bundle"] is None
    assert generated["grounding_manifest"] == validated["manifest"]
    assert generated["grounding_manifest"]["approval"] == approval
    assert "run_path" in json.loads(capsys.readouterr().out)
