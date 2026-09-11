"""Explicit frozen-input, extraction and offline review commands."""

import argparse
import json
import os
import sys
from pathlib import Path

from .bundle import IntegrityError, canonical_bytes
from .extraction_capture import (
    load_extraction_run,
    save_extraction_run,
    verify_extraction_run,
)
from .extraction_intake import ExtractionApproval, build_extraction_inputs
from .extraction_records import ExtractionRecord
from .extraction_review import render_extractions
from .grounding import prepare_grounding, validate_grounding
from .preparation_store import PreparationStore


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-inputs")
    prepare.add_argument("--bundle", type=Path, required=True)
    prepare.add_argument("--store", type=Path, required=True)
    prepare.add_argument("--preparation-id", required=True)
    prepare.add_argument("--assessment-id", required=True)
    prepare.add_argument("--approval", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    grounding = commands.add_parser("prepare-grounding")
    grounding.add_argument("--run", type=Path, required=True)
    grounding.add_argument("--bundle", type=Path, required=True)
    grounding.add_argument("--store", type=Path, required=True)
    grounding.add_argument("--company-context", type=Path, required=True)
    grounding.add_argument("--as-of", required=True)
    grounding.add_argument("--output", type=Path, required=True)
    for name in ("generate", "import-extractions", "review", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        if name in ("generate", "import-extractions"):
            command.add_argument("--output-root", type=Path, required=True)
        if name == "review":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--bundle", type=Path)
            command.add_argument("--store", type=Path)
        if name == "import-extractions":
            command.add_argument("--records", type=Path, required=True)
        if name == "generate":
            command.add_argument("--grounding", type=Path)
            command.add_argument("--grounding-bundle", type=Path)
            command.add_argument("--reference-manifest", type=Path)
            command.add_argument("--model", default="minimax/MiniMax-M2.7")
            command.add_argument(
                "--pipeline", action="append", choices=["technical", "fundamental"]
            )
            command.add_argument("--max-documents", type=int, default=10)
            command.add_argument("--code-revision", required=True)
            command.add_argument("--allow-model-calls", action="store_true")
            command.add_argument("--env-file", type=Path)
    return parser


def _run(args):
    if args.command == "prepare-inputs":
        inputs, manifest = build_extraction_inputs(
            args.bundle,
            PreparationStore(args.store),
            args.preparation_id,
            args.assessment_id,
            json.loads(args.approval.read_text()),
        )
        path = save_extraction_run(
            args.output_root, manifest=manifest, inputs=inputs, records=[]
        )
        return {"run_path": str(path.resolve()), **verify_extraction_run(path)}
    if args.command == "prepare-grounding":
        run = load_extraction_run(args.run)
        packet = prepare_grounding(
            run,
            args.bundle,
            PreparationStore(args.store),
            json.loads(args.company_context.read_text()),
            args.as_of,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            handle.write(canonical_bytes(packet))
        return {
            "grounding_path": str(args.output.resolve()),
            "digest": packet["digest"],
            "context_count": len(packet["contexts"]),
        }
    if args.command == "verify":
        return verify_extraction_run(args.run)
    if args.command == "review":
        return render_extractions(
            args.run,
            args.output,
            base=args.bundle,
            store=PreparationStore(args.store) if args.store else None,
        )
    if args.command == "generate" and not args.allow_model_calls:
        from .extraction_runtime import RuntimeUnavailable

        raise RuntimeUnavailable("model_calls_not_authorized")
    run = load_extraction_run(args.run)
    approval = ExtractionApproval.model_validate(run["manifest"].get("approval", {}))
    if any(
        getattr(approval, key) != run["manifest"][key]
        for key in ("bundle_id", "preparation_id", "assessment_id")
    ):
        raise ValueError("approval_evidence_mismatch")
    if not approval.reviewer.strip() or not approval.reason.strip():
        raise ValueError("approval_attribution_required")
    if args.command == "import-extractions":
        imported = json.loads(args.records.read_text())
        if imported["input_run_id"] != run["run_id"]:
            raise ValueError("extraction_import_run_mismatch")
        records = [ExtractionRecord.model_validate(r) for r in imported["records"]]
        records = [*run["records"], *records]
    else:
        from .extraction_runtime import generate_extractions

        if run["records"]:
            raise ValueError("generation_requires_unprocessed_input_run")
        if args.env_file:
            from dotenv import load_dotenv

            load_dotenv(args.env_file, override=False)
        reference = (
            json.loads(args.reference_manifest.read_text())
            if args.reference_manifest
            else {}
        )
        grounding_packet = None
        if args.grounding:
            grounding_packet = json.loads(args.grounding.read_text())
            validate_grounding(
                grounding_packet,
                run["inputs"],
                run_id=run["run_id"],
                manifest=run["manifest"],
                base=args.grounding_bundle,
            )
        records = generate_extractions(
            run["inputs"],
            eval_database_url=os.environ.get("THEME_EVAL_DATABASE_URL"),
            application_database_url=os.environ.get("DATABASE_URL"),
            reference_manifest=reference,
            model=args.model,
            pipelines=args.pipeline or ["technical", "fundamental"],
            max_documents=args.max_documents,
            code_revision=args.code_revision,
            allow_model_calls=True,
            grounding_packet=grounding_packet,
            grounding_bundle=args.grounding_bundle,
            grounding_manifest=run["manifest"],
        )
    manifest = {
        **run["manifest"],
        "input_run_id": run["run_id"],
        "extraction_status": "partial",
    }
    if (
        run["inputs"]
        and len(records) == len(run["inputs"]) * 2
        and all(r.status == "success" for r in records)
    ):
        manifest["extraction_status"] = "complete"
    path = save_extraction_run(
        args.output_root, manifest=manifest, inputs=run["inputs"], records=records
    )
    return {"run_path": str(path.resolve()), **verify_extraction_run(path)}


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        result = _run(args)
    except IntegrityError:
        print(json.dumps({"error": "extraction_integrity_failed"}), file=sys.stderr)
        return 5
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        # Runtime errors carry a deliberately sanitized stable code.
        from .extraction_runtime import RuntimeUnavailable

        if isinstance(error, RuntimeUnavailable):
            print(json.dumps({"error": error.code}), file=sys.stderr)
            return 4
        print(
            json.dumps({"error": "invalid_extraction_input_or_output"}), file=sys.stderr
        )
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0
