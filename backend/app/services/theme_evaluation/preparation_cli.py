"""Offline-first preparation commands with explicitly enabled network/model use."""

import argparse
import json
import os
import sys
from pathlib import Path

from .bundle import IntegrityError
from .image_preparation import OpenCodeGoVision
from .preparation_pipeline import import_articles, prepare
from .preparation_records import Handoff
from .preparation_review import render_preparation
from .preparation_store import PreparationStore, validate_handoff
from .preparation_translation_import import import_translations


def _parser():
    parser = argparse.ArgumentParser(
        description="Prepare evidence for review; never generate theme extractions."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prepare",
        "import-articles",
        "import-translations",
        "review",
        "verify",
    ):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        if name in ("prepare", "import-articles"):
            command.add_argument("--handoff", type=Path)
            command.add_argument("--prior-preparation")
        else:
            command.add_argument("--preparation", required=True)
        if name == "prepare":
            command.add_argument(
                "--stages",
                nargs="+",
                choices=["article", "text", "image"],
                required=True,
            )
            command.add_argument(
                "--allow-network",
                action="store_true",
                help="Enable public publisher/image downloads.",
            )
            command.add_argument(
                "--allow-model-calls",
                action="store_true",
                help="Enable approved Kimi K2.6 image calls.",
            )
            command.add_argument("--max-images", type=int, default=100)
            command.add_argument("--max-documents", type=int, default=500)
        elif name in ("import-articles", "import-translations"):
            command.add_argument("--records", type=Path, required=True)
        elif name == "review":
            command.add_argument("--output", type=Path, required=True)
    return parser


def _run(args):
    store = PreparationStore(args.output_root)
    if args.command == "review":
        return render_preparation(args.bundle, store, args.preparation, args.output)
    if args.command == "verify":
        store.verify_all()
        manifest = store.load(args.bundle, args.preparation)
        return {
            "preparation_id": args.preparation,
            "integrity": "verified",
            "bindings": len(manifest.bindings),
        }
    if args.command == "import-translations":
        records = json.loads(args.records.read_bytes())
        if (
            set(records) != {"bundle_id", "preparation_id", "translations"}
            or records["bundle_id"] != args.bundle.name
            or records["preparation_id"] != args.preparation
        ):
            raise ValueError("translation_import_base_mismatch")
        pid = import_translations(
            args.bundle, store, args.preparation, records["translations"]
        )
        return {"preparation_id": pid, "evidence_review": "pending"}
    handoff = (
        Handoff.model_validate_json(args.handoff.read_bytes())
        if args.handoff
        else Handoff(bundle_id=args.bundle.name)
    )
    validate_handoff(args.bundle, handoff)
    if args.command == "prepare":
        vision = None
        if args.allow_model_calls:
            if "image" not in args.stages:
                raise ValueError("image_stage_required")
            vision = OpenCodeGoVision(os.environ.get("OPENCODE_GO_API_KEY", ""))
        pid = prepare(
            args.bundle,
            store,
            handoff,
            stages=args.stages,
            vision=vision,
            allow_network=args.allow_network,
            prior_id=args.prior_preparation,
            max_images=args.max_images,
            max_documents=args.max_documents,
        )
    else:
        records = json.loads(args.records.read_bytes())
        if (
            set(records) != {"bundle_id", "articles"}
            or records["bundle_id"] != args.bundle.name
        ):
            raise ValueError("article_import_base_mismatch")
        pid = import_articles(
            args.bundle,
            store,
            handoff,
            records["articles"],
            prior_id=args.prior_preparation,
        )
    return {
        "preparation_id": pid,
        "output_root": str(args.output_root.resolve()),
        "evidence_review": "pending",
    }


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        result = _run(args)
    except IntegrityError:
        print(json.dumps({"error": "preparation_integrity_failed"}), file=sys.stderr)
        return 5
    except (ValueError, OSError, KeyError, TypeError, RuntimeError):
        print(
            json.dumps({"error": "invalid_preparation_input_or_configuration"}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0
