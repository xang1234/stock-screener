"""Explicit local evidence commands; extraction is outside this checkpoint."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .article_intake import apply_followups, import_derivatives, propose_references
from .bundle import IntegrityError, canonical_bytes, load_bundle, seal_bundle, sha256, verify_bundle
from .records import Derivative, Document, Followup, REQUIRED_LIST_IDS
from .review import coverage_summary, render_review
from .xui_intake import import_xui, read_required_lists


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Collect and review theme benchmark evidence before extraction.')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('import-x', 'collect-x'):
        p = commands.add_parser(name)
        p.add_argument('--output-root', type=Path, required=True)
        p.add_argument('--max-posts-per-source', type=int, default=50)
        if name == 'import-x':
            p.add_argument('--first', type=Path, required=True)
            p.add_argument('--second', type=Path, required=True)
            p.add_argument('--mode', choices=['controlled', 'observed_capture'], default='observed_capture')
            p.add_argument('--captured-at', default=None)
            p.add_argument('--requested-limit', type=int, default=None)
        else:
            for arg in ('wrapper', 'python', 'xui-bin', 'config'):
                p.add_argument('--' + arg, type=Path, required=True)
            p.add_argument('--profile', default='default')
            p.add_argument('--limit', type=int, default=50)
    for name in ('references', 'review', 'verify', 'import-articles', 'import-translations'):
        p = commands.add_parser(name)
        p.add_argument('--bundle', type=Path, required=True)
        if name in {'references', 'review'}:
            p.add_argument('--output', type=Path, required=True)
        if name.startswith('import-'):
            p.add_argument('--records', type=Path, required=True)
            p.add_argument('--output-root', type=Path, required=True)
    return parser


def _save_raw(root: Path, raw: bytes) -> str:
    digest = sha256(raw)
    path = root / 'raw' / (digest + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as handle:
            handle.write(raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise IntegrityError('raw_hash_mismatch')
    return digest


def _run(args) -> tuple[dict, int]:
    if args.command in {'import-x', 'collect-x'}:
        captured = datetime.now(timezone.utc)
        hashes = {}
        if args.command == 'import-x':
            captured = args.captured_at or captured
            payloads = {}
            for key, path in zip(REQUIRED_LIST_IDS, (args.first, args.second)):
                raw = path.read_bytes()
                payloads[key] = json.loads(raw)
                hashes[key] = _save_raw(args.output_root, raw)
            limit, mode = args.requested_limit, args.mode
        else:
            payloads = read_required_lists(wrapper=args.wrapper, python=args.python,
                xui_bin=args.xui_bin, config=args.config, profile=args.profile, limit=args.limit)
            for key, payload in payloads.items():
                raw = payload['_capture'].pop('raw_json', None)
                raw = raw.encode() if raw is not None else canonical_bytes(payload)
                hashes[key] = _save_raw(args.output_root, raw)
            limit, mode = args.limit, 'observed_capture'
        bundle = import_xui(payloads, captured_at=captured, max_posts_per_source=args.max_posts_per_source,
                            requested_limit=limit, mode=mode, raw_hashes=hashes)
        bundle = apply_followups(bundle, propose_references(bundle), [])
        path = seal_bundle(args.output_root, bundle)
        failed = any(s.status != 'success' for s in bundle.source_outcomes)
        return {'bundle_path': str(path.resolve()), 'coverage': coverage_summary(bundle)}, (
            3 if args.command == 'collect-x' and failed else 0)
    bundle = load_bundle(args.bundle)
    if args.command == 'verify':
        return verify_bundle(args.bundle), 0
    if args.command == 'review':
        files = render_review(bundle, args.output)
        return {'files': [str(p.resolve()) for p in files], 'coverage': coverage_summary(bundle)}, 0
    if args.command == 'references':
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump([r.model_dump(mode='json') for r in propose_references(bundle)], handle,
                      indent=2, ensure_ascii=False)
        return {'references_path': str(args.output.resolve())}, 0
    records = json.loads(args.records.read_bytes())
    if records['bundle_id'] != args.bundle.name or records['mode'] != bundle.mode:
        raise ValueError('import_base_bundle_mismatch')
    if args.command == 'import-articles':
        if set(records) != {'bundle_id', 'mode', 'followups', 'articles'}:
            raise ValueError('invalid_article_import_fields')
        bundle = apply_followups(bundle, [Followup.model_validate(r) for r in records['followups']],
                                 [Document.model_validate(r) for r in records['articles']])
    else:
        if set(records) != {'bundle_id', 'mode', 'derivatives'}:
            raise ValueError('invalid_translation_import_fields')
        bundle = import_derivatives(bundle, [Derivative.model_validate(r) for r in records['derivatives']])
    path = seal_bundle(args.output_root, bundle)
    return {'bundle_path': str(path.resolve()), 'coverage': coverage_summary(bundle)}, 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result, code = _run(args)
    except IntegrityError as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr)
        return 5
    except (ValueError, OSError, KeyError, TypeError):
        # Validation errors can contain original content; leave it out of terminal logs.
        print(json.dumps({'error': 'invalid_evidence_input_or_output'}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return code
