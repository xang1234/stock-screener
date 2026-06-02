"""Classify a market's universe into IBD groups and write a release bundle.

Runs the hybrid cascade (crosswalk → local embedding → optional LLM tiebreaker)
over every active universe symbol for ``--market`` that lacks an authoritative
group, then writes ``ibd-classification-<market>-<date>-<rev>.json.gz`` plus the
``ibd-classification-latest-<market>.json`` manifest into ``--output-dir``.

Expects the universe + curated CSV already loaded into the DB (the GitHub Action
imports the weekly-reference bundle and loads the CSV in prior steps). The LLM
tiebreaker is configured purely via env vars (``IBD_LLM_*``) — see
``app.services.llm.openai_compatible_client``.

Usage:
    python -m app.scripts.build_ibd_classification_bundle --market SG --output-dir /tmp/ibd
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from app.database import SessionLocal
from app.scripts._runtime import prepare_runtime, repo_root
from app.services.ibd_classification_bundle import (
    bundle_asset_name,
    build_manifest,
    build_payload,
    latest_manifest_name,
    read_bundle,
    write_bundle,
    write_manifest,
)
from app.services.ibd_classification_health import (
    build_health_report,
    health_asset_name,
    write_health_report,
)
from app.services.ibd_classification_service import IBDClassificationService
from app.services.ibd_crosswalk import IBDCrosswalk


# Single source of truth for the embedding model id, recorded in the health
# report's `embedding_model` fingerprint so a model swap is visible in the diff.
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"


def _default_crosswalk_path() -> Path:
    return repo_root() / "data" / "ibd_crosswalk.json"


def _load_crosswalk(path: Path) -> IBDCrosswalk | None:
    if path.exists():
        return IBDCrosswalk.load(path)
    print(f"WARNING: crosswalk file {path} not found; skipping deterministic tier", flush=True)
    return None


def _build_engine(disabled: bool):
    if disabled:
        return None
    try:
        from app.services.theme_embedding_service import ThemeEmbeddingEngine

        return ThemeEmbeddingEngine(EMBEDDING_MODEL_ID)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: embedding engine unavailable ({exc}); skipping embedding tier", flush=True)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True)
    parser.add_argument("--output-dir", default=str(repo_root() / ".tmp" / "ibd-classification"))
    parser.add_argument("--crosswalk", default=str(_default_crosswalk_path()))
    parser.add_argument("--bundle-name", default=None)
    parser.add_argument("--latest-manifest-name", default=None)
    parser.add_argument("--no-llm", action="store_true", help="Disable the LLM tiebreaker tier.")
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Disable the embedding tier. The LLM tiebreaker ranks the embedding "
        "shortlist, so this also disables the LLM tier (crosswalk-only).",
    )
    parser.add_argument("--as-of", default=None, help="Override as-of date (YYYY-MM-DD).")
    parser.add_argument(
        "--prev-bundle",
        default=None,
        help="Path to the previous week's bundle (.json.gz) for the churn diff. "
        "When omitted, the health report's diff is null (e.g. first run).",
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=0.0,
        help="Soft deadline for the classification loop. Once reached, the paid LLM "
        "tier is disabled and remaining symbols use the free deterministic "
        "fallback so the job finishes under the CI cap. 0 = unbounded.",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=0,
        help="Cap on paid LLM tiebreaker calls per run; remaining low-confidence "
        "symbols use the free deterministic fallback. 0 = unbounded.",
    )
    parser.add_argument(
        "--soft-attach",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When the crosswalk/embedding/LLM tiers don't produce a confident "
        "match, attach the nearest deterministic guess (relaxed crosswalk "
        "plurality, then nearest centroid) instead of leaving the symbol "
        "unresolved. Keeps foreign-market coverage high off the paid tier.",
    )
    args = parser.parse_args()

    prepare_runtime()
    market = args.market.strip().upper()

    crosswalk = _load_crosswalk(Path(args.crosswalk))
    engine = _build_engine(args.no_embedding)

    tiebreaker = None
    model_id = None
    if not args.no_llm:
        from app.services.llm.openai_compatible_client import build_ibd_tiebreaker

        tiebreaker, model_id = build_ibd_tiebreaker()

    now = datetime.utcnow().replace(microsecond=0)
    as_of_date = args.as_of or now.date().isoformat()
    as_of_compact = as_of_date.replace("-", "")
    source_revision = f"ibd:{now.strftime('%Y%m%d%H%M%S')}"

    deadline_seconds = args.max_runtime_minutes * 60 if args.max_runtime_minutes > 0 else None
    max_llm_calls = args.max_llm_calls if args.max_llm_calls > 0 else None

    def _print_progress(record: dict) -> None:
        print(f"  progress: {record}", flush=True)

    with SessionLocal() as db:
        service = IBDClassificationService(
            crosswalk=crosswalk,
            embedding_engine=engine,
            llm_tiebreaker=tiebreaker,
            llm_model_id=model_id,
        )
        result = service.classify_market(
            db, market,
            deadline_seconds=deadline_seconds,
            max_llm_calls=max_llm_calls,
            soft_attach=args.soft_attach,
            progress_callback=_print_progress,
        )

    summary = result.summary()
    payload = build_payload(
        market=market,
        as_of_date=as_of_date,
        source_revision=source_revision,
        generated_at=now.isoformat() + "Z",
        model_id=model_id,
        assignments=result.assignments,
        summary=summary,
    )

    output_dir = Path(args.output_dir)
    resolved_bundle_name = args.bundle_name or bundle_asset_name(market, as_of_compact, source_revision)
    resolved_manifest_name = args.latest_manifest_name or latest_manifest_name(market)
    bundle_path = output_dir / resolved_bundle_name
    manifest_path = output_dir / resolved_manifest_name

    sha256 = write_bundle(bundle_path, payload)
    manifest = build_manifest(payload=payload, bundle_name=resolved_bundle_name, sha256=sha256)
    write_manifest(manifest_path, manifest)

    prev_payload = None
    if args.prev_bundle:
        prev_path = Path(args.prev_bundle)
        if prev_path.exists():
            prev_payload = read_bundle(prev_path)
        else:
            print(
                f"WARNING: --prev-bundle {prev_path} not found; "
                "health report churn diff will be null",
                flush=True,
            )
    health = build_health_report(
        payload=payload,
        prev_payload=prev_payload,
        embedding_model=EMBEDDING_MODEL_ID,
    )
    health_path = output_dir / health_asset_name(market)
    write_health_report(health_path, health)

    print("IBD classification bundle complete:")
    print(f"  - market:   {market}")
    print(f"  - summary:  {summary}")
    print(f"  - bundle:   {bundle_path}")
    print(f"  - manifest: {manifest_path}")
    print(f"  - sha256:   {sha256}")
    print(f"  - health:   {health_path}")
    if health.get("diff") is not None:
        print(f"  - churn:    {health['diff']['churn_pct']}%")
    if result.deadline_hit:
        print(
            f"  - NOTE: runtime deadline reached after {result.processed}/{result.candidates} "
            f"symbols; remainder classified via free deterministic fallback (partial-quality run)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
