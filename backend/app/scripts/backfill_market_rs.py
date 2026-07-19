"""Operator CLI for shadow backfill and guarded balanced Market RS activation."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any

from app.config import settings
from app.database import SessionLocal
from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.services.static_site_export_service import StaticSiteExportService
from app.wiring.bootstrap import get_market_rs_rollout_service


class RolloutCommandFailed(RuntimeError):
    pass


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO date: {value}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill and optionally activate balanced Market RS",
    )
    parser.add_argument("--market", required=True)
    parser.add_argument("--through-date", required=True, type=_parse_date)
    parser.add_argument("--start-date", type=_parse_date)
    parser.add_argument("--static-staging-dir", type=Path)
    parser.add_argument("--activate", action="store_true")
    return parser


def _validate_staging_directory(path: Path | None) -> Path:
    if path is None:
        raise RolloutCommandFailed("--activate requires --static-staging-dir")
    if not path.is_absolute():
        raise RolloutCommandFailed("--static-staging-dir must be an absolute path")
    resolved = path.resolve()
    serving_dir = Path(settings.static_export_output_dir).expanduser().resolve()
    if resolved == serving_dir:
        raise RolloutCommandFailed(
            "--static-staging-dir must not be the configured serving directory"
        )
    if resolved.exists() and any(resolved.iterdir()):
        raise RolloutCommandFailed("--static-staging-dir must be empty")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _build_balanced_feature_snapshot(*, market: str, through_date: date) -> int:
    from app.interfaces.tasks.feature_store_tasks import build_daily_snapshot

    result = build_daily_snapshot.run(
        market=market,
        as_of_date_str=through_date.isoformat(),
        universe_name=f"market:{market}",
        publish_pointer_key=(
            f"rollout_rs:{BALANCED_RS_FORMULA_VERSION}:{market}"
        ),
        static_daily_mode=True,
        ignore_runtime_market_gate=True,
        skip_if_published=False,
        rs_formula_version_override=BALANCED_RS_FORMULA_VERSION,
    )
    if not isinstance(result, dict) or result.get("status") != "published":
        raise RolloutCommandFailed(
            f"Balanced Feature snapshot did not publish: {result}"
        )
    run_id = result.get("run_id")
    if run_id is None:
        raise RolloutCommandFailed("Balanced Feature snapshot returned no run ID")
    return int(run_id)


def _export_static_v3(
    *,
    market: str,
    feature_run_id: int,
    static_staging_dir: Path,
) -> None:
    StaticSiteExportService(SessionLocal).export(
        static_staging_dir,
        clean=True,
        markets=(market,),
        rs_formula_version_overrides={
            market: BALANCED_RS_FORMULA_VERSION,
        },
        feature_run_ids_by_market={market: feature_run_id},
    )


def _publish_live_groups(market: str) -> None:
    from app.services.group_rankings_cache import bump_group_rankings_epoch
    from app.services.ui_snapshot_service import safe_publish_groups_bootstrap

    bump_group_rankings_epoch(market)
    if market == "US":
        safe_publish_groups_bootstrap()


def execute_rollout(options: argparse.Namespace) -> dict[str, Any]:
    market = str(options.market).strip().upper()
    staging_dir = (
        _validate_staging_directory(options.static_staging_dir)
        if options.activate
        else None
    )
    db = SessionLocal()
    service = get_market_rs_rollout_service()
    try:
        report = service.backfill(
            db,
            market=market,
            through_date=options.through_date,
            start_date=options.start_date,
        )
        payload: dict[str, Any] = {
            "backfill": report.to_dict(),
            "activated": False,
        }
        if not options.activate:
            return payload
        if not report.ok or report.failed_count:
            raise RolloutCommandFailed(
                "One or more required backfill dates failed; repair the reported dates before activation"
            )

        feature_run_id = _build_balanced_feature_snapshot(
            market=market,
            through_date=options.through_date,
        )
        _export_static_v3(
            market=market,
            feature_run_id=feature_run_id,
            static_staging_dir=staging_dir,
        )
        db.expire_all()
        validation = service.validate_activation(
            db,
            market=market,
            through_date=options.through_date,
            feature_run_id=feature_run_id,
            static_staging_dir=staging_dir,
        )
        payload["validation"] = validation.to_dict()
        if not validation.ok:
            raise RolloutCommandFailed(
                "Activation validation failed: " + "; ".join(validation.errors)
            )
        service.activate(
            db,
            market=market,
            formula_version=BALANCED_RS_FORMULA_VERSION,
            feature_run_id=feature_run_id,
            validation=validation,
            static_staging_dir=staging_dir,
        )
        _publish_live_groups(market)
        payload.update(
            {
                "activated": True,
                "market": market,
                "formula_version": BALANCED_RS_FORMULA_VERSION,
                "feature_run_id": feature_run_id,
                "static_staging_dir": str(staging_dir),
            }
        )
        return payload
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    options = _build_parser().parse_args(argv)
    try:
        payload = execute_rollout(options)
    except RolloutCommandFailed as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    backfill = payload.get("backfill") or {}
    return 1 if int(backfill.get("failed_count", 0) or 0) > 0 else 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
