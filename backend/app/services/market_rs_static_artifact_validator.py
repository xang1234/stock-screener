"""Validate one staged static bundle against a canonical Market RS run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy.orm import Session

from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.models.industry import IBDGroupRank
from app.services.static_groups_rrg_export import (
    StaticGroupsRRGDatabasePayloadSource,
    StaticGroupsRRGUnavailableError,
    StaticGroupsRRGUnavailableReason,
)
from app.services.static_site_export_service import STATIC_SITE_SCHEMA_VERSION


@dataclass(frozen=True)
class StaticRsArtifactDocuments:
    manifest: Mapping[str, Any]
    groups: Mapping[str, Any]
    scan: Mapping[str, Any]


@dataclass(frozen=True)
class StaticArtifactValidationResult:
    manifest_sha256: str | None
    rrg_status: str | None
    errors: tuple[str, ...]


class MarketRsStaticArtifactValidator:
    @staticmethod
    def _json_file(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def manifest_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def validate(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
        latest_run: Any,
        feature_run_id: int,
        static_staging_dir: Path,
    ) -> StaticArtifactValidationResult:
        errors: list[str] = []
        manifest_path = static_staging_dir / "manifest.json"
        if not manifest_path.is_file():
            return StaticArtifactValidationResult(
                manifest_sha256=None,
                rrg_status=None,
                errors=("Missing staged static-site-v3 manifest.",),
            )
        manifest_hash = self.manifest_hash(manifest_path)
        try:
            manifest = self._json_file(manifest_path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return StaticArtifactValidationResult(
                manifest_sha256=manifest_hash,
                rrg_status=None,
                errors=(f"Invalid staged static manifest: {exc}",),
            )
        if manifest.get("schema_version") != STATIC_SITE_SCHEMA_VERSION:
            errors.append("Staged static manifest is not static-site-v3.")
        entry = (manifest.get("markets") or {}).get(market) or {}
        expected_entry = {
            "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
            "market_rs_run_id": latest_run.id,
            "rs_as_of_date": through_date.isoformat(),
            "rs_universe_size": latest_run.eligible_symbol_count,
        }
        for key, expected in expected_entry.items():
            if entry.get(key) != expected:
                errors.append(
                    f"Staged static Market metadata {key}={entry.get(key)!r}; "
                    f"expected {expected!r}."
                )

        market_dir = static_staging_dir / "markets" / market.lower()
        groups_path = market_dir / "groups.json"
        scan_path = market_dir / "scan" / "manifest.json"
        groups_payload: dict[str, Any] = {}
        if not groups_path.is_file():
            errors.append("Missing staged static Groups artifact.")
        else:
            groups_payload = self._json_file(groups_path)
            if (
                groups_payload.get("schema_version") != STATIC_SITE_SCHEMA_VERSION
                or groups_payload.get("rs_formula_version")
                != BALANCED_RS_FORMULA_VERSION
                or groups_payload.get("market_rs_run_id") != latest_run.id
            ):
                errors.append(
                    "Staged Groups artifact has mismatched "
                    "schema/formula/run metadata."
                )
        scan_payload: dict[str, Any] = {}
        if not scan_path.is_file():
            errors.append("Missing staged static Scan manifest.")
        else:
            scan_payload = self._json_file(scan_path)
            if scan_payload.get("run_id") != feature_run_id:
                errors.append("Staged Scan manifest names a different Feature run.")

        documents = StaticRsArtifactDocuments(
            manifest=manifest,
            groups=groups_payload,
            scan=scan_payload,
        )
        self._validate_group_parity(
            db,
            market=market,
            through_date=through_date,
            documents=documents,
            errors=errors,
        )
        self._validate_stock_parity(
            latest_run=latest_run,
            documents=documents,
            errors=errors,
        )
        rrg_status = self._validate_rrg(
            db,
            market=market,
            through_date=through_date,
            market_dir=market_dir,
            errors=errors,
        )
        return StaticArtifactValidationResult(
            manifest_sha256=manifest_hash,
            rrg_status=rrg_status,
            errors=tuple(errors),
        )

    @staticmethod
    def _validate_group_parity(
        db: Session,
        *,
        market: str,
        through_date: date,
        documents: StaticRsArtifactDocuments,
        errors: list[str],
    ) -> None:
        live_groups = (
            db.query(IBDGroupRank)
            .filter(
                IBDGroupRank.market == market,
                IBDGroupRank.date == through_date,
                IBDGroupRank.rs_formula_version == BALANCED_RS_FORMULA_VERSION,
            )
            .order_by(IBDGroupRank.rank)
            .all()
        )
        static_groups = (
            (((documents.groups.get("payload") or {}).get("rankings") or {}).get(
                "rankings"
            ))
            or []
        )
        static_by_name = {row.get("industry_group"): row for row in static_groups}
        parity_fields = (
            "rank",
            "avg_rs_rating",
            "avg_rs_rating_1m",
            "avg_rs_rating_3m",
            "num_stocks",
            "top_symbol",
            "rs_formula_version",
            "market_rs_run_id",
        )
        for live in live_groups:
            static = static_by_name.get(live.industry_group)
            if static is None:
                errors.append(
                    f"Static Groups artifact omits {live.industry_group}."
                )
                continue
            for field in parity_fields:
                live_value = getattr(live, field)
                static_value = static.get(field)
                if isinstance(live_value, float):
                    equal = (
                        math.isclose(
                            live_value,
                            float(static_value),
                            rel_tol=0,
                            abs_tol=1e-9,
                        )
                        if static_value is not None
                        else False
                    )
                else:
                    equal = live_value == static_value
                if not equal:
                    errors.append(
                        "Live/static Group mismatch for "
                        f"{live.industry_group}.{field}."
                    )

    @staticmethod
    def _validate_stock_parity(
        *,
        latest_run: Any,
        documents: StaticRsArtifactDocuments,
        errors: list[str],
    ) -> None:
        scan_rows = list(documents.scan.get("preview_rows") or [])
        if not scan_rows:
            scan_rows = list(documents.scan.get("initial_rows") or [])[:10]
        stock_by_symbol = {row.symbol: row for row in latest_run.rows}
        stock_fields = {
            "rs_rating": "overall_rs",
            "rs_rating_1m": "rs_1m",
            "rs_rating_3m": "rs_3m",
            "rs_rating_12m": "rs_12m",
        }
        compared = 0
        for static_row in sorted(
            scan_rows,
            key=lambda row: str(row.get("symbol")),
        ):
            stock = stock_by_symbol.get(static_row.get("symbol"))
            if stock is None:
                continue
            compared += 1
            for static_field, stock_field in stock_fields.items():
                if static_row.get(static_field) != getattr(stock, stock_field):
                    errors.append(
                        f"Live/static stock mismatch for "
                        f"{stock.symbol}.{static_field}."
                    )
            if static_row.get("rs_formula_version") not in (
                None,
                BALANCED_RS_FORMULA_VERSION,
            ):
                errors.append(
                    f"Static stock {stock.symbol} has a mixed RS formula."
                )
            if static_row.get("market_rs_run_id") not in (None, latest_run.id):
                errors.append(
                    f"Static stock {stock.symbol} has a mixed Market RS run."
                )
        if latest_run.rows and compared == 0:
            errors.append(
                "No deterministic stock sample overlaps the staged Scan preview."
            )

    def _validate_rrg(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
        market_dir: Path,
        errors: list[str],
    ) -> str | None:
        rrg_path = market_dir / "groups_rrg.json"
        try:
            expected_rrg = StaticGroupsRRGDatabasePayloadSource(
                schema_version=STATIC_SITE_SCHEMA_VERSION,
            ).build(
                db=db,
                generated_at="validation",
                expected_as_of_date=through_date,
                market=market,
                formula_version=BALANCED_RS_FORMULA_VERSION,
            )
        except StaticGroupsRRGUnavailableError as exc:
            if (
                exc.reason_code
                is StaticGroupsRRGUnavailableReason.INSUFFICIENT_HISTORY
            ):
                if rrg_path.is_file():
                    payload = self._json_file(rrg_path)
                    if payload.get("available") or payload.get("payload"):
                        errors.append(
                            "Static RRG contains coordinates despite "
                            "insufficient balanced history."
                        )
                return "insufficient_balanced_history"
            errors.append(
                "Balanced RRG validation failed "
                f"({exc.reason_code.value}): {exc.reason}"
            )
            return None
        if not rrg_path.is_file():
            errors.append("Missing staged balanced RRG artifact.")
        else:
            actual_rrg = self._json_file(rrg_path)
            if (
                actual_rrg.get("rs_formula_version")
                != BALANCED_RS_FORMULA_VERSION
            ):
                errors.append("Staged RRG artifact uses a mixed RS formula.")
            if actual_rrg.get("payload") != expected_rrg.get("payload"):
                errors.append("Live/static RRG coordinates diverge.")
        return "available"


__all__ = [
    "MarketRsStaticArtifactValidator",
    "StaticArtifactValidationResult",
    "StaticRsArtifactDocuments",
]
