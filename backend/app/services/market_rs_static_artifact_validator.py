"""Validate one staged static bundle against a canonical Market RS run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.models.industry import IBDGroupRank
from app.services.static_groups_rrg_export import (
    StaticGroupsRRGDatabasePayloadSource,
    StaticGroupsRRGUnavailableError,
    StaticGroupsRRGUnavailableReason,
)
from app.services.static_site_export_service import (
    SCAN_BUNDLE_SCHEMA_VERSION,
    STATIC_SITE_SCHEMA_VERSION,
)


@dataclass(frozen=True)
class StaticRsArtifactDocuments:
    manifest: dict[str, Any]
    groups: dict[str, Any]
    scan_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class StaticArtifactFingerprint:
    sha256: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class StaticArtifactValidationResult:
    bundle_fingerprint: StaticArtifactFingerprint | None
    rrg_status: str | None
    errors: tuple[str, ...]


class MarketRsStaticArtifactValidator:
    @staticmethod
    def _json_file(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def bundle_fingerprint(
        static_staging_dir: Path,
        *,
        market: str,
    ) -> StaticArtifactFingerprint:
        root = Path(static_staging_dir)
        manifest_path = root / "manifest.json"
        market_dir = root / "markets" / market.lower()
        paths = ([manifest_path] if manifest_path.is_file() else []) + sorted(
            path for path in market_dir.rglob("*") if path.is_file()
        )
        digest = hashlib.sha256()
        relative_paths: list[str] = []
        for path in paths:
            relative = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            relative_paths.append(relative)
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative.encode("utf-8"))
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return StaticArtifactFingerprint(
            sha256=digest.hexdigest(),
            files=tuple(relative_paths),
        )

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
                bundle_fingerprint=None,
                rrg_status=None,
                errors=("Missing staged static-site-v3 manifest.",),
            )
        bundle_fingerprint = self.bundle_fingerprint(
            static_staging_dir,
            market=market,
        )
        try:
            manifest = self._json_file(manifest_path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return StaticArtifactValidationResult(
                bundle_fingerprint=bundle_fingerprint,
                rrg_status=None,
                errors=(f"Invalid staged static manifest: {exc}",),
            )
        if manifest.get("schema_version") != STATIC_SITE_SCHEMA_VERSION:
            errors.append("Staged static manifest is not static-site-v3.")
        entry = (manifest.get("markets") or {}).get(market) or {}
        expected_identity = {
            "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
            "market_rs_run_id": latest_run.id,
            "rs_as_of_date": through_date.isoformat(),
            "rs_universe_size": latest_run.eligible_symbol_count,
        }
        for key, expected in expected_identity.items():
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
            try:
                groups_payload = self._json_file(groups_path)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(f"Invalid staged Groups artifact: {exc}")
            if groups_payload.get("schema_version") != STATIC_SITE_SCHEMA_VERSION:
                errors.append(
                    "Staged Groups artifact has mismatched schema metadata."
                )
            self._validate_identity(
                groups_payload,
                expected_identity=expected_identity,
                label="Staged Groups artifact",
                errors=errors,
            )
        scan_payload: dict[str, Any] = {}
        scan_rows: tuple[dict[str, Any], ...] = ()
        if not scan_path.is_file():
            errors.append("Missing staged static Scan manifest.")
        else:
            try:
                scan_payload = self._json_file(scan_path)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(f"Invalid staged Scan manifest: {exc}")
            if scan_payload.get("run_id") != feature_run_id:
                errors.append("Staged Scan manifest names a different Feature run.")
            if scan_payload.get("schema_version") != SCAN_BUNDLE_SCHEMA_VERSION:
                errors.append("Staged Scan manifest has mismatched schema metadata.")
            self._validate_identity(
                scan_payload,
                expected_identity=expected_identity,
                label="Staged Scan manifest",
                errors=errors,
            )
            scan_rows = self._load_scan_rows(
                static_staging_dir,
                market_dir=market_dir,
                scan_payload=scan_payload,
                feature_run_id=feature_run_id,
                expected_identity=expected_identity,
                errors=errors,
            )

        documents = StaticRsArtifactDocuments(
            manifest=manifest,
            groups=groups_payload,
            scan_rows=scan_rows,
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
            through_date=through_date,
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
            bundle_fingerprint=bundle_fingerprint,
            rrg_status=rrg_status,
            errors=tuple(errors),
        )

    @staticmethod
    def _validate_identity(
        payload: dict[str, Any],
        *,
        expected_identity: dict[str, Any],
        label: str,
        errors: list[str],
    ) -> None:
        for key, expected in expected_identity.items():
            if payload.get(key) != expected:
                errors.append(
                    f"{label}.{key}={payload.get(key)!r}; expected {expected!r}."
                )

    def _load_scan_rows(
        self,
        static_staging_dir: Path,
        *,
        market_dir: Path,
        scan_payload: dict[str, Any],
        feature_run_id: int,
        expected_identity: dict[str, Any],
        errors: list[str],
    ) -> tuple[dict[str, Any], ...]:
        root = Path(static_staging_dir).resolve()
        chunk_refs = scan_payload.get("chunks")
        if not isinstance(chunk_refs, list):
            errors.append("Staged Scan manifest.chunks must be a list.")
            return ()

        rows: list[dict[str, Any]] = []
        referenced_paths: set[Path] = set()
        for expected_index, chunk_ref in enumerate(chunk_refs, start=1):
            if not isinstance(chunk_ref, dict) or not isinstance(
                chunk_ref.get("path"), str
            ):
                errors.append(
                    f"Staged Scan chunk reference {expected_index} is invalid."
                )
                continue
            chunk_path = (root / chunk_ref["path"]).resolve()
            if not chunk_path.is_relative_to(root):
                errors.append(
                    f"Staged Scan chunk {chunk_ref['path']!r} escapes the bundle."
                )
                continue
            referenced_paths.add(chunk_path)
            if not chunk_path.is_file():
                errors.append(f"Missing staged Scan chunk {chunk_ref['path']}.")
                continue
            try:
                chunk = self._json_file(chunk_path)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(f"Invalid staged Scan chunk {chunk_ref['path']}: {exc}")
                continue
            if chunk.get("run_id") != feature_run_id:
                errors.append(
                    f"Staged Scan chunk {chunk_ref['path']} names a different Feature run."
                )
            if chunk.get("schema_version") != SCAN_BUNDLE_SCHEMA_VERSION:
                errors.append(
                    f"Staged Scan chunk {chunk_ref['path']} has mismatched schema metadata."
                )
            if chunk.get("chunk_index") != expected_index:
                errors.append(
                    f"Staged Scan chunk {chunk_ref['path']} has an invalid index."
                )
            self._validate_identity(
                chunk,
                expected_identity=expected_identity,
                label=f"Staged Scan chunk {chunk_ref['path']}",
                errors=errors,
            )
            chunk_rows = chunk.get("rows")
            if not isinstance(chunk_rows, list) or not all(
                isinstance(row, dict) for row in chunk_rows
            ):
                errors.append(
                    f"Staged Scan chunk {chunk_ref['path']} has invalid rows."
                )
                continue
            if chunk_ref.get("count") != len(chunk_rows):
                errors.append(
                    f"Staged Scan chunk {chunk_ref['path']} count does not match its rows."
                )
            rows.extend(chunk_rows)

        discovered_paths = {
            path.resolve()
            for path in (market_dir / "scan" / "chunks").glob("*.json")
            if path.is_file()
        }
        if discovered_paths != referenced_paths:
            errors.append(
                "Staged Scan chunk files do not exactly match the manifest."
            )
        if scan_payload.get("rows_total") != len(rows):
            errors.append("Staged Scan rows_total does not match its chunks.")
        return tuple(rows)

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
        through_date: date,
        documents: StaticRsArtifactDocuments,
        errors: list[str],
    ) -> None:
        scan_rows = documents.scan_rows
        stock_by_symbol = {row.symbol: row for row in latest_run.rows}
        stock_fields = {
            "rs_rating": "overall_rs",
            "rs_rating_1m": "rs_1m",
            "rs_rating_3m": "rs_3m",
            "rs_rating_12m": "rs_12m",
        }
        expected_identity = {
            "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
            "market_rs_run_id": latest_run.id,
            "rs_as_of_date": through_date.isoformat(),
            "rs_universe_size": latest_run.eligible_symbol_count,
        }
        compared = 0
        for static_row in sorted(
            scan_rows,
            key=lambda row: str(row.get("symbol")),
        ):
            stock = stock_by_symbol.get(static_row.get("symbol"))
            for field, expected in expected_identity.items():
                if static_row.get(field) != expected:
                    errors.append(
                        f"Static stock {static_row.get('symbol')}.{field}="
                        f"{static_row.get(field)!r}; expected {expected!r}."
                    )
            if stock is None:
                if any(static_row.get(field) is not None for field in stock_fields):
                    errors.append(
                        f"Static stock {static_row.get('symbol')} carries RS ratings "
                        "but is absent from the canonical Market RS run."
                    )
                continue
            compared += 1
            for static_field, stock_field in stock_fields.items():
                if static_row.get(static_field) != getattr(stock, stock_field):
                    errors.append(
                        f"Live/static stock mismatch for "
                        f"{stock.symbol}.{static_field}."
                    )
        if latest_run.rows and compared == 0:
            errors.append(
                "No staged Scan rows overlap the canonical Market RS run."
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
            if exc.reason_code is StaticGroupsRRGUnavailableReason.NOT_ENABLED:
                if rrg_path.is_file():
                    payload = self._json_file(rrg_path)
                    if payload.get("available") or payload.get("payload"):
                        errors.append(
                            "Static RRG contains coordinates despite RRG not "
                            "being enabled for this market."
                        )
                return "not_enabled"
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
    "StaticArtifactFingerprint",
    "StaticArtifactValidationResult",
    "StaticRsArtifactDocuments",
]
