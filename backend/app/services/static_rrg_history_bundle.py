"""Compact rolling history bundle for static-site RRG builds."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.models.industry import IBDGroupRank


STATIC_RRG_HISTORY_SCHEMA_VERSION = "static-rrg-history-v1"
STATIC_RRG_HISTORY_RETENTION_DAYS = 420

_ROW_FIELDS = (
    "industry_group",
    "date",
    "rank",
    "avg_rs_rating",
    "median_rs_rating",
    "weighted_avg_rs_rating",
    "rs_std_dev",
    "num_stocks",
    "num_stocks_rs_above_80",
    "top_symbol",
    "top_rs_rating",
)


class StaticRRGHistoryBundleError(ValueError):
    """Raised when a rolling RRG history bundle is invalid."""


@dataclass(frozen=True)
class StaticRRGHistoryBundleService:
    """Import/export only the group-rank rows needed to carry static RRG forward."""

    retention_days: int = STATIC_RRG_HISTORY_RETENTION_DAYS

    def export_bundle(
        self,
        db: Session,
        *,
        market: str,
        output_path: Path,
        through_date: date,
    ) -> dict[str, Any]:
        normalized_market = _normalize_market(market)
        cutoff = through_date - timedelta(days=max(1, int(self.retention_days)))
        rows = (
            db.query(IBDGroupRank)
            .filter(
                IBDGroupRank.market == normalized_market,
                IBDGroupRank.date >= cutoff,
                IBDGroupRank.date <= through_date,
            )
            .order_by(IBDGroupRank.date.asc(), IBDGroupRank.rank.asc())
            .all()
        )
        if not rows:
            raise StaticRRGHistoryBundleError(
                f"No group-rank history is available for market {normalized_market}."
            )

        payload = {
            "schema_version": STATIC_RRG_HISTORY_SCHEMA_VERSION,
            "market": normalized_market,
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "through_date": through_date.isoformat(),
            "retention_days": int(self.retention_days),
            "rows": [_serialize_row(row) for row in rows],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_payload(output_path, payload)
        return {
            "path": str(output_path),
            "market": normalized_market,
            "through_date": through_date.isoformat(),
            "rows": len(rows),
            "dates": len({row.date for row in rows}),
        }

    def import_bundle(
        self,
        db: Session,
        *,
        market: str,
        input_path: Path,
    ) -> dict[str, Any]:
        normalized_market = _normalize_market(market)
        payload = _read_payload(input_path)
        rows = _validate_payload(payload, expected_market=normalized_market)
        row_dates = [row["date"] for row in rows]
        first_date = min(row_dates)
        last_date = max(row_dates)

        try:
            db.query(IBDGroupRank).filter(
                IBDGroupRank.market == normalized_market,
                IBDGroupRank.date >= first_date,
                IBDGroupRank.date <= last_date,
            ).delete(synchronize_session=False)
            db.bulk_save_objects(
                [IBDGroupRank(market=normalized_market, **row) for row in rows]
            )
            db.commit()
        except Exception:
            db.rollback()
            raise

        return {
            "path": str(input_path),
            "market": normalized_market,
            "through_date": last_date.isoformat(),
            "rows": len(rows),
            "dates": len(set(row_dates)),
        }


def _normalize_market(market: str) -> str:
    normalized = str(market or "").strip().upper()
    if not normalized:
        raise StaticRRGHistoryBundleError("RRG history market is required.")
    return normalized


def _serialize_row(row: IBDGroupRank) -> dict[str, Any]:
    payload = {field: getattr(row, field) for field in _ROW_FIELDS}
    payload["date"] = row.date.isoformat()
    return payload


def _validate_payload(
    payload: Any,
    *,
    expected_market: str,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise StaticRRGHistoryBundleError("RRG history bundle must be a JSON object.")
    if payload.get("schema_version") != STATIC_RRG_HISTORY_SCHEMA_VERSION:
        raise StaticRRGHistoryBundleError("Unsupported RRG history bundle schema version.")
    bundle_market = _normalize_market(payload.get("market"))
    if bundle_market != expected_market:
        raise StaticRRGHistoryBundleError(
            f"RRG history bundle market {bundle_market} does not match {expected_market}."
        )
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise StaticRRGHistoryBundleError("RRG history bundle contains no rows.")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[date, str]] = set()
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise StaticRRGHistoryBundleError("RRG history row must be an object.")
        try:
            row_date = date.fromisoformat(str(raw["date"]))
            industry_group = str(raw["industry_group"]).strip()
            rank = int(raw["rank"])
            avg_rs_rating = float(raw["avg_rs_rating"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StaticRRGHistoryBundleError("Malformed RRG history row.") from exc
        if not industry_group or rank < 1:
            raise StaticRRGHistoryBundleError("Malformed RRG history row identity.")
        key = (row_date, industry_group)
        if key in seen:
            raise StaticRRGHistoryBundleError(
                f"Duplicate RRG history row for {industry_group} on {row_date}."
            )
        seen.add(key)
        rows.append(
            {
                "industry_group": industry_group,
                "date": row_date,
                "rank": rank,
                "avg_rs_rating": avg_rs_rating,
                "median_rs_rating": _optional_float(raw.get("median_rs_rating")),
                "weighted_avg_rs_rating": _optional_float(raw.get("weighted_avg_rs_rating")),
                "rs_std_dev": _optional_float(raw.get("rs_std_dev")),
                "num_stocks": int(raw.get("num_stocks") or 0),
                "num_stocks_rs_above_80": int(raw.get("num_stocks_rs_above_80") or 0),
                "top_symbol": str(raw["top_symbol"]) if raw.get("top_symbol") else None,
                "top_rs_rating": _optional_float(raw.get("top_rs_rating")),
            }
        )
    return rows


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _read_payload(path: Path) -> Any:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticRRGHistoryBundleError(
            f"Unable to read RRG history bundle {path}: {exc}"
        ) from exc


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        return
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")


__all__ = [
    "STATIC_RRG_HISTORY_RETENTION_DAYS",
    "STATIC_RRG_HISTORY_SCHEMA_VERSION",
    "StaticRRGHistoryBundleError",
    "StaticRRGHistoryBundleService",
]
