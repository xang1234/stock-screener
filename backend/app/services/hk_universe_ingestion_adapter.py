"""HK market universe ingestion adapter with deterministic canonicalization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from .security_master_service import security_master_resolver

_HK_EXCHANGE_ALIASES: dict[str, str] = {
    "HKEX": "XHKG",
    "SEHK": "XHKG",
    "XHKG": "XHKG",
}

# Sources that are considered policy-approved for HK universe ingestion.
_APPROVED_HK_SOURCES: frozenset[str] = frozenset(
    {
        "hkex_official",
        "sehk_official",
        "xhkg_official",
        "hk_manual_csv",
        "hk_reference_bundle",
    }
)

_HK_NUMERIC_CODE_RE = re.compile(r"^[0-9]{1,8}$")


@dataclass(frozen=True)
class HKCanonicalUniverseRow:
    """Canonical HK row emitted by the ingestion adapter."""

    symbol: str
    name: str
    market: str
    exchange: str
    currency: str
    timezone: str
    local_code: str
    sector: str
    industry: str
    market_cap: float | None
    source_name: str
    source_symbol: str
    source_row_number: int
    snapshot_id: str
    snapshot_as_of: str | None
    source_metadata: dict[str, Any]
    lineage_hash: str
    row_hash: str


@dataclass(frozen=True)
class HKRejectedUniverseRow:
    """Rejected HK row with reason."""

    source_row_number: int
    source_symbol: str
    reason: str


@dataclass(frozen=True)
class HKCanonicalizationResult:
    """Canonicalization output, split into accepted and rejected rows."""

    canonical_rows: tuple[HKCanonicalUniverseRow, ...]
    rejected_rows: tuple[HKRejectedUniverseRow, ...]


class HKUniverseIngestionAdapter:
    """Normalize and validate HK universe rows for deterministic snapshots."""

    @staticmethod
    def normalize_source_name(source_name: str) -> str:
        normalized = (source_name or "").strip().lower().replace("-", "_")
        if not normalized:
            raise ValueError("source_name must be provided")
        return normalized

    @classmethod
    def is_approved_source(cls, source_name: str) -> bool:
        normalized = cls.normalize_source_name(source_name)
        if normalized in _APPROVED_HK_SOURCES:
            return True
        return normalized.startswith("hkex_") or normalized.startswith("sehk_")

    @staticmethod
    def _normalize_source_symbol(raw_symbol: Any) -> str:
        symbol = str(raw_symbol or "").strip().upper().replace(" ", "")
        if symbol.startswith("$"):
            symbol = symbol[1:]
        return symbol

    @staticmethod
    def _normalize_exchange(raw_exchange: Any) -> str:
        exchange = str(raw_exchange or "").strip().upper() or "XHKG"
        normalized = _HK_EXCHANGE_ALIASES.get(exchange)
        if normalized is None:
            raise ValueError(
                f"Unsupported HK exchange '{exchange}'. Expected one of: "
                "HKEX, SEHK, XHKG"
            )
        return normalized

    @staticmethod
    def _normalize_hk_local_code(source_symbol: str) -> str:
        token = source_symbol
        for prefix in ("HKEX:", "SEHK:", "XHKG:"):
            if token.startswith(prefix):
                token = token[len(prefix):]
                break
        if token.endswith(".HK"):
            token = token[:-3]

        if not _HK_NUMERIC_CODE_RE.fullmatch(token):
            raise ValueError(
                f"Invalid HK symbol '{source_symbol}'. "
                "Expected numeric local code with optional .HK suffix."
            )

        digits = token.lstrip("0")
        if not digits:
            raise ValueError(
                f"Invalid HK symbol '{source_symbol}'. Local code cannot be all zeros."
            )
        if len(digits) <= 4:
            return digits.zfill(4)
        if len(digits) <= 5:
            return digits
        raise ValueError(
            f"Invalid HK local code '{source_symbol}'. "
            "Only 1-5 significant digits are supported."
        )

    @staticmethod
    def _parse_market_cap(raw_value: Any) -> float | None:
        if raw_value is None:
            return None
        if isinstance(raw_value, (int, float)):
            return float(raw_value)

        raw = str(raw_value).strip().upper().replace(",", "")
        if not raw or raw == "-":
            return None

        multiplier = 1.0
        if raw.endswith("B"):
            multiplier = 1e9
            raw = raw[:-1]
        elif raw.endswith("M"):
            multiplier = 1e6
            raw = raw[:-1]
        elif raw.endswith("K"):
            multiplier = 1e3
            raw = raw[:-1]

        return float(raw) * multiplier

    @staticmethod
    def _hash_payload(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def canonicalize_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> HKCanonicalizationResult:
        normalized_source_name = self.normalize_source_name(source_name)
        if not self.is_approved_source(normalized_source_name):
            raise ValueError(
                f"Unapproved HK source '{source_name}'. "
                "Use an approved HK source identifier."
            )

        normalized_snapshot_id = (snapshot_id or "").strip()
        if not normalized_snapshot_id:
            raise ValueError("snapshot_id must be provided")

        metadata = dict(source_metadata or {})
        canonical_by_symbol: dict[str, HKCanonicalUniverseRow] = {}
        rejected_rows: list[HKRejectedUniverseRow] = []

        for index, raw_row in enumerate(rows, start=1):
            source_symbol = self._normalize_source_symbol(
                raw_row.get("symbol")
                or raw_row.get("local_code")
                or raw_row.get("ticker")
            )
            if not source_symbol:
                rejected_rows.append(
                    HKRejectedUniverseRow(
                        source_row_number=index,
                        source_symbol="",
                        reason="Missing symbol/local_code/ticker",
                    )
                )
                continue

            try:
                exchange = self._normalize_exchange(raw_row.get("exchange"))
                local_code = self._normalize_hk_local_code(source_symbol)
                identity = security_master_resolver.resolve_identity(
                    symbol=f"{local_code}.HK",
                    market="HK",
                    exchange=exchange,
                    local_code=local_code,
                )
                row_name = str(raw_row.get("name") or raw_row.get("company") or "").strip()
                row_sector = str(raw_row.get("sector") or "").strip()
                row_industry = str(raw_row.get("industry") or "").strip()
                row_market_cap = self._parse_market_cap(
                    raw_row.get("market_cap") or raw_row.get("marketcap")
                )

                lineage_payload = {
                    "source_name": normalized_source_name,
                    "snapshot_id": normalized_snapshot_id,
                    "source_row_number": index,
                    "source_symbol": source_symbol,
                    "canonical_symbol": identity.canonical_symbol,
                }
                canonical_payload = {
                    "symbol": identity.canonical_symbol,
                    "market": identity.market,
                    "exchange": identity.exchange,
                    "local_code": identity.local_code,
                    "name": row_name,
                    "sector": row_sector,
                    "industry": row_industry,
                    "market_cap": row_market_cap,
                    "source_name": normalized_source_name,
                    "snapshot_id": normalized_snapshot_id,
                }
                canonical_row = HKCanonicalUniverseRow(
                    symbol=identity.canonical_symbol,
                    name=row_name,
                    market=identity.market,
                    exchange=identity.exchange or "XHKG",
                    currency=identity.currency,
                    timezone=identity.timezone,
                    local_code=identity.local_code,
                    sector=row_sector,
                    industry=row_industry,
                    market_cap=row_market_cap,
                    source_name=normalized_source_name,
                    source_symbol=source_symbol,
                    source_row_number=index,
                    snapshot_id=normalized_snapshot_id,
                    snapshot_as_of=snapshot_as_of,
                    source_metadata=metadata,
                    lineage_hash=self._hash_payload(lineage_payload),
                    row_hash=self._hash_payload(canonical_payload),
                )

                existing = canonical_by_symbol.get(canonical_row.symbol)
                if existing is None:
                    canonical_by_symbol[canonical_row.symbol] = canonical_row
                else:
                    existing_key = (existing.source_symbol, existing.source_row_number)
                    candidate_key = (
                        canonical_row.source_symbol,
                        canonical_row.source_row_number,
                    )
                    if candidate_key < existing_key:
                        canonical_by_symbol[canonical_row.symbol] = canonical_row
            except Exception as exc:
                rejected_rows.append(
                    HKRejectedUniverseRow(
                        source_row_number=index,
                        source_symbol=source_symbol,
                        reason=str(exc),
                    )
                )

        canonical_rows = tuple(
            sorted(canonical_by_symbol.values(), key=lambda row: row.symbol)
        )
        rejected_rows_tuple = tuple(
            sorted(rejected_rows, key=lambda row: row.source_row_number)
        )
        return HKCanonicalizationResult(
            canonical_rows=canonical_rows,
            rejected_rows=rejected_rows_tuple,
        )


hk_universe_ingestion_adapter = HKUniverseIngestionAdapter()

