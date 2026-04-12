"""TW market universe ingestion adapter with deterministic canonicalization."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from .security_master_service import security_master_resolver

_TW_EXCHANGE_ALIASES: dict[str, str] = {
    "TWSE": "TWSE",
    "XTAI": "TWSE",
    "TPEX": "TPEX",
    "TWO": "TPEX",
}

# Sources that are considered policy-approved for TW universe ingestion.
_APPROVED_TW_SOURCES: frozenset[str] = frozenset(
    {
        "twse_official",
        "tpex_official",
        "xtai_official",
        "tw_manual_csv",
        "tw_reference_bundle",
    }
)

_TW_LOCAL_CODE_RE = re.compile(r"^[0-9]{3,6}[A-Z]?$")


@dataclass(frozen=True)
class TWCanonicalUniverseRow:
    """Canonical TW row emitted by the ingestion adapter."""

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
class TWRejectedUniverseRow:
    """Rejected TW row with reason."""

    source_row_number: int
    source_symbol: str
    reason: str


@dataclass(frozen=True)
class TWCanonicalizationResult:
    """Canonicalization output, split into accepted and rejected rows."""

    canonical_rows: tuple[TWCanonicalUniverseRow, ...]
    rejected_rows: tuple[TWRejectedUniverseRow, ...]


class TWUniverseIngestionAdapter:
    """Normalize and validate TW universe rows for deterministic snapshots."""

    @staticmethod
    def normalize_source_name(source_name: str) -> str:
        normalized = (source_name or "").strip().lower().replace("-", "_")
        if not normalized:
            raise ValueError("source_name must be provided")
        return normalized

    @classmethod
    def is_approved_source(cls, source_name: str) -> bool:
        normalized = cls.normalize_source_name(source_name)
        if normalized in _APPROVED_TW_SOURCES:
            return True
        return (
            normalized.startswith("twse_")
            or normalized.startswith("tpex_")
            or normalized.startswith("xtai_")
        )

    @staticmethod
    def _normalize_source_symbol(raw_symbol: Any) -> str:
        symbol = str(raw_symbol or "").strip().upper().replace(" ", "")
        if symbol.startswith("$"):
            symbol = symbol[1:]
        return symbol

    @staticmethod
    def _infer_exchange_from_symbol(source_symbol: str) -> str | None:
        token = str(source_symbol or "").strip().upper()
        if token.startswith(("TPEX:", "TWO:")) or token.endswith(".TWO"):
            return "TPEX"
        if token.startswith(("TWSE:", "XTAI:")) or token.endswith(".TW"):
            return "TWSE"
        return None

    @classmethod
    def _normalize_exchange(cls, raw_exchange: Any, *, source_symbol: str = "") -> str:
        exchange = str(raw_exchange or "").strip().upper()
        if not exchange:
            inferred = cls._infer_exchange_from_symbol(source_symbol)
            return inferred or "TWSE"
        normalized = _TW_EXCHANGE_ALIASES.get(exchange)
        if normalized is None:
            raise ValueError(
                f"Unsupported TW exchange '{exchange}'. Expected one of: "
                "TWSE, XTAI, TPEX, TWO"
            )
        return normalized

    @staticmethod
    def _normalize_tw_local_code(source_symbol: str) -> str:
        token = source_symbol
        for prefix in ("TWSE:", "XTAI:", "TPEX:", "TWO:"):
            if token.startswith(prefix):
                token = token[len(prefix):]
                break

        if token.endswith(".TWO"):
            token = token[:-4]
        elif token.endswith(".TW"):
            token = token[:-3]

        if not _TW_LOCAL_CODE_RE.fullmatch(token):
            raise ValueError(
                f"Invalid TW symbol '{source_symbol}'. "
                "Expected TW local code with optional .TW/.TWO suffix."
            )
        return token

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

    @staticmethod
    def _selection_key(row: TWCanonicalUniverseRow) -> tuple[str, int]:
        return (row.source_symbol, row.source_row_number)

    @staticmethod
    def _prefer_text(primary: str, fallback: str) -> str:
        return primary if primary.strip() else fallback

    def _canonical_payload(
        self,
        row: TWCanonicalUniverseRow,
        *,
        name: str,
        sector: str,
        industry: str,
        market_cap: float | None,
    ) -> dict[str, Any]:
        return {
            "symbol": row.symbol,
            "market": row.market,
            "exchange": row.exchange,
            "local_code": row.local_code,
            "name": name,
            "sector": sector,
            "industry": industry,
            "market_cap": market_cap,
            "source_name": row.source_name,
            "snapshot_id": row.snapshot_id,
        }

    def _merge_duplicate_rows(
        self,
        first: TWCanonicalUniverseRow,
        second: TWCanonicalUniverseRow,
    ) -> TWCanonicalUniverseRow:
        if self._selection_key(first) <= self._selection_key(second):
            primary = first
            secondary = second
        else:
            primary = second
            secondary = first

        merged_name = self._prefer_text(primary.name, secondary.name)
        merged_sector = self._prefer_text(primary.sector, secondary.sector)
        merged_industry = self._prefer_text(primary.industry, secondary.industry)
        merged_market_cap = primary.market_cap if primary.market_cap is not None else secondary.market_cap
        merged_source_metadata = (
            primary.source_metadata if primary.source_metadata else secondary.source_metadata
        )
        merged_row_hash = self._hash_payload(
            self._canonical_payload(
                primary,
                name=merged_name,
                sector=merged_sector,
                industry=merged_industry,
                market_cap=merged_market_cap,
            )
        )
        return replace(
            primary,
            name=merged_name,
            sector=merged_sector,
            industry=merged_industry,
            market_cap=merged_market_cap,
            source_metadata=merged_source_metadata,
            row_hash=merged_row_hash,
        )

    def canonicalize_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> TWCanonicalizationResult:
        normalized_source_name = self.normalize_source_name(source_name)
        if not self.is_approved_source(normalized_source_name):
            raise ValueError(
                f"Unapproved TW source '{source_name}'. "
                "Use an approved TW source identifier."
            )

        normalized_snapshot_id = (snapshot_id or "").strip()
        if not normalized_snapshot_id:
            raise ValueError("snapshot_id must be provided")

        metadata = dict(source_metadata or {})
        canonical_by_symbol: dict[str, TWCanonicalUniverseRow] = {}
        rejected_rows: list[TWRejectedUniverseRow] = []

        for index, raw_row in enumerate(rows, start=1):
            source_symbol = self._normalize_source_symbol(
                raw_row.get("symbol")
                or raw_row.get("local_code")
                or raw_row.get("ticker")
            )
            if not source_symbol:
                rejected_rows.append(
                    TWRejectedUniverseRow(
                        source_row_number=index,
                        source_symbol="",
                        reason="Missing symbol/local_code/ticker",
                    )
                )
                continue

            try:
                exchange = self._normalize_exchange(
                    raw_row.get("exchange"),
                    source_symbol=source_symbol,
                )
                local_code = self._normalize_tw_local_code(source_symbol)
                identity = security_master_resolver.resolve_identity(
                    symbol=f"{local_code}.TW",
                    market="TW",
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
                canonical_row = TWCanonicalUniverseRow(
                    symbol=identity.canonical_symbol,
                    name=row_name,
                    market=identity.market,
                    exchange=identity.exchange or exchange,
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
                    canonical_by_symbol[canonical_row.symbol] = self._merge_duplicate_rows(
                        existing,
                        canonical_row,
                    )
            except Exception as exc:
                rejected_rows.append(
                    TWRejectedUniverseRow(
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
        return TWCanonicalizationResult(
            canonical_rows=canonical_rows,
            rejected_rows=rejected_rows_tuple,
        )


tw_universe_ingestion_adapter = TWUniverseIngestionAdapter()
