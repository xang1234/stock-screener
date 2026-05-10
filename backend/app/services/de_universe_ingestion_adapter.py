"""DE market universe ingestion adapter with deterministic canonicalization.

Germany lists primarily on Deutsche Börse Xetra (XETR) with secondary trading
on Frankfurt floor (XFRA / FWB). Source rows arrive carrying ticker symbols
shaped like ``SAP``, ``SIE.DE``, or ``ALV.F``; this adapter canonicalizes them
to ``<TICKER>.DE`` (Xetra) or ``<TICKER>.F`` (Frankfurt) and emits a single
deterministic row per canonical symbol.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from .security_master_service import security_master_resolver

_DE_EXCHANGE_ALIASES: dict[str, str] = {
    "XETR": "XETR",
    "XETRA": "XETR",
    "XFRA": "XFRA",
    "FRA": "XFRA",
    "FWB": "XFRA",
}

# Sources that are considered policy-approved for DE universe ingestion.
_APPROVED_DE_SOURCES: frozenset[str] = frozenset(
    {
        "dbg_official",
        "xetra_official",
        "deutsche_boerse_official",
        "de_manual_csv",
        "de_reference_bundle",
    }
)

# German tickers are alphanumeric, typically 3-4 chars (SAP, BMW, 1COV, RHM)
# with occasional preferred-share suffix digits (HEN3, MUV2). Cap at 8 chars
# to leave headroom for ETF tickers like EXS1 / iShares variants.
_DE_LOCAL_CODE_RE = re.compile(r"^[A-Z0-9]{1,8}$")


@dataclass(frozen=True)
class DECanonicalUniverseRow:
    """Canonical DE row emitted by the ingestion adapter."""

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
class DERejectedUniverseRow:
    """Rejected DE row with reason."""

    source_row_number: int
    source_symbol: str
    reason: str


@dataclass(frozen=True)
class DECanonicalizationResult:
    """Canonicalization output, split into accepted and rejected rows."""

    canonical_rows: tuple[DECanonicalUniverseRow, ...]
    rejected_rows: tuple[DERejectedUniverseRow, ...]


class DEUniverseIngestionAdapter:
    """Normalize and validate DE universe rows for deterministic snapshots."""

    @staticmethod
    def normalize_source_name(source_name: str) -> str:
        normalized = (source_name or "").strip().lower().replace("-", "_")
        if not normalized:
            raise ValueError("source_name must be provided")
        return normalized

    @classmethod
    def is_approved_source(cls, source_name: str) -> bool:
        normalized = cls.normalize_source_name(source_name)
        if normalized in _APPROVED_DE_SOURCES:
            return True
        return (
            normalized.startswith("dbg_")
            or normalized.startswith("xetra_")
            or normalized.startswith("de_")
        )

    @staticmethod
    def _normalize_source_symbol(raw_symbol: Any) -> str:
        symbol = str(raw_symbol or "").strip().upper().replace(" ", "")
        if symbol.startswith("$"):
            symbol = symbol[1:]
        return symbol

    @staticmethod
    def _normalize_exchange(raw_exchange: Any, *, default: str = "XETR") -> str:
        exchange = str(raw_exchange or "").strip().upper() or default
        normalized = _DE_EXCHANGE_ALIASES.get(exchange)
        if normalized is None:
            raise ValueError(
                f"Unsupported DE exchange '{exchange}'. Expected one of: "
                "XETR, XETRA, XFRA, FRA, FWB"
            )
        return normalized

    @classmethod
    def _normalize_de_local_code(cls, source_symbol: str) -> tuple[str, str | None]:
        """Return (local_code, explicit_suffix) — None when the symbol has no
        explicit ``.DE``/``.F`` suffix or exchange-prefix marker. The caller
        falls back to the row's exchange column in that case so that a naked
        ``ALV`` with exchange ``XFRA`` lands as ``ALV.F``, not ``ALV.DE``.
        """
        token = source_symbol
        suffix: str | None = None
        for prefix in ("XETR:", "XETRA:", "XFRA:", "FRA:", "FWB:"):
            if token.startswith(prefix):
                token = token[len(prefix):]
                suffix = ".F" if prefix in ("XFRA:", "FRA:", "FWB:") else ".DE"
                break
        if token.endswith(".DE"):
            token = token[:-3]
            suffix = ".DE"
        elif token.endswith(".F"):
            token = token[:-2]
            suffix = ".F"

        if not _DE_LOCAL_CODE_RE.fullmatch(token):
            raise ValueError(
                f"Invalid DE symbol '{source_symbol}'. "
                "Expected 1-8 alphanumeric chars with optional .DE or .F suffix."
            )
        return token, suffix

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

        try:
            return float(raw) * multiplier
        except ValueError:
            return None

    @staticmethod
    def _hash_payload(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _selection_key(row: DECanonicalUniverseRow) -> tuple[str, int]:
        return (row.source_symbol, row.source_row_number)

    @staticmethod
    def _prefer_text(primary: str, fallback: str) -> str:
        return primary if primary.strip() else fallback

    def _canonical_payload(
        self,
        row: DECanonicalUniverseRow,
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
        first: DECanonicalUniverseRow,
        second: DECanonicalUniverseRow,
    ) -> DECanonicalUniverseRow:
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
    ) -> DECanonicalizationResult:
        normalized_source_name = self.normalize_source_name(source_name)
        if not self.is_approved_source(normalized_source_name):
            raise ValueError(
                f"Unapproved DE source '{source_name}'. "
                "Use an approved DE source identifier."
            )

        normalized_snapshot_id = (snapshot_id or "").strip()
        if not normalized_snapshot_id:
            raise ValueError("snapshot_id must be provided")

        metadata = dict(source_metadata or {})
        canonical_by_symbol: dict[str, DECanonicalUniverseRow] = {}
        rejected_rows: list[DERejectedUniverseRow] = []

        for index, raw_row in enumerate(rows, start=1):
            source_symbol = self._normalize_source_symbol(
                raw_row.get("symbol")
                or raw_row.get("local_code")
                or raw_row.get("ticker")
            )
            if not source_symbol:
                rejected_rows.append(
                    DERejectedUniverseRow(
                        source_row_number=index,
                        source_symbol="",
                        reason="Missing symbol/local_code/ticker",
                    )
                )
                continue

            try:
                local_code, explicit_suffix = self._normalize_de_local_code(source_symbol)
                # Resolution policy: an explicit suffix on the symbol
                # (``.DE`` / ``.F``) wins over the row's exchange column.
                # When the symbol has no explicit suffix, the exchange
                # column decides — naked ``ALV`` with ``XFRA`` becomes
                # ``ALV.F``. With neither, default to Xetra.
                row_exchange = self._normalize_exchange(
                    raw_row.get("exchange"), default="XETR",
                )
                if explicit_suffix is not None:
                    exchange = "XETR" if explicit_suffix == ".DE" else "XFRA"
                    symbol_suffix = explicit_suffix
                else:
                    exchange = row_exchange
                    symbol_suffix = ".DE" if exchange == "XETR" else ".F"

                identity = security_master_resolver.resolve_identity(
                    symbol=f"{local_code}{symbol_suffix}",
                    market="DE",
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
                canonical_exchange = identity.exchange or exchange
                canonical_payload = {
                    "symbol": identity.canonical_symbol,
                    "market": identity.market,
                    "exchange": canonical_exchange,
                    "local_code": identity.local_code,
                    "name": row_name,
                    "sector": row_sector,
                    "industry": row_industry,
                    "market_cap": row_market_cap,
                    "source_name": normalized_source_name,
                    "snapshot_id": normalized_snapshot_id,
                }
                canonical_row = DECanonicalUniverseRow(
                    symbol=identity.canonical_symbol,
                    name=row_name,
                    market=identity.market,
                    exchange=canonical_exchange,
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
            except ValueError as exc:
                rejected_rows.append(
                    DERejectedUniverseRow(
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
        return DECanonicalizationResult(
            canonical_rows=canonical_rows,
            rejected_rows=rejected_rows_tuple,
        )


de_universe_ingestion_adapter = DEUniverseIngestionAdapter()
