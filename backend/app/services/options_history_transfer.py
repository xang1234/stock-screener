"""Portable, checksummed aggregate history for ephemeral static builds."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, date, datetime
from typing import Any

from app.use_cases.options_analytics import (
    OPTIONS_ANALYTICS_CALCULATION_VERSION,
    OPTIONS_ANALYTICS_SCHEMA_VERSION,
)

OPTIONS_HISTORY_TRANSFER_SCHEMA_VERSION = "options-history-transfer-v1"


class OptionsHistoryTransferError(ValueError):
    """Raised when a history transfer bundle is unsafe or incompatible."""


def _canonical_payload(bundle: dict[str, Any]) -> bytes:
    content = {key: value for key, value in bundle.items() if key != "payload_checksum"}
    try:
        encoded = json.dumps(
            content,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise OptionsHistoryTransferError(
            "non-finite or invalid history payload"
        ) from exc
    return encoded.encode("utf-8")


def _checksum(bundle: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(bundle)).hexdigest()


def _require_finite(value: Any, *, location: str = "bundle") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise OptionsHistoryTransferError(f"non-finite value at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite(item, location=f"{location}[{index}]")


def _parse_date(value: Any, *, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise OptionsHistoryTransferError(f"invalid {field}") from exc


class OptionsHistoryTransfer:
    def __init__(
        self,
        repository: Any,
        *,
        published_reader: Any | None = None,
        market: str = "US",
        calculation_version: str = OPTIONS_ANALYTICS_CALCULATION_VERSION,
    ) -> None:
        self._repository = repository
        self._published_reader = published_reader or repository
        self._market = market.strip().upper()
        self._calculation_version = calculation_version

    def export_bundle(
        self,
        *,
        exported_at: datetime | None = None,
        required_published_run_id: int | None = None,
    ) -> dict[str, Any]:
        if required_published_run_id is not None:
            published = self._published_reader.get_published_run(
                self._market,
                self._calculation_version,
            )
            if published is None or published.id != required_published_run_id:
                raise OptionsHistoryTransferError(
                    "required newly published options run is not current"
                )
        observations = [
            dict(row)
            for row in self._repository.export_history_observations(
                self._market,
                self._calculation_version,
            )
        ]
        observations.sort(
            key=lambda row: (
                row["as_of_date"],
                row["external_source_feature_run_key"],
                row["symbol"],
            )
        )
        memberships: dict[str, dict[str, Any]] = {}
        for row in observations:
            if row.get("candidate_kind") != "current":
                continue
            symbol = str(row["symbol"]).strip().upper()
            ranks = [
                int(rank)
                for rank in (row.get("candidate_rank"), row.get("leader_rank"))
                if rank is not None
            ]
            candidate = {
                "symbol": symbol,
                "as_of_date": row["as_of_date"],
                "prior_best_rank": min(ranks) if ranks else 10_000,
            }
            incumbent = memberships.get(symbol)
            if incumbent is None or (
                candidate["as_of_date"],
                -candidate["prior_best_rank"],
            ) > (incumbent["as_of_date"], -incumbent["prior_best_rank"]):
                memberships[symbol] = candidate
        timestamp = (exported_at or datetime.now(UTC)).astimezone(UTC)
        bundle = {
            "schema_version": OPTIONS_HISTORY_TRANSFER_SCHEMA_VERSION,
            "calculation_version": self._calculation_version,
            "market": self._market,
            "exported_at": timestamp.isoformat().replace("+00:00", "Z"),
            "observations": observations,
            "last_current_memberships": [
                memberships[symbol] for symbol in sorted(memberships)
            ],
        }
        bundle["payload_checksum"] = _checksum(bundle)
        return bundle

    def import_bundle(
        self,
        bundle: dict[str, Any],
        *,
        today: date | None = None,
    ) -> dict[str, Any]:
        if not isinstance(bundle, dict):
            raise OptionsHistoryTransferError("history bundle must be an object")
        if bundle.get("schema_version") != OPTIONS_HISTORY_TRANSFER_SCHEMA_VERSION:
            raise OptionsHistoryTransferError("unsupported history schema version")
        if str(bundle.get("market") or "").upper() != self._market:
            raise OptionsHistoryTransferError("history market is incompatible")
        if bundle.get("calculation_version") != self._calculation_version:
            raise OptionsHistoryTransferError(
                "history calculation version is incompatible"
            )
        observations = bundle.get("observations")
        memberships = bundle.get("last_current_memberships")
        if not isinstance(observations, list) or not isinstance(memberships, list):
            raise OptionsHistoryTransferError("history observations are invalid")
        _require_finite(bundle)
        forbidden = {"strike_points", "raw_contracts", "raw_contract"}
        identities: set[tuple[str, str]] = set()
        run_identity: dict[str, tuple[str, str, str]] = {}
        current_memberships: dict[str, dict[str, Any]] = {}
        required_observation_fields = {
            "external_source_feature_run_key",
            "as_of_date",
            "schema_version",
            "provider",
            "symbol",
            "candidate_kind",
            "observation_state",
            "core_valid",
        }
        cutoff = today or datetime.now(UTC).date()
        for row in observations:
            if not isinstance(row, dict):
                raise OptionsHistoryTransferError(
                    "history observation must be an object"
                )
            if forbidden.intersection(row):
                raise OptionsHistoryTransferError("history contains forbidden raw data")
            missing = required_observation_fields.difference(row)
            if missing:
                raise OptionsHistoryTransferError(
                    "history observation fields are missing"
                )
            if row["schema_version"] != OPTIONS_ANALYTICS_SCHEMA_VERSION:
                raise OptionsHistoryTransferError(
                    "history observation schema is incompatible"
                )
            if row["candidate_kind"] not in {"current", "continuity"}:
                raise OptionsHistoryTransferError("history candidate kind is invalid")
            identity = (
                str(row.get("external_source_feature_run_key") or ""),
                str(row.get("symbol") or "").strip().upper(),
            )
            if not all(identity):
                raise OptionsHistoryTransferError(
                    "history observation identity is missing"
                )
            if identity in identities:
                raise OptionsHistoryTransferError(
                    "duplicate history observation identity"
                )
            identities.add(identity)
            as_of_date = _parse_date(row.get("as_of_date"), field="observation date")
            if as_of_date > cutoff:
                raise OptionsHistoryTransferError("future history observation date")
            observation_at = row.get("observation_at")
            if observation_at is not None:
                try:
                    observed_date = datetime.fromisoformat(
                        str(observation_at).replace("Z", "+00:00")
                    ).date()
                except ValueError as exc:
                    raise OptionsHistoryTransferError(
                        "invalid history observation timestamp"
                    ) from exc
                if observed_date > cutoff:
                    raise OptionsHistoryTransferError(
                        "future history observation timestamp"
                    )
            external_key = identity[0]
            metadata = (
                str(row["as_of_date"]),
                str(row["schema_version"]),
                str(row["provider"]),
            )
            if external_key in run_identity and run_identity[external_key] != metadata:
                raise OptionsHistoryTransferError("inconsistent history run identity")
            run_identity[external_key] = metadata
            if row["candidate_kind"] == "current":
                ranks = [
                    int(rank)
                    for rank in (row.get("candidate_rank"), row.get("leader_rank"))
                    if rank is not None
                ]
                membership = {
                    "symbol": identity[1],
                    "as_of_date": row["as_of_date"],
                    "prior_best_rank": min(ranks) if ranks else 10_000,
                }
                incumbent = current_memberships.get(identity[1])
                if incumbent is None or (
                    membership["as_of_date"],
                    -membership["prior_best_rank"],
                ) > (incumbent["as_of_date"], -incumbent["prior_best_rank"]):
                    current_memberships[identity[1]] = membership
        membership_symbols: set[str] = set()
        for membership in memberships:
            if not isinstance(membership, dict):
                raise OptionsHistoryTransferError(
                    "history membership must be an object"
                )
            symbol = str(membership.get("symbol") or "").strip().upper()
            if not symbol or symbol in membership_symbols:
                raise OptionsHistoryTransferError(
                    "duplicate or invalid history membership"
                )
            membership_symbols.add(symbol)
            if (
                _parse_date(membership.get("as_of_date"), field="membership date")
                > cutoff
            ):
                raise OptionsHistoryTransferError("future history membership date")
        expected_memberships = [
            current_memberships[symbol] for symbol in sorted(current_memberships)
        ]
        normalized_memberships = sorted(
            (
                {
                    "symbol": str(row.get("symbol") or "").strip().upper(),
                    "as_of_date": row.get("as_of_date"),
                    "prior_best_rank": row.get("prior_best_rank"),
                }
                for row in memberships
            ),
            key=lambda row: row["symbol"],
        )
        if normalized_memberships != expected_memberships:
            raise OptionsHistoryTransferError(
                "history memberships do not match observations"
            )
        if bundle.get("payload_checksum") != _checksum(bundle):
            raise OptionsHistoryTransferError("history payload checksum mismatch")
        return self._repository.import_history_transfer(
            tuple(dict(row) for row in observations),
            market=self._market,
            calculation_version=self._calculation_version,
            schema_version=OPTIONS_HISTORY_TRANSFER_SCHEMA_VERSION,
        )


__all__ = [
    "OPTIONS_HISTORY_TRANSFER_SCHEMA_VERSION",
    "OptionsHistoryTransfer",
    "OptionsHistoryTransferError",
]
