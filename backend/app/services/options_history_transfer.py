"""Portable, checksummed aggregate history for ephemeral static builds."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Protocol

from pydantic import ValidationError

from app.schemas.options_history_transfer import (
    OptionsHistoryBundle,
    OptionsHistoryObservation,
)
from app.use_cases.options_analytics import (
    OPTIONS_ANALYTICS_CALCULATION_VERSION,
    OPTIONS_ANALYTICS_SCHEMA_VERSION,
)

OPTIONS_HISTORY_TRANSFER_SCHEMA_VERSION = "options-history-transfer-v1"


class OptionsHistoryTransferError(ValueError):
    """Raised when a history transfer bundle is unsafe or incompatible."""


class OptionsHistoryRepository(Protocol):
    def export_history_observations(
        self,
        market: str,
        calculation_version: str,
    ) -> Sequence[OptionsHistoryObservation | Mapping[str, object]]: ...

    def import_history_transfer(
        self,
        observations: Sequence[OptionsHistoryObservation],
        *,
        market: str,
        calculation_version: str,
        schema_version: str,
    ) -> dict[str, int | str]: ...


class PublishedRunReader(Protocol):
    def get_published_run(self, market: str, calculation_version: str) -> object | None: ...


def _canonical_payload(bundle: Mapping[str, object]) -> bytes:
    content = {key: value for key, value in bundle.items() if key != "payload_checksum"}
    try:
        return json.dumps(
            content,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OptionsHistoryTransferError(
            "non-finite or invalid history payload"
        ) from exc


def _checksum(bundle: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_payload(bundle)).hexdigest()


class OptionsHistoryTransfer:
    def __init__(
        self,
        repository: OptionsHistoryRepository,
        *,
        published_reader: PublishedRunReader | None = None,
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
    ) -> dict[str, object]:
        if required_published_run_id is not None:
            published = self._published_reader.get_published_run(
                self._market,
                self._calculation_version,
            )
            if published is None or published.id != required_published_run_id:
                raise OptionsHistoryTransferError(
                    "required newly published options run is not current"
                )
        observations = tuple(
            sorted(
                (
                    OptionsHistoryObservation.model_validate(row)
                    for row in self._repository.export_history_observations(
                        self._market,
                        self._calculation_version,
                    )
                ),
                key=lambda row: (row.as_of_date, *row.identity),
            )
        )
        payload: dict[str, object] = {
            "schema_version": OPTIONS_HISTORY_TRANSFER_SCHEMA_VERSION,
            "calculation_version": self._calculation_version,
            "market": self._market,
            "exported_at": (exported_at or datetime.now(UTC))
            .astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "observations": [
                row.model_dump(mode="json") for row in observations
            ],
        }
        payload["payload_checksum"] = _checksum(payload)
        return payload

    def import_bundle(
        self,
        bundle: dict[str, object],
        *,
        today: date | None = None,
    ) -> dict[str, int | str]:
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
        _canonical_payload(bundle)
        try:
            parsed = OptionsHistoryBundle.model_validate(bundle)
        except ValidationError as exc:
            raise OptionsHistoryTransferError(f"invalid history payload: {exc}") from exc
        self._validate_observations(parsed.observations, today=today)
        if parsed.payload_checksum != _checksum(bundle):
            raise OptionsHistoryTransferError("history payload checksum mismatch")
        return self._repository.import_history_transfer(
            parsed.observations,
            market=self._market,
            calculation_version=self._calculation_version,
            schema_version=OPTIONS_HISTORY_TRANSFER_SCHEMA_VERSION,
        )

    @staticmethod
    def _validate_observations(
        observations: Sequence[OptionsHistoryObservation],
        *,
        today: date | None,
    ) -> None:
        cutoff = today or datetime.now(UTC).date()
        identities: set[tuple[str, str]] = set()
        runs: dict[str, tuple[date, str, str]] = {}
        for row in observations:
            if row.schema_version != OPTIONS_ANALYTICS_SCHEMA_VERSION:
                raise OptionsHistoryTransferError(
                    "history observation schema is incompatible"
                )
            if row.identity in identities:
                raise OptionsHistoryTransferError(
                    "duplicate history observation identity"
                )
            identities.add(row.identity)
            if row.as_of_date > cutoff:
                raise OptionsHistoryTransferError("future history observation date")
            if row.observation_at is not None and row.observation_at.date() > cutoff:
                raise OptionsHistoryTransferError(
                    "future history observation timestamp"
                )
            external_key = row.external_source_feature_run_key
            if external_key in runs and runs[external_key] != row.run_identity:
                raise OptionsHistoryTransferError("inconsistent history run identity")
            runs[external_key] = row.run_identity


__all__ = [
    "OPTIONS_HISTORY_TRANSFER_SCHEMA_VERSION",
    "OptionsHistoryTransfer",
    "OptionsHistoryTransferError",
]
