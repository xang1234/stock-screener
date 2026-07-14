"""Cross-runtime field contract for scan filtering, sorting, and builder UX."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal


FilterKind = Literal["range", "categorical", "boolean", "text"]
FieldValueType = Literal["number", "date", "string", "boolean"]
_CONTRACT_PATH = Path(__file__).with_name("scan_filter_fields.json")


@dataclass(frozen=True, slots=True)
class ScanFieldCapability:
    """One logical scan field loaded from the shared browser/backend contract."""

    field: str
    filter_kind: FilterKind | None = None
    value_type: FieldValueType = "string"
    sortable: bool = False
    api_filter: bool = True
    legacy_key: str | None = None


def _default_value_type(kind: FilterKind | None) -> FieldValueType:
    if kind == "range":
        return "number"
    if kind == "boolean":
        return "boolean"
    return "string"


def _load_contract() -> tuple[ScanFieldCapability, ...]:
    payload = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("fields"), list
    ):
        raise RuntimeError("Unsupported scan filter field contract")

    capabilities: list[ScanFieldCapability] = []
    for raw in payload["fields"]:
        if not isinstance(raw, dict):
            raise RuntimeError("Scan filter field entries must be objects")
        field = raw.get("field")
        kind = raw.get("kind")
        if not isinstance(field, str) or not field:
            raise RuntimeError("Scan filter fields must use non-empty names")
        if kind not in {None, "range", "categorical", "boolean", "text"}:
            raise RuntimeError(f"Unsupported scan filter kind for {field!r}")

        value_type = raw.get("value_type", _default_value_type(kind))
        if value_type not in {"number", "date", "string", "boolean"}:
            raise RuntimeError(f"Unsupported value type for {field!r}")

        capabilities.append(
            ScanFieldCapability(
                field=field,
                filter_kind=kind,
                value_type=value_type,
                sortable=raw.get("sortable") is True,
                api_filter=raw.get("api_filter") is not False,
                legacy_key=raw.get("legacy_key"),
            )
        )
    return tuple(capabilities)


SCAN_FIELD_CAPABILITIES: Final = _load_contract()
_capability_map = {item.field: item for item in SCAN_FIELD_CAPABILITIES}
if len(_capability_map) != len(SCAN_FIELD_CAPABILITIES):
    raise RuntimeError("Duplicate logical scan field capability")
FIELD_CAPABILITIES: Final = MappingProxyType(_capability_map)

ALL_FILTER_FIELD_KINDS: Final = MappingProxyType(
    {
        item.field: item.filter_kind
        for item in SCAN_FIELD_CAPABILITIES
        if item.filter_kind is not None
    }
)
FILTER_FIELD_KINDS: Final = MappingProxyType(
    {
        item.field: item.filter_kind
        for item in SCAN_FIELD_CAPABILITIES
        if item.api_filter and item.filter_kind is not None
    }
)
RANGE_FIELDS: Final = frozenset(
    field for field, kind in FILTER_FIELD_KINDS.items() if kind == "range"
)
CATEGORICAL_FIELDS: Final = frozenset(
    field for field, kind in FILTER_FIELD_KINDS.items() if kind == "categorical"
)
BOOLEAN_FIELDS: Final = frozenset(
    field for field, kind in FILTER_FIELD_KINDS.items() if kind == "boolean"
)
TEXT_FIELDS: Final = frozenset(
    field for field, kind in FILTER_FIELD_KINDS.items() if kind == "text"
)
SORT_FIELDS: Final = frozenset(
    item.field for item in SCAN_FIELD_CAPABILITIES if item.sortable
)
LEGACY_RANGE_FILTER_FIELDS: Final = MappingProxyType(
    {
        item.legacy_key: item.field
        for item in SCAN_FIELD_CAPABILITIES
        if item.legacy_key and item.filter_kind == "range"
    }
)
LEGACY_BOOLEAN_FILTER_FIELDS: Final = MappingProxyType(
    {
        item.legacy_key: item.field
        for item in SCAN_FIELD_CAPABILITIES
        if item.legacy_key and item.filter_kind == "boolean"
    }
)

__all__ = [
    "ALL_FILTER_FIELD_KINDS",
    "BOOLEAN_FIELDS",
    "CATEGORICAL_FIELDS",
    "FIELD_CAPABILITIES",
    "FILTER_FIELD_KINDS",
    "LEGACY_BOOLEAN_FILTER_FIELDS",
    "LEGACY_RANGE_FILTER_FIELDS",
    "RANGE_FIELDS",
    "SCAN_FIELD_CAPABILITIES",
    "SORT_FIELDS",
    "TEXT_FIELDS",
    "ScanFieldCapability",
]
