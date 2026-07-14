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
    filter_kinds: frozenset[FilterKind] = frozenset()
    api_filter_kinds: frozenset[FilterKind] = frozenset()
    value_type: FieldValueType = "string"
    sortable: bool = False
    legacy_key: str | None = None

    def supports(self, kind: str) -> bool:
        return kind in self.filter_kinds

    def supports_api(self, kind: str) -> bool:
        return kind in self.api_filter_kinds


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

        raw_filter_kinds = raw.get(
            "filter_kinds",
            [] if kind is None else [kind],
        )
        if not isinstance(raw_filter_kinds, list) or any(
            item not in {"range", "categorical", "boolean", "text"}
            for item in raw_filter_kinds
        ):
            raise RuntimeError(f"Unsupported filter kinds for {field!r}")
        filter_kinds = frozenset(raw_filter_kinds)
        if kind is not None and kind not in filter_kinds:
            raise RuntimeError(f"Primary filter kind missing for {field!r}")

        raw_api_filter_kinds = raw.get("api_filter_kinds")
        if raw.get("api_filter") is False:
            api_filter_kinds: frozenset[FilterKind] = frozenset()
        elif raw_api_filter_kinds is None:
            api_filter_kinds = filter_kinds
        elif not isinstance(raw_api_filter_kinds, list) or any(
            item not in filter_kinds for item in raw_api_filter_kinds
        ):
            raise RuntimeError(f"Unsupported API filter kinds for {field!r}")
        else:
            api_filter_kinds = frozenset(raw_api_filter_kinds)

        value_type = raw.get("value_type", _default_value_type(kind))
        if value_type not in {"number", "date", "string", "boolean"}:
            raise RuntimeError(f"Unsupported value type for {field!r}")

        capabilities.append(
            ScanFieldCapability(
                field=field,
                filter_kind=kind,
                filter_kinds=filter_kinds,
                api_filter_kinds=api_filter_kinds,
                value_type=value_type,
                sortable=raw.get("sortable") is True,
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
        item.field: item.filter_kinds
        for item in SCAN_FIELD_CAPABILITIES
        if item.filter_kinds
    }
)
FILTER_FIELD_KINDS: Final = MappingProxyType(
    {
        item.field: item.api_filter_kinds
        for item in SCAN_FIELD_CAPABILITIES
        if item.api_filter_kinds
    }
)
RANGE_FIELDS: Final = frozenset(
    field for field, kinds in FILTER_FIELD_KINDS.items() if "range" in kinds
)
CATEGORICAL_FIELDS: Final = frozenset(
    field for field, kinds in FILTER_FIELD_KINDS.items() if "categorical" in kinds
)
BOOLEAN_FIELDS: Final = frozenset(
    field for field, kinds in FILTER_FIELD_KINDS.items() if "boolean" in kinds
)
TEXT_FIELDS: Final = frozenset(
    field for field, kinds in FILTER_FIELD_KINDS.items() if "text" in kinds
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
