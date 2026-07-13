"""Guard the public filter contract against persistence-adapter drift."""

from app.domain.scanning.filter_capabilities import (
    BOOLEAN_FIELDS,
    CATEGORICAL_FIELDS,
    FILTER_FIELD_KINDS,
    RANGE_FIELDS,
    SORT_FIELDS,
    TEXT_FIELDS,
    filter_field_catalog_payload,
)
from app.infra.query import feature_store_query, scan_result_query


def _adapter_fields(module) -> set[str]:
    return set(module._COLUMN_MAP) | set(module._JSON_FIELD_MAP)


def test_every_filter_field_is_supported_by_both_persistence_adapters():
    ordinary_fields = RANGE_FIELDS | CATEGORICAL_FIELDS | BOOLEAN_FIELDS
    text_fields = TEXT_FIELDS - {"listing_search"}

    for adapter in (scan_result_query, feature_store_query):
        supported = _adapter_fields(adapter)
        assert ordinary_fields <= supported
        assert text_fields <= supported


def test_every_sort_field_is_supported_by_both_persistence_adapters():
    scan_result_sorts = _adapter_fields(scan_result_query) | set(
        scan_result_query._PYTHON_SORT_FIELDS
    )

    assert SORT_FIELDS <= scan_result_sorts
    assert SORT_FIELDS <= _adapter_fields(feature_store_query)


def test_builder_catalog_is_a_typed_subset_of_the_server_contract():
    catalog = filter_field_catalog_payload()
    fields = [item["field"] for item in catalog]

    assert len(fields) == len(set(fields))
    assert set(fields) <= set(FILTER_FIELD_KINDS)
    assert all(item["type"] == FILTER_FIELD_KINDS[item["field"]] for item in catalog)
    assert all(item["sortable"] == (item["field"] in SORT_FIELDS) for item in catalog)
