"""Guard the public filter contract against persistence-adapter drift."""

from app.domain.scanning.filter_capabilities import (
    BOOLEAN_FIELDS,
    CATEGORICAL_FIELDS,
    FIELD_CAPABILITIES,
    RANGE_FIELDS,
    SORT_FIELDS,
    TEXT_FIELDS,
)
from app.infra.query import feature_store_query, scan_result_query


def test_every_filter_field_is_supported_by_both_persistence_adapters():
    ordinary_fields = RANGE_FIELDS | CATEGORICAL_FIELDS | BOOLEAN_FIELDS
    text_fields = TEXT_FIELDS - {"listing_search"}

    for adapter in (scan_result_query, feature_store_query):
        supported = adapter.supported_filter_fields()
        assert ordinary_fields <= supported
        assert text_fields <= supported


def test_every_sort_field_is_supported_by_both_persistence_adapters():
    assert SORT_FIELDS <= scan_result_query.supported_sort_fields()
    assert SORT_FIELDS <= feature_store_query.supported_sort_fields()


def test_date_metadata_is_available_to_domain_validation():
    assert FIELD_CAPABILITIES["ipo_date"].value_type == "date"
