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
    public_fields = RANGE_FIELDS | CATEGORICAL_FIELDS | BOOLEAN_FIELDS | TEXT_FIELDS

    for adapter in (scan_result_query, feature_store_query):
        supported = adapter.supported_filter_fields()
        assert public_fields - {"listing_search"} <= supported


def test_every_sort_field_is_supported_by_both_persistence_adapters():
    assert SORT_FIELDS <= scan_result_query.supported_sort_fields()
    assert SORT_FIELDS <= feature_store_query.supported_sort_fields()


def test_date_metadata_is_available_to_domain_validation():
    assert FIELD_CAPABILITIES["ipo_date"].value_type == "date"


def test_real_fields_can_support_multiple_filter_kinds_with_api_subsets():
    symbol = FIELD_CAPABILITIES["symbol"]
    industry = FIELD_CAPABILITIES["ibd_industry_group"]

    assert symbol.filter_kinds == frozenset({"text", "categorical"})
    assert symbol.api_filter_kinds == frozenset({"text"})
    assert industry.filter_kinds == frozenset({"categorical", "text"})
    assert industry.api_filter_kinds == frozenset({"categorical"})
