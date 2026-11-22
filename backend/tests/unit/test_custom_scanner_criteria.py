"""CustomScanner reads criteria exactly as the feature-store compiler does."""

from app.domain.scanning.custom_criteria_compiler import compile_custom_criteria
from app.scanners.custom_scanner import CustomScanner

from tests.unit.conftest import _make_stub_stock_data


def test_null_custom_filters_falls_back_to_top_level_keys():
    criteria = {"custom_filters": None, "market_cap_min": 1}

    requirements = CustomScanner().get_data_requirements(criteria)

    assert requirements.needs_fundamentals is True
    assert compile_custom_criteria(criteria, screeners=["custom"]).representable_keys


def test_null_min_score_applies_the_compilers_default_gate():
    criteria = {"min_score": None, "custom_filters": {"price_min": 1}}

    result = CustomScanner().scan_stock("TEST", _make_stub_stock_data("TEST"), criteria)

    compiled = compile_custom_criteria(criteria, screeners=["custom"])
    assert result.details["min_score_required"] == compiled.min_score == 70.0


def test_nested_custom_filters_must_be_a_mapping():
    import pytest
    from app.domain.scanning.custom_criteria_compiler import flatten_custom_filters

    with pytest.raises(TypeError):
        flatten_custom_filters({"custom_filters": [("price_min", 10)]})


def test_scan_request_rejects_non_object_custom_filters():
    import pytest
    from pydantic import ValidationError
    from app.schemas.scanning import ScanCreateRequest

    with pytest.raises(ValidationError, match="custom_filters must be an object"):
        ScanCreateRequest(criteria={"custom_filters": [["price_min", 10]]})
