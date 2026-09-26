from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from app.domain.company_exposure.manifest import (
    CaseLayerRequirement,
    CaseRunReport,
    collect_case_inventory,
    load_requirements,
    validate_case_report,
)


@dataclass
class _Mark:
    args: tuple


class _Item:
    def __init__(self, nodeid, cases=(), layers=(), slices=()):
        self.nodeid = nodeid
        self._marks = {
            "case": [_Mark((c,)) for c in cases],
            "exposure_layer": [_Mark((layer,)) for layer in layers],
            "exposure_slice": [_Mark((s,)) for s in slices],
        }

    def iter_markers(self, name):
        return iter(self._marks.get(name, []))


R02_UNIT = CaseLayerRequirement("R02", "unit")


def _report(items, outcomes, mode="recorded"):
    return CaseRunReport(
        inventory=collect_case_inventory(items), outcomes=outcomes, execution_mode=mode
    )


def test_one_case_can_label_several_real_tests():
    items = [
        _Item("a.py::test_one", ["R02"], ["unit"]),
        _Item("b.py::test_two", ["R02"], ["unit"]),
    ]
    decision = validate_case_report(
        _report(items, {"a.py::test_one": "passed", "b.py::test_two": "passed"}),
        {R02_UNIT},
    )
    assert decision.passed


@pytest.mark.parametrize("outcome", ["skipped", "xfailed", "xpassed", "failed", None])
def test_any_nonpassing_tagged_test_fails_gate_even_with_passing_sibling(outcome):
    items = [
        _Item("a.py::test_one", ["R02"], ["unit"]),
        _Item("b.py::test_two", ["R02"], ["unit"]),
    ]
    outcomes = {"a.py::test_one": "passed"}
    if outcome is not None:
        outcomes["b.py::test_two"] = outcome
    decision = validate_case_report(_report(items, outcomes), {R02_UNIT})
    assert not decision.passed
    assert decision.failed_nodeids == ("b.py::test_two",)


def test_missing_layer_coverage_fails():
    items = [_Item("a.py::test_one", ["R02"], ["postgres"])]
    decision = validate_case_report(
        _report(items, {"a.py::test_one": "passed"}), {R02_UNIT}
    )
    assert decision.missing_requirements == (R02_UNIT,)
    assert not decision.passed


def test_unknown_ids_and_layers_are_errors():
    inventory = collect_case_inventory(
        [
            _Item("a.py::test_x", ["R99"], ["unit"]),
            _Item("a.py::test_y", ["R02"], ["galaxy"]),
            _Item("a.py::test_z", ["R02"], []),
        ]
    )
    assert inventory.errors == (
        "a.py::test_x:unknown_case_id:R99",
        "a.py::test_y:unknown_layer:galaxy",
        "a.py::test_z:case_without_layer",
    )
    with pytest.raises(ValueError):
        CaseLayerRequirement("R99", "unit")


def test_parameterized_node_ids_are_recorded_individually():
    items = [
        _Item("a.py::test_p[0]", ["R04"], ["unit"]),
        _Item("a.py::test_p[1]", ["R04"], ["unit"]),
    ]
    decision = validate_case_report(
        _report(items, {"a.py::test_p[0]": "passed", "a.py::test_p[1]": "failed"}),
        {CaseLayerRequirement("R04", "unit")},
    )
    assert decision.failed_nodeids == ("a.py::test_p[1]",)


def test_wrong_execution_mode_fails():
    items = [_Item("a.py::test_one", ["R02"], ["unit"])]
    decision = validate_case_report(
        _report(items, {"a.py::test_one": "passed"}, mode="live"),
        {R02_UNIT},
        required_execution_mode="recorded",
    )
    assert not decision.passed
    assert decision.errors == ("wrong_execution_mode:live",)


def test_requirements_load_by_slice(tmp_path):
    path = tmp_path / "req.json"
    path.write_text(
        json.dumps({"slices": {"S1": {"required": [{"case": "R02", "layer": "unit"}]}}})
    )
    assert load_requirements(path, "S1") == frozenset({R02_UNIT})
