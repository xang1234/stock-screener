"""Acceptance-case traceability for the company exposure map.

Requirement IDs (E01..E15, I01..I11, R01..R15) label real tests through
``@pytest.mark.case``; they never select application behaviour. This module
validates a collected/executed test inventory against required
``(case, layer)`` pairs. It does not run tests itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

CASE_IDS = frozenset(
    {f"E{i:02}" for i in range(1, 16)}
    | {f"I{i:02}" for i in range(1, 12)}
    | {f"R{i:02}" for i in range(1, 16)}
)
LAYERS = frozenset(
    {"unit", "schema", "postgres", "api", "integration", "deployment", "frontend"}
)
PASSING = "passed"


@dataclass(frozen=True, slots=True)
class CaseLayerRequirement:
    case_id: str
    layer: str

    def __post_init__(self) -> None:
        if self.case_id not in CASE_IDS:
            raise ValueError(f"unknown_case_id:{self.case_id}")
        if self.layer not in LAYERS:
            raise ValueError(f"unknown_layer:{self.layer}")


@dataclass(frozen=True, slots=True)
class CollectedCase:
    nodeid: str
    case_ids: tuple[str, ...]
    layer: str | None
    slices: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CaseInventory:
    items: tuple[CollectedCase, ...]
    errors: tuple[str, ...] = ()

    def nodes_for(self, requirement: CaseLayerRequirement) -> tuple[str, ...]:
        return tuple(
            item.nodeid
            for item in self.items
            if requirement.case_id in item.case_ids
            and item.layer == requirement.layer
        )


@dataclass(frozen=True, slots=True)
class CaseRunReport:
    inventory: CaseInventory
    outcomes: Mapping[str, str]
    execution_mode: str


@dataclass(frozen=True, slots=True)
class CaseGateDecision:
    passed: bool
    missing_requirements: tuple[CaseLayerRequirement, ...] = ()
    failed_nodeids: tuple[str, ...] = ()
    execution_mode: str = ""
    errors: tuple[str, ...] = field(default_factory=tuple)


def collect_case_inventory(items: Iterable) -> CaseInventory:
    """Build an inventory from pytest items (or objects with the same API)."""

    collected = []
    errors = []
    for item in items:
        case_ids = tuple(
            str(mark.args[0]) for mark in item.iter_markers(name="case") if mark.args
        )
        layers = [
            str(mark.args[0])
            for mark in item.iter_markers(name="exposure_layer")
            if mark.args
        ]
        slices = tuple(
            str(mark.args[0])
            for mark in item.iter_markers(name="exposure_slice")
            if mark.args
        )
        if not case_ids and not layers:
            continue
        for case_id in case_ids:
            if case_id not in CASE_IDS:
                errors.append(f"{item.nodeid}:unknown_case_id:{case_id}")
        if len(set(layers)) > 1:
            errors.append(f"{item.nodeid}:multiple_layers")
        layer = layers[0] if layers else None
        if layer is not None and layer not in LAYERS:
            errors.append(f"{item.nodeid}:unknown_layer:{layer}")
        if case_ids and layer is None:
            errors.append(f"{item.nodeid}:case_without_layer")
        collected.append(
            CollectedCase(
                nodeid=item.nodeid,
                case_ids=tuple(sorted(set(case_ids))),
                layer=layer,
                slices=slices,
            )
        )
    return CaseInventory(items=tuple(collected), errors=tuple(errors))


def validate_case_report(
    report: CaseRunReport,
    required: set[CaseLayerRequirement] | frozenset[CaseLayerRequirement],
    *,
    required_execution_mode: str | None = None,
) -> CaseGateDecision:
    """Every required case/layer needs ≥1 collected test, and every tagged
    test applicable to a required requirement must have passed."""

    errors = list(report.inventory.errors)
    if required_execution_mode and report.execution_mode != required_execution_mode:
        errors.append(f"wrong_execution_mode:{report.execution_mode}")
    missing = []
    failed = []
    for requirement in sorted(required, key=lambda r: (r.case_id, r.layer)):
        nodes = report.inventory.nodes_for(requirement)
        if not nodes:
            missing.append(requirement)
            continue
        for node in nodes:
            if report.outcomes.get(node, "not_run") != PASSING:
                failed.append(node)
    failed_nodes = tuple(dict.fromkeys(failed))
    return CaseGateDecision(
        passed=not missing and not failed_nodes and not errors,
        missing_requirements=tuple(missing),
        failed_nodeids=failed_nodes,
        execution_mode=report.execution_mode,
        errors=tuple(errors),
    )


def load_contract_cases(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if set(cases) != CASE_IDS:
        raise ValueError("contract_case_manifest_incomplete")
    return cases


def load_requirements(path: Path, slice_id: str) -> frozenset[CaseLayerRequirement]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["slices"][slice_id]["required"]
    return frozenset(CaseLayerRequirement(row["case"], row["layer"]) for row in rows)
