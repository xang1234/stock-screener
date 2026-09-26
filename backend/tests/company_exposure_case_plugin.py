"""Pytest plugin recording company-exposure acceptance-case tags.

Tests carry ``@pytest.mark.case("R02")``, ``@pytest.mark.exposure_layer("unit")``
and optionally ``@pytest.mark.exposure_slice("S1")``. The plugin validates
tags at collection (unknown case IDs or layers fail collection) and, when
``--exposure-case-report PATH`` is given, writes the collected node IDs and
their outcomes for ``scripts/run_required_company_exposure_postgres.py``.

It never chooses which application code runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.company_exposure.manifest import collect_case_inventory

_INVENTORY_KEY = pytest.StashKey[object]()
_OUTCOMES_KEY = pytest.StashKey[dict]()


def pytest_addoption(parser):
    parser.addoption(
        "--exposure-case-report",
        action="store",
        default=None,
        help="Write company-exposure case/layer inventory and outcomes as JSON.",
    )


def pytest_configure(config):
    # Markers are registered in pytest.ini so --strict-markers accepts them.
    config.stash[_OUTCOMES_KEY] = {}


def pytest_collection_modifyitems(session, config, items):
    inventory = collect_case_inventory(items)
    config.stash[_INVENTORY_KEY] = inventory
    if inventory.errors:
        raise pytest.UsageError(
            "invalid company-exposure case tags: " + "; ".join(inventory.errors)
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    outcomes = item.config.stash.get(_OUTCOMES_KEY, None)
    if outcomes is None:
        return
    if report.when == "call" or (
        report.when == "setup" and (report.failed or report.skipped)
    ):
        if hasattr(report, "wasxfail"):
            value = "xpassed" if report.passed else "xfailed"
        elif report.passed:
            value = "passed"
        elif report.skipped:
            value = "skipped"
        else:
            value = "failed"
        outcomes[report.nodeid] = value
    elif report.when == "teardown" and report.failed:
        outcomes[report.nodeid] = "failed"


def pytest_sessionfinish(session, exitstatus):
    path = session.config.getoption("--exposure-case-report")
    if not path:
        return
    inventory = session.config.stash.get(_INVENTORY_KEY, None)
    outcomes = session.config.stash.get(_OUTCOMES_KEY, {})
    payload = {
        "exit_status": int(exitstatus),
        "items": [
            {
                "nodeid": item.nodeid,
                "case_ids": list(item.case_ids),
                "layer": item.layer,
                "slices": list(item.slices),
            }
            for item in (inventory.items if inventory else ())
        ],
        "errors": list(inventory.errors) if inventory else [],
        "outcomes": outcomes,
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), "utf-8")
