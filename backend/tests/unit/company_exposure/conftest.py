from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixtures.company_exposure.factory import FixedClock

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "company_exposure"


@pytest.fixture
def contract_manifest() -> dict[str, dict]:
    payload = json.loads((FIXTURE_DIR / "contract_cases.json").read_text("utf-8"))
    return payload["cases"]


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock()
