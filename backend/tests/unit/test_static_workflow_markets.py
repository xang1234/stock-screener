"""Static-site workflow market matrix coverage."""

from __future__ import annotations

import re
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _workflow_matrix_markets(path: str) -> list[str]:
    content = (_PROJECT_ROOT / path).read_text(encoding="utf-8")
    match = re.search(r"market:\s*\[([^\]]+)\]", content)
    assert match is not None, f"{path} does not declare a market matrix"
    return [market.strip() for market in match.group(1).split(",")]


def test_static_and_weekly_reference_workflows_cover_supported_markets():
    expected = ["US", "HK", "IN", "JP", "KR", "TW", "CN", "CA"]

    assert _workflow_matrix_markets(".github/workflows/static-site.yml") == expected
    assert _workflow_matrix_markets(".github/workflows/weekly-reference-data.yml") == expected
