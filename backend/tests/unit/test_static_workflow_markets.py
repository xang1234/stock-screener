"""Static-site workflow market matrix coverage."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.domain.markets import market_registry


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _workflow_matrix_markets(path: str) -> list[str]:
    content = (_PROJECT_ROOT / path).read_text(encoding="utf-8")
    match = re.search(r"market:\s*\[([^\]]+)\]", content)
    if match is not None:
        return [market.strip() for market in match.group(1).split(",")]

    dispatch_matrix_match = re.search(r"\|\|\s*'(\[[^']+\])'", content)
    assert dispatch_matrix_match is not None, f"{path} does not declare a market matrix"
    return list(json.loads(dispatch_matrix_match.group(1)))


def test_static_and_weekly_reference_workflows_cover_supported_markets():
    expected = list(market_registry.supported_market_codes())

    assert _workflow_matrix_markets(".github/workflows/static-site.yml") == expected
    assert _workflow_matrix_markets(".github/workflows/weekly-reference-data.yml") == expected
