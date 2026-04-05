from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "spec_relpath",
    (
        Path("backend/desktop/StockScanner.spec"),
        Path("backend/desktop/StockScanner.macos.spec"),
    ),
)
def test_bundle_includes_validation_route_module(spec_relpath: Path):
    spec_path = REPO_ROOT / spec_relpath
    assert '"app.api.v1.validation"' in spec_path.read_text(encoding="utf-8")
