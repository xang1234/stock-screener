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
def test_bundle_includes_digest_validation_and_strategy_profile_route_modules(spec_relpath: Path):
    spec_path = REPO_ROOT / spec_relpath
    contents = spec_path.read_text(encoding="utf-8")
    assert '"app.api.v1.validation"' in contents
    assert '"app.api.v1.digest"' in contents
    assert '"app.api.v1.strategy_profiles"' in contents
