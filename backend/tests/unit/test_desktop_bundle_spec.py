from pathlib import Path


def test_windows_bundle_includes_validation_route_module():
    spec_path = Path("backend/desktop/StockScanner.spec")
    assert '"app.api.v1.validation"' in spec_path.read_text(encoding="utf-8")


def test_macos_bundle_includes_validation_route_module():
    spec_path = Path("backend/desktop/StockScanner.macos.spec")
    assert '"app.api.v1.validation"' in spec_path.read_text(encoding="utf-8")
