from __future__ import annotations

from pathlib import Path


def test_backend_app_pct_change_calls_specify_fill_method():
    backend_app = Path(__file__).resolve().parents[2] / "app"
    offenders: list[str] = []

    for path in sorted(backend_app.rglob("*.py")):
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if ".pct_change(" not in line:
                continue
            if "fill_method=" in line:
                continue
            offenders.append(f"{path.relative_to(backend_app.parent)}:{line_no}:{line.strip()}")

    assert offenders == []
