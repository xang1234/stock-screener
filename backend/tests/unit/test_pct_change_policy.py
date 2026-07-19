from __future__ import annotations

import ast
from pathlib import Path


def test_backend_app_pct_change_calls_specify_fill_method():
    backend_app = Path(__file__).resolve().parents[2] / "app"
    offenders: list[str] = []

    for path in sorted(backend_app.rglob("*.py")):
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "pct_change":
                continue
            if any(keyword.arg == "fill_method" for keyword in node.keywords):
                continue
            line = lines[node.lineno - 1].strip()
            offenders.append(
                f"{path.relative_to(backend_app.parent)}:{node.lineno}:{line}"
            )

    assert offenders == []
