#!/usr/bin/env python3
"""Fail when the public application/build graph acquires xui-reader directly."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import NamedTuple


PRIVATE_REFERENCE = re.compile(
    r"git\+ssh://git@github\.com/xang1234/xui\.git|"
    r"github\.com/xang1234/xui(?:\.git)?|"
    r"(?<![A-Za-z0-9])xui[-_]reader(?![A-Za-z0-9])",
    re.IGNORECASE,
)
PRIVATE_STAGE_MARKER = "FROM builder AS social-xui-builder"
DEPENDENCY_NAMES = {
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "requirements.txt",
    "requirements-runtime.txt",
    "requirements-theme-ml.txt",
    "requirements-test.txt",
    "pyproject.toml",
    "poetry.lock",
    "pdm.lock",
    "uv.lock",
}


class Violation(NamedTuple):
    category: str
    path: str
    detail: str


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _record_if_private(
    violations: list[Violation],
    *,
    category: str,
    root: Path,
    path: Path,
    text: str,
) -> None:
    match = PRIVATE_REFERENCE.search(text)
    if match is not None:
        violations.append(
            Violation(category, _relative(root, path), match.group(0))
        )


def _application_import_violations(root: Path) -> list[Violation]:
    violations: list[Violation] = []
    backend = root / "backend" / "app"
    if backend.exists():
        for path in backend.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                for name in names:
                    if name == "xui_reader" or name.startswith("xui_reader."):
                        violations.append(
                            Violation(
                                "application-import",
                                _relative(root, path),
                                name,
                            )
                        )
    frontend = root / "frontend" / "src"
    import_pattern = re.compile(
        r"(?:import\s+.*?\s+from\s+|import\s*\(|require\s*\()"
        r"[^\n;]*xui[-_]reader",
        re.IGNORECASE,
    )
    if frontend.exists():
        for path in frontend.rglob("*"):
            if path.suffix not in {".js", ".jsx", ".ts", ".tsx"} or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            match = import_pattern.search(text)
            if match is not None:
                violations.append(
                    Violation(
                        "application-import",
                        _relative(root, path),
                        match.group(0),
                    )
                )
    return violations


def verify(root: Path) -> list[Violation]:
    root = Path(root).resolve()
    violations: list[Violation] = []

    for directory in (root, root / "backend", root / "frontend"):
        for name in DEPENDENCY_NAMES:
            path = directory / name
            if path.is_file():
                _record_if_private(
                    violations,
                    category="public-dependency",
                    root=root,
                    path=path,
                    text=path.read_text(encoding="utf-8"),
                )

    dockerfile = root / "backend" / "Dockerfile"
    if dockerfile.exists():
        public_stages = dockerfile.read_text(encoding="utf-8").split(
            PRIVATE_STAGE_MARKER,
            1,
        )[0]
        _record_if_private(
            violations,
            category="default-docker-stage",
            root=root,
            path=dockerfile,
            text=public_stages,
        )

    workflows = root / ".github" / "workflows"
    if workflows.exists():
        for path in workflows.iterdir():
            if path.name == "private-social-worker.yml" or path.suffix not in {".yml", ".yaml"}:
                continue
            _record_if_private(
                violations,
                category="public-workflow",
                root=root,
                path=path,
                text=path.read_text(encoding="utf-8"),
            )

    violations.extend(_application_import_violations(root))
    return sorted(set(violations))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    violations = verify(root)
    if violations:
        print("Private xui-reader boundary violations:")
        for violation in violations:
            print(
                f"- {violation.category}: {violation.path}: {violation.detail}"
            )
        return 1
    print("Private xui-reader boundary: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
