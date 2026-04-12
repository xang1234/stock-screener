#!/usr/bin/env python3
"""Static architecture gate for strict dependency injection boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    roots: tuple[Path, ...]


RULES: tuple[Rule, ...] = (
    Rule(
        name="Service modules must not reference SessionLocal",
        pattern=re.compile(r"\bSessionLocal\b"),
        roots=(ROOT / "app" / "services",),
    ),
    Rule(
        name="API/task/app-script entrypoints must not call SessionLocal(...)",
        pattern=re.compile(r"\bSessionLocal\s*\("),
        roots=(
            ROOT / "app" / "api",
            ROOT / "app" / "tasks",
            ROOT / "app" / "scripts",
            ROOT / "scripts",
        ),
    ),
    Rule(
        name="Legacy get_instance singleton usage is forbidden",
        pattern=re.compile(r"\bget_instance\s*\("),
        roots=(ROOT / "app", ROOT / "scripts"),
    ),
    Rule(
        name="Constructor defaults must not bind SessionLocal as the session factory",
        pattern=re.compile(r"session_factory\s*=\s*SessionLocal\b"),
        roots=(ROOT / "app", ROOT / "scripts"),
    ),
)


def _iter_python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    self_path = Path(__file__).resolve()
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file() and path.resolve() != self_path
    )


def _find_violations(rule: Rule) -> list[str]:
    violations: list[str] = []
    for search_root in rule.roots:
        for path in _iter_python_files(search_root):
            text = path.read_text(encoding="utf-8")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if rule.pattern.search(line):
                    rel_path = path.relative_to(ROOT)
                    violations.append(
                        f"{rel_path}:{line_no}: {rule.name}: {line.strip()}"
                    )
    return violations


def main() -> int:
    failures: list[str] = []
    for rule in RULES:
        failures.extend(_find_violations(rule))

    if failures:
        for failure in failures:
            print(failure)
        return 1

    print("Strict DI architecture gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
