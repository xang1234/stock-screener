"""Regression checks for the static Alembic baseline revision."""

from __future__ import annotations

from pathlib import Path
import re
from collections import defaultdict


def _baseline_source() -> str:
    baseline_path = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / "20260408_0001_baseline.py"
    )
    return baseline_path.read_text()


def test_baseline_uses_quoted_manual_trigger_source_default():
    source = _baseline_source()

    assert "server_default='manual'" in source
    assert "server_default=sa.text('manual')" not in source


def test_baseline_avoids_redundant_indexes():
    source = _baseline_source().splitlines()

    indexes: dict[tuple[str, tuple[str, ...], bool], list[str]] = defaultdict(list)
    primary_key_id_tables: set[str] = set()
    unique_constraints: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    current_table: str | None = None

    for line in source:
        table_match = re.search(r"op\.create_table\('([^']+)'", line)
        if table_match:
            current_table = table_match.group(1)
        if current_table and re.search(r"sa\.Column\('id',.*primary_key=True", line):
            primary_key_id_tables.add(current_table)
        if current_table:
            constraint_match = re.search(r"sa\.UniqueConstraint\((.*)\)", line)
            if constraint_match:
                cols = tuple(
                    re.findall(r"'([^']+)'", constraint_match.group(1).split("name=")[0])
                )
                if cols:
                    unique_constraints[current_table].add(cols)
        if line.strip() == ")" and current_table:
            current_table = None

        index_match = re.search(
            r"op\.create_index\('([^']+)', '([^']+)', \[([^\]]+)\](?:, unique=(True|False))?\)",
            line,
        )
        if not index_match:
            continue
        name, table, cols, unique = index_match.groups()
        columns = tuple(part.strip().strip("'") for part in cols.split(","))
        indexes[(table, columns, unique == "True")].append(name)

        assert not (
            columns == ("id",) and table in primary_key_id_tables
        ), f"redundant primary-key index left in baseline: {name}"
        assert (
            columns not in unique_constraints.get(table, set())
        ), f"index duplicates unique constraint in baseline: {name}"

    duplicate_indexes = {
        key: names
        for key, names in indexes.items()
        if len(names) > 1
    }
    assert duplicate_indexes == {}, duplicate_indexes
