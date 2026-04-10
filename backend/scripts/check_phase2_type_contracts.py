#!/usr/bin/env python3
"""Focused Phase 2 type-contract gate for touched reliability modules."""

from __future__ import annotations

import ast
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGET_MODULES = [
    ROOT / "app/api/v1/themes_content_pipeline.py",
    ROOT / "app/services/theme_extraction_service.py",
    ROOT / "app/services/theme_merging_service.py",
    ROOT / "app/services/assistant_gateway_service.py",
    ROOT / "app/services/cache/price_cache_freshness.py",
    ROOT / "app/services/cache/price_cache_warmup.py",
    ROOT / "app/services/cache/price_cache_failure_telemetry.py",
    ROOT / "app/theme_platform/contracts.py",
]

REQUIRED_RETURN_ANNOTATIONS: dict[Path, set[str]] = {
    ROOT / "app/services/theme_extraction_service.py": {
        "_load_pipeline_config",
        "_load_configured_model",
        "_load_reprocessing_config",
        "_init_client",
        "_rate_limit",
        "extract_from_content",
    },
    ROOT / "app/services/theme_merging_service.py": {
        "__init__",
        "_load_merge_model_config",
        "_init_llm_client",
        "execute_merge",
        "approve_suggestion",
    },
    ROOT / "app/services/cache/price_cache_warmup.py": {
        "get_warmup_metadata",
        "get_heartbeat_info",
        "get_task_progress",
    },
}

REQUIRED_TYPED_DICTS = {
    "PipelineRunStatusPayload",
    "MergeActionResult",
    "WarmupHeartbeatState",
    "WarmupStateSnapshot",
}


def _load_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _check_return_annotations(path: Path) -> list[str]:
    errors: list[str] = []
    required_names = REQUIRED_RETURN_ANNOTATIONS.get(path)
    if not required_names:
        return errors

    tree = _load_tree(path)
    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = node

    for name in sorted(required_names):
        node = found.get(name)
        if node is None:
            errors.append(f"{path}: missing required function '{name}'")
            continue
        if node.returns is None:
            errors.append(f"{path}:{node.lineno} function '{name}' missing return annotation")
    return errors


def _check_typed_dict_contracts(path: Path) -> list[str]:
    errors: list[str] = []
    if path.name != "contracts.py":
        return errors
    tree = _load_tree(path)
    typed_dict_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "TypedDict":
                    typed_dict_names.add(node.name)
    missing = REQUIRED_TYPED_DICTS - typed_dict_names
    for name in sorted(missing):
        errors.append(f"{path}: missing required TypedDict '{name}'")
    return errors


def main() -> int:
    errors: list[str] = []

    for path in TARGET_MODULES:
        py_compile.compile(str(path), doraise=True)
        errors.extend(_check_return_annotations(path))
        errors.extend(_check_typed_dict_contracts(path))

    if errors:
        for err in errors:
            print(err)
        return 1

    print("Phase 2 type-contract gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
