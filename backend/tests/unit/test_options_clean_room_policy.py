from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[2]
PRODUCTION_ROOTS = (ROOT / "app", ROOT.parent / "frontend" / "src")
FORBIDDEN = (
    "net_premium_inflow",
    "option_flow_signal",
    "options_force_release",
    "max_pain_task",
    "gex_task",
)


def test_options_surface_contains_no_abandoned_pr_flow_or_pipeline_vocabulary():
    offenders = []
    for root in PRODUCTION_ROOTS:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".js", ".jsx", ".ts", ".tsx"}:
                continue
            if "options" not in path.as_posix().lower():
                continue
            text = path.read_text(encoding="utf-8").lower()
            for term in FORBIDDEN:
                if term in text:
                    offenders.append(f"{path.relative_to(ROOT.parent)}: {term}")
    assert offenders == []
