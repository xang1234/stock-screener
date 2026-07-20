from pathlib import Path

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = BACKEND_ROOT.parent
LIMITS = {
    "app/services/ibd_group_rank_service.py": 900,
    "app/services/market_rs_rollout_service.py": 250,
    "app/tasks/group_rank_tasks.py": 800,
    "app/wiring/bootstrap.py": 1050,
}
EXTRACTED_MODULES = (
    "app/services/group_rank_input_loader.py",
    "app/services/group_rank_historical_calculator.py",
    "app/services/group_ranking_calculator.py",
    "app/services/group_ranking_repository.py",
    "app/services/static_chart_bundle_exporter.py",
    "app/services/static_breadth_section_builder.py",
    "app/services/static_group_section_builder.py",
    "app/services/static_artifact_combiner.py",
    "app/services/feature_run_group_enrichment.py",
    "app/services/market_rs_activation_validator.py",
    "app/services/market_rs_activator.py",
    "app/services/market_rs_backfill_service.py",
    "app/services/market_rs_rollout_contracts.py",
    "app/services/market_rs_static_artifact_validator.py",
    "app/wiring/canonical_rs_runtime.py",
)


@pytest.mark.parametrize("relative_path,limit", LIMITS.items())
def test_rs_touched_production_modules_stay_bounded(relative_path, limit):
    path = BACKEND_ROOT / relative_path
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    assert line_count <= limit, (
        f"{relative_path} has {line_count} lines; limit is {limit}"
    )


def test_new_extracted_modules_stay_below_seven_hundred_lines():
    for relative_path in EXTRACTED_MODULES:
        line_count = len(
            (BACKEND_ROOT / relative_path).read_text(encoding="utf-8").splitlines()
        )
        assert line_count <= 700, f"{relative_path} has {line_count} lines"


def test_group_ranking_uses_composition_without_runtime_wiring_imports():
    facade = (BACKEND_ROOT / "app/services/ibd_group_rank_service.py").read_text(
        encoding="utf-8"
    )
    input_loader = (BACKEND_ROOT / "app/services/group_rank_input_loader.py").read_text(
        encoding="utf-8"
    )
    historical = (
        BACKEND_ROOT / "app/services/group_rank_historical_calculator.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "LegacyGroupRankDataMixin" not in facade
    assert "LegacyGroupRankBackfillMixin" not in facade
    assert "wiring.bootstrap" not in input_loader
    assert "wiring.bootstrap" not in historical
    assert "self.input_loader" in facade
    assert "self.historical_calculator" in facade


def test_live_group_rankings_page_stays_bounded():
    path = REPOSITORY_ROOT / "frontend/src/pages/GroupRankingsPage.jsx"
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    assert line_count < 1000, f"GroupRankingsPage.jsx has {line_count} lines"
