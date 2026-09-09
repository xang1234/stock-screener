"""OpenCode Go social defaults are safe on fresh and existing installations."""

import importlib.util
import json
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.app_settings import AppSetting
from app.services.social_llm_budget_service import SocialLLMBudgetService


MODEL = "opencode-go/deepseek-v4-flash"


def _migration():
    path = Path(__file__).resolve().parents[2] / "alembic/versions/20260908_0038_seed_opencode_go_social.py"
    spec = importlib.util.spec_from_file_location("opencode_go_social_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


def test_migration_seeds_model_and_conservative_peak_pricing(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'opencode.sqlite'}")
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            _migration().upgrade()
    sessions = sessionmaker(engine, expire_on_commit=False)

    with sessions() as db:
        selected = db.scalar(select(AppSetting).where(
            AppSetting.key == "social_llm_extraction_model"
        ))
        raw_price = db.scalar(select(AppSetting).where(AppSetting.key == "social_llm_pricing"))
        assert selected.value == MODEL
        models = json.loads(raw_price.value)["models"]
        assert set(models) == {MODEL}
        assert models[MODEL] == {
            "provider": "openai",
            "actual_models": ["deepseek-v4-flash", "openai/deepseek-v4-flash"],
            "input_usd_per_million": "0.44",
            "output_usd_per_million": "1.32",
        }
    price = SocialLLMBudgetService(sessions).price(MODEL)
    assert str(price.input_rate) == "0.440000000000"
    assert str(price.output_rate) == "1.320000000000"
    engine.dispose()


def test_migration_preserves_an_existing_explicit_model_selection(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'existing.sqlite'}")
    Base.metadata.create_all(engine)
    with sessionmaker(engine).begin() as db:
        db.add_all([
            AppSetting(
                key="social_llm_extraction_model",
                value="openai/glm-4.7-flash",
                category="social",
            ),
            AppSetting(
                key="social_llm_pricing",
                value=json.dumps({
                    "version": "old-v1",
                    "models": {"openai/glm-4.7-flash": {
                        "provider": "openai",
                        "actual_models": ["glm-4.7-flash"],
                        "input_usd_per_million": "1",
                        "output_usd_per_million": "1",
                    }},
                }),
                category="social",
            ),
        ])
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            _migration().upgrade()

    with sessionmaker(engine)() as db:
        selected = db.scalar(select(AppSetting).where(
            AppSetting.key == "social_llm_extraction_model"
        ))
        pricing = db.scalar(select(AppSetting).where(AppSetting.key == "social_llm_pricing"))
        assert selected.value == "openai/glm-4.7-flash"
        existing_pricing = json.loads(pricing.value)
        assert existing_pricing["version"] == "old-v1"
        assert set(existing_pricing["models"]) == {"openai/glm-4.7-flash"}
    engine.dispose()
