"""Seed OpenCode Go as the fresh-install social extraction provider."""

import json

from alembic import op
import sqlalchemy as sa


revision = "20260908_0038"
down_revision = "20260907_0037"
branch_labels = None
depends_on = None

MODEL = "opencode-go/deepseek-v4-flash"
PRICING_VERSION = "opencode-go-2026-09-08-peak-v1"
PRICE = {
    "provider": "openai",
    "actual_models": ["deepseek-v4-flash", "openai/deepseek-v4-flash"],
    # Use the documented peak rates so the dollar guard remains conservative.
    "input_usd_per_million": "0.44",
    "output_usd_per_million": "1.32",
}


def upgrade():
    settings = sa.table(
        "app_settings", sa.column("key"), sa.column("value"), sa.column("category")
    )
    connection = op.get_bind()

    selected = connection.execute(sa.select(settings.c.value).where(
        settings.c.key == "social_llm_extraction_model"
    )).first()
    if selected is None:
        connection.execute(settings.insert().values(
            key="social_llm_extraction_model", value=MODEL, category="social"
        ))
    else:
        # An administrator explicitly selected a Social model before this
        # provider shipped; preserve its matching price contract unchanged.
        return

    pricing_row = connection.execute(sa.select(settings.c.value).where(
        settings.c.key == "social_llm_pricing"
    )).first()
    pricing = json.dumps(
        {"version": PRICING_VERSION, "models": {MODEL: PRICE}}, sort_keys=True
    )
    if pricing_row is None:
        connection.execute(settings.insert().values(
            key="social_llm_pricing", value=pricing, category="social"
        ))
    else:
        connection.execute(settings.update().where(
            settings.c.key == "social_llm_pricing"
        ).values(value=pricing))


def downgrade():
    # Preserve administrator-selected models and price contracts across rollback.
    pass
