"""Tests for merge-model selection behavior in ThemeMergingService."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.app_settings import AppSetting
from app.models.theme import ThemeCluster
from app.services.theme_merging_service import ThemeMergingService


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _llm_json_response(payload: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=payload),
            )
        ]
    )


def test_load_merge_model_config_reads_llm_merge_model_setting(db_session, monkeypatch):
    db_session.add_all(
        [
            AppSetting(key="llm_merge_model", value="ollama_chat/qwen3:14b", category="llm"),
            AppSetting(key="ollama_api_base", value="http://localhost:22434", category="llm"),
        ]
    )
    db_session.commit()
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)

    service = ThemeMergingService.__new__(ThemeMergingService)
    service.db = db_session
    service.merge_model_id = ThemeMergingService.DEFAULT_MERGE_MODEL

    service._load_merge_model_config()

    assert service.merge_model_id == "ollama_chat/qwen3:14b"
    assert "OLLAMA_API_BASE" in os.environ
    assert os.environ["OLLAMA_API_BASE"] == "http://localhost:22434"


def test_verify_merge_with_llm_uses_configured_merge_model_without_fallbacks():
    service = ThemeMergingService.__new__(ThemeMergingService)
    service.llm = MagicMock()
    service.merge_model_id = "groq/qwen/qwen3-32b"
    service._last_llm_request = 0
    service._min_llm_interval = 0

    service.llm.completion_sync.return_value = _llm_json_response(
        '{"should_merge": true, "confidence": 0.93, "relationship": "identical", "reasoning": "same concept", "canonical_name": "AI Infrastructure"}'
    )

    theme1 = ThemeCluster(
        id=1,
        canonical_key="ai_infrastructure",
        display_name="AI Infrastructure",
        name="AI Infra",
        aliases=["AI Infrastructure"],
        description="Datacenter capex",
        category="technology",
        pipeline="technical",
    )
    theme2 = ThemeCluster(
        id=2,
        canonical_key="ai_datacenter_buildout",
        display_name="AI Datacenter Buildout",
        name="AI Datacenter Buildout",
        aliases=["AI Infra Buildout"],
        description="Datacenter build cycle",
        category="technology",
        pipeline="technical",
    )

    result = service.verify_merge_with_llm(theme1, theme2, 0.91)

    assert result["should_merge"] is True
    assert result["confidence"] == 0.93
    kwargs = service.llm.completion_sync.call_args.kwargs
    assert kwargs["model"] == "groq/qwen/qwen3-32b"
    assert kwargs["allow_fallbacks"] is False
