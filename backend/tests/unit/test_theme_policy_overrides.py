"""Tests for admin-configurable theme matcher/lifecycle policy overrides."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.app_settings import AppSetting
from app.services.theme_extraction_service import ThemeExtractionService


def test_theme_extraction_uses_admin_matcher_overrides():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        session.add(
            AppSetting(
                key="theme_policy_overrides",
                value=(
                    '{"technical":{"matcher":{"match_default_threshold":0.87,'
                    '"fuzzy_attach_threshold":0.93,"fuzzy_review_threshold":0.80,'
                    '"fuzzy_ambiguity_margin":0.05,"embedding_attach_threshold":0.86,'
                    '"embedding_review_threshold":0.79,"embedding_ambiguity_margin":0.04}}}'
                ),
                category="theme",
            )
        )
        session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = session
        service.pipeline = "technical"
        service.match_threshold_config = None
        service.theme_policy_overrides = service._load_theme_policy_overrides()

        threshold_config = service._get_match_threshold_config()
        fuzzy_attach, fuzzy_review, fuzzy_margin = service._resolve_fuzzy_thresholds("news")
        emb_attach, emb_review, emb_margin = service._resolve_embedding_thresholds("news")

        assert threshold_config.default_threshold == 0.87
        assert fuzzy_attach == 0.93
        assert fuzzy_review == 0.80
        assert fuzzy_margin == 0.05
        assert emb_attach == 0.86
        assert emb_review == 0.79
        assert emb_margin == 0.04
    finally:
        session.close()
