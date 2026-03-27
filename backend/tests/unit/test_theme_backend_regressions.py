from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.task_execution import TaskExecutionHistory
from app.schemas.theme import ThemeAlertResponse, ThemeMergeSuggestionResponse
from app.services.task_registry_service import TaskRegistryService
from app.services.theme_discovery_service import ThemeDiscoveryService


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *args, **kwargs):
        return self

    def all(self):
        return self._rows


class _FakeDb:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *args, **kwargs):
        return _FakeQuery(self._rows)


def test_lifecycle_snapshot_normalizes_aware_mentions_with_naive_now():
    service = ThemeDiscoveryService.__new__(ThemeDiscoveryService)
    service.db = _FakeDb(
        [
            (
                datetime(2026, 3, 27, 8, 0, tzinfo=timezone.utc),
                0.9,
                "news",
                "Macro Desk",
            )
        ]
    )
    service.pipeline = "technical"

    observation = service._lifecycle_snapshot(101, now=datetime(2026, 3, 27, 12, 0, 0))

    assert observation["mentions_7d"] == 1
    assert observation["mentions_30d"] == 1
    assert observation["days_since_last_mention"] == 0


def test_task_registry_completion_handles_mixed_timezone_datetimes():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[TaskExecutionHistory.__table__])
    session = sessionmaker(bind=engine)()
    session.add(
        TaskExecutionHistory(
            task_name="daily-breadth-calculation",
            task_function="app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill",
            task_id="task-123",
            status="running",
            started_at=datetime.utcnow() - timedelta(seconds=30),
            triggered_by="manual",
        )
    )
    session.commit()

    TaskRegistryService()._update_execution_completed(
        session,
        "task-123",
        "completed",
        result={"status": "ok"},
    )

    execution = session.query(TaskExecutionHistory).filter(
        TaskExecutionHistory.task_id == "task-123"
    ).first()
    assert execution is not None
    assert execution.status == "completed"
    assert execution.duration_seconds is not None
    assert execution.duration_seconds >= 0


def test_theme_alert_response_decodes_json_string_fields():
    alert = ThemeAlertResponse.model_validate(
        SimpleNamespace(
            id=1,
            alert_type="lifecycle_transition",
            title="Theme reactivated",
            description="Recovered after renewed coverage",
            severity="info",
            related_tickers="[]",
            metrics='{"from_state":"dormant","to_state":"reactivated"}',
            triggered_at=datetime.now(timezone.utc),
            is_read=False,
        )
    )

    assert alert.related_tickers == []
    assert alert.metrics == {"from_state": "dormant", "to_state": "reactivated"}


def test_theme_merge_suggestion_response_decodes_alias_lists():
    suggestion = ThemeMergeSuggestionResponse(
        id=7,
        source_theme_id=11,
        source_theme_name="Silver Mining (Tier-1 Undervalued Assets)",
        source_aliases='["Silver Mining (Tier-1 Undervalued Assets)"]',
        target_theme_id=12,
        target_theme_name="Tier-1 Silver Assets",
        target_aliases='["Tier-1 Silver Assets"]',
        similarity_score=0.98,
        llm_confidence=0.93,
        relationship_type="identical",
        reasoning="Equivalent phrasing.",
        suggested_name="Tier-1 Silver Assets",
        status="pending",
        created_at="2026-03-27T08:36:12+00:00",
        source_cluster_id=11,
        source_name="Silver Mining (Tier-1 Undervalued Assets)",
        target_cluster_id=12,
        target_name="Tier-1 Silver Assets",
        embedding_similarity=0.98,
        llm_reasoning="Equivalent phrasing.",
        llm_relationship="identical",
        suggested_canonical_name="Tier-1 Silver Assets",
    )

    assert suggestion.source_aliases == ["Silver Mining (Tier-1 Undervalued Assets)"]
    assert suggestion.target_aliases == ["Tier-1 Silver Assets"]
