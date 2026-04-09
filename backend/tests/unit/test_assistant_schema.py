from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.schemas.assistant import AssistantMessageResponse, AssistantReferenceItem


def test_assistant_reference_item_accepts_internal_and_https_urls():
    internal = AssistantReferenceItem(type="internal", title="NVDA", url="/stocks/NVDA")
    external = AssistantReferenceItem(type="web", title="Reuters", url="https://example.com/reuters")

    assert internal.url == "/stocks/NVDA"
    assert external.url == "https://example.com/reuters"


def test_assistant_reference_item_rejects_non_web_schemes():
    with pytest.raises(ValidationError):
        AssistantReferenceItem(type="web", title="Bad", url="javascript:alert(1)")

    with pytest.raises(ValidationError):
        AssistantReferenceItem(type="web", title="Bad", url="//example.com/path")


def test_assistant_message_response_omits_internal_reasoning_fields():
    message = AssistantMessageResponse.model_validate(
        SimpleNamespace(
            id=1,
            conversation_id="conv-1",
            role="assistant",
            content="Hello",
            agent_type="hermes",
            tool_name=None,
            tool_input=None,
            tool_output=None,
            reasoning="internal",
            tool_calls=[{"tool": "stock_snapshot"}],
            thinking_traces=[{"step": "hidden"}],
            source_references=[],
            created_at=datetime(2026, 4, 9, tzinfo=UTC),
        )
    )

    payload = message.model_dump()
    assert payload["tool_calls"] == [{"tool": "stock_snapshot"}]
    assert "reasoning" not in payload
    assert "thinking_traces" not in payload
