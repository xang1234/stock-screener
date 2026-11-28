from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from sqlalchemy import select

from app.database import SessionLocal
from app.infra.db.models.social_analysis import SocialLLMAttempt
from app.infra.db.models.social_signals import SocialSourceRegistry
from app.infra.db.repositories.economic_taxonomy_work_repo import (
    LEASE_DURATION,
    EconomicTaxonomyWorkRepository,
)
from app.models.app_settings import AppSetting
from app.models.economic_taxonomy_runtime import (
    ProcessingRequest,
    ProviderAttempt,
    TaxonomyAuthority,
)
from app.services import economic_taxonomy_llm_provider as provider_module
from app.services.economic_exposure_extraction import (
    EconomicExposureExtractor,
    RetryableProviderError,
)
from app.services.economic_source_admission import (
    EconomicSourceAdmissionService,
    EvidenceAdmission,
)
from app.services.economic_taxonomy_llm_provider import (
    EconomicTaxonomyLLMProvider,
    RoutedEconomicReservationManager,
    build_economic_taxonomy_provider,
)
from app.services.llm.llm_service import LLMPreDispatchError, LLMService
from app.services.social_anthropic_llm import AnthropicMeteredLLM
from app.tasks.economic_taxonomy_tasks import (
    POLICY_BUNDLE_VERSION,
    EconomicTaxonomyTaskService,
)

HAIKU = "anthropic/claude-haiku-5-5"


class _LLM:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def completion(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _response(payload: str):
    return SimpleNamespace(
        id="request-1",
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload))],
    )


def _anthropic_create(monkeypatch, create):
    monkeypatch.setattr(
        anthropic,
        "AsyncAnthropic",
        lambda **_kwargs: SimpleNamespace(messages=SimpleNamespace(create=create)),
    )


def _credentials(monkeypatch, *, anthropic="", minimax="", zai="", opencode=""):
    settings = provider_module.settings
    monkeypatch.setattr(settings, "anthropic_api_key", anthropic)
    monkeypatch.setattr(settings, "minimax_api_key", minimax)
    monkeypatch.setattr(settings, "zai_api_key", zai)
    monkeypatch.setattr(settings, "zai_api_keys", "")
    monkeypatch.setattr(settings, "opencode_go_api_key", opencode)


def test_provider_uses_single_metered_json_dispatch():
    llm = _LLM(_response('{"status":"successful_empty","candidates":[]}'))

    result = EconomicTaxonomyLLMProvider(lambda: llm, model="minimax/MiniMax-M2.7").extract(
        evidence={"original_text": "No economic theme."},
        policy_version="economic-extraction-v1",
    )

    assert result == {
        "status": "successful_empty",
        "candidates": [],
        "provider_request_id": "request-1",
        "actual_cost": 0,
    }
    assert len(llm.calls) == 1
    assert llm.calls[0]["metered"] is True
    assert llm.calls[0]["allow_fallbacks"] is False
    assert llm.calls[0]["num_retries"] == 0
    assert 0 < 2 * llm.calls[0]["timeout"] < LEASE_DURATION.total_seconds()


def test_provider_configuration_failure_is_retryable_before_dispatch():
    provider = EconomicTaxonomyLLMProvider(
        lambda: _LLM(LLMPreDispatchError("missing credentials")), model="minimax/MiniMax-M2.7"
    )

    with pytest.raises(RetryableProviderError) as error:
        provider.extract(evidence={}, policy_version="economic-extraction-v1")

    assert error.value.dispatch_confirmed is False


@pytest.mark.parametrize(
    ("credentials", "expected"),
    [
        ({"anthropic": "a", "minimax": "m", "zai": "z", "opencode": "o"}, HAIKU),
        ({"minimax": "m", "zai": "z", "opencode": "o"}, "minimax/MiniMax-M2.7"),
        ({"zai": "z", "opencode": "o"}, "openai/glm-4.7-flash"),
        ({"opencode": "o"}, "opencode-go/deepseek-v4-flash"),
        ({"anthropic": "  "}, None),
    ],
)
def test_first_configured_credential_selects_the_provider(
    monkeypatch, credentials, expected
):
    _credentials(monkeypatch, **credentials)

    provider, reservations = build_economic_taxonomy_provider(SessionLocal)

    if expected is None:
        assert (provider, reservations) == (None, None)
        return
    assert provider.model == reservations.model == expected
    assert isinstance(provider.llm(), AnthropicMeteredLLM if expected == HAIKU else LLMService)


def test_haiku_dispatch_sends_free_json_and_settles_social_cost(monkeypatch):
    _credentials(monkeypatch, anthropic="test-key", minimax="m")
    with SessionLocal.begin() as db:
        db.add(
            AppSetting(
                key="social_llm_pricing",
                category="social",
                value=json.dumps(
                    {
                        "version": "haiku-v1",
                        "models": {
                            HAIKU: {
                                "provider": "anthropic",
                                "actual_models": ["claude-haiku-5-5"],
                                "input_usd_per_million": "0.10",
                                "output_usd_per_million": "0.50",
                            }
                        },
                    }
                ),
            )
        )
    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            id="msg_1",
            model="claude-haiku-5-5",
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(
                    type="text",
                    text='```json\n{"status":"successful_empty","candidates":[]}\n```',
                ),
            ],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
        )

    _anthropic_create(monkeypatch, create)
    provider, _reservations = build_economic_taxonomy_provider(SessionLocal)

    result = provider.extract(
        evidence={"original_text": "No economic theme."},
        policy_version="economic-extraction-v1",
    )

    assert sent["model"] == "claude-haiku-5-5"
    assert sent["max_tokens"] == RoutedEconomicReservationManager.OUTPUT_TOKEN_LIMIT
    assert sent["output_config"] == {"effort": "medium"}
    assert 0 < 2 * sent["timeout"] < LEASE_DURATION.total_seconds()
    assert "temperature" not in sent and "response_format" not in sent
    assert result == {
        "status": "successful_empty",
        "candidates": [],
        "provider_request_id": "msg_1",
        "actual_cost": provider.budget.price(HAIKU).cost(1000, 200),
    }
    assert result["actual_cost"] == Decimal("0.0002")


class _MessagesAPI(BaseHTTPRequestHandler):
    # Keep-alive lets the SDK pool the connection across dispatches.
    protocol_version = "HTTP/1.1"
    served = 0

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        type(self).served += 1
        body = json.dumps(
            {
                "id": f"msg_{self.served}",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": '{"status":"successful_empty","candidates":[]}'}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def messages_api(monkeypatch):
    _MessagesAPI.served = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MessagesAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    yield _MessagesAPI
    server.shutdown()
    server.server_close()


def test_haiku_extract_then_review_both_reach_the_provider(monkeypatch, messages_api):
    _credentials(monkeypatch, anthropic="test-key")
    provider, _reservations = build_economic_taxonomy_provider(SessionLocal)

    extraction = provider.extract(evidence={"original_text": "x"}, policy_version="v1")
    review = provider.review(
        extraction=extraction, policy_version="v1", facet_catalog_semantic_hash="h"
    )

    assert (extraction["provider_request_id"], review["provider_request_id"]) == (
        "msg_1",
        "msg_2",
    )
    assert messages_api.served == 2


NOW = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)
HAIKU_PRICE = {
    "provider": "anthropic",
    "actual_models": ["claude-haiku-5-5"],
    "input_usd_per_million": "0.10",
    "output_usd_per_million": "0.50",
}


def status_error(status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError(
        "refused", response=httpx2.Response(status, request=request), body=None
    )


def social_request(monkeypatch, *effects):
    """One Social-backed processing request served by a scripted Haiku client."""
    _credentials(monkeypatch, anthropic="test-key")
    with SessionLocal.begin() as db:
        db.add_all(
            [
                TaxonomyAuthority(
                    id=1,
                    mode="shadow",
                    processing_head_revision=1,
                    authority_epoch=7,
                    writes_fenced=False,
                    semantic_invalidation_revision=0,
                    cutover_catch_up_cursor=[],
                    rollback_state="ready",
                ),
                SocialSourceRegistry(id=1),
                AppSetting(
                    key="social_llm_pricing",
                    category="social",
                    value=json.dumps({"version": "haiku-v1", "models": {HAIKU: HAIKU_PRICE}}),
                ),
            ]
        )
    with SessionLocal.begin() as db:
        admitted = EconomicSourceAdmissionService(db).admit_content(
            EvidenceAdmission(
                provider="x",
                canonical_item_id="haiku-post",
                capture_route="social",
                original_text="Memory pricing rose.",
                preparation_version="social-v1",
                available_at=NOW,
                evidence_channels=("narrative",),
                source_metadata={"social_work_id": 1},
            )
        )
        request_id = EconomicTaxonomyWorkRepository(db).enqueue_request(
            source_lineage_id=admitted.source_lineage_id,
            evidence_packet_id=admitted.packet_id,
            policy_bundle_version=POLICY_BUNDLE_VERSION,
            available_at=NOW,
        ).id

    effects = list(effects)

    async def create(**_kwargs):
        effect = effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return SimpleNamespace(
            id="msg_ok",
            model="claude-haiku-5-5",
            content=[SimpleNamespace(type="text", text=json.dumps(effect))],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
        )

    _anthropic_create(monkeypatch, create)
    provider, reservations = build_economic_taxonomy_provider(SessionLocal)
    extractor = EconomicExposureExtractor(
        SessionLocal,
        provider=provider,
        extraction_policy_version="economic-extraction-v1",
        approved_dimensions={"industry"},
        reservations=reservations,
    )

    def process(rid, lease_token):
        extractor.extract(rid)
        with SessionLocal() as db:
            EconomicTaxonomyWorkRepository(db).complete(
                rid, lease_token=lease_token, result_payload={}
            )
            db.commit()

    service = EconomicTaxonomyTaskService(
        SessionLocal, pipeline=SimpleNamespace(process=process), clock=lambda: NOW
    )
    return request_id, service


def outcomes(request_id):
    with SessionLocal() as db:
        request = db.get(ProcessingRequest, request_id)
        attempts = db.scalars(
            select(ProviderAttempt)
            .where(ProviderAttempt.logical_request_id == request_id)
            .order_by(ProviderAttempt.attempt_number)
        ).all()
        social = db.scalars(select(SocialLLMAttempt).order_by(SocialLLMAttempt.id)).all()
        return (
            (request.status, request.completion_code),
            [attempt.outcome for attempt in attempts],
            [(row.state, row.actual_usd) for row in social],
        )


def test_rate_limited_haiku_dispatch_leaves_the_request_retryable_at_zero_cost(
    monkeypatch,
):
    request_id, service = social_request(monkeypatch, status_error(429))

    result = service.process(limit=1, worker_id="worker:test")

    assert (result["retryable"], result["held"], result["terminal_failures"]) == (1, 0, 0)
    assert outcomes(request_id) == (
        ("retryable", "provider_retryable"),
        ["retryable_failure"],
        [("reconciled", Decimal("0"))],
    )
