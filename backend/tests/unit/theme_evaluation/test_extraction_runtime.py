import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from app.services.theme_evaluation import extraction_runtime as runtime
from app.services.theme_evaluation.extraction_records import (
    ExtractionRecord,
    make_input,
)

EVAL_URL = "postgresql://readonly@example-eval-db.internal/theme_evaluation"
APP_URL = "postgresql://application@example-app-db.internal/stocks"


class _Result:
    def __init__(self, *, scalar=None, rows=None):
        self.scalar = scalar
        self.rows = rows or []

    def scalar_one_or_none(self):
        return self.scalar

    def mappings(self):
        return self

    def all(self):
        return self.rows


class _Session:
    def __init__(self, marker="isolated-v1", rows=None, settings=None):
        self.marker = marker
        self.rows = (
            rows
            if rows is not None
            else [{"symbol": "NVDA", "name": "NVIDIA", "is_active": True}]
        )
        self.settings = settings if settings is not None else []
        self.queries = []

    def execute(self, statement, values=None):
        query = str(statement)
        self.queries.append(query)
        if "FROM app_settings WHERE key = :key" in query:
            return _Result(scalar=self.marker)
        if "FROM stock_universe" in query:
            return _Result(rows=self.rows)
        return _Result(rows=self.settings)


def _reference(session):
    digest, count, _ = runtime._snapshot(session)
    return {"reference_sha256": digest, "active_stock_count": count}


def _fake_context(session):
    @contextmanager
    def context(_url):
        yield session

    return context


def test_preflight_rejects_absent_eval_dsn_before_opening_database(monkeypatch):
    opened = False

    @contextmanager
    def unexpected(_url):
        nonlocal opened
        opened = True
        yield _Session()

    monkeypatch.setattr(runtime, "_open_evaluation_session", unexpected)

    with pytest.raises(
        runtime.RuntimeUnavailable, match="evaluation_database_url_required"
    ) as raised:
        runtime.preflight(
            eval_database_url=None,
            application_database_url=APP_URL,
            reference_manifest={},
        )

    assert raised.value.code == "evaluation_database_url_required"
    assert opened is False


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://readonly@localhost/theme_evaluation",
        "postgresql://readonly@127.0.0.1/theme_evaluation",
        "postgresql://readonly@[::1]/theme_evaluation",
    ],
)
def test_preflight_rejects_local_aliases_without_opening_database(monkeypatch, url):
    monkeypatch.setattr(
        runtime, "_open_evaluation_session", lambda _url: pytest.fail("opened")
    )

    with pytest.raises(
        runtime.RuntimeUnavailable, match="evaluation_database_host_ambiguous"
    ):
        runtime.preflight(
            eval_database_url=url,
            application_database_url=APP_URL,
            reference_manifest={},
        )


def test_preflight_requires_marker_nonempty_rows_and_matching_digest(monkeypatch):
    session = _Session()
    manifest = _reference(session)
    monkeypatch.setattr(runtime, "_open_evaluation_session", _fake_context(session))

    result = runtime.preflight(
        eval_database_url=EVAL_URL,
        application_database_url=APP_URL,
        reference_manifest=manifest,
    )

    assert result.reference_sha256 == manifest["reference_sha256"]
    assert result.active_stock_count == 1
    assert any("stock_universe" in query for query in session.queries)

    for marker, rows, code in [
        ("wrong", session.rows, "evaluation_database_marker_missing"),
        ("isolated-v1", [], "reference_data_unavailable"),
    ]:
        blocked = _Session(marker=marker, rows=rows)
        monkeypatch.setattr(runtime, "_open_evaluation_session", _fake_context(blocked))
        with pytest.raises(runtime.RuntimeUnavailable, match=code):
            runtime.preflight(
                eval_database_url=EVAL_URL,
                application_database_url=APP_URL,
                reference_manifest=manifest,
            )


def test_preflight_never_accepts_same_parsed_database_identity(monkeypatch):
    monkeypatch.setattr(
        runtime, "_open_evaluation_session", lambda _url: pytest.fail("opened")
    )

    with pytest.raises(
        runtime.RuntimeUnavailable, match="evaluation_database_not_isolated"
    ):
        runtime.preflight(
            eval_database_url="postgresql+psycopg://first@example.internal:5432/stocks",
            application_database_url="postgresql://second@example.internal/stocks",
            reference_manifest={},
        )


def test_preflight_only_rejects_ambiguous_evaluation_host(monkeypatch):
    session = _Session()
    monkeypatch.setattr(runtime, "_open_evaluation_session", _fake_context(session))

    result = runtime.preflight(
        eval_database_url=EVAL_URL,
        application_database_url="postgresql://application@localhost/stocks",
        reference_manifest=_reference(session),
    )

    assert result.active_stock_count == 1


def test_generation_is_disabled_by_default_before_opening_session(monkeypatch):
    monkeypatch.setattr(
        runtime, "_open_evaluation_session", lambda _url: pytest.fail("opened")
    )

    with pytest.raises(runtime.RuntimeUnavailable, match="model_calls_not_enabled"):
        runtime.generate_extractions(
            [],
            eval_database_url=EVAL_URL,
            application_database_url=APP_URL,
            reference_manifest={},
            model=runtime.MINIMAX_MODEL,
            pipelines=["technical"],
            max_documents=1,
            code_revision="revision",
        )


def test_generation_rejects_duplicate_inputs_before_opening_session(monkeypatch):
    source = make_input(
        source_id="post:1",
        source_kind="post",
        source_url="https://example.com/post/1",
        title="Supplier orders",
        text="Orders increased.",
        language="en",
        published_at=datetime(2026, 9, 8, tzinfo=timezone.utc),
        available_at=datetime(2026, 9, 8, tzinfo=timezone.utc),
        original_text_sha256=hashlib.sha256(b"Orders increased.").hexdigest(),
        result_ids=[],
        input_kind="original",
        normalization_policy=None,
        warnings=[],
    )
    monkeypatch.setattr(
        runtime, "_open_evaluation_session", lambda _url: pytest.fail("opened")
    )

    with pytest.raises(runtime.RuntimeUnavailable, match="duplicate_extraction_input"):
        runtime.generate_extractions(
            [source, source],
            eval_database_url=EVAL_URL,
            application_database_url=APP_URL,
            reference_manifest={},
            model=runtime.MINIMAX_MODEL,
            pipelines=["technical"],
            max_documents=2,
            code_revision="revision",
            allow_model_calls=True,
        )


def test_isolated_runtime_services_supports_llm_client_without_process_runtime():
    """Evaluation client setup must not depend on application startup wiring."""

    from app.services.llm import LLMService
    from app.wiring import runtime_context

    caller_runtime = runtime_context.current_runtime_services()
    evaluation_session = _Session()
    evaluation_session_factory = lambda: evaluation_session

    with runtime._isolated_runtime_services(evaluation_session_factory):
        bound_runtime = runtime_context.resolve_runtime_services()
        assert bound_runtime.session_factory() is evaluation_session_factory
        assert bound_runtime.session_factory()() is evaluation_session
    assert runtime_context.current_runtime_services() is caller_runtime

    with (
        pytest.raises(ValueError, match="restore"),
        runtime._isolated_runtime_services(evaluation_session_factory),
    ):
        raise ValueError("restore")
    assert runtime_context.current_runtime_services() is caller_runtime

    runtime_context.clear_runtime_services()
    try:
        with pytest.raises(RuntimeError, match="RuntimeServices are not initialized"):
            LLMService(use_case="extraction")

        with runtime._isolated_runtime_services(evaluation_session_factory):
            LLMService(use_case="extraction")
        assert runtime_context.current_runtime_services() is None
    finally:
        runtime_context.set_runtime_services(caller_runtime, bind_process=True)


def test_generation_rejects_missing_primary_client_before_provider_attempts(
    monkeypatch,
):
    class ClientlessThemeExtractionService:
        def __init__(self, _session, pipeline):
            self.llm = None
            self.pipeline = pipeline

    source = make_input(
        source_id="post:1",
        source_kind="post",
        source_url="https://example.com/post/1",
        title="Supplier orders",
        text="Orders increased.",
        language="en",
        published_at=datetime(2026, 9, 8, tzinfo=timezone.utc),
        available_at=datetime(2026, 9, 8, tzinfo=timezone.utc),
        original_text_sha256=hashlib.sha256(b"Orders increased.").hexdigest(),
        result_ids=[],
        input_kind="original",
        normalization_policy=None,
        warnings=[],
    )
    session = _Session()
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-kimi-key")
    monkeypatch.setattr(runtime, "_open_evaluation_session", _fake_context(session))
    monkeypatch.setattr(
        "app.services.theme_extraction_service.ThemeExtractionService",
        ClientlessThemeExtractionService,
    )
    monkeypatch.setattr(
        runtime,
        "_ApprovedExtractionRoute",
        lambda *_args: pytest.fail("provider route should not be constructed"),
    )

    with pytest.raises(
        runtime.RuntimeUnavailable, match="extraction_primary_client_unavailable"
    ):
        runtime.generate_extractions(
            [source],
            eval_database_url=EVAL_URL,
            application_database_url=APP_URL,
            reference_manifest=_reference(session),
            model=runtime.MINIMAX_MODEL,
            pipelines=["technical"],
            max_documents=1,
            code_revision="revision",
            allow_model_calls=True,
        )


@pytest.mark.parametrize("grounded", [False, True])
@pytest.mark.parametrize("review_failure", [False, True])
def test_generation_reuses_one_kimi_client_for_all_batch_records(
    monkeypatch, grounded, review_failure
):
    class ExtractionService:
        def __init__(self, _session, pipeline):
            self.llm = SimpleNamespace(preset=SimpleNamespace(primary=None))
            self.pipeline = pipeline

        def extract_from_content(self, _item, *, grounding_context=None):
            assert grounding_context is not None
            assert grounding_context.primary_input_id == _item.external_id
            assert grounding_context.warnings == (
                ["frozen_test"] if grounded else ["grounding_not_enabled"]
            )
            if review_failure:
                from app.services.theme_claim_review import ClaimReviewError

                self.last_claim_review = {
                    "status": "unavailable",
                    "candidates": [],
                    "decisions": [],
                }
                raise ClaimReviewError(
                    "claim_review_unavailable"
                ) from runtime.RuntimeUnavailable("kimi_fallback_rate_limited")
            return []

    created = []

    class KimiClient:
        def __init__(self, api_key):
            self.api_key = api_key
            created.append(self)

    inputs = [
        make_input(
            source_id=f"post:{index}",
            source_kind="post",
            source_url=f"https://example.com/post/{index}",
            title="Supplier orders",
            text=f"Orders increased {index}.",
            language="en",
            published_at=datetime(2026, 9, 8, tzinfo=timezone.utc),
            available_at=datetime(2026, 9, 8, tzinfo=timezone.utc),
            original_text_sha256=hashlib.sha256(
                f"Orders increased {index}.".encode()
            ).hexdigest(),
            result_ids=[],
            input_kind="original",
            normalization_policy=None,
            warnings=[],
        )
        for index in range(2)
    ]
    session = _Session()
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-kimi-key")
    monkeypatch.setattr(runtime, "_open_evaluation_session", _fake_context(session))
    monkeypatch.setattr(
        "app.services.theme_extraction_service.ThemeExtractionService",
        ExtractionService,
    )
    monkeypatch.setattr(runtime, "_KimiExtractionClient", KimiClient)

    from app.services.theme_evaluation.bundle import canonical_bytes
    from app.services.theme_grounding_context import GroundingContext

    manifest = {
        "bundle_id": "a" * 64,
        "preparation_id": "b" * 64,
        "assessment_id": "c" * 64,
        "approval": {"reviewer": "test"},
    }
    packet = {
        "policy_version": "grounding-v1",
        "run_id": "d" * 64,
        **manifest,
        "as_of": "2026-09-09T00:00:00Z",
        "contexts": {
            item.input_id: GroundingContext(
                primary_input_id=item.input_id,
                context_available_at="2026-09-09T00:00:00Z",
                warnings=["frozen_test"],
            ).model_dump(mode="json")
            for item in inputs
        },
    }
    packet["digest"] = hashlib.sha256(canonical_bytes(packet)).hexdigest()
    records = runtime.generate_extractions(
        inputs,
        grounding_packet=packet if grounded else None,
        grounding_manifest=manifest if grounded else None,
        eval_database_url=EVAL_URL,
        application_database_url=APP_URL,
        reference_manifest=_reference(session),
        model=runtime.MINIMAX_MODEL,
        pipelines=["technical"],
        max_documents=2,
        code_revision="revision",
        allow_model_calls=True,
    )

    assert len(records) == 2
    assert all(
        record.status == ("failed" if review_failure else "success")
        for record in records
    )
    if review_failure:
        assert all(
            record.error_code == "kimi_fallback_rate_limited" for record in records
        )
        assert all(record.claim_review["status"] == "unavailable" for record in records)
    assert all((record.grounding_context is not None) == grounded for record in records)
    assert len(created) == 1


def test_kimi_extraction_session_header_is_stable_and_supports_an_explicit_id(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "opencode_go_api_base", "https://gateway.example/v1/")
    sessions = []

    def handle(request):
        assert str(request.url) == "https://gateway.example/v1/chat/completions"
        sessions.append(request.headers["x-opencode-session"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "[]"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handle)
    first = runtime._KimiExtractionClient("test-key", transport=transport)
    first.complete([], max_tokens=1)
    first.complete([], max_tokens=1)
    explicit = runtime._KimiExtractionClient(
        "test-key", session_id="evaluation-batch-42", transport=transport
    )
    explicit.complete([], max_tokens=1)
    second = runtime._KimiExtractionClient("test-key", transport=transport)
    second.complete([], max_tokens=1)

    assert sessions[0] == sessions[1]
    assert sessions[2] == "evaluation-batch-42"
    assert sessions[3] != sessions[0]


@pytest.mark.asyncio
async def test_approved_route_disables_zai_and_records_primary_then_kimi(monkeypatch):
    class Primary:
        preset = SimpleNamespace(
            primary=SimpleNamespace(model_id=runtime.MINIMAX_MODEL)
        )

        async def completion(self, **kwargs):
            assert kwargs["model"] == runtime.MINIMAX_MODEL
            assert kwargs["allow_fallbacks"] is False
            raise RuntimeError("private provider message")

    kimi = runtime._KimiExtractionClient("not-a-real-key")

    async def fallback(**kwargs):
        assert kwargs["model"] == runtime.KIMI_MODEL
        assert kwargs["allow_fallbacks"] is False
        return SimpleNamespace(
            model=runtime.KIMI_MODEL,
            provider=runtime.KIMI_PROVIDER,
            usage={},
            choices=[],
        )

    monkeypatch.setattr(kimi, "completion", fallback)
    route = runtime._ApprovedExtractionRoute(Primary(), kimi)

    response = await route.completion(
        messages=[{"role": "user", "content": "source text"}],
        model="openai/glm-4.7-flash",
        allow_fallbacks=True,
        temperature=0.2,
        max_tokens=2000,
    )

    assert response.model == runtime.KIMI_MODEL
    assert [call["status"] for call in route.calls] == ["failed", "success"]
    assert [call["requested_model"] for call in route.calls] == [
        runtime.MINIMAX_MODEL,
        runtime.KIMI_MODEL,
    ]
    assert "private provider message" not in str(route.calls)
    assert route.calls[0]["parameters"]["num_retries"] == 0
    assert route.calls[0]["parameters"]["metered"] is True
    await route.completion(
        messages=[{"role": "user", "content": "review candidates"}],
        model=runtime.MINIMAX_MODEL,
        max_tokens=2000,
    )
    assert [call["status"] for call in route.calls] == [
        "failed",
        "success",
        "failed",
        "success",
    ]
    assert route.calls[0]["messages_sha256"] != route.calls[2]["messages_sha256"]


@pytest.mark.asyncio
async def test_kimi_fallback_records_its_actual_public_request_and_response():
    class Primary:
        preset = SimpleNamespace(
            primary=SimpleNamespace(model_id=runtime.MINIMAX_MODEL)
        )

        async def completion(self, **kwargs):
            raise RuntimeError("private MiniMax failure")

    def handler(request):
        payload = json.loads(request.content)
        assert payload["model"] == runtime.KIMI_MODEL
        assert payload["thinking"] == {"type": "disabled"}
        assert "temperature" not in payload
        assert payload["max_tokens"] == 2000
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "[]"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    route = runtime._ApprovedExtractionRoute(
        Primary(),
        runtime._KimiExtractionClient(
            "test-key", transport=httpx.MockTransport(handler)
        ),
    )

    response = await route.completion(
        messages=[{"role": "user", "content": "source text"}],
        model=runtime.MINIMAX_MODEL,
        allow_fallbacks=True,
        temperature=0.2,
        max_tokens=2000,
    )
    record = ExtractionRecord(
        input_id="a" * 64,
        pipeline="technical",
        status="success",
        mentions=[],
        error_code=None,
        generated_at="2026-09-08T10:00:00Z",
        requested_model=runtime.MINIMAX_MODEL,
        reference_sha256="b" * 64,
        code_revision="revision",
        calls=route.calls,
    )

    assert response.choices[0].message.content == "[]"
    assert record.calls[1]["actual_model"] is None
    assert record.calls[1]["provider"] == runtime.KIMI_PROVIDER
    assert record.calls[1]["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "[]"},
            "finish_reason": "stop",
        }
    ]
    assert record.calls[1]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
    }
    assert "temperature" not in record.calls[1]["parameters"]
    assert record.calls[1]["parameters"]["max_tokens"] == 2000
    assert record.calls[1]["parameters"]["allow_fallbacks"] is False
    assert record.calls[1]["parameters"]["thinking"] == {"type": "disabled"}
