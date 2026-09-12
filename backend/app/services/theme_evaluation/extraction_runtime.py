"""Bounded, read-only runtime for replaying the production theme extractor.

This module intentionally owns no persisted evaluation state.  Its only database
connection is the explicitly supplied evaluation database, and every query and
extractor read happens in one repeatable-read, read-only transaction.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker

EVALUATION_MARKER_KEY = "theme_evaluation_database"
EVALUATION_MARKER_VALUE = "isolated-v1"
MINIMAX_MODEL = "minimax/MiniMax-M2.7"
KIMI_PROVIDER = "opencode-go"
KIMI_MODEL = "kimi-k2.6"
_POSTGRES_DEFAULT_PORT = 5432
_LOCAL_DATABASE_HOSTS = frozenset(
    {"localhost", "local", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}
)
_EXTRACTOR_SETTING_KEYS = (
    "llm_extraction_model",
    "reprocessing_max_age_days",
    "theme_policy_overrides",
)
_TELEMETRY_PATCH_LOCK = threading.RLock()


class RuntimeUnavailable(RuntimeError):
    """A stable, secret-free prerequisite error for the evaluation CLI."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PreflightResult:
    reference_sha256: str
    active_stock_count: int
    configuration_sha256: str


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _identity(url: URL, *, reject_ambiguous_host: bool) -> tuple[str, int, str]:
    if url.get_backend_name() != "postgresql":
        raise RuntimeUnavailable("evaluation_database_not_postgresql")
    host = (url.host or "").strip().lower().rstrip(".")
    database = (url.database or "").strip()
    if not host or not database:
        raise RuntimeUnavailable("evaluation_database_identity_invalid")
    if reject_ambiguous_host and (
        host in _LOCAL_DATABASE_HOSTS or host.startswith("127.")
    ):
        # A local alias is too easy to accidentally point at the application DB.
        raise RuntimeUnavailable("evaluation_database_host_ambiguous")
    return host, int(url.port or _POSTGRES_DEFAULT_PORT), database


def _validated_urls(
    eval_database_url: str | None, application_database_url: str | None
) -> tuple[URL, tuple[str, int, str]]:
    if not isinstance(eval_database_url, str) or not eval_database_url.strip():
        raise RuntimeUnavailable("evaluation_database_url_required")
    if (
        not isinstance(application_database_url, str)
        or not application_database_url.strip()
    ):
        raise RuntimeUnavailable("application_database_url_required")
    try:
        eval_url = make_url(eval_database_url)
        application_url = make_url(application_database_url)
        eval_identity = _identity(eval_url, reject_ambiguous_host=True)
        # This URL is parsed only; it is never connected to.  A developer may
        # legitimately run the application on a local database, while the eval
        # database still has to use an unambiguous dedicated host.
        application_identity = _identity(application_url, reject_ambiguous_host=False)
    except RuntimeUnavailable:
        raise
    except Exception:  # noqa: BLE001 - URL parsing errors must never reveal a DSN.
        raise RuntimeUnavailable("evaluation_database_url_invalid") from None
    if eval_identity == application_identity:
        raise RuntimeUnavailable("evaluation_database_not_isolated")
    return eval_url, eval_identity


@contextmanager
def _open_evaluation_session(eval_url: URL) -> Iterator[Session]:
    """Open one transaction that cannot mutate the evaluation database."""

    engine = create_engine(eval_url, pool_pre_ping=True)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = factory()
    try:
        # Do this before *any* ORM or raw SQL query.  The separate database role
        # should also be SELECT-only; transaction mode protects this process too.
        session.execute(
            text("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        )
        yield session
    except RuntimeUnavailable:
        raise
    except Exception:  # noqa: BLE001 - database diagnostics must stay secret-free.
        raise RuntimeUnavailable("evaluation_database_unavailable") from None
    finally:
        try:
            session.rollback()
        except Exception as ignored:  # noqa: BLE001 - best-effort cleanup only.
            _ = ignored
        session.close()
        engine.dispose()


def _manifest_digest(reference_manifest: dict) -> str:
    if not isinstance(reference_manifest, dict):
        raise RuntimeUnavailable("reference_manifest_invalid")
    values = {
        str(reference_manifest[key]).lower()
        for key in ("reference_sha256", "sha256", "digest")
        if isinstance(reference_manifest.get(key), str)
    }
    if len(values) != 1 or len(next(iter(values))) != 64:
        raise RuntimeUnavailable("reference_manifest_invalid")
    value = next(iter(values))
    if any(character not in "0123456789abcdef" for character in value):
        raise RuntimeUnavailable("reference_manifest_invalid")
    return value


def _snapshot(session: Session) -> tuple[str, int, str]:
    marker = session.execute(
        text("SELECT value FROM app_settings WHERE key = :key"),
        {"key": EVALUATION_MARKER_KEY},
    ).scalar_one_or_none()
    if marker != EVALUATION_MARKER_VALUE:
        raise RuntimeUnavailable("evaluation_database_marker_missing")

    rows = (
        session.execute(
            text(
                "SELECT symbol, name, is_active FROM stock_universe "
                "WHERE is_active IS TRUE ORDER BY symbol ASC, name ASC"
            )
        )
        .mappings()
        .all()
    )
    if not rows:
        raise RuntimeUnavailable("reference_data_unavailable")
    settings = (
        session.execute(
            text(
                "SELECT key, value FROM app_settings "
                "WHERE key IN ('llm_extraction_model', 'reprocessing_max_age_days', "
                "'theme_policy_overrides') ORDER BY key ASC"
            ),
        )
        .mappings()
        .all()
    )
    reference = {
        "schema": "theme-evaluation-reference-v1",
        # ``is_active`` is included because it is the field in active_filter().
        "active_stock_universe": [
            {
                "symbol": row["symbol"],
                "name": row["name"],
                "is_active": row["is_active"],
            }
            for row in rows
        ],
        "extractor_settings": [dict(row) for row in settings],
    }
    return _sha256(reference), len(rows), _sha256(reference["extractor_settings"])


def _preflight_in_session(
    session: Session, reference_manifest: dict
) -> PreflightResult:
    expected = _manifest_digest(reference_manifest)
    actual, count, configuration_sha = _snapshot(session)
    if actual != expected:
        raise RuntimeUnavailable("reference_manifest_mismatch")
    expected_count = reference_manifest.get("active_stock_count")
    if expected_count is not None and expected_count != count:
        raise RuntimeUnavailable("reference_manifest_mismatch")
    return PreflightResult(
        reference_sha256=actual,
        active_stock_count=count,
        configuration_sha256=configuration_sha,
    )


def preflight(
    *,
    eval_database_url: str | None,
    application_database_url: str | None,
    reference_manifest: dict,
) -> PreflightResult:
    """Verify isolated reference data without importing a provider or app DB session."""

    eval_url, _ = _validated_urls(eval_database_url, application_database_url)
    with _open_evaluation_session(eval_url) as session:
        return _preflight_in_session(session, reference_manifest)


def _model_credentials_available() -> bool:
    return bool(
        isinstance(os.environ.get("MINIMAX_API_KEY"), str)
        and os.environ["MINIMAX_API_KEY"].strip()
        and isinstance(os.environ.get("OPENCODE_GO_API_KEY"), str)
        and os.environ["OPENCODE_GO_API_KEY"].strip()
    )


def _validate_generation_request(
    *,
    model: str,
    pipelines: list[str] | tuple[str, ...],
    max_documents: int,
    allow_model_calls: bool,
) -> tuple[str, ...]:
    if model != MINIMAX_MODEL:
        raise RuntimeUnavailable("extraction_model_not_sanctioned")
    if (
        not isinstance(max_documents, int)
        or isinstance(max_documents, bool)
        or max_documents < 1
    ):
        raise RuntimeUnavailable("max_documents_invalid")
    normalized = tuple(str(value).strip().lower() for value in pipelines)
    if not normalized or any(
        value not in {"technical", "fundamental"} for value in normalized
    ):
        raise RuntimeUnavailable("extraction_pipeline_invalid")
    if len(set(normalized)) != len(normalized):
        raise RuntimeUnavailable("extraction_pipeline_duplicate")
    if not allow_model_calls:
        raise RuntimeUnavailable("model_calls_not_enabled")
    if not _model_credentials_available():
        raise RuntimeUnavailable("model_credentials_unavailable")
    return normalized


def _validated_inputs(inputs: list[Any], max_documents: int) -> list[Any]:
    """Validate the full admitted set before opening the database or a provider."""

    if not isinstance(inputs, list):
        raise RuntimeUnavailable("extraction_inputs_invalid")
    from .extraction_records import ExtractionInput

    try:
        values = [ExtractionInput.model_validate(value) for value in inputs]
    except (TypeError, ValueError):
        raise RuntimeUnavailable("extraction_inputs_invalid") from None
    if len(values) != len({value.input_id for value in values}):
        raise RuntimeUnavailable("duplicate_extraction_input")
    return values[:max_documents]


class _NoopTelemetry:
    def record_extraction(self, *args: Any, **kwargs: Any) -> None:
        return None


@contextmanager
def _isolated_runtime_services(
    session_factory: Callable[[], Session],
) -> Iterator[None]:
    """Bind extractor-only wiring to the supplied evaluation session factory."""

    # LLMService resolves its key managers through RuntimeServices even though
    # constructing the client does not need a database or Redis connection.
    # Bind a fresh, context-local container so that lookup never falls through
    # to application process state.  Resetting the token restores any caller
    # context on both normal and exceptional exits.
    from app.wiring.bootstrap import RuntimeServices
    from app.wiring.runtime_context import reset_runtime_services, set_runtime_services

    token = set_runtime_services(RuntimeServices(session_factory=session_factory))
    try:
        yield
    finally:
        reset_runtime_services(token)


class _PublicJson:
    """A public, JSON-safe provider value for RecordingLLM.model_dump()."""

    def __init__(self, value: dict[str, Any]):
        self._value = value

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        if mode != "json":
            raise ValueError("public_json_mode_required")
        return self._value


class _PublicChoice(_PublicJson):
    """Keep the extractor-compatible message attribute and public choice dump."""

    def __init__(self, value: dict[str, Any], content: str):
        super().__init__(value)
        self.message = SimpleNamespace(content=content)


@contextmanager
def _isolated_telemetry() -> Iterator[None]:
    """Prevent the production extractor's finally block from emitting telemetry."""

    # ThemeExtractionService has no injection point for telemetry.  Its dynamic
    # import makes this narrow, process-local replacement sufficient; the lock
    # avoids leaking the no-op hook to concurrent production extraction calls.
    from app.services import telemetry

    with _TELEMETRY_PATCH_LOCK:
        original = telemetry.get_telemetry
        telemetry.get_telemetry = lambda: _NoopTelemetry()
        try:
            yield
        finally:
            telemetry.get_telemetry = original


class _KimiExtractionClient:
    """The approved OpenCode Go fallback with the existing Kimi conventions."""

    provider = KIMI_PROVIDER
    model = KIMI_MODEL

    def __init__(
        self,
        api_key: str,
        *,
        session_id: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        from .kimi_client import _session_id

        self._api_key = api_key
        self._transport = transport
        self._session_id = _session_id(session_id)
        self.preset = SimpleNamespace(primary=SimpleNamespace(model_id=self.model))

    async def completion(self, **kwargs: Any) -> Any:
        return await asyncio.to_thread(
            self.complete,
            kwargs.get("messages", []),
            max_tokens=int(kwargs.get("max_tokens") or 4000),
        )

    def complete(self, messages: list[dict], *, max_tokens: int) -> Any:
        # This is deliberately separate from OpenCodeGoKimi.complete_json(): the
        # production extractor's normal parser expects an array, not an object.
        from .kimi_client import (
            _MAX_PROVIDER_RESPONSE_BYTES,
            _OPENCODE_GO_ENDPOINT,
            _retry_after_seconds,
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},
        }
        try:
            with (
                httpx.Client(
                    transport=self._transport,
                    timeout=httpx.Timeout(45.0, connect=5.0),
                    trust_env=False,
                ) as client,
                client.stream(
                    "POST",
                    _OPENCODE_GO_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "User-Agent": "stockscreen-theme-evaluation/1.0",
                        "x-opencode-session": self._session_id,
                    },
                    json=payload,
                ) as response,
            ):
                if not 200 <= response.status_code < 300:
                    # Deliberately classify the error without keeping a body.
                    status = response.status_code
                    if status == 429:
                        _retry_after_seconds(response.headers.get("retry-after"))
                        raise RuntimeUnavailable("kimi_fallback_rate_limited")
                    if status in {401, 403}:
                        raise RuntimeUnavailable("kimi_fallback_auth_failed")
                    raise RuntimeUnavailable("kimi_fallback_unavailable")
                length = response.headers.get("content-length")
                if length is not None and (
                    not length.isdigit() or int(length) > _MAX_PROVIDER_RESPONSE_BYTES
                ):
                    raise RuntimeUnavailable("kimi_fallback_response_invalid")
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > _MAX_PROVIDER_RESPONSE_BYTES:
                        raise RuntimeUnavailable("kimi_fallback_response_invalid")
        except RuntimeUnavailable:
            raise
        except httpx.TimeoutException:
            raise RuntimeUnavailable("kimi_fallback_timeout") from None
        except httpx.HTTPError:
            raise RuntimeUnavailable("kimi_fallback_unavailable") from None
        try:
            envelope = json.loads(raw)
            choice = envelope["choices"][0]
            if not isinstance(choice, dict):
                raise TypeError
            if choice.get("finish_reason") != "stop":
                raise TypeError
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise TypeError
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            raise RuntimeUnavailable("kimi_fallback_response_invalid") from None
        usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else {}
        response_model = (
            envelope.get("model") if isinstance(envelope.get("model"), str) else None
        )
        return SimpleNamespace(
            model=response_model,
            provider=self.provider,
            usage=_PublicJson(usage),
            choices=[_PublicChoice(choice, content)],
        )


class _ApprovedExtractionRoute:
    """MiniMax primary route plus the explicitly approved Kimi fallback only."""

    def __init__(self, primary: Any, kimi: _KimiExtractionClient):
        self.primary = primary
        self.kimi = kimi
        self.preset = getattr(primary, "preset", None)
        self.calls: list[dict[str, Any]] = []

    async def completion(self, **kwargs: Any) -> Any:
        from .extraction_capture import RecordingLLM

        request = dict(kwargs)
        request["model"] = MINIMAX_MODEL
        request["allow_fallbacks"] = False
        request["num_retries"] = 0
        request["metered"] = True
        primary = RecordingLLM(self.primary)
        try:
            response = await primary.completion(**request)
        except Exception:  # noqa: BLE001 - any primary provider failure uses Kimi once.
            # RecordingLLM has already captured the stable primary failure with
            # no provider body/error text.  The Kimi request is a new recorded
            # actual provider call, never an LLMService fallback.
            fallback = RecordingLLM(self.kimi)
            fallback_request = dict(request)
            fallback_request["model"] = KIMI_MODEL
            # OpenCode Go uses disabled thinking and deliberately has no
            # temperature field; record the actual effective request shape.
            fallback_request.pop("temperature", None)
            fallback_request.pop("num_retries", None)
            fallback_request.pop("metered", None)
            fallback_request["thinking"] = {"type": "disabled"}
            try:
                response = await fallback.completion(**fallback_request)
            except Exception:
                self.calls.extend(primary.calls + fallback.calls)
                raise
            self.calls.extend(primary.calls + fallback.calls)
            return response
        self.calls.extend(primary.calls)
        return response


def _record_failure(
    input_id: str,
    pipeline: str,
    *,
    model: str,
    reference: str,
    code_revision: str,
    code: str,
    calls: list[dict] | None = None,
    grounding_fields: dict | None = None,
):
    from .extraction_records import ExtractionRecord

    return ExtractionRecord(
        input_id=input_id,
        pipeline=pipeline,
        status="failed",
        mentions=[],
        error_code=code,
        generated_at=datetime.now(timezone.utc),
        requested_model=model,
        reference_sha256=reference,
        code_revision=code_revision,
        calls=calls or [],
        **(grounding_fields or {}),
    )


def generate_extractions(
    inputs: list[Any],
    *,
    eval_database_url: str | None,
    application_database_url: str | None,
    reference_manifest: dict,
    model: str,
    pipelines: list[str] | tuple[str, ...],
    max_documents: int,
    code_revision: str,
    allow_model_calls: bool = False,
    grounding_packet: dict | None = None,
    grounding_bundle: Path | None = None,
    grounding_manifest: dict | None = None,
) -> list[Any]:
    """Extract from admitted inputs only; never cluster, persist, or use app DB."""

    # URL validation is intentionally first so an absent evaluation DSN neither
    # imports a provider nor constructs a production service.
    eval_url, _ = _validated_urls(eval_database_url, application_database_url)
    if (
        not isinstance(max_documents, int)
        or isinstance(max_documents, bool)
        or max_documents < 1
    ):
        raise RuntimeUnavailable("max_documents_invalid")
    # Validate identity-bound inputs before prerequisites which could otherwise
    # reach a database or provider on a malformed/duplicate run.
    admitted = _validated_inputs(inputs, max_documents)
    from app.services.theme_grounding_context import GroundingContext

    from .grounding import validate_grounding

    contexts = (
        validate_grounding(
            grounding_packet,
            _validated_inputs(inputs, len(inputs)),
            base=grounding_bundle,
            manifest=grounding_manifest,
        )
        if grounding_packet is not None
        else {}
    )
    selected_pipelines = _validate_generation_request(
        model=model,
        pipelines=pipelines,
        max_documents=max_documents,
        allow_model_calls=allow_model_calls,
    )
    if not isinstance(code_revision, str) or not code_revision.strip():
        raise RuntimeUnavailable("code_revision_required")

    with _open_evaluation_session(eval_url) as session:
        result = _preflight_in_session(session, reference_manifest)
        # Delayed production imports ensure preflight remains available in a
        # database/provider-free environment.
        from app.models.theme import ContentItem
        from app.services.security_master_service import SecurityMasterResolver
        from app.services.theme_extraction_service import ThemeExtractionService

        from .extraction_records import ExtractionRecord

        records = []
        with _isolated_runtime_services(lambda: session):
            kimi = _KimiExtractionClient(os.environ["OPENCODE_GO_API_KEY"].strip())
            for extraction_input in admitted:
                context = contexts.get(extraction_input.input_id) or GroundingContext(
                    primary_input_id=extraction_input.input_id,
                    warnings=["grounding_not_enabled"],
                )
                grounding_fields = (
                    {"grounding_context": context.model_dump(mode="json")}
                    if grounding_packet is not None
                    else {}
                )
                for pipeline in selected_pipelines:
                    service = ThemeExtractionService(session, pipeline=pipeline)
                    if service.llm is None:
                        raise RuntimeUnavailable(
                            "extraction_primary_client_unavailable"
                        )
                    service.configured_model = MINIMAX_MODEL
                    service._security_master = SecurityMasterResolver()
                    route = _ApprovedExtractionRoute(
                        service.llm,
                        kimi,
                    )
                    service.llm = route
                    service.provider = "litellm"
                    item = ContentItem(
                        source_id=None,
                        source_type=extraction_input.source_kind,
                        source_name=extraction_input.source_id,
                        external_id=extraction_input.input_id,
                        title=extraction_input.title,
                        content=extraction_input.text,
                        url=extraction_input.source_url,
                        published_at=extraction_input.published_at,
                        source_language=extraction_input.language,
                    )
                    try:
                        with _isolated_telemetry():
                            mentions = service.extract_from_content(
                                item, grounding_context=context
                            )
                        records.append(
                            ExtractionRecord(
                                input_id=extraction_input.input_id,
                                pipeline=pipeline,
                                status="success",
                                mentions=mentions,
                                error_code=None,
                                generated_at=datetime.now(timezone.utc),
                                requested_model=model,
                                reference_sha256=result.reference_sha256,
                                code_revision=code_revision,
                                calls=list(route.calls),
                                claim_review=getattr(
                                    service, "last_claim_review", None
                                ),
                                **grounding_fields,
                            )
                        )
                    except RuntimeUnavailable as exc:
                        records.append(
                            _record_failure(
                                extraction_input.input_id,
                                pipeline,
                                model=model,
                                reference=result.reference_sha256,
                                code_revision=code_revision,
                                code=exc.code,
                                calls=list(route.calls),
                                grounding_fields={
                                    **grounding_fields,
                                    "claim_review": getattr(
                                        service, "last_claim_review", None
                                    ),
                                },
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 - persist a stable extraction failure.
                        from app.services.theme_claim_review import ClaimReviewError

                        code = "extraction_failed"
                        if isinstance(exc, ClaimReviewError):
                            code = (
                                exc.__cause__.code
                                if isinstance(exc.__cause__, RuntimeUnavailable)
                                else str(exc)
                            )
                        records.append(
                            _record_failure(
                                extraction_input.input_id,
                                pipeline,
                                model=model,
                                reference=result.reference_sha256,
                                code_revision=code_revision,
                                code=code,
                                calls=list(route.calls),
                                grounding_fields={
                                    **grounding_fields,
                                    "claim_review": getattr(
                                        service, "last_claim_review", None
                                    ),
                                },
                            )
                        )
        return records
