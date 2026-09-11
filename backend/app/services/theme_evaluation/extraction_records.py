"""Strict, immutable records for frozen extraction sidecars.

These records deliberately live outside ``Bundle``.  A sealed source bundle is
never rewritten when a provider result is imported or generated.
"""

from dataclasses import asdict
from hashlib import sha256
from typing import Any, Literal

from pydantic import (
    AwareDatetime,
    Field,
    TypeAdapter,
    model_serializer,
    model_validator,
)

from .bundle import canonical_bytes
from .records import SHA, URL, Record

_SECRET_FIELD_PARTS = (
    "api_key",
    "authorization",
    "cookie",
    "password",
    "secret",
    "access_token",
    "refresh_token",
    "auth_token",
)
_JSON_VALUE = TypeAdapter(dict[str, Any])


def _input_payload(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "input_id"}


def extraction_input_id(value: dict[str, Any]) -> str:
    """Return the content ID bound to every declared extraction input field."""
    payload = _JSON_VALUE.dump_python(_input_payload(value), mode="json")
    return sha256(canonical_bytes(payload)).hexdigest()


def _assert_json_safe(value: Any) -> None:
    """Reject private request material and values that cannot enter canonical JSON."""
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError("non_string_call_key")  # noqa: TRY004 - Pydantic validation contract
            lowered = key.lower()
            if lowered == "token" or any(
                part in lowered for part in _SECRET_FIELD_PARTS
            ):
                raise ValueError("private_call_metadata")
            _assert_json_safe(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_json_safe(nested)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError("non_json_call_value")


class ExtractionInput(Record):
    input_id: SHA
    source_id: str
    source_kind: str
    source_url: URL
    title: str
    text: str
    language: str | None
    published_at: AwareDatetime | None
    available_at: AwareDatetime
    original_text_sha256: SHA
    result_ids: list[SHA]
    input_kind: Literal["original", "translation", "image_transcription", "article"]
    normalization_policy: str | None
    normalization: dict | None = None
    warnings: list[str]

    @model_validator(mode="after")
    def input_id_binds_content(self):
        expected = extraction_input_id(self.model_dump(mode="json"))
        if self.input_id != expected:
            raise ValueError("input_id_mismatch")
        raw_text = self.text
        if self.normalization is not None:
            from .quantity_display import normalize_quantities

            original = self.normalization.get("original")
            if not isinstance(original, str):
                raise ValueError("normalization_original_required")
            expected_view = asdict(normalize_quantities(original))
            if (
                canonical_bytes(expected_view) != canonical_bytes(self.normalization)
                or self.normalization_policy != expected_view["policy_version"]
                or self.text != expected_view["text"]
                or expected_view["issues"]
            ):
                raise ValueError("normalization_evidence_mismatch")
            raw_text = original
        elif self.normalization_policy is not None:
            raise ValueError("normalization_evidence_required")
        if (
            self.input_kind == "original"
            and self.original_text_sha256 != sha256(raw_text.encode()).hexdigest()
        ):
            raise ValueError("original_input_text_hash_mismatch")
        if len(self.result_ids) != len(set(self.result_ids)):
            raise ValueError("duplicate_input_result")
        return self


def make_input(**fields: Any) -> ExtractionInput:
    """Construct an input and calculate its ID; supplied IDs must agree."""
    supplied = fields.pop("input_id", None)
    value = dict(fields)
    value.setdefault("normalization", None)
    value["input_id"] = extraction_input_id(value)
    if supplied is not None and supplied != value["input_id"]:
        raise ValueError("input_id_mismatch")
    return ExtractionInput.model_validate(value)


class ExtractionMention(Record):
    claim_support: (
        dict[str, Literal["supported", "inferred", "unsupported", "absent"]] | None
    ) = None
    theme: str = Field(min_length=1)
    development: str | None = Field(default=None, max_length=1000)
    tickers: list[str]
    sentiment: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0, le=1)
    excerpt: str = Field(max_length=500)

    @model_validator(mode="after")
    def nonblank_strings(self):
        if not self.theme.strip() or any(not ticker.strip() for ticker in self.tickers):
            raise ValueError("blank_extraction_mention")
        return self


class ExtractionCall(Record):
    requested_model: str | None
    messages_sha256: SHA
    parameters_sha256: SHA
    parameters: dict
    status: Literal["success", "failed"]
    actual_model: str | None
    provider: str | None
    usage: dict
    choices: list[dict]
    captured_at: AwareDatetime
    error_code: str | None = None

    @model_validator(mode="after")
    def consistent_call(self):
        if (
            sha256(canonical_bytes(self.parameters)).hexdigest()
            != self.parameters_sha256
        ):
            raise ValueError("call_parameter_hash_mismatch")
        if self.status == "failed" and (not self.error_code or self.choices):
            raise ValueError("failed_call_provenance_invalid")
        if self.status == "success" and self.error_code:
            raise ValueError("successful_call_has_error")
        return self


class ExtractionRecord(Record):
    input_id: SHA
    pipeline: Literal["technical", "fundamental"]
    status: Literal["success", "failed"]
    mentions: list[dict]
    error_code: str | None
    generated_at: AwareDatetime
    requested_model: str
    reference_sha256: SHA
    code_revision: str
    calls: list[dict]
    grounding_context: dict | None = None
    claim_review: dict | None = None

    @model_serializer(mode="wrap")
    def preserve_legacy_shape(self, handler):
        data = handler(self)
        for optional in ("grounding_context", "claim_review"):
            if optional not in self.model_fields_set:
                data.pop(optional, None)
        return data

    @model_validator(mode="after")
    def successful_and_failed_results_are_unambiguous(self):
        if self.grounding_context is not None:
            from app.services.theme_grounding_context import GroundingContext

            context = GroundingContext.model_validate(self.grounding_context)
            if context.primary_input_id != self.input_id:
                raise ValueError("grounding_input_mismatch")
        if self.status == "success":
            if self.error_code is not None:
                raise ValueError("successful_extraction_has_error")
        elif self.mentions or not self.error_code:
            raise ValueError("failed_extraction_requires_no_mentions_and_error")
        for mention in self.mentions:
            ExtractionMention.model_validate(mention)
        for call in self.calls:
            _assert_json_safe(call)
            ExtractionCall.model_validate(call)
        try:
            canonical_bytes({"calls": self.calls})
        except (TypeError, ValueError) as exc:
            raise ValueError("non_json_call_value") from exc
        return self
