"""Prepare one live article or image attachment for source-grounded extraction.

This module deliberately has no database or task-queue dependency.  The caller owns
deduplication, retries, and persistence; this boundary only converts one safely
downloaded attachment into bounded, attributable evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

from app.services.language_detection_service import (
    detect_language as detect_content_language,
)

from .theme_evaluation.article_recovery import ArticleRecovery, parse_article
from .theme_evaluation.image_preparation import (
    OpenCodeGoVision,
    VisionClient,
    prepare_image,
)
from .theme_evaluation.kimi_translation import OpenCodeGoTranslator
from .theme_evaluation.multilingual_preparation import (
    detect_language as detect_preparation_language,
)
from .theme_evaluation.multilingual_preparation import prepare_text
from .theme_evaluation.preparation_failures import PreparationFailure
from .theme_evaluation.public_fetch import PublicResponse, fetch_public
from .theme_evaluation.quantity_display import POLICY_VERSION as QUANTITY_POLICY_VERSION
from .theme_evaluation.quantity_display import normalize_quantities

AttachmentKind = Literal["article", "image"]
AttachmentStatus = Literal["complete", "partial"]

MAX_ARTICLE_TEXT_CHARS = 10_000
MAX_IMAGE_TEXT_CHARS = 10_000


@dataclass(frozen=True)
class PreparedAttachment:
    """Evidence ready for grounding, with the unaltered source text alongside it."""

    text: str
    original_text: str
    content_sha256: str
    final_url: str
    status: AttachmentStatus
    provenance: dict[str, Any]


class LiveAttachmentPreparationError(PreparationFailure):
    """A persistable attachment failure with a deliberately small public code set."""


def _failure(
    code: str, *, http_status: int | None = None
) -> LiveAttachmentPreparationError:
    return LiveAttachmentPreparationError(code, http_status=http_status)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _bounded(value: str, *, limit: int, warning: str) -> tuple[str, list[str]]:
    if len(value) <= limit:
        return value, []
    return value[:limit], [warning]


def _source_provenance(source_text: str, *, truncated: bool) -> dict[str, Any]:
    return {
        "text_sha256": sha256(source_text.encode()).hexdigest(),
        "text_chars": len(source_text),
        "truncated": truncated,
    }


def _http_status(value: str) -> int | None:
    prefix = "http_status_"
    if not value.startswith(prefix) or not value[len(prefix) :].isdigit():
        return None
    status = int(value[len(prefix) :])
    return status if 100 <= status <= 599 else None


def _fetch_failure(error: Exception) -> LiveAttachmentPreparationError:
    detail = str(error) if isinstance(error, ValueError) else ""
    status = _http_status(detail)
    if (
        detail == "public_fetch_failed"
        or status == 429
        or (status is not None and 500 <= status <= 599)
    ):
        return _failure("attachment_fetch_transient", http_status=status)
    return _failure("attachment_fetch_failed", http_status=status)


def _source_language(text: str) -> str:
    """Use the production detector for English and the preparation detector for Korean."""
    fallback, _warnings = detect_preparation_language(text)
    if fallback != "und":
        return fallback
    return detect_content_language(text)


def _translation_provenance(
    prepared, translator: Callable | None, retry: dict[str, Any]
) -> dict[str, Any]:
    provider = getattr(translator, "provider", "injected" if translator else None)
    model = getattr(translator, "model", None) if translator else None
    policy = getattr(translator, "policy_version", None) if translator else None
    return {
        "status": prepared.status,
        "source_language": prepared.source_language,
        "target_language": prepared.target_language,
        "warnings": list(prepared.warnings),
        "segments": [segment.status for segment in prepared.segments],
        "provider": provider,
        "model": model,
        "policy_version": policy,
        "retry": retry,
    }


def _prepare_text(text: str, *, translator: Callable | None):
    language = _source_language(text)
    retry = {"needed": False, "codes": []}

    def traced_translator(original: str, source: str, target: str):
        if translator is None:
            return None
        try:
            return translator(original, source, target)
        except PreparationFailure as error:
            retry["codes"].append(error.code)
            retry["needed"] = retry["needed"] or error.retryable
            raise
        except Exception:
            retry["codes"].append("translation_failed")
            raise

    prepared = prepare_text(
        text, language=language, translator=traced_translator if translator else None
    )
    rendered = "".join(
        segment.translated if segment.translated is not None else segment.original
        for segment in prepared.segments
    )
    quantities = normalize_quantities(rendered)
    warnings = list(prepared.warnings)
    retry["codes"] = _dedupe(retry["codes"])
    if retry["needed"]:
        warnings.append("translation_retryable")
    if quantities.issues:
        warnings.append("quantity_normalization_issue")
    provenance = {
        "translation": _translation_provenance(prepared, translator, retry),
        "quantity_normalization": {
            "policy_version": QUANTITY_POLICY_VERSION,
            "quantities": [
                {
                    "original": quantity.original,
                    "value": quantity.value,
                    "unit": quantity.unit,
                    "display": quantity.display,
                }
                for quantity in quantities.quantities
            ],
            "issues": [issue.code for issue in quantities.issues],
        },
    }
    return quantities.text, warnings, provenance, prepared.status


def _article_text(recovery: ArticleRecovery) -> str:
    parts = [part.strip() for part in (recovery.title, recovery.text) if part.strip()]
    if not recovery.text.strip() or not parts:
        raise _failure("article_content_unavailable")
    return "\n".join(parts)


def _prepare_article(
    url: str,
    *,
    fetcher: Callable[[str], PublicResponse],
    parser: Callable[[bytes, str], ArticleRecovery],
    translator: Callable | None,
    max_article_text_chars: int,
) -> PreparedAttachment:
    try:
        response = fetcher(url)
    except PreparationFailure:
        raise
    except Exception as exc:
        raise _fetch_failure(exc) from exc
    if not isinstance(response, PublicResponse):
        raise _failure("attachment_response_invalid")
    if response.content_type and not any(
        marker in response.content_type.lower() for marker in ("html", "xhtml")
    ):
        raise _failure("article_content_unavailable")
    try:
        recovery = parser(response.body, response.final_url)
    except PreparationFailure:
        raise
    except Exception as exc:
        raise _failure("article_content_unavailable") from exc
    if not isinstance(recovery, ArticleRecovery):
        raise _failure("article_content_unavailable")

    source_text = _article_text(recovery)
    original_text, truncation_warnings = _bounded(
        source_text,
        limit=max_article_text_chars,
        warning="article_text_truncated",
    )
    text, preparation_warnings, preparation, translation_status = _prepare_text(
        original_text, translator=translator
    )
    warnings = _dedupe(
        [*recovery.warnings, *truncation_warnings, *preparation_warnings]
    )
    status: AttachmentStatus = (
        "complete"
        if recovery.capture_status == "full"
        and not truncation_warnings
        and translation_status == "success"
        else "partial"
    )
    return PreparedAttachment(
        text=text,
        original_text=original_text,
        content_sha256=sha256(response.body).hexdigest(),
        final_url=response.final_url,
        status=status,
        provenance={
            "kind": "article",
            "warnings": warnings,
            "source": _source_provenance(
                source_text, truncated=bool(truncation_warnings)
            ),
            "article": {
                "method": recovery.method,
                "capture_status": recovery.capture_status,
                "canonical_url": recovery.canonical_url,
                "response_sha256": recovery.response_sha256,
                "body_sha256": recovery.body_sha256,
            },
            **preparation,
        },
    )


def _image_text(observation) -> str:
    sections = [observation.transcription.strip()]
    if observation.observations:
        sections.append("Observations:\n" + "\n".join(observation.observations))
    return "\n\n".join(section for section in sections if section)


def _prepare_image(
    url: str,
    *,
    fetcher: Callable[..., PublicResponse],
    vision: VisionClient | None,
    translator: Callable | None,
    max_image_text_chars: int,
) -> PreparedAttachment:
    if vision is None:
        raise _failure("vision_provider_unavailable")
    try:
        response = fetcher(url, max_bytes=10 * 1024 * 1024)
    except PreparationFailure:
        raise
    except Exception as exc:
        raise _fetch_failure(exc) from exc
    if not isinstance(response, PublicResponse):
        raise _failure("attachment_response_invalid")
    try:
        observation = prepare_image(response.body, model_client=vision)
    except PreparationFailure as error:
        if error.code == "model_schema_invalid":
            raise _failure("image_content_unavailable") from error
        raise
    except ValueError as exc:
        raise _failure("image_validation_failed") from exc
    source_text = _image_text(observation)
    original_text, truncation_warnings = _bounded(
        source_text,
        limit=max_image_text_chars,
        warning="image_observation_truncated",
    )
    if not original_text:
        raise _failure("image_content_unavailable")
    text, preparation_warnings, preparation, translation_status = _prepare_text(
        original_text, translator=translator
    )
    warnings = _dedupe(
        [*truncation_warnings, *preparation_warnings, *observation.uncertainties]
    )
    status: AttachmentStatus = (
        "complete"
        if not truncation_warnings
        and not observation.uncertainties
        and translation_status == "success"
        else "partial"
    )
    return PreparedAttachment(
        text=text,
        original_text=original_text,
        content_sha256=sha256(response.body).hexdigest(),
        final_url=response.final_url,
        status=status,
        provenance={
            "kind": "image",
            "warnings": warnings,
            "source": _source_provenance(
                source_text, truncated=bool(truncation_warnings)
            ),
            "image": {
                "image_type": observation.image_type.value,
                "uncertainties": list(observation.uncertainties),
                "provider": getattr(vision, "provider", "injected"),
                "model": getattr(vision, "model", None),
                "policy_version": getattr(vision, "policy_version", None),
            },
            **preparation,
        },
    )


def process_attachment(
    kind: str,
    url: str,
    *,
    fetcher: Callable = fetch_public,
    parser: Callable[[bytes, str], ArticleRecovery] = parse_article,
    vision: VisionClient | None = None,
    translator: Callable | None = None,
    max_article_text_chars: int = MAX_ARTICLE_TEXT_CHARS,
    max_image_text_chars: int = MAX_IMAGE_TEXT_CHARS,
) -> PreparedAttachment:
    """Safely recover and prepare one attachment without persistence or retries."""
    if any(
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        for limit in (max_article_text_chars, max_image_text_chars)
    ):
        raise _failure("attachment_limit_invalid")
    if kind == "article":
        return _prepare_article(
            url,
            fetcher=fetcher,
            parser=parser,
            translator=translator,
            max_article_text_chars=max_article_text_chars,
        )
    if kind == "image":
        return _prepare_image(
            url,
            fetcher=fetcher,
            vision=vision,
            translator=translator,
            max_image_text_chars=max_image_text_chars,
        )
    raise _failure("attachment_kind_unsupported")


class LiveAttachmentPreparer:
    """Configured callable for normal ingestion workers using approved Kimi services."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        fetcher: Callable = fetch_public,
        parser: Callable[[bytes, str], ArticleRecovery] = parse_article,
        vision: VisionClient | None = None,
        translator: Callable | None = None,
        max_article_text_chars: int = MAX_ARTICLE_TEXT_CHARS,
        max_image_text_chars: int = MAX_IMAGE_TEXT_CHARS,
    ) -> None:
        if api_key is not None:
            vision = vision or OpenCodeGoVision(api_key)
            translator = translator or OpenCodeGoTranslator(api_key)
        self.fetcher = fetcher
        self.parser = parser
        self.vision = vision
        self.translator = translator
        self.max_article_text_chars = max_article_text_chars
        self.max_image_text_chars = max_image_text_chars

    def __call__(self, kind: str, url: str) -> PreparedAttachment:
        return process_attachment(
            kind,
            url,
            fetcher=self.fetcher,
            parser=self.parser,
            vision=self.vision,
            translator=self.translator,
            max_article_text_chars=self.max_article_text_chars,
            max_image_text_chars=self.max_image_text_chars,
        )


__all__ = [
    "MAX_ARTICLE_TEXT_CHARS",
    "MAX_IMAGE_TEXT_CHARS",
    "AttachmentKind",
    "AttachmentStatus",
    "LiveAttachmentPreparationError",
    "LiveAttachmentPreparer",
    "PreparedAttachment",
    "process_attachment",
]
