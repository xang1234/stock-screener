"""Prepare an exact, already captured X translation without another model call."""

from .multilingual_preparation import (
    TextPreparation,
    TranslationSegment,
    detect_language,
)
from .preparation_results import TextRequest, TextResult


def captured_translation_result(doc):
    capture = doc.source_metadata.x_translation
    if capture is None or capture.status != "captured":
        return None
    capture.check_source(doc.document_id.removeprefix("post:"), doc.text)
    language = capture.source_language or doc.original_language
    source, warnings = detect_language(doc.text, language)
    return TextResult(
        request=TextRequest(
            input_sha256=doc.text_sha256,
            provider=capture.provider,
            model=None,
            policy_version="x-rendered-v1",
            language=language,
            method="translation_import",
        ),
        payload=TextPreparation(
            source_language=source,
            supplied_language=language,
            target_language="en",
            language_warnings=warnings,
            segments=[
                TranslationSegment(
                    original=doc.text, translated=capture.text, status="translated"
                )
            ],
        ),
        created_at=capture.captured_at,
    )
