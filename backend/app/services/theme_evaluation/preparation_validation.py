"""Validate derivative meaning as well as its serialized hash."""

from .article_recovery import ArticleRecovery
from .bundle import sha256
from .multilingual_preparation import TextPreparation
from .preparation_models import ImageObservation


def validate_result(result):
    if not result.payload:
        if result.status != "unavailable":
            raise ValueError("preparation_payload_required")
        return
    stage = result.request.stage
    if stage == "text":
        text = TextPreparation.model_validate(result.payload)
        if text.status != result.status:
            raise ValueError("translation_status_mismatch")
        original = "".join(s.original for s in text.segments)
        if sha256(original.encode()) != result.request.input_sha256:
            raise ValueError("translation_source_hash_mismatch")
        missing = 0
        for segment in text.segments:
            if segment.status == "unavailable":
                if segment.translated is not None:
                    raise ValueError("unavailable_translation_has_text")
                missing += 1
            elif segment.translated is None or (
                segment.original.strip() and not segment.translated.strip()
            ):
                raise ValueError("translation_segment_missing")
            elif (
                segment.status == "identity" and segment.original != segment.translated
            ):
                raise ValueError("identity_translation_changed")
        if missing and result.status not in ("partial", "unavailable"):
            raise ValueError("incomplete_translation")
        if not missing and result.status in ("partial", "unavailable"):
            raise ValueError("translation_status_inconsistent")
    elif stage == "image":
        image = ImageObservation.model_validate(result.payload)
        if result.request.input_sha256 not in result.assets:
            raise ValueError("image_asset_required")
        if image.uncertainties and result.status == "success":
            raise ValueError("image_uncertainty_requires_review")
    else:
        article = ArticleRecovery.model_validate(result.payload)
        if (
            article.response_sha256 != result.request.input_sha256
            or article.response_sha256 not in result.assets
        ):
            raise ValueError("article_asset_mismatch")
        if result.status == "success" and (
            not article.text.strip() or article.capture_status != "full"
        ):
            raise ValueError("incomplete_article")
