"""Live attachment preparation keeps external evidence bounded and attributable."""

from __future__ import annotations

import hashlib
import io

import pytest
from app.services.theme_evaluation.article_recovery import ArticleRecovery
from app.services.theme_evaluation.public_fetch import PublicResponse
from PIL import Image


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (1, 1), "white").save(output, format="PNG")
    return output.getvalue()


def test_article_preparation_preserves_source_and_translates_recovered_text():
    """Dropping source text or treating unverified article recovery as complete is a bug."""
    from app.services.live_attachment_preparation import LiveAttachmentPreparer

    raw = b"""
    <html><head><title>\xec\x82\xbc\xec\x84\xb1 \xeb\xa7\xa4\xec\xb6\x9c</title></head><body>
    <script type=\"application/ld+json\">
    {\"@context\": \"https://schema.org\", \"@type\": \"NewsArticle\",
     \"articleBody\": \"\xec\x82\xbc\xec\x84\xb1\xec\x9d\x98 \xeb\xa7\xa4\xec\xb6\x9c\xec\x9d\xb4 \xec\xa6\x9d\xea\xb0\x80\xed\x96\x88\xeb\x8b\xa4.\"}
    </script></body></html>
    """

    result = LiveAttachmentPreparer(
        fetcher=lambda _: PublicResponse(
            raw, "https://publisher.example/report", "text/html"
        ),
        translator=lambda text, _source, _target: {
            "\uc0bc\uc131 \ub9e4\ucd9c\n\uc0bc\uc131\uc758 \ub9e4\ucd9c\uc774 \uc99d\uac00\ud588\ub2e4.": "Samsung revenue\nSamsung revenue increased.",
        }[text],
    )("article", "https://short.example/post")

    assert (
        result.original_text
        == "\uc0bc\uc131 \ub9e4\ucd9c\n\uc0bc\uc131\uc758 \ub9e4\ucd9c\uc774 \uc99d\uac00\ud588\ub2e4."
    )
    assert result.text == "Samsung revenue\nSamsung revenue increased."
    assert result.content_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.final_url == "https://publisher.example/report"
    assert result.status == "partial"
    assert result.provenance["translation"]["status"] == "success"
    assert "completeness_unverified" in result.provenance["warnings"]


def test_image_preparation_keeps_transcription_and_only_factual_observations():
    """A vision result must remain attributable evidence, not an inferred theme summary."""
    from app.services.live_attachment_preparation import LiveAttachmentPreparer

    image = image_bytes()

    class Vision:
        provider = "test-vision"
        model = "test-model"
        policy_version = "test-image-v1"

        def describe_image(self, _data, _mime_type):
            return {
                "transcription": "\uc0bc\uc131 \ub9e4\ucd9c",
                "observations": ["\ub9c9\ub300 \uadf8\ub798\ud504"],
                "image_type": "chart",
                "uncertainties": [],
            }

    result = LiveAttachmentPreparer(
        fetcher=lambda _url, **_kwargs: PublicResponse(
            image, "https://images.example/chart.png", "image/png"
        ),
        vision=Vision(),
        translator=lambda text, _source, _target: {
            "\uc0bc\uc131 \ub9e4\ucd9c\n\n": "Samsung revenue\n\n",
            "Observations:\n\ub9c9\ub300 \uadf8\ub798\ud504": "Observations:\nBar chart",
        }[text],
    )("image", "https://images.example/chart.png")

    assert (
        result.original_text
        == "\uc0bc\uc131 \ub9e4\ucd9c\n\nObservations:\n\ub9c9\ub300 \uadf8\ub798\ud504"
    )
    assert result.text == "Samsung revenue\n\nObservations:\nBar chart"
    assert result.content_sha256 == hashlib.sha256(image).hexdigest()
    assert result.status == "complete"
    assert result.provenance["image"] == {
        "image_type": "chart",
        "uncertainties": [],
        "provider": "test-vision",
        "model": "test-model",
        "policy_version": "test-image-v1",
    }


def test_failed_attachment_download_raises_a_safe_typed_error():
    """Persisting a transport exception or calling it complete would leak and mislead."""
    from app.services.live_attachment_preparation import (
        LiveAttachmentPreparationError,
        process_attachment,
    )

    def unavailable(_url):
        raise RuntimeError("upstream said token=secret")

    with pytest.raises(LiveAttachmentPreparationError) as failure:
        process_attachment(
            "article", "https://publisher.example/report", fetcher=unavailable
        )

    assert failure.value.code == "attachment_fetch_failed"


@pytest.mark.parametrize(
    ("fetch_error", "expected_code", "retryable", "http_status"),
    [
        ("public_fetch_failed", "attachment_fetch_transient", True, None),
        ("http_status_429", "attachment_fetch_transient", True, 429),
        ("http_status_503", "attachment_fetch_transient", True, 503),
        ("http_status_403", "attachment_fetch_failed", False, 403),
        ("public_address_required", "attachment_fetch_failed", False, None),
    ],
)
def test_fetch_retry_policy_separates_transient_and_terminal_public_errors(
    fetch_error, expected_code, retryable, http_status
):
    """Retrying SSRF rejection or auth failure would burn capacity without new evidence."""
    from app.services.live_attachment_preparation import (
        LiveAttachmentPreparationError,
        process_attachment,
    )

    def unavailable(_url):
        raise ValueError(fetch_error)

    with pytest.raises(LiveAttachmentPreparationError) as failure:
        process_attachment(
            "article", "https://publisher.example/report", fetcher=unavailable
        )

    assert failure.value.code == expected_code
    assert failure.value.retryable is retryable
    assert failure.value.http_status == http_status


def test_transient_translation_is_partial_with_explicit_retry_metadata():
    """Caching a timeout as a successful translation would permanently lose evidence."""
    from app.services.live_attachment_preparation import LiveAttachmentPreparer
    from app.services.theme_evaluation.preparation_failures import PreparationFailure

    raw = b"article"
    recovery = ArticleRecovery(
        title="",
        text="\uc0bc\uc131\uc758 \ub9e4\ucd9c\uc774 \uc99d\uac00\ud588\ub2e4.",
        final_url="https://publisher.example/report",
        method="semantic_html",
        capture_status="full",
        response_sha256=hashlib.sha256(raw).hexdigest(),
    )

    def timeout(_text, _source, _target):
        raise PreparationFailure("model_timeout")

    result = LiveAttachmentPreparer(
        fetcher=lambda _url: PublicResponse(
            raw, "https://publisher.example/report", "text/html"
        ),
        parser=lambda _body, _url: recovery,
        translator=timeout,
    )("article", "https://publisher.example/report")

    assert result.status == "partial"
    assert "translation_retryable" in result.provenance["warnings"]
    assert result.provenance["translation"]["retry"] == {
        "needed": True,
        "codes": ["model_timeout"],
    }


def test_article_text_over_the_live_limit_is_explicitly_partial():
    """Removing the live 10k bound would allow one atypical attachment to dominate work."""
    from app.services.live_attachment_preparation import LiveAttachmentPreparer

    raw = b"html response"
    recovery = ArticleRecovery(
        title="",
        text="123456789",
        final_url="https://publisher.example/report",
        method="semantic_html",
        capture_status="full",
        response_sha256=hashlib.sha256(raw).hexdigest(),
    )
    result = LiveAttachmentPreparer(
        fetcher=lambda _url: PublicResponse(
            raw, "https://publisher.example/report", "text/html"
        ),
        parser=lambda _body, _url: recovery,
        max_article_text_chars=5,
    )("article", "https://publisher.example/report")

    assert result.original_text == "12345"
    assert result.text == "12345"
    assert result.status == "partial"
    assert "article_text_truncated" in result.provenance["warnings"]
    assert result.provenance["source"] == {
        "text_sha256": hashlib.sha256(b"123456789").hexdigest(),
        "text_chars": 9,
        "truncated": True,
    }


def test_empty_image_observation_is_a_safe_terminal_content_gap():
    """An empty model response must never be persisted as a completed attachment."""
    from app.services.live_attachment_preparation import (
        LiveAttachmentPreparationError,
        LiveAttachmentPreparer,
    )

    class EmptyVision:
        def describe_image(self, _data, _mime_type):
            return {
                "transcription": "",
                "observations": [],
                "image_type": "other",
                "uncertainties": [],
            }

    with pytest.raises(LiveAttachmentPreparationError) as failure:
        LiveAttachmentPreparer(
            fetcher=lambda _url, **_kwargs: PublicResponse(
                image_bytes(), "https://images.example/chart.png", "image/png"
            ),
            vision=EmptyVision(),
        )("image", "https://images.example/chart.png")

    assert failure.value.code == "image_content_unavailable"
    assert failure.value.retryable is False


def test_invalid_live_attachment_limit_is_rejected_before_preparation():
    """A zero limit must not silently turn a complete capture into misleading evidence."""
    from app.services.live_attachment_preparation import (
        LiveAttachmentPreparationError,
        process_attachment,
    )

    with pytest.raises(LiveAttachmentPreparationError) as failure:
        process_attachment(
            "article",
            "https://publisher.example/report",
            fetcher=lambda _url: PublicResponse(
                b"body", "https://publisher.example/report", "text/html"
            ),
            parser=lambda _body, _url: ArticleRecovery(
                title="title",
                text="body",
                final_url="https://publisher.example/report",
                method="semantic_html",
                capture_status="full",
                response_sha256=hashlib.sha256(b"body").hexdigest(),
            ),
            max_article_text_chars=0,
        )

    assert failure.value.code == "attachment_limit_invalid"
