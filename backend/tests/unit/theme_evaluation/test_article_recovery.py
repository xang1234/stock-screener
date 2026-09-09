"""Reject unsafe fetches and misleading article completeness."""

from pathlib import Path

import httpx
import pytest

from app.services.theme_evaluation.bundle import sha256


FIXTURES = Path(__file__).parent / "fixtures" / "articles"


def fixture(name):
    return (FIXTURES / name).read_bytes()


def api():
    from app.services.theme_evaluation import article_recovery, public_fetch

    return article_recovery, public_fetch


def test_public_fetch_pins_address_and_checks_redirect_before_request():
    _, f = api()
    calls = []

    def respond(request):
        calls.append(request)
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "publisher.example"
        return httpx.Response(302, headers={"Location": "http://127.0.0.1/private"})

    with pytest.raises(ValueError, match="public"):
        f.fetch_public(
            "https://publisher.example/story",
            transport=httpx.MockTransport(respond),
            resolver=lambda host: ["93.184.216.34"] if host != "127.0.0.1" else [host],
        )
    assert len(calls) == 1


def test_stream_size_limit_rejects_oversized_body():
    _, f = api()
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"x" * 101))
    with pytest.raises(ValueError, match="size"):
        f.fetch_public(
            "https://publisher.example/story",
            transport=transport,
            resolver=lambda host: ["93.184.216.34"],
            max_bytes=100,
        )


def test_semantic_article_keeps_full_text_but_not_assumed_complete():
    a, _ = api()
    raw = b"<html><title>Orders</title><nav>Ignore navigation</nav><article><p>First paragraph.</p><p>Last paragraph.</p></article></html>"
    result = a.parse_article(raw, "https://publisher.example/story")
    assert result.text == "First paragraph.\n\nLast paragraph."
    assert result.capture_status == "partial"
    assert "completeness_unverified" in result.warnings


def test_paywall_jsonld_does_not_make_article_complete():
    a, _ = api()
    raw = b'<script type="application/ld+json">{"@type":"NewsArticle","isAccessibleForFree":false,"articleBody":"An excerpt behind access controls."}</script><main>Subscribe to continue reading</main>'
    result = a.parse_article(raw, "https://publisher.example/story")
    assert result.capture_status == "partial"
    assert "access_restricted" in result.warnings


def test_challenge_and_multiple_article_bodies_are_not_merged():
    a, _ = api()
    challenge = a.parse_article(
        b"<title>Just a moment...</title><main>Verify you are human</main>",
        "https://publisher.example/story",
    )
    assert challenge.text == ""
    assert "access_interstitial" in challenge.warnings
    multiple = a.parse_article(
        b'<script type="application/ld+json">[{"@type":"Article","articleBody":"First story"},{"@type":"Article","articleBody":"Different story"}]</script>',
        "https://publisher.example/story",
    )
    assert multiple.text == ""
    assert "body_ambiguous" in multiple.warnings


def test_redirect_does_not_leak_cookies_between_hosts_sharing_an_ip():
    _, f = api()
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                302,
                headers={
                    "Location": "https://other.example/story",
                    "Set-Cookie": "publisher_private=abc; Path=/; Secure",
                },
            )
        assert "cookie" not in request.headers
        return httpx.Response(200, content=b"article")

    result = f.fetch_public(
        "https://publisher.example/story",
        transport=httpx.MockTransport(respond),
        resolver=lambda host: ["93.184.216.34"],
    )
    assert result.final_url == "https://other.example/story"


def test_encoded_response_is_rejected_before_decompression():
    _, f = api()
    import gzip

    body = gzip.compress(b"x" * 2_000_000)
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200, headers={"Content-Encoding": "gzip"}, stream=httpx.ByteStream(body)
        )
    )
    with pytest.raises(ValueError, match="encoding"):
        f.fetch_public(
            "https://publisher.example/story",
            transport=transport,
            resolver=lambda host: ["93.184.216.34"],
            max_bytes=1000,
        )


def test_script_only_page_is_not_an_article_body():
    a, _ = api()
    result = a.parse_article(
        fixture("script_only.html"), "https://example.com/news/123"
    )
    assert result.text == ""
    assert result.capture_status == "partial"
    assert "body_missing" in result.warnings
    assert "render_required" in result.warnings


def test_reader_controls_are_excluded_without_removing_body_quotes():
    a, _ = api()
    result = a.parse_article(
        fixture("quote_with_controls.html"), "https://example.com/news/123"
    )
    assert "Translate" not in result.text
    assert "Revenue rose 10%." in result.text
    assert "Demand is strong." in result.text
    assert "Subscribe demand accelerated." in result.text


def test_paywall_excerpt_stays_partial_and_reviewable():
    a, _ = api()
    result = a.parse_article(
        fixture("paywall_excerpt.html"), "https://example.com/news/paid-story"
    )
    assert result.text == "The disclosed excerpt contains a concrete demand signal."
    assert result.capture_status == "partial"
    assert "access_restricted" in result.warnings
    assert "completeness_unverified" in result.warnings


def test_conflicting_json_ld_bodies_are_ambiguous():
    a, _ = api()
    result = a.parse_article(
        fixture("conflicting_bodies.html"), "https://example.com/news/123"
    )
    assert result.text == ""
    assert result.method == "unavailable"
    assert "body_ambiguous" in result.warnings


def test_daum_owned_body_excludes_translation_and_related_news_controls():
    a, _ = api()
    result = a.parse_article(
        fixture("daum_article.html"), "https://v.daum.net/v/20260909114236353"
    )
    assert result.text == (
        "Revenue rose 10% as memory demand strengthened.\n\n"
        "Management said “Demand is strong.”"
    )
    assert result.method == "semantic_html"


def test_oracle_announcement_body_is_selected_from_inspected_region():
    a, _ = api()
    result = a.parse_article(
        fixture("oracle_announcement.html"),
        "https://investor.oracle.com/investor-news/news-details/2026/"
        "Oracle-Sets-the-Date/default.aspx",
    )
    assert result.text == (
        "Oracle will release fiscal 2027 first-quarter results on September 10.\n\n"
        "A conference call will follow at 4:00 p.m. Central Time."
    )
    assert "Investor Relations" not in result.text


def test_unrelated_canonical_host_is_not_trusted_for_article_identity():
    a, _ = api()
    result = a.parse_article(
        b'<link rel="canonical" href="https://unrelated.example/stolen">'
        b"<article><p>Owned publisher text.</p></article>",
        "https://publisher.example/news/123",
    )
    assert result.canonical_url is None
    assert "untrusted_canonical_url" in result.warnings


def test_same_origin_canonical_is_retained():
    a, _ = api()
    result = a.parse_article(
        b'<link rel="canonical" href="/news/123">'
        b"<article><p>Owned publisher text.</p></article>",
        "https://publisher.example/news/123?utm_source=x",
    )
    assert result.canonical_url == "https://publisher.example/news/123"


def test_invalid_canonical_is_ignored_without_losing_the_body():
    a, _ = api()
    result = a.parse_article(
        b'<link rel="canonical" href="https://publisher.example:bad/story">'
        b"<article><p>Owned publisher text.</p></article>",
        "https://publisher.example/news/123",
    )
    assert result.canonical_url is None
    assert result.text == "Owned publisher text."
    assert "invalid_canonical_url" in result.warnings


def test_new_parse_stores_body_hash_but_legacy_serialization_shape_is_stable():
    a, _ = api()
    parsed = a.parse_article(
        b"<article><p>Body hash evidence.</p></article>",
        "https://publisher.example/news/123",
    )
    assert parsed.body_sha256 == sha256(b"Body hash evidence.")

    legacy = {
        "title": "Legacy",
        "text": "Previously stored body.",
        "final_url": "https://publisher.example/legacy",
        "canonical_url": None,
        "method": "semantic_html",
        "capture_status": "partial",
        "warnings": ["completeness_unverified"],
        "response_sha256": sha256(b"legacy response"),
        "retrieved_at": "2026-09-09T12:16:01Z",
        "match_basis": None,
    }
    assert a.ArticleRecovery.model_validate(legacy).model_dump(mode="json") == legacy
