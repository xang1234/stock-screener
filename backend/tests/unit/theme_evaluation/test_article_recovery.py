"""Reject unsafe fetches and misleading article completeness."""

import httpx
import pytest


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
    assert "ambiguous_article_bodies" in multiple.warnings


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
