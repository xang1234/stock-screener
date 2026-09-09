"""Reference routing preserves source records while deduplicating article work."""

import json
from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent / "fixtures" / "articles"


def routing_api():
    from app.services.theme_evaluation import reference_routing

    return reference_routing


@pytest.mark.parametrize(
    "case",
    json.loads((FIXTURES / "reference_routes.json").read_text())["classifications"],
)
def test_reference_classification_is_structural(case):
    r = routing_api()
    assert (
        r.classify_reference(case["url"], parent_post_id=case["parent_post_id"])
        == case["expected"]
    )


def test_destination_normalization_removes_only_known_tracking_parameters():
    r = routing_api()
    assert r.normalize_destination(
        "HTTPS://Publisher.Example/story?id=17&utm_source=x&token=keep#fragment"
    ) == "https://publisher.example/story?id=17&token=keep"
    assert r.normalize_destination(
        "https://publisher.example/story?id=18"
    ) != r.normalize_destination("https://publisher.example/story?id=17")


def test_two_references_to_one_final_destination_share_work_but_both_survive():
    r = routing_api()
    routes = [
        r.route_reference(
            reference_id="ref:short",
            source_post_id="post:1",
            original_url="https://t.co/one",
            final_url="https://publisher.example/story/17?utm_source=x",
        ),
        r.route_reference(
            reference_id="ref:direct",
            source_post_id="post:2",
            original_url="https://publisher.example/story/17",
            final_url="https://publisher.example/story/17",
        ),
    ]
    groups = r.group_article_destinations(routes)
    assert len(groups) == 1
    assert groups[0].identity_url == "https://publisher.example/story/17"
    assert [item.reference_id for item in groups[0].references] == [
        "ref:short",
        "ref:direct",
    ]
    assert [item.source_post_id for item in groups[0].references] == [
        "post:1",
        "post:2",
    ]


def test_canonical_alias_requires_same_origin_and_nonconflicting_article_identity():
    r = routing_api()
    assert r.destination_identity(
        "https://publisher.example/story/17?utm_source=x",
        "https://publisher.example/story/17",
    ) == "https://publisher.example/story/17"
    assert r.destination_identity(
        "https://publisher.example/story/17",
        "https://other.example/story/17",
    ) == "https://publisher.example/story/17"
    assert r.destination_identity(
        "https://publisher.example/story/17",
        "https://publisher.example/story/18",
    ) == "https://publisher.example/story/17"


@pytest.mark.parametrize(
    "final_url,canonical_url",
    [
        (
            "https://example.com/news/2026/09/article-a-123",
            "https://example.com/news/2026/09/article-b-456",
        ),
        ("https://example.com/news/123", "https://example.com/"),
        (
            "https://example.com/view?article_id=123",
            "https://example.com/view?article_id=456",
        ),
        ("https://example.com/view?page=123", "https://example.com/view"),
    ],
)
def test_canonical_alias_cannot_discard_article_identity(final_url, canonical_url):
    r = routing_api()
    assert r.destination_identity(final_url, canonical_url) == final_url


def test_linked_post_manifest_deduplicates_and_bounds_one_hop():
    r = routing_api()
    routes = [
        r.route_reference(
            reference_id="ref:queue",
            source_post_id="100",
            original_url="https://t.co/a",
            final_url="https://x.com/analyst/status/200",
        ),
        r.route_reference(
            reference_id="ref:duplicate",
            source_post_id="101",
            original_url="https://t.co/b",
            final_url="https://twitter.com/other/status/200",
        ),
        r.route_reference(
            reference_id="ref:base-cycle",
            source_post_id="100",
            original_url="https://t.co/c",
            final_url="https://x.com/analyst/status/101",
        ),
        r.route_reference(
            reference_id="ref:second",
            source_post_id="100",
            original_url="https://t.co/d",
            final_url="https://x.com/analyst/status/300",
        ),
        r.route_reference(
            reference_id="ref:excess",
            source_post_id="100",
            original_url="https://t.co/e",
            final_url="https://x.com/analyst/status/400",
        ),
        r.route_reference(
            reference_id="ref:second-hop",
            source_post_id="200",
            original_url="https://t.co/f",
            final_url="https://x.com/analyst/status/500",
            source_depth=1,
        ),
    ]
    manifest = r.build_linked_post_manifest(
        routes, base_post_ids={"100", "101"}, max_posts=2
    )
    assert manifest.post_ids == ("200", "300")
    assert {item.reference_id: item.disposition for item in manifest.entries} == {
        "ref:queue": "queued",
        "ref:duplicate": "duplicate",
        "ref:base-cycle": "already_in_base",
        "ref:second": "queued",
        "ref:excess": "limit_exceeded",
        "ref:second-hop": "one_hop_limit",
    }


def test_article_stage_fetches_distinct_short_links_then_reuses_final_capture():
    from app.services.theme_evaluation import article_stage
    from app.services.theme_evaluation.public_fetch import PublicResponse

    calls = []

    def fetch(url):
        calls.append(url)
        return PublicResponse(
            body=b"<article><p>Shared article body.</p></article>",
            final_url="https://publisher.example/story/17",
            content_type="text/html",
        )

    outcome = article_stage.recover_references(
        [
            article_stage.ReferenceInput("ref:a", "post:1", "https://t.co/a"),
            article_stage.ReferenceInput("ref:b", "post:2", "https://t.co/b"),
        ],
        fetcher=fetch,
    )
    assert calls == ["https://t.co/a", "https://t.co/b"]
    assert len(outcome.captures) == 1
    assert outcome.capture_for("ref:a") is outcome.capture_for("ref:b")
    assert [route.reference_id for route in outcome.routes] == ["ref:a", "ref:b"]


def test_article_stage_exposes_credible_canonical_identity_on_each_route():
    from app.services.theme_evaluation import article_stage
    from app.services.theme_evaluation.public_fetch import PublicResponse

    def fetch(_url):
        return PublicResponse(
            body=(
                b'<link rel="canonical" href="https://publisher.example/story/17">'
                b"<article><p>Canonical article body.</p></article>"
            ),
            final_url="https://publisher.example/story/17?utm_source=x",
            content_type="text/html",
        )

    outcome = article_stage.recover_references(
        [article_stage.ReferenceInput("ref:a", "post:1", "https://t.co/a")],
        fetcher=fetch,
    )
    assert outcome.routes[0].canonical_url == "https://publisher.example/story/17"
    assert list(outcome.captures) == ["https://publisher.example/story/17"]


def test_article_stage_does_not_reuse_body_across_conflicting_canonical_articles():
    from app.services.theme_evaluation import article_stage
    from app.services.theme_evaluation.public_fetch import PublicResponse

    responses = {
        "https://t.co/a": PublicResponse(
            body=(
                b'<link rel="canonical" href="/news/2026/09/article-b-456">'
                b"<article><p>Article A body.</p></article>"
            ),
            final_url="https://publisher.example/news/2026/09/article-a-123",
            content_type="text/html",
        ),
        "https://t.co/b": PublicResponse(
            body=b"<article><p>Article B body.</p></article>",
            final_url="https://publisher.example/news/2026/09/article-b-456",
            content_type="text/html",
        ),
    }
    outcome = article_stage.recover_references(
        [
            article_stage.ReferenceInput("ref:a", "post:1", "https://t.co/a"),
            article_stage.ReferenceInput("ref:b", "post:2", "https://t.co/b"),
        ],
        fetcher=responses.__getitem__,
    )
    assert len(outcome.captures) == 2
    assert outcome.capture_for("ref:a").recovery.text == "Article A body."
    assert outcome.capture_for("ref:b").recovery.text == "Article B body."
    assert outcome.capture_for("ref:a") is not outcome.capture_for("ref:b")


@pytest.mark.parametrize(
    "message,expected",
    [("http_status_401", "http_401"), ("http_status_403", "http_403")],
)
def test_article_stage_preserves_specific_http_access_failure(message, expected):
    from app.services.theme_evaluation import article_stage

    def fail(_url):
        raise ValueError(message)

    outcome = article_stage.recover_references(
        [article_stage.ReferenceInput("ref:a", "post:1", "https://example.com/a")],
        fetcher=fail,
    )
    assert outcome.assessment_for("ref:a") == expected
