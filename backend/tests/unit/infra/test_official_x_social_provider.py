import json
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import pytest

from app.domain.social_signals.records import SocialReadRequest
from app.infra.providers.official_x_social_provider import OfficialXSocialProvider

NOW = datetime(2026, 9, 7, 3, 0, tzinfo=timezone.utc)
FIXTURE = Path(__file__).parents[2] / "fixtures/social/official_list_read.json"


def read_request(**changes):
    values = dict(request_id="read-1", source_id="source-9", list_id="1986290701492232693",
                  intent="incremental", observed_at=NOW, limit=2,
                  target_published_after=datetime(2026, 8, 24, tzinfo=timezone.utc))
    values.update(changes)
    return SocialReadRequest(**values)


def make_provider(handler, *, token="secret", reservations=None, sleeps=None, **changes):
    reservations = reservations if reservations is not None else []
    sleeps = sleeps if sleeps is not None else []
    def reserve(day, requested_posts, daily_limit):
        reservations.append((day, requested_posts, daily_limit))
        return requested_posts
    return OfficialXSocialProvider(
        bearer_token=token, reservation=reserve,
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.x.com"),
        daily_post_limit=2000, budget_timezone="Asia/Singapore",
        sleep=lambda seconds: sleeps.append(seconds), retry_delay_seconds=0.25, **changes)


def test_reads_approved_dialect_and_normalizes_posts_with_proposed_progress():
    payload, seen, reservations = json.loads(FIXTURE.read_text()), [], []
    def handler(req):
        seen.append(req)
        return httpx.Response(200, json=payload)
    batch = make_provider(handler, reservations=reservations).read_source(read_request())
    assert seen[0].url.path == "/2/lists/1986290701492232693/tweets"
    assert dict(seen[0].url.params) == {
        "tweet.fields":"created_at,public_metrics,author_id,referenced_tweets,entities",
        "expansions":"author_id", "user.fields":"username", "max_results":"5"}
    assert seen[0].headers["authorization"] == "Bearer secret"
    assert reservations == [(date(2026, 9, 7), 5, 2000)]
    assert batch.outcome.proposed_progress == "page-2"
    assert batch.outcome.committed_progress is None
    assert batch.posts[0].author_handle == "alpha_research"
    assert batch.posts[0].url == batch.posts[0].canonical_url == "https://x.com/alpha_research/status/syn-101"
    assert batch.posts[0].source_id == "source-9"
    assert (batch.posts[0].likes, batch.posts[0].views, batch.posts[0].quotes, batch.posts[0].bookmarks) == (0, 12, None, None)
    assert batch.posts[1].quoted_text is None and batch.posts[1].created_at.tzinfo == timezone.utc


def test_caps_pages_to_run_limit_and_resumes_from_application_progress():
    calls, reservations = [], []
    payloads = [
        {"data":[{"id":"1","author_id":"u","text":"one","created_at":"2026-09-05T01:00:00Z","public_metrics":{}}],"includes":{"users":[{"id":"u","username":"a"}]},"meta":{"next_token":"next"}},
        {"data":[{"id":"2","author_id":"u","text":"two","created_at":"2026-09-05T00:00:00Z","public_metrics":{}}],"includes":{"users":[{"id":"u","username":"a"}]},"meta":{"next_token":"unused"}},]
    def handler(req):
        calls.append(dict(req.url.params)); return httpx.Response(200, json=payloads[len(calls)-1])
    batch = make_provider(handler, reservations=reservations, max_results_per_page=5).read_source(
        read_request(intent="initial", limit=2, application_progress="resume")
    )
    assert [c.get("pagination_token") for c in calls] == ["resume", "next"]
    assert [c["max_results"] for c in calls] == ["5", "5"]
    assert reservations == [(date(2026, 9, 7), 5, 2000)] * 2
    assert [p.provider_post_id for p in batch.posts] == ["1", "2"]
    assert batch.outcome.proposed_progress == "unused"


def test_retains_minimum_page_overflow_before_advancing_cursor():
    def raw_post(post_id):
        return {
            "id": str(post_id),
            "author_id": "u",
            "text": f"post {post_id}",
            "created_at": "2026-09-05T01:00:00Z",
            "public_metrics": {},
        }

    payloads = [
        {
            "data": [raw_post(index) for index in range(100)],
            "includes": {"users": [{"id": "u", "username": "a"}]},
            "meta": {"next_token": "page-2"},
        },
        {
            "data": [raw_post(index) for index in range(100, 105)],
            "includes": {"users": [{"id": "u", "username": "a"}]},
            "meta": {"next_token": "page-3"},
        },
    ]
    calls = []

    def handler(req):
        calls.append(dict(req.url.params))
        return httpx.Response(200, json=payloads[len(calls) - 1])

    batch = make_provider(handler).read_source(
        read_request(intent="initial", limit=101)
    )

    assert [call["max_results"] for call in calls] == ["100", "5"]
    assert len(batch.posts) == batch.outcome.received_count == 105
    assert batch.request.limit == 105
    assert batch.outcome.proposed_progress == "page-3"
    assert batch.posts[-1].provider_post_id == "104"


def test_sub_five_reservation_is_exhausted_without_sending_invalid_request():
    requested = []
    provider = OfficialXSocialProvider(
        bearer_token="secret", reservation=lambda *_: 4,
        client=httpx.Client(transport=httpx.MockTransport(
            lambda request: requested.append(request) or httpx.Response(200, json={"data": []})
        )), daily_post_limit=10, budget_timezone="Asia/Singapore",
    )

    batch = provider.read_source(read_request(limit=2))

    assert requested == []
    assert batch.outcome.error_code == "daily_limit_exhausted"


def test_incremental_starts_at_head_and_test_read_never_proposes_progress():
    calls = []
    response = {"data":[], "meta":{"next_token":"next"}}
    def handler(req): calls.append(dict(req.url.params)); return httpx.Response(200, json=response)
    incremental = make_provider(handler).read_source(read_request(application_progress="stale"))
    diagnostic = make_provider(handler).read_source(read_request(intent="test", limit=1, application_progress="stale"))
    assert "pagination_token" not in calls[0]
    assert "pagination_token" not in calls[1]
    assert incremental.outcome.proposed_progress == "next"
    assert diagnostic.outcome.proposed_progress is None


def test_blank_token_and_exhausted_allowance_make_no_request():
    requested = []
    def handler(req): requested.append(req); return httpx.Response(500)
    blank = make_provider(handler, token="  ").read_source(read_request())
    exhausted = OfficialXSocialProvider(bearer_token="secret", reservation=lambda *_: 0,
        client=httpx.Client(transport=httpx.MockTransport(handler)), daily_post_limit=1,
        budget_timezone="Asia/Singapore").read_source(read_request())
    assert requested == []
    assert blank.outcome.error_code == "reauthentication_required"
    assert exhausted.outcome.error_code == "daily_limit_exhausted"


@pytest.mark.parametrize("grant", ["5", 6, -1])
def test_invalid_reservation_grant_fails_closed_without_request(grant):
    requested = []
    def handler(req): requested.append(req); return httpx.Response(200, json={"data":[],"meta":{}})
    social_provider = OfficialXSocialProvider(bearer_token="secret", reservation=lambda *_: grant,
        client=httpx.Client(transport=httpx.MockTransport(handler)), daily_post_limit=10,
        budget_timezone="Asia/Singapore")
    batch = social_provider.read_source(read_request(limit=2))
    assert requested == []
    assert batch.outcome.error_code == "invalid_reservation_grant"


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_map_to_stable_code(status):
    batch = make_provider(lambda _: httpx.Response(status)).read_source(read_request())
    assert batch.posts == () and batch.outcome.error_code == "reauthentication_required"


def test_rate_limit_exposes_utc_reset_time_without_progress():
    batch = make_provider(lambda _: httpx.Response(429, headers={"x-rate-limit-reset":"1788753600"})).read_source(read_request())
    assert batch.outcome.error_code == "rate_limited"
    assert batch.outcome.rate_limit_reset_at == datetime.fromtimestamp(1788753600, timezone.utc)
    assert batch.outcome.proposed_progress is None


def test_one_network_failure_is_delayed_retried_and_reserved_again():
    attempts, sleeps, reservations = 0, [], []
    def handler(_):
        nonlocal attempts
        attempts += 1
        if attempts == 1: raise httpx.ConnectError("synthetic")
        return httpx.Response(200, json={"data":[],"meta":{"result_count":0}})
    batch = make_provider(handler, sleeps=sleeps, reservations=reservations).read_source(read_request())
    assert batch.outcome.read_status == "success"
    assert attempts == 2 and sleeps == [0.25]
    assert reservations == [(date(2026, 9, 7), 5, 2000)] * 2


def test_invalid_json_and_invalid_schema_fail_closed():
    bad_json = make_provider(lambda _: httpx.Response(200, content=b"not-json")).read_source(read_request())
    bad_schema = make_provider(lambda _: httpx.Response(200, json={"data":[{"id":"1"}]})).read_source(read_request())
    assert bad_json.outcome.error_code == "invalid_provider_json"
    assert bad_schema.outcome.error_code == "invalid_provider_schema"


def test_provider_error_envelope_and_malformed_reference_fail_closed():
    error = make_provider(lambda _: httpx.Response(200, json={"errors":[{"title":"bad"}]})).read_source(read_request())
    response={"data":[{"id":"1","author_id":"u","text":"x","created_at":"2026-09-05T01:00:00Z","referenced_tweets":["bad"],"public_metrics":{}}],"includes":{"users":[{"id":"u","username":"a"}]},"meta":{}}
    malformed = make_provider(lambda _: httpx.Response(200,json=response)).read_source(read_request(limit=1))
    assert error.outcome.error_code == "invalid_provider_schema"
    assert malformed.outcome.error_code == "invalid_provider_schema"


def test_response_aliases_normalize_but_conflicts_are_rejected():
    post = {"id":"1","author_id":"u","text":"x","created_at":"2026-09-05T01:00:00Z",
            "referenced_posts":[{"type":"reposted","id":"0"}],"public_metrics":{"repost_count":3}}
    response = {"data":[post],"includes":{"users":[{"id":"u","username":"a"}]},"meta":{}}
    normalized = make_provider(lambda _: httpx.Response(200, json=response)).read_source(read_request(limit=1))
    assert normalized.posts[0].reposts == 3 and normalized.posts[0].is_repost is True
    post["referenced_tweets"], post["public_metrics"]["retweet_count"] = [], 2
    rejected = make_provider(lambda _: httpx.Response(200, json=response)).read_source(read_request(limit=1))
    assert rejected.outcome.error_code == "invalid_provider_schema"


def test_future_provider_timestamp_fails_closed_against_dispatch_clock():
    response={"data":[{"id":"1","author_id":"u","text":"x","created_at":"2026-09-07T03:06:00Z","public_metrics":{}}],"includes":{"users":[{"id":"u","username":"a"}]},"meta":{}}
    batch=make_provider(lambda _: httpx.Response(200,json=response)).read_source(read_request(limit=1))
    assert batch.outcome.error_code == "invalid_provider_schema"


def test_diagnostic_is_capped_at_five_and_history_stops_before_boundary():
    calls = []
    response = {"data":[
        {"id":"new","author_id":"u","text":"new","created_at":"2026-09-05T01:00:00Z","public_metrics":{}},
        {"id":"old","author_id":"u","text":"old","created_at":"2026-08-20T01:00:00Z","public_metrics":{}},
    ],"includes":{"users":[{"id":"u","username":"a"}]},"meta":{"next_token":"older"}}
    def handler(req): calls.append(dict(req.url.params)); return httpx.Response(200,json=response)
    batch = make_provider(handler).read_source(read_request(intent="test", limit=100))
    assert calls == [{
        "tweet.fields":"created_at,public_metrics,author_id,referenced_tweets,entities",
        "expansions":"author_id", "user.fields":"username", "max_results":"5"}]
    assert [post.provider_post_id for post in batch.posts] == ["new"]
    assert batch.outcome.proposed_progress is None


def test_initial_read_crossing_requested_boundary_marks_history_observed():
    response = {"data":[
        {"id":"new","author_id":"u","text":"new","created_at":"2026-09-05T01:00:00Z","public_metrics":{}},
        {"id":"old","author_id":"u","text":"old","created_at":"2026-08-20T01:00:00Z","public_metrics":{}},
    ],"includes":{"users":[{"id":"u","username":"a"}]},"meta":{"next_token":"older"}}
    batch = make_provider(lambda _: httpx.Response(200, json=response)).read_source(
        read_request(intent="initial", limit=100)
    )
    assert [post.provider_post_id for post in batch.posts] == ["new"]
    assert batch.outcome.history_status == "observed_window"
    assert batch.outcome.coverage_reason_codes == ()
