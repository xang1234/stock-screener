import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.domain.social_signals.records import SocialReadRequest
from app.infra.providers.xui_cli_social_provider import XuiCliSocialProvider


NOW = datetime(2026, 9, 7, 3, 0, tzinfo=timezone.utc)
FIXTURE = Path(__file__).parents[2] / "fixtures/social/xui_list_read.json"


def request(**changes):
    values = dict(
        request_id="read-1", source_id="content-source-9",
        list_id="1522014550211457024", intent="incremental", observed_at=NOW,
        limit=20, target_published_after=datetime(2026, 8, 24, tzinfo=timezone.utc),
    )
    values.update(changes)
    return SocialReadRequest(**values)


class SyntheticRunner:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def completed(payload, returncode=0, stderr=""):
    stdout = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


def auth(*, authenticated=True, status_code="authenticated", returncode=0, **extra):
    return completed({"profile":"bot", "storage_state_path":"/private/session.json",
                      "authenticated":authenticated, "status_code":status_code,
                      "message":"synthetic", "next_steps":[], **extra}, returncode)


def provider(runner, **changes):
    return XuiCliSocialProvider(config_path="/private/config.toml", profile="bot",
                                runner=runner, timeout_seconds=45, cooldown_seconds=600,
                                **changes)


def test_executes_exact_noninteractive_commands_and_normalizes_fixture():
    runner = SyntheticRunner(auth(), completed(json.loads(FIXTURE.read_text())))
    batch = provider(runner).read_source(request())

    assert runner.calls == [
        (["xui", "auth", "status", "--path", "/private/config.toml", "--profile", "bot", "--json"],
         {"capture_output": True, "text": True, "timeout": 45, "shell": False}),
        (["xui", "read", "--path", "/private/config.toml", "--profile", "bot", "--limit", "20",
          "--json", "--sources", "list:1522014550211457024"],
         {"capture_output": True, "text": True, "timeout": 45, "shell": False}),
    ]
    assert [post.source_id for post in batch.posts] == ["content-source-9"] * 2
    assert [post.reposts for post in batch.posts] == [7, 1]
    assert batch.posts[0].author_handle == "@alpha_research"
    assert batch.posts[0].canonical_url == "https://x.com/alpha_research/status/syn-101"
    assert batch.outcome.read_status == "success"
    assert batch.outcome.committed_progress is None
    assert batch.outcome.proposed_progress is None


def test_normalizes_nullable_repost_flag_from_xui_as_false():
    payload = json.loads(FIXTURE.read_text())
    payload["items"][0]["is_repost"] = None

    batch = provider(SyntheticRunner(auth(), completed(payload))).read_source(request())

    assert batch.outcome.read_status == "success"
    assert batch.outcome.error_code is None
    assert batch.posts[0].is_repost is False


def test_preserves_photo_and_expanded_article_references_but_drops_video_thumbnails():
    payload = json.loads(FIXTURE.read_text())
    payload["items"][0]["image_urls"] = [
        "https://pbs.twimg.com/media/chart.jpg",
        "https://pbs.twimg.com/ext_tw_video_thumb/clip.jpg",
    ]
    payload["items"][0]["article_urls"] = [
        "https://publisher.example.com/semiconductor-report",
        "https://x.com/a/status/other-post",
    ]

    batch = provider(SyntheticRunner(auth(), completed(payload))).read_source(request())

    assert [(attachment.kind, attachment.url) for attachment in batch.posts[0].attachments] == [
        ("image", "https://pbs.twimg.com/media/chart.jpg"),
        ("article", "https://publisher.example.com/semiconductor-report"),
    ]


def test_test_intent_caps_cli_limit_at_five_and_never_uses_application_progress():
    payload = json.loads(FIXTURE.read_text())
    runner = SyntheticRunner(auth(), completed(payload))
    batch = provider(runner).read_source(request(intent="test", limit=99, application_progress="cursor"))
    command = runner.calls[1][0]
    assert command[command.index("--limit") + 1] == "5"
    assert "--new" not in command and "--checkpoint-mode" not in command
    assert batch.outcome.proposed_progress is None


@pytest.mark.parametrize(("intent", "requested", "expected"), [
    ("initial", 999, "999"),
    ("initial", 1000, "1000"),
    ("initial", 1001, "1000"),
    ("incremental", 199, "199"),
    ("incremental", 200, "200"),
    ("incremental", 201, "200"),
    ("test", 4, "4"),
    ("test", 5, "5"),
    ("test", 6, "5"),
])
def test_intent_limits_respect_smaller_requests_and_enforce_hard_caps(intent, requested, expected):
    runner = SyntheticRunner(auth(), completed(json.loads(FIXTURE.read_text())))
    provider(runner).read_source(request(intent=intent, limit=requested))
    command = runner.calls[1][0]
    assert command[command.index("--limit") + 1] == expected


def test_identical_retries_are_repeatable_and_do_not_claim_progress():
    payload = json.loads(FIXTURE.read_text())
    runner = SyntheticRunner(auth(), completed(payload), auth(), completed(payload))
    social_provider = provider(runner)
    first = social_provider.read_source(request())
    retried = social_provider.read_source(request())
    assert retried.posts == first.posts
    assert first.outcome.committed_progress is retried.outcome.committed_progress is None


@pytest.mark.parametrize("bad_payload", [
    [], {}, {"succeeded_sources":1, "failed_sources":0, "outcomes":[], "items":[], "mystery":1},
    {"succeeded_sources":1, "failed_sources":0, "outcomes":[], "items":[]},
])
def test_malformed_or_unknown_success_shapes_fail_closed(bad_payload):
    runner = SyntheticRunner(auth(), completed(bad_payload))
    batch = provider(runner).read_source(request())
    assert batch.posts == ()
    assert batch.outcome.error_code == "invalid_provider_schema"


@pytest.mark.parametrize("mutation", ["bad_summary", "missing_outcome_field", "string_repost"])
def test_incoherent_summary_and_malformed_nested_values_fail_closed(mutation):
    payload = json.loads(FIXTURE.read_text())
    if mutation == "bad_summary":
        payload["succeeded_sources"] = 2
    elif mutation == "missing_outcome_field":
        payload["outcomes"][0].pop("observed_ids")
    else:
        payload["items"][0]["is_repost"] = "false"
    batch = provider(SyntheticRunner(auth(), completed(payload))).read_source(request())
    assert batch.outcome.error_code == "invalid_provider_schema"


@pytest.mark.parametrize(("field", "value"), [
    ("page_loads", True), ("page_loads", -1), ("page_loads", "1"),
    ("scroll_rounds", True), ("scroll_rounds", -1), ("scroll_rounds", "1"),
    ("seen_items", True), ("seen_items", -1), ("seen_items", "2"),
])
def test_required_envelope_counters_are_nonnegative_integers(field, value):
    payload = json.loads(FIXTURE.read_text())
    payload[field] = value
    batch = provider(SyntheticRunner(auth(), completed(payload))).read_source(request())
    assert batch.outcome.error_code == "invalid_provider_schema"


@pytest.mark.parametrize(("envelope_field", "outcome_field"), [
    ("page_loads", "page_loads"),
    ("scroll_rounds", "scroll_rounds"),
    ("seen_items", "observed_ids"),
])
def test_single_source_envelope_counters_match_the_documented_outcome_totals(envelope_field, outcome_field):
    payload = json.loads(FIXTURE.read_text())
    payload[envelope_field] = payload["outcomes"][0][outcome_field] + 1
    batch = provider(SyntheticRunner(auth(), completed(payload))).read_source(request())
    assert batch.outcome.error_code == "invalid_provider_schema"


def test_stderr_is_never_parsed_as_success_or_exposed(caplog):
    secret = "/private/profiles/bot/session/storage_state.json cookie=secret"
    runner = SyntheticRunner(auth(), completed("not json", stderr=secret))
    batch = provider(runner).read_source(request())
    assert batch.outcome.error_code == "invalid_provider_json"
    assert secret not in caplog.text and "/private/config.toml" not in caplog.text


def test_missing_executable_is_unavailable_without_retry():
    runner = SyntheticRunner(FileNotFoundError("xui"))
    batch = provider(runner).read_source(request())
    assert batch.outcome.error_code == "provider_unavailable"
    assert len(runner.calls) == 1


def test_timed_out_list_read_retries_once_after_a_delay(monkeypatch):
    delays = []
    monkeypatch.setattr("time.sleep", lambda seconds: delays.append(seconds))
    runner = SyntheticRunner(
        auth(),
        subprocess.TimeoutExpired(["xui", "read"], 45),
        completed(json.loads(FIXTURE.read_text())),
    )

    batch = provider(runner).read_source(request())

    assert batch.outcome.read_status == "success"
    assert [call[0][1] for call in runner.calls] == ["auth", "read", "read"]
    assert delays == [30]


def test_timed_out_list_read_stops_after_the_single_retry(monkeypatch):
    delays = []
    monkeypatch.setattr("time.sleep", lambda seconds: delays.append(seconds))
    runner = SyntheticRunner(
        auth(),
        subprocess.TimeoutExpired(["xui", "read"], 45),
        subprocess.TimeoutExpired(["xui", "read"], 45),
    )

    batch = provider(runner).read_source(request())

    assert batch.outcome.error_code == "provider_timeout"
    assert [call[0][1] for call in runner.calls] == ["auth", "read", "read"]
    assert delays == [30]


def test_explicit_network_failure_retries_once_without_reauthenticating(monkeypatch):
    delays = []
    monkeypatch.setattr("time.sleep", lambda seconds: delays.append(seconds))
    failed = {
        "succeeded_sources": 0, "failed_sources": 1,
        "page_loads": 1, "scroll_rounds": 0, "seen_items": 0, "items": [],
        "outcomes": [{
            "source_id": "list:1522014550211457024", "source_kind": "list",
            "ok": False, "item_count": 0, "page_loads": 1,
            "scroll_rounds": 0, "observed_ids": 0,
            "error": "temporary network connection failure",
            "html_artifact_path": None, "selector_report_path": None,
        }],
    }
    runner = SyntheticRunner(
        auth(), completed(failed, returncode=2),
        completed(json.loads(FIXTURE.read_text())),
    )

    batch = provider(runner).read_source(request())

    assert batch.outcome.read_status == "success"
    assert [call[0][1] for call in runner.calls] == ["auth", "read", "read"]
    assert delays == [30]


def test_failed_auth_prevents_read_and_starts_reauthentication_cooldown():
    runner = SyntheticRunner(auth(authenticated=False, status_code="login_wall", returncode=2))
    social_provider = provider(runner)
    failed = social_provider.read_source(request())
    cooled = social_provider.read_source(request(observed_at=NOW + timedelta(minutes=5)))
    assert len(runner.calls) == 1
    assert failed.outcome.error_code == "reauthentication_required"
    assert failed.outcome.rate_limit_reset_at == NOW + timedelta(minutes=10)
    assert cooled.outcome.error_code == "reauthentication_required"
    assert cooled.outcome.rate_limit_reset_at == NOW + timedelta(minutes=10)


@pytest.mark.parametrize(("error_text", "expected"), [
    ("challenge required", "reauthentication_required"),
    ("login wall detected", "reauthentication_required"),
    ("reauthentication_required", "reauthentication_required"),
    ("author selector missing", "provider_error"),
    ("selector drift report at /private/artifact.html", "provider_error"),
])
def test_failed_list_outcome_starts_cooldown_and_cannot_be_complete(error_text, expected):
    payload = {"succeeded_sources":0, "failed_sources":1, "page_loads":1, "scroll_rounds":0,
               "seen_items":0, "items":[], "outcomes":[{"source_id":"list:1522014550211457024",
               "source_kind":"list", "ok":False, "item_count":0, "page_loads":1,
               "scroll_rounds":0, "observed_ids":0, "error":error_text,
               "html_artifact_path":"/private/a.html", "selector_report_path":"/private/b.json"}]}
    runner = SyntheticRunner(auth(), completed(payload, returncode=2))
    batch = provider(runner).read_source(request())
    assert batch.outcome.error_code == expected
    assert batch.outcome.processing_status == "failed"
    assert batch.outcome.rate_limit_reset_at == NOW + timedelta(minutes=10)
