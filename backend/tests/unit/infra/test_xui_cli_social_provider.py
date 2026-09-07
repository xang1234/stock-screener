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


def test_test_intent_caps_cli_limit_at_five_and_never_uses_application_progress():
    payload = json.loads(FIXTURE.read_text())
    runner = SyntheticRunner(auth(), completed(payload))
    batch = provider(runner).read_source(request(intent="test", limit=99, application_progress="cursor"))
    command = runner.calls[1][0]
    assert command[command.index("--limit") + 1] == "5"
    assert "--new" not in command and "--checkpoint-mode" not in command
    assert batch.outcome.proposed_progress is None


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


def test_stderr_is_never_parsed_as_success_or_exposed(caplog):
    secret = "/private/profiles/bot/session/storage_state.json cookie=secret"
    runner = SyntheticRunner(auth(), completed("not json", stderr=secret))
    batch = provider(runner).read_source(request())
    assert batch.outcome.error_code == "invalid_provider_json"
    assert secret not in caplog.text and "/private/config.toml" not in caplog.text


@pytest.mark.parametrize("error", [FileNotFoundError("xui"), subprocess.TimeoutExpired(["xui"], 45)])
def test_missing_or_stalled_executable_is_unavailable(error):
    batch = provider(SyntheticRunner(error)).read_source(request())
    assert batch.outcome.error_code == "provider_unavailable"


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
