import io
from types import SimpleNamespace

import pytest
from app.services.theme_evaluation.image_stage import ImageStage
from app.services.theme_evaluation.preparation_failures import PreparationFailure
from app.services.theme_evaluation.preparation_store import PreparationStore
from PIL import Image


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (4, 3), "white").save(output, format="PNG")
    return output.getvalue()


def observation():
    return {
        "transcription": "売上 100億円",
        "observations": ["Three bars are visible."],
        "image_type": "chart",
        "uncertainties": [],
    }


class Vision:
    provider = "opencode-go"
    model = "kimi-k2.6"
    policy_version = "image-v1"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def describe_image(self, data, mime_type):
        self.calls.append((data, mime_type))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_transient_model_failure_retries_the_saved_validated_bytes_once(tmp_path):
    data = png_bytes()
    image = tmp_path / "image.png"
    image.write_bytes(data)
    vision = Vision([PreparationFailure("model_timeout"), observation()])
    sleeps = []
    store = PreparationStore(tmp_path / "prepared")
    stage = ImageStage(store, vision=vision, sleep=sleeps.append)

    outcome = stage.process(str(image), is_local=True)

    assert len(vision.calls) == 2
    assert vision.calls == [(data, "image/png"), (data, "image/png")]
    assert sleeps == [1.0]
    assert outcome.request_count == 2
    assert stage.model_request_count == 2
    assert len(outcome.attempt_ids) == 2
    first = store.load_result(outcome.attempt_ids[0])
    final = store.load_result(outcome.result_id)
    assert first.failure_reasons == ["model_timeout"]
    assert first.assets == final.assets
    assert final.status == "success"
    assert outcome.result_id == outcome.attempt_ids[-1]


def test_rejected_image_records_exact_validation_code_without_model_call(tmp_path):
    image = tmp_path / "bad.png"
    image.write_bytes(b"not an image")
    vision = Vision([observation()])
    store = PreparationStore(tmp_path / "prepared")
    stage = ImageStage(store, vision=vision)

    outcome = stage.process(str(image), is_local=True)

    result = store.load_result(outcome.result_id)
    assert result.failure_reasons == ["invalid_image"]
    assert result.assets == []
    assert outcome.request_count == 0
    assert vision.calls == []


def test_remote_download_failure_is_sanitized_before_validation(tmp_path):
    vision = Vision([observation()])
    store = PreparationStore(tmp_path / "prepared")

    def fetcher(*args, **kwargs):
        raise ValueError("private URL response body")

    outcome = ImageStage(
        store, vision=vision, fetcher=fetcher, allow_network=True
    ).process("https://example.com/image.png", is_local=False)

    result = store.load_result(outcome.result_id)
    assert result.failure_reasons == ["image_download_failed"]
    assert "private" not in result.model_dump_json()
    assert outcome.request_count == 0
    assert outcome.download_count == 1
    assert vision.calls == []


@pytest.mark.parametrize(
    ("source_code", "diagnostic_code"),
    [
        ("public_fetch_failed", "image_download_connection_failed"),
        ("http_status_408", "image_download_timeout"),
        ("http_status_429", "image_download_rate_limited"),
        ("http_status_503", "image_download_server_error"),
    ],
)
def test_transient_download_failure_retries_once_before_model_processing(
    tmp_path, source_code, diagnostic_code
):
    data = png_bytes()
    responses = iter([ValueError(source_code), SimpleNamespace(body=data)])
    fetch_calls = []

    def fetcher(url, *, max_bytes):
        fetch_calls.append((url, max_bytes))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    store = PreparationStore(tmp_path / "prepared")
    sleeps = []
    outcome = ImageStage(
        store,
        vision=Vision([observation()]),
        fetcher=fetcher,
        allow_network=True,
        sleep=sleeps.append,
    ).process("https://example.com/image.png", is_local=False)

    assert len(fetch_calls) == 2
    assert outcome.download_count == 2
    assert sleeps == [1.0]
    assert outcome.download_count == 2
    assert outcome.request_count == 1
    assert len(outcome.attempt_ids) == 2
    assert store.load_result(outcome.attempt_ids[0]).failure_reasons == [
        diagnostic_code,
        source_code,
    ]
    assert store.load_result(outcome.result_id).status == "success"


@pytest.mark.parametrize(
    ("source_code", "diagnostic_code"),
    [
        ("http_status_401", "image_download_auth_failed"),
        ("response_size_limit", "image_download_size_limit"),
        ("public_address_required", "image_download_address_rejected"),
    ],
)
def test_deterministic_download_failure_is_terminal_and_keeps_safe_detail(
    tmp_path, source_code, diagnostic_code
):
    calls = []

    def fetcher(*args, **kwargs):
        calls.append(1)
        raise ValueError(source_code)

    store = PreparationStore(tmp_path / "prepared")
    sleeps = []
    outcome = ImageStage(
        store,
        vision=Vision([]),
        fetcher=fetcher,
        allow_network=True,
        sleep=sleeps.append,
    ).process("https://example.com/private.png", is_local=False)

    assert calls == [1]
    assert sleeps == []
    assert outcome.download_count == 1
    assert store.load_result(outcome.result_id).failure_reasons == [
        diagnostic_code,
        source_code,
    ]


def test_download_server_retry_budget_is_shared_by_url(tmp_path):
    calls = []

    def fetcher(*args, **kwargs):
        calls.append(1)
        raise ValueError("http_status_503")

    store = PreparationStore(tmp_path / "prepared")
    stage = ImageStage(
        store,
        vision=Vision([]),
        fetcher=fetcher,
        allow_network=True,
        sleep=lambda _: None,
    )

    first = stage.process("https://example.com/down.png", is_local=False)
    second = stage.process("https://example.com/down.png", is_local=False)

    assert first == second
    assert calls == [1, 1]
    assert stage.download_request_count == 2
    assert first.download_count == 2
    assert len(first.attempt_ids) == 2
    assert store.load_result(first.result_id).failure_reasons == [
        "image_download_server_error",
        "http_status_503",
    ]


def test_network_disabled_and_local_read_failures_have_distinct_codes(tmp_path):
    store = PreparationStore(tmp_path / "prepared")
    stage = ImageStage(store, vision=Vision([]))

    remote = stage.process("https://example.com/image.png", is_local=False)
    local = stage.process(str(tmp_path / "missing.png"), is_local=True)

    assert store.load_result(remote.result_id).failure_reasons == ["network_disabled"]
    assert store.load_result(local.result_id).failure_reasons == [
        "image_file_read_failed"
    ]


def test_remote_image_uses_the_bounded_fetch_boundary(tmp_path):
    data = png_bytes()
    calls = []

    def fetcher(url, *, max_bytes):
        calls.append((url, max_bytes))
        return SimpleNamespace(body=data)

    store = PreparationStore(tmp_path / "prepared")
    outcome = ImageStage(
        store,
        vision=Vision([observation()]),
        fetcher=fetcher,
        allow_network=True,
    ).process("https://example.com/image.png", is_local=False)

    assert calls == [("https://example.com/image.png", 10 * 1024 * 1024)]
    assert store.load_result(outcome.result_id).source_url == (
        "https://example.com/image.png"
    )


def test_validated_image_is_saved_when_vision_provider_is_unavailable(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(png_bytes())
    store = PreparationStore(tmp_path / "prepared")

    outcome = ImageStage(store, vision=None).process(str(image), is_local=True)

    result = store.load_result(outcome.result_id)
    assert result.failure_reasons == ["vision_provider_unavailable"]
    assert result.assets == [result.request.input_sha256]
    assert outcome.request_count == 0


def test_duplicate_references_share_input_retry_accounting(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(png_bytes())
    second.write_bytes(png_bytes())
    vision = Vision(
        [PreparationFailure("model_server_error", http_status=503), observation()]
    )
    store = PreparationStore(tmp_path / "prepared")
    stage = ImageStage(store, vision=vision, sleep=lambda _: None)

    first_outcome = stage.process(str(first), is_local=True)
    second_outcome = stage.process(str(second), is_local=True)

    assert second_outcome == first_outcome
    assert len(vision.calls) == 2
    assert stage.model_request_count == 2
    assert first_outcome.request_count == 2


def test_successful_store_cache_avoids_a_new_model_attempt(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(png_bytes())
    store = PreparationStore(tmp_path / "prepared")
    first_vision = Vision([observation()])
    first = ImageStage(store, vision=first_vision).process(str(image), is_local=True)
    second_vision = Vision([])

    second = ImageStage(store, vision=second_vision).process(str(image), is_local=True)

    assert second.result_id == first.result_id
    assert second.request_count == 0
    assert second.attempt_ids == (first.result_id,)
    assert second_vision.calls == []


def test_invalid_model_schema_is_recorded_without_retry(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(png_bytes())
    vision = Vision([{"transcription": "partial private output"}])
    sleeps = []
    store = PreparationStore(tmp_path / "prepared")

    outcome = ImageStage(store, vision=vision, sleep=sleeps.append).process(
        str(image), is_local=True
    )

    result = store.load_result(outcome.result_id)
    assert result.failure_reasons == ["model_schema_invalid"]
    assert "partial private output" not in result.model_dump_json()
    assert outcome.request_count == 1
    assert sleeps == []


def test_unknown_model_exception_is_terminal_and_sanitized(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(png_bytes())
    vision = Vision([RuntimeError("secret credential and request body")])
    store = PreparationStore(tmp_path / "prepared")

    outcome = ImageStage(store, vision=vision).process(str(image), is_local=True)

    result = store.load_result(outcome.result_id)
    assert result.failure_reasons == ["model_failure_unknown"]
    assert "secret credential" not in result.model_dump_json()
    assert outcome.request_count == 1
    assert len(vision.calls) == 1


def test_rate_limit_honors_bounded_retry_after(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(png_bytes())
    vision = Vision(
        [
            PreparationFailure(
                "model_rate_limited", http_status=429, retry_after_seconds=12
            ),
            observation(),
        ]
    )
    sleeps = []

    outcome = ImageStage(
        PreparationStore(tmp_path / "prepared"),
        vision=vision,
        sleep=sleeps.append,
    ).process(str(image), is_local=True)

    assert outcome.request_count == 2
    assert sleeps == [12.0]


def test_long_rate_limit_wait_is_deferred_without_retry(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(png_bytes())
    vision = Vision(
        [
            PreparationFailure(
                "model_rate_limited", http_status=429, retry_after_seconds=31
            )
        ]
    )
    sleeps = []
    store = PreparationStore(tmp_path / "prepared")

    outcome = ImageStage(store, vision=vision, sleep=sleeps.append).process(
        str(image), is_local=True
    )

    assert store.load_result(outcome.result_id).failure_reasons == [
        "model_rate_limited",
        "model_retry_deferred",
    ]
    assert outcome.request_count == 1
    assert len(vision.calls) == 1
    assert sleeps == []


def test_exhausted_retry_budget_is_shared_by_later_references(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(png_bytes())
    second.write_bytes(png_bytes())
    vision = Vision(
        [PreparationFailure("model_timeout"), PreparationFailure("model_timeout")]
    )
    store = PreparationStore(tmp_path / "prepared")
    stage = ImageStage(store, vision=vision, sleep=lambda _: None)

    first_outcome = stage.process(str(first), is_local=True)
    second_outcome = stage.process(str(second), is_local=True)

    assert first_outcome == second_outcome
    assert first_outcome.request_count == 2
    assert len(first_outcome.attempt_ids) == 2
    assert len(vision.calls) == 2
    assert store.load_result(first_outcome.result_id).failure_reasons == [
        "model_timeout"
    ]
