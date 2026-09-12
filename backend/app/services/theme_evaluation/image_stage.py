"""Run-scoped image preparation with bounded, input-keyed model retries."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .bundle import sha256
from .image_preparation import (
    MAX_IMAGE_BYTES,
    VisionClient,
    prepare_image,
    validate_image,
)
from .preparation_failures import PreparationFailure
from .preparation_results import ImageRequest, ImageResult
from .preparation_store import PreparationStore

_IMAGE_VALIDATION_CODES = frozenset(
    {
        "animated_image",
        "image_pixel_limit",
        "image_too_large",
        "invalid_image",
        "invalid_image_dimensions",
        "unsupported_image_type",
    }
)


@dataclass(frozen=True)
class ImageStageOutcome:
    result_id: str
    attempt_ids: tuple[str, ...]
    request_count: int
    download_count: int = 0


class ImageStage:
    """Prepare images while sharing one retry budget across duplicate references."""

    def __init__(
        self,
        store: PreparationStore,
        *,
        vision: VisionClient | None,
        fetcher=None,
        allow_network: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.store = store
        self.vision = vision
        self.fetcher = fetcher
        self.allow_network = allow_network
        self.sleep = sleep
        self._outcomes: dict[tuple[str, str, str | None, str], ImageStageOutcome] = {}
        self._downloads: dict[str, tuple[bytes, tuple[str, ...], int]] = {}
        self._download_failures: dict[str, ImageStageOutcome] = {}
        self.model_request_count = 0
        self.download_request_count = 0

    def _request(self, digest: str) -> ImageRequest:
        client = self.vision
        return ImageRequest(
            input_sha256=digest,
            provider=client.provider if client else "local",
            model=client.model if client else None,
            policy_version=client.policy_version if client else "image-v1",
        )

    def _save_unavailable(
        self, location: str, *, is_local: bool, reasons: list[str]
    ) -> str:
        result_id = self.store.save_result(
            ImageResult(
                request=self._request(sha256(location.encode())),
                failure_reasons=reasons,
                source_url=None if is_local else location,
            )
        )
        return result_id

    @staticmethod
    def _download_diagnosis(error: Exception) -> tuple[PreparationFailure, str | None]:
        detail = str(error) if isinstance(error, ValueError) else ""
        status = None
        if detail.startswith("http_status_") and detail[12:].isdigit():
            parsed_status = int(detail[12:])
            status = parsed_status if 100 <= parsed_status <= 599 else None
        code = (
            "image_download_connection_failed"
            if detail == "public_fetch_failed"
            else "image_download_timeout"
            if status == 408
            else "image_download_rate_limited"
            if status == 429
            else "image_download_auth_failed"
            if status in {401, 403}
            else "image_download_server_error"
            if status is not None and 500 <= status <= 599
            else "image_download_http_error"
            if status is not None
            else "image_download_size_limit"
            if detail == "response_size_limit"
            else "image_download_encoding_unsupported"
            if detail == "response_encoding_unsupported"
            else "image_download_redirect_failed"
            if detail in {"redirect_limit", "redirect_without_destination"}
            else "image_download_address_rejected"
            if detail in {"public_address_required", "public_port_required"}
            else "image_download_failed"
        )
        safe_detail = detail if detail in {
            "public_fetch_failed",
            "response_size_limit",
            "response_encoding_unsupported",
            "redirect_limit",
            "redirect_without_destination",
            "public_address_required",
            "public_port_required",
        } or status is not None else None
        return PreparationFailure(code, http_status=status), safe_detail

    def _download(self, location: str):
        cached = self._downloads.get(location)
        if cached:
            return cached
        failed = self._download_failures.get(location)
        if failed:
            return failed

        attempts = []
        count = 0
        while True:
            count += 1
            self.download_request_count += 1
            try:
                data = self.fetcher(location, max_bytes=MAX_IMAGE_BYTES).body
            except Exception as error:  # noqa: BLE001 - retain only classified codes
                failure, detail = self._download_diagnosis(error)
                result_id = self._save_unavailable(
                    location,
                    is_local=False,
                    reasons=[failure.code, *([detail] if detail else [])],
                )
                attempts.append(result_id)
                if failure.retryable and count == 1:
                    self.sleep(1.0)
                    continue
                outcome = ImageStageOutcome(
                    result_id, tuple(attempts), request_count=0, download_count=count
                )
                self._download_failures[location] = outcome
                return outcome
            downloaded = (data, tuple(attempts), count)
            self._downloads[location] = downloaded
            return downloaded

    @staticmethod
    def _combine(
        prefix_ids: tuple[str, ...], download_count: int, outcome: ImageStageOutcome
    ) -> ImageStageOutcome:
        return ImageStageOutcome(
            outcome.result_id,
            prefix_ids + outcome.attempt_ids,
            outcome.request_count,
            download_count,
        )

    def process(self, location: str, *, is_local: bool) -> ImageStageOutcome:
        if is_local:
            try:
                with Path(location).open("rb") as handle:
                    data = handle.read(MAX_IMAGE_BYTES + 1)
            except OSError:
                result_id = self._save_unavailable(
                    location, is_local=True, reasons=["image_file_read_failed"]
                )
                return ImageStageOutcome(result_id, (result_id,), 0)
            prefix_ids = ()
            download_count = 0
        elif not self.allow_network:
            result_id = self._save_unavailable(
                location, is_local=False, reasons=["network_disabled"]
            )
            return ImageStageOutcome(result_id, (result_id,), 0)
        else:
            downloaded = self._download(location)
            if isinstance(downloaded, ImageStageOutcome):
                return downloaded
            data, prefix_ids, download_count = downloaded
        try:
            metadata = validate_image(data)
        except ValueError as failure:
            code = str(failure)
            result_id = self._save_unavailable(
                location,
                is_local=is_local,
                reasons=[code if code in _IMAGE_VALIDATION_CODES else "invalid_image"],
            )
            return ImageStageOutcome(
                result_id, prefix_ids + (result_id,), 0, download_count
            )
        asset_id = self.store.save_asset(data)
        request = self._request(str(metadata["sha256"]))
        key = (
            request.input_sha256,
            request.provider,
            request.model,
            request.policy_version,
        )
        previous = self._outcomes.get(key)
        if previous:
            return self._combine(prefix_ids, download_count, previous)
        cached = self.store.cached(request)
        if cached:
            outcome = ImageStageOutcome(cached[0], (cached[0],), 0)
            self._outcomes[key] = outcome
            return self._combine(prefix_ids, download_count, outcome)
        if self.vision is None:
            result_id = self.store.save_result(
                ImageResult(
                    request=request,
                    assets=[asset_id],
                    failure_reasons=["vision_provider_unavailable"],
                    source_url=None if is_local else location,
                )
            )
            outcome = ImageStageOutcome(result_id, (result_id,), 0)
            self._outcomes[key] = outcome
            return self._combine(prefix_ids, download_count, outcome)

        attempts = []
        request_count = 0
        while True:
            request_count += 1
            self.model_request_count += 1
            try:
                payload = prepare_image(data, model_client=self.vision)
                result = ImageResult(
                    request=request,
                    payload=payload,
                    assets=[asset_id],
                    source_url=None if is_local else location,
                )
            except PreparationFailure as failure:
                deferred = (
                    failure.code == "model_rate_limited"
                    and failure.retry_after_seconds is not None
                    and failure.retry_after_seconds > 30
                )
                result = ImageResult(
                    request=request,
                    assets=[asset_id],
                    failure_reasons=[
                        failure.code,
                        *(["model_retry_deferred"] if deferred else []),
                    ],
                    source_url=None if is_local else location,
                )
                attempt_id = self.store.save_result(result)
                attempts.append(attempt_id)
                if failure.retryable and request_count == 1 and not deferred:
                    delay = (
                        failure.retry_after_seconds
                        if failure.retry_after_seconds is not None
                        else 1.0
                    )
                    self.sleep(delay)
                    continue
                outcome = ImageStageOutcome(attempt_id, tuple(attempts), request_count)
                self._outcomes[key] = outcome
                return self._combine(prefix_ids, download_count, outcome)
            except Exception:  # noqa: BLE001 - preserve only a safe unknown diagnosis
                result = ImageResult(
                    request=request,
                    assets=[asset_id],
                    failure_reasons=["model_failure_unknown"],
                    source_url=None if is_local else location,
                )
                attempt_id = self.store.save_result(result)
                attempts.append(attempt_id)
                outcome = ImageStageOutcome(attempt_id, tuple(attempts), request_count)
                self._outcomes[key] = outcome
                return self._combine(prefix_ids, download_count, outcome)
            attempt_id = self.store.save_result(result)
            attempts.append(attempt_id)
            outcome = ImageStageOutcome(attempt_id, tuple(attempts), request_count)
            self._outcomes[key] = outcome
            return self._combine(prefix_ids, download_count, outcome)
