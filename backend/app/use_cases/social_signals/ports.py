"""Provider-neutral ports for social-signal use cases."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from app.domain.social_signals.records import (
    ConfirmationInput,
    DispatchResult,
    QueuePage,
    SocialReadRequest,
    SocialRunResult,
    SocialSourceBatch,
)


class SocialProvider(Protocol):
    def read_source(self, request: SocialReadRequest) -> SocialSourceBatch: ...


class SocialWriter(Protocol):
    def persist_observations(self, batch: SocialSourceBatch) -> SocialSourceBatch: ...

    def publish(self, run_id: str, expected_mode_version: int) -> SocialRunResult: ...


class ConfirmationReader(Protocol):
    def read(
        self, market: str, symbols: tuple[str, ...], now: datetime
    ) -> tuple[ConfirmationInput, ...]: ...


class PublishedReader(Protocol):
    def queue(
        self,
        market: str,
        window_days: int,
        view: str,
        rank_mode: str,
        page: int,
        page_size: int,
    ) -> QueuePage: ...


class ProviderReadLease(Protocol):
    def acquire(self, owner: str, ttl_seconds: int) -> bool: ...

    def release(self, owner: str) -> None: ...


class SocialDispatcher(Protocol):
    def refresh(self, origin: str) -> DispatchResult: ...

    def test_source(self, source_id: str, actor: str) -> DispatchResult: ...

