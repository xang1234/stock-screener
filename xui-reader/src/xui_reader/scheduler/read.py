"""Multi-source read orchestration helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from xui_reader.auth import storage_state_path
from xui_reader.browser.session import PlaywrightBrowserSession
from xui_reader.collectors.timeline import TimelineCollector
from xui_reader.config import RuntimeConfig
from xui_reader.errors import CollectError, SchedulerError
from xui_reader.extract.tweets import PrimaryFallbackTweetExtractor
from xui_reader.models import SourceRef, TweetItem
from xui_reader.scheduler.merge import merge_tweet_items


ReadSourceFn = Callable[[SourceRef], tuple[TweetItem, ...]]


@dataclass(frozen=True)
class SourceReadOutcome:
    source_id: str
    source_kind: str
    ok: bool
    item_count: int
    error: str | None = None


@dataclass(frozen=True)
class MultiSourceReadResult:
    items: tuple[TweetItem, ...]
    outcomes: tuple[SourceReadOutcome, ...]

    @property
    def succeeded(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.ok)

    @property
    def failed(self) -> int:
        return sum(1 for outcome in self.outcomes if not outcome.ok)


def run_multi_source_read(
    sources: Iterable[SourceRef],
    read_source: ReadSourceFn,
) -> MultiSourceReadResult:
    """Read each source independently and merge all successful results deterministically."""
    successful_batches: list[tuple[TweetItem, ...]] = []
    outcomes: list[SourceReadOutcome] = []

    for source in sources:
        try:
            items = tuple(read_source(source))
            successful_batches.append(items)
            outcomes.append(
                SourceReadOutcome(
                    source_id=source.source_id,
                    source_kind=source.kind.value,
                    ok=True,
                    item_count=len(items),
                    error=None,
                )
            )
        except Exception as exc:
            outcomes.append(
                SourceReadOutcome(
                    source_id=source.source_id,
                    source_kind=source.kind.value,
                    ok=False,
                    item_count=0,
                    error=str(exc),
                )
            )

    merged = merge_tweet_items(successful_batches)
    return MultiSourceReadResult(items=merged, outcomes=tuple(outcomes))


def run_configured_read(
    config: RuntimeConfig,
    *,
    profile_name: str | None = None,
    config_path: str | Path | None = None,
    limit: int = 100,
    read_source: ReadSourceFn | None = None,
) -> MultiSourceReadResult:
    """Execute one multi-source read from enabled config sources."""
    enabled_sources = tuple(source for source in config.sources if source.enabled)
    if not enabled_sources:
        raise SchedulerError("No enabled sources configured. Add a [[sources]] entry with enabled = true.")
    if limit <= 0:
        raise SchedulerError("limit must be > 0.")

    if read_source is None:
        read_source = lambda source: collect_source_items(  # noqa: E731
            config,
            source,
            profile_name=profile_name,
            config_path=config_path,
            limit=limit,
        )
    return run_multi_source_read(enabled_sources, read_source)


def collect_source_items(
    config: RuntimeConfig,
    source: SourceRef,
    *,
    profile_name: str | None = None,
    config_path: str | Path | None = None,
    limit: int = 100,
) -> tuple[TweetItem, ...]:
    """Collect and extract one source using the configured browser/session pipeline."""
    if limit <= 0:
        raise SchedulerError("limit must be > 0.")

    selected_profile = profile_name or config.app.default_profile
    state_path = storage_state_path(selected_profile, config_path)
    if not state_path.exists():
        raise CollectError(
            f"Missing storage_state for profile '{selected_profile}' at '{state_path}'. "
            "Run `xui auth login` before read/watch/doctor smoke checks."
        )
    if not state_path.is_file():
        raise CollectError(f"storage_state path '{state_path}' is not a file.")

    extractor = PrimaryFallbackTweetExtractor()
    with PlaywrightBrowserSession(config, storage_state=state_path) as session:
        collector = TimelineCollector(config, session)
        batch = collector.collect(source, limit=limit)
        extracted = extractor.extract(
            {
                "source_id": source.source_id,
                "dom_snapshots": batch.dom_snapshots,
            }
        )
        if extracted:
            return extracted
        return batch.items


def run_source_smoke_check(
    config: RuntimeConfig,
    source: SourceRef,
    profile_name: str | None = None,
    config_path: str | Path | None = None,
    limit: int = 20,
) -> int:
    """Run one bounded smoke collection for a source and return emitted item count."""
    items = collect_source_items(
        config,
        source,
        profile_name=profile_name,
        config_path=config_path,
        limit=limit,
    )
    return len(items)
