"""Multi-source read orchestration helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from xui_reader.auth import storage_state_path
from xui_reader.browser.session import PlaywrightBrowserSession
from xui_reader.collectors.timeline import TimelineCollector
from xui_reader.config import RuntimeConfig
from xui_reader.diagnostics.artifacts import redact_text, write_html_artifact
from xui_reader.diagnostics.events import JsonlEventLogger
from xui_reader.errors import CollectError, SchedulerError
from xui_reader.extract.tweets import PrimaryFallbackTweetExtractor
from xui_reader.models import SourceRef, TweetItem
from xui_reader.profiles import profiles_root
from xui_reader.scheduler.merge import merge_tweet_items


ReadSourceFn = Callable[[SourceRef], "SourceReadResult | tuple[TweetItem, ...]"]
SourceFailureHookFn = Callable[[SourceRef, Exception], tuple[str | None, str | None]]


@dataclass(frozen=True)
class SourceReadResult:
    items: tuple[TweetItem, ...]
    page_loads: int = 0
    scroll_rounds: int = 0
    observed_ids: int = 0
    dom_snapshots: int = 0


@dataclass(frozen=True)
class SourceReadOutcome:
    source_id: str
    source_kind: str
    ok: bool
    item_count: int
    page_loads: int = 0
    scroll_rounds: int = 0
    observed_ids: int = 0
    error: str | None = None
    html_artifact_path: str | None = None
    selector_report_path: str | None = None


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

    @property
    def total_page_loads(self) -> int:
        return sum(outcome.page_loads for outcome in self.outcomes)

    @property
    def total_scroll_rounds(self) -> int:
        return sum(outcome.scroll_rounds for outcome in self.outcomes)

    @property
    def total_observed_ids(self) -> int:
        return sum(outcome.observed_ids for outcome in self.outcomes)


def run_multi_source_read(
    sources: Iterable[SourceRef],
    read_source: ReadSourceFn,
    *,
    on_source_failure: SourceFailureHookFn | None = None,
) -> MultiSourceReadResult:
    """Read each source independently and merge all successful results deterministically."""
    successful_batches: list[tuple[TweetItem, ...]] = []
    outcomes: list[SourceReadOutcome] = []

    for source in sources:
        try:
            read_result = _coerce_source_read_result(read_source(source))
            successful_batches.append(read_result.items)
            outcomes.append(
                SourceReadOutcome(
                    source_id=source.source_id,
                    source_kind=source.kind.value,
                    ok=True,
                    item_count=len(read_result.items),
                    page_loads=max(0, read_result.page_loads),
                    scroll_rounds=max(0, read_result.scroll_rounds),
                    observed_ids=max(0, read_result.observed_ids),
                    error=None,
                )
            )
        except Exception as exc:
            html_artifact_path = None
            selector_report_path = None
            if on_source_failure is not None:
                html_artifact_path, selector_report_path = on_source_failure(source, exc)
            outcomes.append(
                SourceReadOutcome(
                    source_id=source.source_id,
                    source_kind=source.kind.value,
                    ok=False,
                    item_count=0,
                    error=str(exc),
                    html_artifact_path=html_artifact_path,
                    selector_report_path=selector_report_path,
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
    run_id: str | None = None,
    enable_debug_artifacts: bool = False,
    raw_html_opt_in: bool | None = None,
    event_logger: JsonlEventLogger | None = None,
) -> MultiSourceReadResult:
    """Execute one multi-source read from enabled config sources."""
    enabled_sources = tuple(source for source in config.sources if source.enabled)
    if not enabled_sources:
        raise SchedulerError("No enabled sources configured. Add a [[sources]] entry with enabled = true.")
    if limit <= 0:
        raise SchedulerError("limit must be > 0.")

    selected_profile = profile_name or config.app.default_profile
    resolved_run_id = run_id or _new_run_id("read")
    artifacts_dir = _artifacts_dir_for_profile(selected_profile, config_path)

    if read_source is None:
        read_source = lambda source: collect_source_result(  # noqa: E731
            config,
            source,
            profile_name=selected_profile,
            config_path=config_path,
            limit=limit,
        )
    failure_hook: SourceFailureHookFn | None = None
    if enable_debug_artifacts:
        failure_hook = lambda source, exc: _write_source_failure_artifacts(  # noqa: E731
            artifacts_dir=artifacts_dir,
            run_id=resolved_run_id,
            source=source,
            error=exc,
            raw_html_opt_in=raw_html_opt_in,
        )
    result = run_multi_source_read(
        enabled_sources,
        read_source,
        on_source_failure=failure_hook,
    )
    if event_logger is not None:
        event_logger.append(
            "read_run",
            run_id=resolved_run_id,
            payload={
                "succeeded_sources": result.succeeded,
                "failed_sources": result.failed,
                "page_loads": result.total_page_loads,
                "scroll_rounds": result.total_scroll_rounds,
                "seen_items": result.total_observed_ids,
                "emitted_items": len(result.items),
                "source_outcomes": [
                    {
                        "source_id": outcome.source_id,
                        "source_kind": outcome.source_kind,
                        "ok": outcome.ok,
                        "item_count": outcome.item_count,
                        "page_loads": outcome.page_loads,
                        "scroll_rounds": outcome.scroll_rounds,
                        "observed_ids": outcome.observed_ids,
                        "error": outcome.error,
                    }
                    for outcome in result.outcomes
                ],
            },
        )
    return result


def collect_source_result(
    config: RuntimeConfig,
    source: SourceRef,
    *,
    profile_name: str | None = None,
    config_path: str | Path | None = None,
    limit: int = 100,
) -> SourceReadResult:
    """Collect and extract one source plus counters used by watch/logging budgets."""
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
        stats = batch.stats
        page_loads = 1
        scroll_rounds = stats.scroll_rounds if stats is not None else 0
        observed_ids = stats.observed_ids if stats is not None else len(batch.items)
        dom_snapshots = len(batch.dom_snapshots)
        try:
            extracted = extractor.extract(
                {
                    "source_id": source.source_id,
                    "dom_snapshots": batch.dom_snapshots,
                }
            )
        except Exception as exc:
            if batch.dom_snapshots:
                setattr(exc, "dom_snapshot", str(batch.dom_snapshots[-1]))
            raise
        if extracted:
            return SourceReadResult(
                items=tuple(extracted),
                page_loads=page_loads,
                scroll_rounds=scroll_rounds,
                observed_ids=observed_ids,
                dom_snapshots=dom_snapshots,
            )
        if batch.items:
            error = CollectError(
                f"Extractor produced no items for source '{source.source_id}' "
                f"from {len(batch.dom_snapshots)} DOM snapshot(s)."
            )
            if batch.dom_snapshots:
                setattr(error, "dom_snapshot", str(batch.dom_snapshots[-1]))
            raise error
        return SourceReadResult(
            items=(),
            page_loads=page_loads,
            scroll_rounds=scroll_rounds,
            observed_ids=observed_ids,
            dom_snapshots=dom_snapshots,
        )


def collect_source_items(
    config: RuntimeConfig,
    source: SourceRef,
    *,
    profile_name: str | None = None,
    config_path: str | Path | None = None,
    limit: int = 100,
) -> tuple[TweetItem, ...]:
    """Backward-compatible item-only source collection helper."""
    return collect_source_result(
        config,
        source,
        profile_name=profile_name,
        config_path=config_path,
        limit=limit,
    ).items


def run_source_smoke_check(
    config: RuntimeConfig,
    source: SourceRef,
    profile_name: str | None = None,
    config_path: str | Path | None = None,
    limit: int = 20,
) -> int:
    """Run one bounded smoke collection for a source and return emitted item count."""
    result = collect_source_result(
        config,
        source,
        profile_name=profile_name,
        config_path=config_path,
        limit=limit,
    )
    return len(result.items)


def _coerce_source_read_result(value: SourceReadResult | tuple[TweetItem, ...]) -> SourceReadResult:
    if isinstance(value, SourceReadResult):
        return value
    items = tuple(value)
    return SourceReadResult(items=items, observed_ids=len(items))


def _artifacts_dir_for_profile(profile_name: str, config_path: str | Path | None) -> Path:
    return profiles_root(config_path) / profile_name / "artifacts"


def _write_source_failure_artifacts(
    *,
    artifacts_dir: Path,
    run_id: str,
    source: SourceRef,
    error: Exception,
    raw_html_opt_in: bool | None,
) -> tuple[str | None, str | None]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    html = _extract_dom_snapshot(error)
    html_path = write_html_artifact(
        artifacts_dir,
        run_id=run_id,
        source_id=source.source_id,
        raw_html=html,
        raw_html_opt_in=raw_html_opt_in,
    )
    selector_report_path = artifacts_dir / f"{_slug(run_id)}_{_slug(source.source_id)}_selector-report.json"
    status_matches = len(re.findall(r"/status/(\d+)", html))
    selector_report = {
        "schema_version": "v1",
        "run_id": run_id,
        "source_id": source.source_id,
        "source_kind": source.kind.value,
        "error": redact_text(str(error)),
        "dom_snapshot_chars": len(html),
        "status_id_matches": status_matches,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    selector_report_path.write_text(
        json.dumps(selector_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(html_path), str(selector_report_path)


def _extract_dom_snapshot(error: Exception) -> str:
    candidate: Any = getattr(error, "dom_snapshot", "")
    if isinstance(candidate, str):
        return candidate
    return ""


def _new_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{stamp}"


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "unknown"
