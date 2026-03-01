"""Twitter source ingestion via xui-reader (X web UI)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib
import logging
from pathlib import Path
import sys
from typing import Any, Callable
from urllib.parse import urlparse, urlunparse

from ..config import settings
from ..models.theme import ContentSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _XUIBindings:
    load_runtime_config: Callable[..., Any]
    run_configured_read: Callable[..., Any]
    parse_list_id: Callable[[str], str]
    parse_handle: Callable[[str], str]
    probe_auth_status: Callable[..., Any]
    source_ref_cls: type
    source_kind_enum: Any


def _repo_root_from_here() -> Path | None:
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        candidate = parent / "xui-reader" / "src"
        if candidate.exists():
            return parent
    return None


def _ensure_xui_module_path() -> None:
    repo_root = _repo_root_from_here()
    if repo_root is None:
        return
    xui_src = repo_root / "xui-reader" / "src"
    xui_src_text = str(xui_src)
    if xui_src.exists() and xui_src_text not in sys.path:
        sys.path.insert(0, xui_src_text)


def _load_xui_bindings() -> _XUIBindings:
    try:
        return _import_xui_bindings()
    except ModuleNotFoundError:
        _ensure_xui_module_path()
        try:
            return _import_xui_bindings()
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "xui-reader is not available. Install with `pip install -e ./xui-reader` "
                "and ensure Playwright Chromium is installed (`python -m playwright install chromium`)."
            ) from exc


def _import_xui_bindings() -> _XUIBindings:
    config_mod = importlib.import_module("xui_reader.config")
    scheduler_mod = importlib.import_module("xui_reader.scheduler.read")
    collectors_mod = importlib.import_module("xui_reader.collectors.timeline")
    auth_mod = importlib.import_module("xui_reader.auth")
    models_mod = importlib.import_module("xui_reader.models")
    return _XUIBindings(
        load_runtime_config=getattr(config_mod, "load_runtime_config"),
        run_configured_read=getattr(scheduler_mod, "run_configured_read"),
        parse_list_id=getattr(collectors_mod, "parse_list_id"),
        parse_handle=getattr(collectors_mod, "parse_handle"),
        probe_auth_status=getattr(auth_mod, "probe_auth_status"),
        source_ref_cls=getattr(models_mod, "SourceRef"),
        source_kind_enum=getattr(models_mod, "SourceKind"),
    )


class XUITwitterFetcher:
    """Fetch twitter content through xui-reader UI collection."""

    provider_name = "xui"

    def fetch(self, source: ContentSource, since: datetime | None = None) -> list[dict[str, Any]]:
        if not settings.xui_enabled:
            raise RuntimeError("xui twitter ingestion is disabled (XUI_ENABLED=false).")

        bindings = _load_xui_bindings()
        locator = _normalized_locator(source)
        source_ref, source_kind, fallback_author = _to_source_ref(bindings, locator, source)

        auth_status = bindings.probe_auth_status(
            profile_name=settings.xui_profile,
            config_path=settings.xui_config_path,
        )
        if not auth_status.authenticated:
            login_cmd = (
                f"xui auth login --profile {settings.xui_profile} --path {settings.xui_config_path}"
            )
            raise RuntimeError(
                f"XUI auth not ready ({auth_status.status_code}): {auth_status.message} "
                f"Next step: run `{login_cmd}`."
            )

        runtime_config = bindings.load_runtime_config(settings.xui_config_path)
        single_source_config = _with_single_source(runtime_config, source_ref)
        result = bindings.run_configured_read(
            single_source_config,
            profile_name=settings.xui_profile,
            config_path=settings.xui_config_path,
            limit=settings.xui_limit_per_source,
            new_only=settings.xui_new_only,
            checkpoint_mode=settings.xui_checkpoint_mode,
        )

        if result.failed > 0:
            failed = next((outcome for outcome in result.outcomes if not outcome.ok), None)
            details = failed.error if failed and failed.error else "unknown read failure"
            raise RuntimeError(
                f"XUI read failed for source '{source.name}' ({source_kind}): {details}"
            )

        since_bound = _normalize_datetime(since)
        records: list[dict[str, Any]] = []
        for item in result.items:
            published_at = _normalize_datetime(item.created_at)
            if since_bound is not None and published_at is not None and published_at < since_bound:
                continue
            record = {
                "external_id": hashlib.md5(f"twitter:{item.tweet_id}".encode("utf-8")).hexdigest(),
                "title": "",
                "content": item.text or "",
                "url": _tweet_url(item.tweet_id, item.author_handle, fallback_author),
                "author": item.author_handle or source.name,
                "published_at": published_at,
            }
            records.append(record)

        logger.info(
            "twitter_fetch provider=%s source_kind=%s source=%s auth_status=%s items_fetched=%d",
            self.provider_name,
            source_kind,
            source.name,
            auth_status.status_code,
            len(records),
        )
        return records


def _normalized_locator(source: ContentSource) -> str:
    raw = (source.url or source.name or "").strip()
    if not raw:
        raise RuntimeError(f"Twitter source '{source.name}' is missing url/identifier.")
    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    if host in {"twitter.com", "www.twitter.com", "mobile.twitter.com"}:
        parsed = parsed._replace(netloc="x.com")
        raw = urlunparse(parsed)
    return raw


def _to_source_ref(bindings: _XUIBindings, locator: str, source: ContentSource) -> tuple[Any, str, str | None]:
    try:
        list_id = bindings.parse_list_id(locator)
        return (
            bindings.source_ref_cls(
                source_id=f"list:{list_id}",
                kind=bindings.source_kind_enum.LIST,
                value=list_id,
                enabled=True,
                label=source.name,
            ),
            "list",
            None,
        )
    except Exception:
        pass

    try:
        handle = bindings.parse_handle(locator)
        return (
            bindings.source_ref_cls(
                source_id=f"user:{handle}",
                kind=bindings.source_kind_enum.USER,
                value=handle,
                enabled=True,
                label=source.name,
            ),
            "user",
            handle,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to parse twitter source locator '{locator}' for source '{source.name}'. "
            "Expected @handle, x.com/<handle>, or x.com/i/lists/<id>."
        ) from exc


def _tweet_url(tweet_id: str, author_handle: str | None, fallback_author: str | None) -> str:
    author = (author_handle or fallback_author or "").strip().lstrip("@")
    if author:
        return f"https://x.com/{author}/status/{tweet_id}"
    return f"https://x.com/i/web/status/{tweet_id}"


def _normalize_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _with_single_source(runtime_config: Any, source_ref: Any) -> Any:
    try:
        return replace(runtime_config, sources=(source_ref,))
    except TypeError:
        if hasattr(runtime_config, "sources"):
            runtime_config.sources = (source_ref,)
            return runtime_config
        raise RuntimeError("xui runtime config object is missing 'sources' field.")
