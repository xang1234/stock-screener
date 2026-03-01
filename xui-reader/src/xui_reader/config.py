"""Shared configuration contracts and validation helpers for xui-reader."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigError
from .models import SourceKind, SourceRef

try:
    from platformdirs import user_config_dir
except ModuleNotFoundError:
    # Keep config importable in minimal environments without optional deps.
    def user_config_dir(appname: str, appauthor: bool = False) -> str:  # type: ignore[override]
        return str(Path.home() / ".config" / appname)


VALID_OUTPUT_FORMATS = {"pretty", "plain", "json", "jsonl"}
VALID_SOURCE_KINDS = {"list", "user"}
VALID_BROWSER_ENGINES = {"chromium", "firefox", "webkit"}
VALID_CHECKPOINT_MODES = {"id", "time"}
DEFAULT_CONFIG_FILENAME = "config.toml"

DEFAULT_CONFIG_TEMPLATE = """[app]
default_profile = "default"
timezone = "UTC"
default_format = "pretty"

[browser]
engine = "chromium"
headless = true
navigation_timeout_ms = 30000
action_timeout_ms = 10000
block_resources = true
viewport_width = 1280
viewport_height = 720
locale = "en-US"

[collection]
limit = 50
max_scrolls = 10
scroll_delay_ms = 1250
scroll_jitter_ms = 250
stagnation_rounds = 2
expand_truncated = false
include_reposts = true
include_pinned = false

[checkpoints]
mode = "id"
stop_early_on_old = true
old_streak_to_stop = 12

[scheduler]
interval_sec = 3600
jitter_pct = 0.07
shutdown_local = "01:00-10:00"
max_runs_per_day = 24
daily_budget_page_loads = 200
daily_budget_scrolls = 500

[storage]
db_filename = "tweets.sqlite3"
keep_days = 30
store_raw_html = false
store_raw_json = true

[selectors]
pack = "default"
override_filename = "selectors/override.json"

[[sources]]
id = "list:84839422"
kind = "list"
list_id = "84839422"
label = "Tech List"
enabled = true

[[sources]]
id = "user:somehandle"
kind = "user"
handle = "somehandle"
tab = "posts"
label = "@somehandle"
enabled = true
"""


@dataclass(frozen=True)
class AppConfig:
    default_profile: str = "default"
    timezone: str = "UTC"
    default_format: str = "pretty"
    debug: bool = False


@dataclass(frozen=True)
class BrowserConfig:
    engine: str = "chromium"
    headless: bool = True
    navigation_timeout_ms: int = 30_000
    action_timeout_ms: int = 10_000
    block_resources: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    locale: str = "en-US"


@dataclass(frozen=True)
class CollectionConfig:
    limit: int = 50
    max_scrolls: int = 10
    scroll_delay_ms: int = 1250
    scroll_jitter_ms: int = 250
    stagnation_rounds: int = 2
    expand_truncated: bool = False
    include_reposts: bool = True
    include_pinned: bool = False


@dataclass(frozen=True)
class CheckpointsConfig:
    mode: str = "id"
    stop_early_on_old: bool = True
    old_streak_to_stop: int = 12


@dataclass(frozen=True)
class SchedulerConfig:
    interval_sec: int = 3600
    jitter_pct: float = 0.07
    shutdown_local: str = "01:00-10:00"
    max_runs_per_day: int = 24
    daily_budget_page_loads: int = 200
    daily_budget_scrolls: int = 500


@dataclass(frozen=True)
class StorageConfig:
    db_filename: str = "tweets.sqlite3"
    keep_days: int = 30
    store_raw_html: bool = False
    store_raw_json: bool = True


@dataclass(frozen=True)
class SelectorsConfig:
    pack: str = "default"
    override_filename: str = "selectors/override.json"


@dataclass(frozen=True)
class RuntimeConfig:
    app: AppConfig = field(default_factory=AppConfig)
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    checkpoints: CheckpointsConfig = field(default_factory=CheckpointsConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    selectors: SelectorsConfig = field(default_factory=SelectorsConfig)
    sources: tuple[SourceRef, ...] = ()


def default_config() -> RuntimeConfig:
    return RuntimeConfig()


def default_config_toml() -> str:
    return DEFAULT_CONFIG_TEMPLATE


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path:
        return Path(config_path).expanduser()

    try:
        import os

        env_value = os.getenv("XUI_CONFIG")
    except Exception:
        env_value = None

    if env_value:
        return Path(env_value).expanduser()

    config_dir = Path(user_config_dir("xui-reader", appauthor=False))
    return config_dir / DEFAULT_CONFIG_FILENAME


def init_default_config(config_path: str | Path | None = None, force: bool = False) -> Path:
    path = resolve_config_path(config_path)
    if path.exists() and path.is_dir():
        raise ConfigError(
            f"Config path '{path}' is a directory; expected a TOML file path (for example '{path / DEFAULT_CONFIG_FILENAME}')."
        )
    if path.exists() and not force:
        raise ConfigError(
            f"Config file already exists at '{path}'. Re-run with --force to overwrite."
        )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(default_config_toml(), encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"Could not write config file at '{path}': {exc}. "
            "Check path permissions or choose a writable location with `--path`."
        ) from exc
    return path


def load_runtime_config(config_path: str | Path | None = None) -> RuntimeConfig:
    path = resolve_config_path(config_path)
    if not path.exists():
        raise ConfigError(
            f"Config file not found at '{path}'. Run `xui config init --path \"{path}\"` to generate defaults."
        )
    if path.is_dir():
        raise ConfigError(
            f"Config path '{path}' is a directory; pass a file path ending in '{DEFAULT_CONFIG_FILENAME}'."
        )

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"Could not read config file '{path}': {exc}. "
            "Check file permissions and that the path points to a readable TOML file."
        ) from exc
    raw = _load_toml(text, path)
    return _parse_runtime_config(raw)


def config_to_dict(config: RuntimeConfig) -> dict[str, Any]:
    return asdict(config)


def _load_toml(text: str, path: Path) -> dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:
            raise ConfigError(
                "TOML parsing requires Python 3.11+ or `tomli` installed. "
                f"Could not parse config file '{path}'."
            ) from exc

    try:
        data = tomllib.loads(text)
    except Exception as exc:
        raise ConfigError(
            f"Config file '{path}' contains invalid TOML: {exc}. "
            "Fix the syntax or regenerate defaults with `xui config init --force`."
        ) from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Config file '{path}' must parse to a TOML table.")
    return data


def _parse_runtime_config(data: dict[str, Any]) -> RuntimeConfig:
    app_raw = _expect_table(data, "app", default={})
    browser_raw = _expect_table(data, "browser", default={})
    collection_raw = _expect_table(data, "collection", default={})
    checkpoints_raw = _expect_table(data, "checkpoints", default={})
    scheduler_raw = _expect_table(data, "scheduler", default={})
    storage_raw = _expect_table(data, "storage", default={})
    selectors_raw = _expect_table(data, "selectors", default={})
    sources_raw = data.get("sources", [])

    if not isinstance(sources_raw, list):
        raise ConfigError("Invalid [sources]: expected an array of tables (`[[sources]]`).")

    app_config = AppConfig(
        default_profile=_expect_non_empty_string(app_raw, "app.default_profile", "default"),
        timezone=_expect_non_empty_string(app_raw, "app.timezone", "UTC"),
        default_format=_expect_choice(
            app_raw,
            "app.default_format",
            default="pretty",
            valid_values=VALID_OUTPUT_FORMATS,
        ),
        debug=_expect_bool(app_raw, "app.debug", default=False),
    )

    browser_config = BrowserConfig(
        engine=_expect_choice(
            browser_raw,
            "browser.engine",
            default="chromium",
            valid_values=VALID_BROWSER_ENGINES,
        ),
        headless=_expect_bool(browser_raw, "browser.headless", default=True),
        navigation_timeout_ms=_expect_positive_int(
            browser_raw, "browser.navigation_timeout_ms", default=30_000
        ),
        action_timeout_ms=_expect_positive_int(browser_raw, "browser.action_timeout_ms", default=10_000),
        block_resources=_expect_bool(browser_raw, "browser.block_resources", default=True),
        viewport_width=_expect_positive_int(browser_raw, "browser.viewport_width", default=1280),
        viewport_height=_expect_positive_int(browser_raw, "browser.viewport_height", default=720),
        locale=_expect_non_empty_string(browser_raw, "browser.locale", "en-US"),
    )

    collection_config = CollectionConfig(
        limit=_expect_positive_int(collection_raw, "collection.limit", default=50),
        max_scrolls=_expect_non_negative_int(collection_raw, "collection.max_scrolls", default=10),
        scroll_delay_ms=_expect_non_negative_int(
            collection_raw,
            "collection.scroll_delay_ms",
            default=1250,
        ),
        scroll_jitter_ms=_expect_non_negative_int(
            collection_raw,
            "collection.scroll_jitter_ms",
            default=250,
        ),
        stagnation_rounds=_expect_positive_int(
            collection_raw,
            "collection.stagnation_rounds",
            default=2,
        ),
        expand_truncated=_expect_bool(
            collection_raw,
            "collection.expand_truncated",
            default=False,
        ),
        include_reposts=_expect_bool(collection_raw, "collection.include_reposts", default=True),
        include_pinned=_expect_bool(collection_raw, "collection.include_pinned", default=False),
    )

    checkpoints_config = CheckpointsConfig(
        mode=_expect_choice(
            checkpoints_raw,
            "checkpoints.mode",
            default="id",
            valid_values=VALID_CHECKPOINT_MODES,
        ),
        stop_early_on_old=_expect_bool(
            checkpoints_raw,
            "checkpoints.stop_early_on_old",
            default=True,
        ),
        old_streak_to_stop=_expect_positive_int(
            checkpoints_raw,
            "checkpoints.old_streak_to_stop",
            default=12,
        ),
    )

    scheduler_config = SchedulerConfig(
        interval_sec=_expect_positive_int(scheduler_raw, "scheduler.interval_sec", default=3600),
        jitter_pct=_expect_float_range(
            scheduler_raw,
            "scheduler.jitter_pct",
            default=0.07,
            min_value=0.0,
            max_value=1.0,
        ),
        shutdown_local=_expect_non_empty_string(
            scheduler_raw,
            "scheduler.shutdown_local",
            "01:00-10:00",
        ),
        max_runs_per_day=_expect_positive_int(
            scheduler_raw,
            "scheduler.max_runs_per_day",
            default=24,
        ),
        daily_budget_page_loads=_expect_positive_int(
            scheduler_raw,
            "scheduler.daily_budget_page_loads",
            default=200,
        ),
        daily_budget_scrolls=_expect_positive_int(
            scheduler_raw,
            "scheduler.daily_budget_scrolls",
            default=500,
        ),
    )

    storage_config = StorageConfig(
        db_filename=_expect_non_empty_string(
            storage_raw,
            "storage.db_filename",
            "tweets.sqlite3",
        ),
        keep_days=_expect_non_negative_int(storage_raw, "storage.keep_days", default=30),
        store_raw_html=_expect_bool(storage_raw, "storage.store_raw_html", default=False),
        store_raw_json=_expect_bool(storage_raw, "storage.store_raw_json", default=True),
    )

    selectors_config = SelectorsConfig(
        pack=_expect_non_empty_string(selectors_raw, "selectors.pack", "default"),
        override_filename=_expect_non_empty_string(
            selectors_raw,
            "selectors.override_filename",
            "selectors/override.json",
        ),
    )

    parsed_sources: list[SourceRef] = []
    for index, source in enumerate(sources_raw):
        if not isinstance(source, dict):
            raise ConfigError(f"sources[{index}] must be a table, got {type(source).__name__}.")

        source_kind = _expect_choice(
            source,
            f"sources[{index}].kind",
            default=None,
            valid_values=VALID_SOURCE_KINDS,
        )
        source_id = _expect_non_empty_string(source, f"sources[{index}].id", default=None)
        enabled = _expect_bool(source, f"sources[{index}].enabled", default=True)
        label = _optional_non_empty_string(source, f"sources[{index}].label")

        if source_kind == "list":
            value = _expect_non_empty_string(source, f"sources[{index}].list_id", default=None)
            tab = None
        else:
            value = _expect_non_empty_string(source, f"sources[{index}].handle", default=None)
            tab = _optional_non_empty_string(source, f"sources[{index}].tab")

        parsed_sources.append(
            SourceRef(
                source_id=source_id,
                kind=SourceKind(source_kind),
                value=value,
                enabled=enabled,
                label=label,
                tab=tab,
            )
        )

    return RuntimeConfig(
        app=app_config,
        browser=browser_config,
        collection=collection_config,
        checkpoints=checkpoints_config,
        scheduler=scheduler_config,
        storage=storage_config,
        selectors=selectors_config,
        sources=tuple(parsed_sources),
    )


def _expect_table(data: dict[str, Any], key: str, default: dict[str, Any]) -> dict[str, Any]:
    value = data.get(key, default)
    if not isinstance(value, dict):
        raise ConfigError(f"Invalid [{key}] table: expected table, got {type(value).__name__}.")
    return value


def _expect_non_empty_string(
    data: dict[str, Any],
    key: str,
    default: str | None,
) -> str:
    field = key.split(".")[-1]
    if field in data:
        value = data[field]
    else:
        if default is None:
            raise ConfigError(f"Missing required value '{key}'.")
        value = default

    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Invalid value for '{key}': expected non-empty string.")
    return value


def _optional_non_empty_string(data: dict[str, Any], key: str) -> str | None:
    field = key.split(".")[-1]
    if field not in data:
        return None
    value = data[field]
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Invalid value for '{key}': expected non-empty string.")
    return value


def _expect_positive_int(data: dict[str, Any], key: str, default: int) -> int:
    field = key.split(".")[-1]
    value = data.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"Invalid value for '{key}': expected positive integer.")
    return value


def _expect_non_negative_int(data: dict[str, Any], key: str, default: int) -> int:
    field = key.split(".")[-1]
    value = data.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"Invalid value for '{key}': expected non-negative integer.")
    return value


def _expect_float_range(
    data: dict[str, Any],
    key: str,
    default: float,
    *,
    min_value: float,
    max_value: float,
) -> float:
    field = key.split(".")[-1]
    raw = data.get(field, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ConfigError(f"Invalid value for '{key}': expected number.")
    value = float(raw)
    if value < min_value or value > max_value:
        raise ConfigError(
            f"Invalid value for '{key}': expected number between {min_value} and {max_value}."
        )
    return value


def _expect_bool(data: dict[str, Any], key: str, default: bool) -> bool:
    field = key.split(".")[-1]
    value = data.get(field, default)
    if not isinstance(value, bool):
        raise ConfigError(f"Invalid value for '{key}': expected boolean true/false.")
    return value


def _expect_choice(
    data: dict[str, Any],
    key: str,
    default: str | None,
    valid_values: set[str],
) -> str:
    field = key.split(".")[-1]
    if field in data:
        value = data[field]
    else:
        if default is None:
            raise ConfigError(f"Missing required value '{key}'.")
        value = default

    if not isinstance(value, str) or value not in valid_values:
        choices = ", ".join(sorted(valid_values))
        raise ConfigError(f"Invalid value for '{key}': expected one of [{choices}].")
    return value
