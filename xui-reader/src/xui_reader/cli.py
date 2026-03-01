"""Typer CLI for xui-reader workflows."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shlex

import typer

from . import __version__
from .auth import (
    DEFAULT_LOGIN_URL,
    auth_logout_to_dict,
    auth_status_to_dict,
    login_and_save_storage_state,
    logout_profile,
    probe_auth_status,
)
from .collectors.timeline import parse_handle, parse_list_id
from .config import (
    VALID_OUTPUT_FORMATS,
    RuntimeConfig,
    config_to_dict,
    init_default_config,
    load_runtime_config,
    resolve_config_path,
)
from .diagnostics.artifacts import redact_value
from .diagnostics.doctor import run_doctor_preflight
from .diagnostics.events import JsonlEventLogger
from .errors import (
    AuthError,
    CollectError,
    ConfigError,
    DiagnosticsError,
    ProfileError,
    RenderError,
    SchedulerError,
    StoreError,
)
from .models import SourceKind, SourceRef, TweetItem
from .profiles import create_profile, delete_profile, list_profiles, profiles_root, switch_profile
from .render import render_items
from .render.jsonout import tweet_item_to_dict
from .scheduler.read import MultiSourceReadResult, run_configured_read
from .scheduler.watch import (
    WatchExitCode,
    WatchRunResult,
    determine_watch_exit_code,
    run_configured_watch,
)

app = typer.Typer(help="Read-only X UI timeline/list reader.")

auth_app = typer.Typer(help="Authentication commands.")
profiles_app = typer.Typer(help="Profile management commands.")
list_app = typer.Typer(help="List helpers.")
user_app = typer.Typer(help="User helpers.")
config_app = typer.Typer(help="Config commands.")

app.add_typer(auth_app, name="auth")
app.add_typer(profiles_app, name="profiles")
app.add_typer(list_app, name="list")
app.add_typer(user_app, name="user")
app.add_typer(config_app, name="config")


@auth_app.command("login")
def auth_login(
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Profile to store session state in (defaults to configured app.default_profile).",
    ),
    path: str | None = typer.Option(
        None, "--path", help="Optional config TOML path (defaults to platform config dir)."
    ),
    login_url: str = typer.Option(
        DEFAULT_LOGIN_URL,
        "--login-url",
        help="Login URL to open before capturing Playwright storage_state.",
    ),
) -> None:
    try:
        storage_path = login_and_save_storage_state(
            profile_name=profile,
            config_path=path,
            login_url=login_url,
        )
    except (ConfigError, AuthError) as exc:
        typer.secho(f"Auth login failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    typer.echo(f"Saved storage_state to {storage_path}")


@auth_app.command("status")
def auth_status(
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Profile to probe (defaults to configured app.default_profile).",
    ),
    path: str | None = typer.Option(
        None, "--path", help="Optional config TOML path (defaults to platform config dir)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Render auth status result as JSON."),
) -> None:
    try:
        result = probe_auth_status(profile_name=profile, config_path=path)
    except (ConfigError, AuthError) as exc:
        typer.secho(f"Auth status failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    if as_json:
        typer.echo(json.dumps(auth_status_to_dict(result), indent=2, sort_keys=True))
    else:
        typer.echo(f"Profile: {result.profile}")
        typer.echo(f"Storage state: {result.storage_state_path}")
        typer.echo(f"Status: {result.status_code}")
        typer.echo(result.message)
        for step in result.next_steps:
            typer.echo(f"- {step}")

    if result.authenticated:
        return
    raise typer.Exit(2)


@auth_app.command("logout")
def auth_logout(
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Profile to log out (defaults to configured app.default_profile).",
    ),
    path: str | None = typer.Option(
        None, "--path", help="Optional config TOML path (defaults to platform config dir)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Render logout result as JSON."),
) -> None:
    try:
        result = logout_profile(profile_name=profile, config_path=path)
    except (ConfigError, AuthError) as exc:
        typer.secho(f"Auth logout failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    if as_json:
        typer.echo(json.dumps(auth_logout_to_dict(result), indent=2, sort_keys=True))
        return

    typer.echo(result.message)
    typer.echo(f"Storage state: {result.storage_state_path}")
    for step in result.next_steps:
        typer.echo(f"- Re-login: `{step}`")


@profiles_app.command("list")
def profiles_list_cmd(
    path: str | None = typer.Option(
        None, "--path", help="Optional config TOML path (defaults to platform config dir)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Render profiles as JSON."),
) -> None:
    try:
        profiles, active_name = list_profiles(path)
    except (ConfigError, ProfileError) as exc:
        typer.secho(f"Profiles list failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    payload = {
        "active_profile": active_name,
        "profiles": [{"name": profile.name, "active": profile.active} for profile in profiles],
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    if not profiles:
        typer.echo("No profiles found.")
    else:
        for profile in profiles:
            marker = "*" if profile.active else " "
            typer.echo(f"{marker} {profile.name}")

    if any(profile.active for profile in profiles):
        typer.echo(f"Active profile: {active_name}")
        return
    typer.echo(
        f"Configured active profile '{active_name}' has no directory yet. "
        f"Run `xui profiles create {active_name}` to bootstrap it."
    )


@profiles_app.command("create")
def profiles_create_cmd(
    name: str,
    path: str | None = typer.Option(
        None, "--path", help="Optional config TOML path (defaults to platform config dir)."
    ),
    switch: bool = typer.Option(False, "--switch", help="Switch to the profile after creation."),
    force: bool = typer.Option(False, "--force", help="Re-bootstrap if profile already exists."),
) -> None:
    try:
        profile_path = create_profile(name, path, switch=switch, force=force)
    except (ConfigError, ProfileError) as exc:
        typer.secho(f"Profiles create failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    typer.echo(f"Profile '{name}' ready at {profile_path}")
    if switch:
        typer.echo(f"Active profile set to '{name}'.")


@profiles_app.command("delete")
def profiles_delete_cmd(
    name: str,
    path: str | None = typer.Option(
        None, "--path", help="Optional config TOML path (defaults to platform config dir)."
    ),
) -> None:
    try:
        deleted_path = delete_profile(name, path)
    except (ConfigError, ProfileError) as exc:
        typer.secho(f"Profiles delete failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc
    typer.echo(f"Deleted profile '{name}' at {deleted_path}")


@profiles_app.command("switch")
def profiles_switch_cmd(
    name: str,
    path: str | None = typer.Option(
        None, "--path", help="Optional config TOML path (defaults to platform config dir)."
    ),
    create: bool = typer.Option(
        False, "--create", help="Create the profile directory if it does not exist."
    ),
) -> None:
    try:
        switched_path = switch_profile(name, path, create_missing=create)
    except (ConfigError, ProfileError) as exc:
        typer.secho(f"Profiles switch failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc
    typer.echo(f"Active profile set to '{name}' ({switched_path})")


@list_app.command("read")
def list_read(
    ctx: typer.Context,
    list_id_or_url: str,
    path: str | None = typer.Option(
        None, "--path", help="Optional config TOML path (defaults to platform config dir)."
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Profile to use for storage_state (defaults to configured app.default_profile).",
    ),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Collection limit."),
    new: bool = typer.Option(False, "--new", help="Emit only unseen items for this source."),
    checkpoint_mode: str = typer.Option(
        "auto",
        "--checkpoint-mode",
        help="Checkpoint mode: auto|id|time.",
    ),
) -> None:
    try:
        config = load_runtime_config(path)
        selected_profile = _resolve_profile(profile, ctx)
        effective_limit = limit if limit is not None else config.collection.limit
        list_id = parse_list_id(list_id_or_url)
        source = SourceRef(
            source_id=f"list:{list_id}",
            kind=SourceKind.LIST,
            value=list_id,
            enabled=True,
            label=f"list:{list_id}",
        )
        result = _run_single_source_read(
            config=config,
            source=source,
            profile_name=selected_profile,
            config_path=path,
            limit=effective_limit,
            new_only=new,
            checkpoint_mode=checkpoint_mode,
            debug_enabled=_resolve_debug(ctx),
        )
    except (ConfigError, AuthError, CollectError, SchedulerError, StoreError, RenderError) as exc:
        typer.secho(f"List read failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    _emit_read_items(result.items, ctx=ctx, config=config)
    if result.failed == len(result.outcomes):
        for outcome in result.outcomes:
            if outcome.error:
                typer.secho(
                    f"Source '{outcome.source_id}' failed: {outcome.error}",
                    err=True,
                    fg=typer.colors.RED,
                )
        if _all_source_failures_are_auth_related(result):
            resolved_profile = selected_profile or config.app.default_profile
            typer.echo(
                f"Next step: `{_auth_login_command_hint(resolved_profile, path)}` "
                "then re-run `xui list read`."
            )
        raise typer.Exit(2)


@list_app.command("parse-id")
def list_parse_id(list_url: str) -> None:
    try:
        typer.echo(parse_list_id(list_url))
    except CollectError as exc:
        typer.secho(f"List parse-id failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc


@user_app.command("read")
def user_read(
    ctx: typer.Context,
    handle_or_url: str,
    path: str | None = typer.Option(
        None, "--path", help="Optional config TOML path (defaults to platform config dir)."
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Profile to use for storage_state (defaults to configured app.default_profile).",
    ),
    tab: str = typer.Option("posts", "--tab", help="Timeline tab: posts|replies|media."),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Collection limit."),
    new: bool = typer.Option(False, "--new", help="Emit only unseen items for this source."),
    checkpoint_mode: str = typer.Option(
        "auto",
        "--checkpoint-mode",
        help="Checkpoint mode: auto|id|time.",
    ),
) -> None:
    try:
        config = load_runtime_config(path)
        selected_profile = _resolve_profile(profile, ctx)
        effective_limit = limit if limit is not None else config.collection.limit
        handle = parse_handle(handle_or_url)
        source = SourceRef(
            source_id=f"user:{handle}",
            kind=SourceKind.USER,
            value=handle,
            enabled=True,
            label=f"@{handle}",
            tab=tab,
        )
        result = _run_single_source_read(
            config=config,
            source=source,
            profile_name=selected_profile,
            config_path=path,
            limit=effective_limit,
            new_only=new,
            checkpoint_mode=checkpoint_mode,
            debug_enabled=_resolve_debug(ctx),
        )
    except (ConfigError, AuthError, CollectError, SchedulerError, StoreError, RenderError) as exc:
        typer.secho(f"User read failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    _emit_read_items(result.items, ctx=ctx, config=config)
    if result.failed == len(result.outcomes):
        for outcome in result.outcomes:
            if outcome.error:
                typer.secho(
                    f"Source '{outcome.source_id}' failed: {outcome.error}",
                    err=True,
                    fg=typer.colors.RED,
                )
        if _all_source_failures_are_auth_related(result):
            resolved_profile = selected_profile or config.app.default_profile
            typer.echo(
                f"Next step: `{_auth_login_command_hint(resolved_profile, path)}` "
                "then re-run `xui user read`."
            )
        raise typer.Exit(2)


@user_app.command("parse-handle")
def user_parse_handle(user_url: str) -> None:
    try:
        typer.echo(parse_handle(user_url))
    except CollectError as exc:
        typer.secho(f"User parse-handle failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc


@config_app.command("init")
def config_init(
    path: str | None = typer.Option(
        None, "--path", help="Optional config TOML path (defaults to platform config dir)."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing config file."),
) -> None:
    try:
        written_path = init_default_config(path, force=force)
    except ConfigError as exc:
        typer.secho(f"Config init failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    typer.echo(f"Wrote default config to {written_path}")


@config_app.command("show")
def config_show(
    path: str | None = typer.Option(
        None, "--path", help="Optional config TOML path (defaults to platform config dir)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Render resolved config as JSON."),
) -> None:
    resolved_path = resolve_config_path(path)
    try:
        config = load_runtime_config(path)
    except ConfigError as exc:
        typer.secho(f"Config show failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    payload = {
        "path": str(resolved_path),
        "config": config_to_dict(config),
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    typer.echo(f"Resolved config path: {payload['path']}")
    typer.echo(f"Default profile: {config.app.default_profile}")
    typer.echo(f"Default format: {config.app.default_format}")
    typer.echo(f"Sources: {len(config.sources)}")


@app.command("read")
def read(
    ctx: typer.Context,
    path: str | None = typer.Option(
        None, "--path", help="Optional config TOML path (defaults to platform config dir)."
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Profile to use for storage_state (defaults to configured app.default_profile).",
    ),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Per-source collection limit."),
    sources: str | None = typer.Option(
        None,
        "--sources",
        help="Override config sources, e.g. list:84839422,user:somehandle",
    ),
    new: bool = typer.Option(False, "--new", help="Emit only unseen items per source."),
    checkpoint_mode: str = typer.Option(
        "auto",
        "--checkpoint-mode",
        help="Checkpoint mode: auto|id|time.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Render read result as JSON summary."),
) -> None:
    try:
        config = load_runtime_config(path)
        selected_profile = _resolve_profile(profile, ctx)
        resolved_profile = selected_profile or config.app.default_profile
        effective_limit = limit if limit is not None else config.collection.limit
        effective_config = _config_with_optional_sources(config, sources)
        debug_enabled = _resolve_debug(ctx)
        event_logger = _event_logger_for_profile(resolved_profile, path) if debug_enabled else None
        result = run_configured_read(
            effective_config,
            profile_name=selected_profile,
            config_path=path,
            limit=effective_limit,
            new_only=new,
            checkpoint_mode=checkpoint_mode,
            enable_debug_artifacts=debug_enabled,
            event_logger=event_logger,
        )
    except (ConfigError, AuthError, CollectError, SchedulerError, StoreError, RenderError) as exc:
        typer.secho(f"Read failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    payload = _read_result_payload(result)
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        output_format = _resolve_output_format(ctx, config_default=effective_config.app.default_format)
        if output_format == "pretty":
            typer.echo(
                f"Read summary: {payload['succeeded_sources']} succeeded, "
                f"{payload['failed_sources']} failed, {len(result.items)} merged items."
            )
            typer.echo("Source outcomes:")
            for outcome in payload["outcomes"]:
                status = "ok" if outcome["ok"] else "failed"
                suffix = f" ({outcome['error']})" if outcome["error"] else ""
                typer.echo(f"- {outcome['source_id']} [{status}] items={outcome['item_count']}{suffix}")
        rendered = render_items(result.items, output_format)
        if rendered:
            typer.echo(rendered)
        if _all_source_failures_are_auth_related(result):
            typer.echo(
                f"Next step: `{_auth_login_command_hint(resolved_profile, path)}` "
                "then re-run `xui read`."
            )

    if result.failed == len(result.outcomes):
        raise typer.Exit(2)


@app.command("watch")
def watch(
    ctx: typer.Context,
    path: str | None = typer.Option(
        None, "--path", help="Optional config TOML path (defaults to platform config dir)."
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Profile to use for storage_state (defaults to configured app.default_profile).",
    ),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Per-source collection limit."),
    interval_seconds: int | None = typer.Option(
        None, "--interval-seconds", min=1, help="Watch interval seconds."
    ),
    jitter_ratio: float | None = typer.Option(
        None,
        "--jitter-ratio",
        min=0.0,
        max=1.0,
        help="Randomized +/- ratio applied to interval.",
    ),
    shutdown_window: str | None = typer.Option(
        None,
        "--shutdown-window",
        help="Optional local shutdown window in HH:MM-HH:MM format.",
    ),
    max_cycles: int | None = typer.Option(None, "--max-cycles", min=1, help="Number of watch cycles."),
    max_page_loads: int | None = typer.Option(
        None,
        "--max-page-loads",
        min=1,
        help="Optional page-load budget across the watch run.",
    ),
    max_scroll_rounds: int | None = typer.Option(
        None,
        "--max-scroll-rounds",
        min=1,
        help="Optional scroll-round budget across the watch run.",
    ),
    new: bool = typer.Option(False, "--new", help="Emit only unseen items per source."),
    checkpoint_mode: str = typer.Option(
        "auto",
        "--checkpoint-mode",
        help="Checkpoint mode: auto|id|time.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Render watch result as JSON."),
) -> None:
    try:
        config = load_runtime_config(path)
        selected_profile = _resolve_profile(profile, ctx)
        resolved_profile = selected_profile or config.app.default_profile
        effective_limit = limit if limit is not None else config.collection.limit
        debug_enabled = _resolve_debug(ctx)
        event_logger = _event_logger_for_profile(resolved_profile, path) if debug_enabled else None
        result = run_configured_watch(
            config,
            profile_name=selected_profile,
            config_path=path,
            limit=effective_limit,
            interval_seconds=interval_seconds,
            jitter_ratio=jitter_ratio,
            shutdown_window=shutdown_window,
            max_cycles=max_cycles,
            max_page_loads=max_page_loads,
            max_scroll_rounds=max_scroll_rounds,
            new_only=new,
            checkpoint_mode=checkpoint_mode,
            enable_debug_artifacts=debug_enabled,
            event_logger=event_logger,
        )
    except (ConfigError, AuthError, CollectError, SchedulerError, StoreError) as exc:
        typer.secho(f"Watch failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    effective_max_cycles = max_cycles if max_cycles is not None else config.scheduler.max_runs_per_day
    exit_code = determine_watch_exit_code(result, max_cycles=effective_max_cycles)
    payload = _watch_result_payload(result, exit_code=exit_code)
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Watch completed: {len(result.cycles)} cycle(s), "
            f"exit_state={_watch_exit_state_name(exit_code)} exit_code={int(exit_code)}."
        )
        for cycle in result.cycles:
            next_run = cycle.next_run_at.isoformat() if cycle.next_run_at else "none"
            typer.echo(
                f"- cycle={cycle.cycle} emitted={cycle.emitted_items} "
                f"sources_ok={cycle.succeeded_sources} sources_failed={cycle.failed_sources} "
                f"page_loads={cycle.page_loads} scroll_rounds={cycle.scroll_rounds} "
                f"sleep={cycle.sleep_seconds:.2f}s next={next_run}"
            )
        if exit_code is WatchExitCode.AUTH_FAIL:
            typer.echo(
                f"Next step: `{_auth_login_command_hint(resolved_profile, path)}` "
                "then re-run `xui watch`."
            )
    if exit_code is not WatchExitCode.SUCCESS:
        raise typer.Exit(int(exit_code))


@app.command("doctor")
def doctor(
    ctx: typer.Context,
    path: str | None = typer.Option(
        None, "--path", help="Optional config TOML path (defaults to platform config dir)."
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Profile to use for auth/smoke checks (defaults to configured app.default_profile).",
    ),
    max_sources: int = typer.Option(
        2,
        "--max-sources",
        min=1,
        help="Maximum number of configured enabled sources to use for optional smoke checks.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Render doctor preflight report as JSON."),
) -> None:
    try:
        config = load_runtime_config(path)
        selected_profile = _resolve_profile(profile, ctx)
        report = run_doctor_preflight(
            config,
            profile_name=selected_profile,
            config_path=path,
            max_sources=max_sources,
        )
    except (ConfigError, DiagnosticsError) as exc:
        typer.secho(f"Doctor failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    redacted_report_details = _redact_details(report.details)
    redacted_sections = [
        {
            "name": section.name,
            "ok": section.ok,
            "summary": section.summary,
            "details": _redact_details(section.details),
        }
        for section in report.sections
    ]

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "ok": report.ok,
                    "checks": list(report.checks),
                    "details": redacted_report_details,
                    "sections": redacted_sections,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    typer.echo("Doctor preflight")
    typer.echo(f"Status: {'ok' if report.ok else 'fail'}")
    if redacted_sections:
        for section in redacted_sections:
            marker = "PASS" if section["ok"] else "FAIL"
            typer.echo(f"[{marker}] {section['name']}: {section['summary']}")
            for key, value in section["details"].items():
                if value:
                    typer.echo(f"  {key}: {value}")
    else:
        for check in report.checks:
            typer.echo(f"- {check}")

    selected_ids = redacted_report_details.get("selected_source_ids", "")
    if selected_ids:
        typer.echo(f"Selected smoke sources: {selected_ids}")
    else:
        typer.echo("Selected smoke sources: none")

    guidance = redacted_report_details.get("guidance")
    if guidance:
        typer.echo("Guidance:")
        for line in guidance.splitlines():
            typer.echo(f"- {line}")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show xui-reader version and exit."),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Active profile override used by commands when they omit --profile.",
    ),
    output_format: str | None = typer.Option(
        None,
        "--format",
        help="Output format: pretty|plain|json|jsonl.",
    ),
    headful: bool = typer.Option(True, "--headful/--headless", help="Browser mode placeholder."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug diagnostics."),
    timeout_ms: int = typer.Option(30_000, "--timeout-ms", help="Timeout in ms."),
) -> None:
    ctx.obj = {
        "profile": profile,
        "output_format": output_format,
        "headful": headful,
        "debug": debug,
        "timeout_ms": timeout_ms,
    }
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def _run_single_source_read(
    *,
    config: RuntimeConfig,
    source: SourceRef,
    profile_name: str | None,
    config_path: str | None,
    limit: int,
    new_only: bool,
    checkpoint_mode: str,
    debug_enabled: bool,
) -> MultiSourceReadResult:
    single = replace(config, sources=(source,))
    event_logger = _event_logger_for_profile(profile_name or config.app.default_profile, config_path) if debug_enabled else None
    return run_configured_read(
        single,
        profile_name=profile_name,
        config_path=config_path,
        limit=limit,
        new_only=new_only,
        checkpoint_mode=checkpoint_mode,
        enable_debug_artifacts=debug_enabled,
        event_logger=event_logger,
    )


def _emit_read_items(items: tuple[TweetItem, ...], *, ctx: typer.Context, config: RuntimeConfig) -> None:
    output_format = _resolve_output_format(ctx, config_default=config.app.default_format)
    rendered = render_items(items, output_format)
    if rendered:
        typer.echo(rendered)


def _config_with_optional_sources(config: RuntimeConfig, raw_sources: str | None) -> RuntimeConfig:
    if raw_sources is None or not raw_sources.strip():
        return config

    parsed_sources: list[SourceRef] = []
    for raw in (part.strip() for part in raw_sources.split(",")):
        if not raw:
            continue
        if raw.startswith("list:"):
            list_id = parse_list_id(raw.split(":", 1)[1])
            parsed_sources.append(
                SourceRef(
                    source_id=f"list:{list_id}",
                    kind=SourceKind.LIST,
                    value=list_id,
                    enabled=True,
                )
            )
            continue
        if raw.startswith("user:"):
            handle = parse_handle(raw.split(":", 1)[1])
            parsed_sources.append(
                SourceRef(
                    source_id=f"user:{handle}",
                    kind=SourceKind.USER,
                    value=handle,
                    enabled=True,
                )
            )
            continue
        raise RenderError(
            f"Invalid --sources entry '{raw}'. Expected list:<id_or_url> or user:<handle_or_url>."
        )

    if not parsed_sources:
        raise RenderError("No valid --sources entries found.")
    return replace(config, sources=tuple(parsed_sources))


def _read_result_payload(result: MultiSourceReadResult) -> dict[str, object]:
    return {
        "succeeded_sources": result.succeeded,
        "failed_sources": result.failed,
        "page_loads": result.total_page_loads,
        "scroll_rounds": result.total_scroll_rounds,
        "seen_items": result.total_observed_ids,
        "outcomes": [
            {
                "source_id": outcome.source_id,
                "source_kind": outcome.source_kind,
                "ok": outcome.ok,
                "item_count": outcome.item_count,
                "page_loads": outcome.page_loads,
                "scroll_rounds": outcome.scroll_rounds,
                "observed_ids": outcome.observed_ids,
                "error": outcome.error,
                "html_artifact_path": outcome.html_artifact_path,
                "selector_report_path": outcome.selector_report_path,
            }
            for outcome in result.outcomes
        ],
        "items": [tweet_item_to_dict(item) for item in result.items],
    }


def _watch_result_payload(
    result: WatchRunResult,
    *,
    exit_code: WatchExitCode | None = None,
) -> dict[str, object]:
    resolved_exit = exit_code if exit_code is not None else WatchExitCode.SUCCESS
    return {
        "exit_code": int(resolved_exit),
        "exit_state": _watch_exit_state_name(resolved_exit),
        "budget_stop_reason": result.budget_stop_reason,
        "interrupted": result.interrupted,
        "counters_state_path": result.counters_state_path,
        "cycles": [
            {
                "cycle": cycle.cycle,
                "started_at": cycle.started_at.isoformat(),
                "next_run_at": cycle.next_run_at.isoformat() if cycle.next_run_at else None,
                "sleep_seconds": cycle.sleep_seconds,
                "emitted_items": cycle.emitted_items,
                "seen_items": cycle.seen_items,
                "page_loads": cycle.page_loads,
                "scroll_rounds": cycle.scroll_rounds,
                "succeeded_sources": cycle.succeeded_sources,
                "failed_sources": cycle.failed_sources,
                "auth_failed_sources": cycle.auth_failed_sources,
            }
            for cycle in result.cycles
        ],
    }


def _resolve_profile(command_profile: str | None, ctx: typer.Context | None) -> str | None:
    if command_profile:
        return command_profile
    if ctx is None:
        return None
    if not isinstance(ctx.obj, dict):
        return None
    configured = ctx.obj.get("profile")
    if isinstance(configured, str) and configured:
        return configured
    return None


def _resolve_output_format(ctx: typer.Context | None, *, config_default: str) -> str:
    configured: str | None = None
    if ctx is not None and isinstance(ctx.obj, dict):
        value = ctx.obj.get("output_format")
        if isinstance(value, str) and value:
            configured = value
    resolved = configured or config_default
    if resolved not in VALID_OUTPUT_FORMATS:
        supported = ", ".join(sorted(VALID_OUTPUT_FORMATS))
        raise RenderError(
            f"Invalid output format '{resolved}'. Supported formats: {supported}."
        )
    return resolved


def _resolve_debug(ctx: typer.Context | None) -> bool:
    if ctx is None or not isinstance(ctx.obj, dict):
        return False
    return bool(ctx.obj.get("debug", False))


def _event_logger_for_profile(profile_name: str | None, config_path: str | None) -> JsonlEventLogger | None:
    if not profile_name:
        return None
    logs_dir = create_profile_logs_dir(profile_name, config_path)
    return JsonlEventLogger(logs_dir / "debug-events.jsonl")


def create_profile_logs_dir(profile_name: str, config_path: str | None) -> Path:
    root = profiles_root(config_path) / profile_name / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _watch_exit_state_name(exit_code: WatchExitCode) -> str:
    return exit_code.name.lower()


def _redact_details(details: dict[str, str]) -> dict[str, str]:
    sanitized = redact_value(details)
    if not isinstance(sanitized, dict):
        return {}
    return {str(key): str(value) for key, value in sanitized.items()}


def _auth_login_command_hint(profile_name: str, config_path: str | None) -> str:
    parts = ["xui", "auth", "login", "--profile", profile_name]
    if config_path:
        parts.extend(["--path", str(Path(config_path).expanduser())])
    return " ".join(shlex.quote(part) for part in parts)


def _all_source_failures_are_auth_related(result: MultiSourceReadResult) -> bool:
    if not result.outcomes:
        return False
    return all(
        (not outcome.ok) and bool(outcome.error) and "Missing storage_state" in str(outcome.error)
        for outcome in result.outcomes
    )
