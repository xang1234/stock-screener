"""Typer CLI scaffold matching the v2 command tree."""

from __future__ import annotations

import json

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
from .config import config_to_dict, init_default_config, load_runtime_config, resolve_config_path
from .diagnostics.doctor import run_doctor_preflight
from .errors import AuthError, CollectError, ConfigError, DiagnosticsError, ProfileError, SchedulerError
from .models import TweetItem
from .profiles import create_profile, delete_profile, list_profiles, switch_profile
from .scheduler.read import MultiSourceReadResult, run_configured_read
from .scheduler.watch import WatchRunResult, run_configured_watch

app = typer.Typer(help="xui-reader scaffold CLI with stable entrypoint wiring.")

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
def list_read(list_id_or_url: str) -> None:
    typer.echo(f"Not implemented yet: list read {list_id_or_url}.")


@list_app.command("parse-id")
def list_parse_id(list_url: str) -> None:
    typer.echo(f"Not implemented yet: list parse-id {list_url}.")


@user_app.command("read")
def user_read(handle_or_url: str) -> None:
    typer.echo(f"Not implemented yet: user read {handle_or_url}.")


@user_app.command("parse-handle")
def user_parse_handle(user_url: str) -> None:
    typer.echo(f"Not implemented yet: user parse-handle {user_url}.")


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
    limit: int = typer.Option(100, "--limit", min=1, help="Per-source collection limit."),
    as_json: bool = typer.Option(False, "--json", help="Render read result as JSON."),
) -> None:
    try:
        config = load_runtime_config(path)
        selected_profile = _resolve_profile(profile, ctx)
        result = run_configured_read(
            config,
            profile_name=selected_profile,
            config_path=path,
            limit=limit,
        )
    except (ConfigError, AuthError, CollectError, SchedulerError) as exc:
        typer.secho(f"Read failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    payload = _read_result_payload(result)
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Read summary: {payload['succeeded_sources']} succeeded, "
            f"{payload['failed_sources']} failed, {len(result.items)} merged items."
        )
        typer.echo("Source outcomes:")
        for outcome in payload["outcomes"]:
            status = "ok" if outcome["ok"] else "failed"
            suffix = f" ({outcome['error']})" if outcome["error"] else ""
            typer.echo(f"- {outcome['source_id']} [{status}] items={outcome['item_count']}{suffix}")
        if result.items:
            typer.echo("Merged order:")
            for item in result.items:
                created = item.created_at.isoformat() if item.created_at else "none"
                typer.echo(f"- {created} {item.source_id} {item.tweet_id}")

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
    limit: int = typer.Option(100, "--limit", min=1, help="Per-source collection limit."),
    interval_seconds: int = typer.Option(300, "--interval-seconds", min=1, help="Watch interval."),
    jitter_ratio: float = typer.Option(
        0.0,
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
    max_cycles: int = typer.Option(1, "--max-cycles", min=1, help="Number of watch cycles to run."),
    as_json: bool = typer.Option(False, "--json", help="Render watch result as JSON."),
) -> None:
    try:
        config = load_runtime_config(path)
        selected_profile = _resolve_profile(profile, ctx)
        result = run_configured_watch(
            config,
            profile_name=selected_profile,
            config_path=path,
            limit=limit,
            interval_seconds=interval_seconds,
            jitter_ratio=jitter_ratio,
            shutdown_window=shutdown_window,
            max_cycles=max_cycles,
        )
    except (ConfigError, AuthError, CollectError, SchedulerError) as exc:
        typer.secho(f"Watch failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    payload = _watch_result_payload(result)
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Watch completed: {len(result.cycles)} cycle(s).")
        for cycle in result.cycles:
            next_run = cycle.next_run_at.isoformat() if cycle.next_run_at else "none"
            typer.echo(
                f"- cycle={cycle.cycle} emitted={cycle.emitted_items} "
                f"sources_ok={cycle.succeeded_sources} sources_failed={cycle.failed_sources} "
                f"sleep={cycle.sleep_seconds:.2f}s next={next_run}"
            )


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

    if as_json:
        sections = [
            {
                "name": section.name,
                "ok": section.ok,
                "summary": section.summary,
                "details": section.details,
            }
            for section in report.sections
        ]
        typer.echo(
            json.dumps(
                {
                    "ok": report.ok,
                    "checks": list(report.checks),
                    "details": report.details,
                    "sections": sections,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    typer.echo("Doctor preflight")
    typer.echo(f"Status: {'ok' if report.ok else 'fail'}")
    if report.sections:
        for section in report.sections:
            marker = "PASS" if section.ok else "FAIL"
            typer.echo(f"[{marker}] {section.name}: {section.summary}")
            for key, value in section.details.items():
                if value:
                    typer.echo(f"  {key}: {value}")
    else:
        for check in report.checks:
            typer.echo(f"- {check}")

    selected_ids = report.details.get("selected_source_ids", "")
    if selected_ids:
        typer.echo(f"Selected smoke sources: {selected_ids}")
    else:
        typer.echo("Selected smoke sources: none")

    guidance = report.details.get("guidance")
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
    output_format: str = typer.Option(
        "pretty",
        "--format",
        help="Output format placeholder: pretty|plain|json|jsonl.",
    ),
    headful: bool = typer.Option(True, "--headful/--headless", help="Browser mode placeholder."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug diagnostics placeholders."),
    timeout_ms: int = typer.Option(30_000, "--timeout-ms", help="Timeout placeholder in ms."),
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


def _read_result_payload(result: MultiSourceReadResult) -> dict[str, object]:
    return {
        "succeeded_sources": result.succeeded,
        "failed_sources": result.failed,
        "outcomes": [
            {
                "source_id": outcome.source_id,
                "source_kind": outcome.source_kind,
                "ok": outcome.ok,
                "item_count": outcome.item_count,
                "error": outcome.error,
            }
            for outcome in result.outcomes
        ],
        "items": [_tweet_item_to_dict(item) for item in result.items],
    }


def _watch_result_payload(result: WatchRunResult) -> dict[str, object]:
    return {
        "cycles": [
            {
                "cycle": cycle.cycle,
                "started_at": cycle.started_at.isoformat(),
                "next_run_at": cycle.next_run_at.isoformat() if cycle.next_run_at else None,
                "sleep_seconds": cycle.sleep_seconds,
                "emitted_items": cycle.emitted_items,
                "succeeded_sources": cycle.succeeded_sources,
                "failed_sources": cycle.failed_sources,
            }
            for cycle in result.cycles
        ]
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


def _tweet_item_to_dict(item: TweetItem) -> dict[str, object]:
    return {
        "tweet_id": item.tweet_id,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "author_handle": item.author_handle,
        "text": item.text,
        "source_id": item.source_id,
        "is_reply": item.is_reply,
        "is_repost": item.is_repost,
        "is_pinned": item.is_pinned,
        "has_quote": item.has_quote,
        "quote_tweet_id": item.quote_tweet_id,
    }
