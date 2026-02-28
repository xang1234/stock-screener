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
from .errors import AuthError, ConfigError, DiagnosticsError, ProfileError
from .profiles import create_profile, delete_profile, list_profiles, switch_profile

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
def read() -> None:
    typer.echo("Not implemented yet: read.")


@app.command("watch")
def watch() -> None:
    typer.echo("Not implemented yet: watch.")


@app.command("doctor")
def doctor(
    path: str | None = typer.Option(
        None, "--path", help="Optional config TOML path (defaults to platform config dir)."
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
        report = run_doctor_preflight(config, max_sources=max_sources)
    except (ConfigError, DiagnosticsError) as exc:
        typer.secho(f"Doctor failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    if as_json:
        typer.echo(
            json.dumps(
                {"ok": report.ok, "checks": list(report.checks), "details": report.details},
                indent=2,
                sort_keys=True,
            )
        )
        return

    typer.echo("Doctor preflight")
    typer.echo(f"Status: {'ok' if report.ok else 'fail'}")
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
    profile: str = typer.Option("default", "--profile", help="Active profile name."),
    output_format: str = typer.Option(
        "pretty",
        "--format",
        help="Output format placeholder: pretty|plain|json|jsonl.",
    ),
    headful: bool = typer.Option(True, "--headful/--headless", help="Browser mode placeholder."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug diagnostics placeholders."),
    timeout_ms: int = typer.Option(30_000, "--timeout-ms", help="Timeout placeholder in ms."),
) -> None:
    _ = (profile, output_format, headful, debug, timeout_ms)
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
