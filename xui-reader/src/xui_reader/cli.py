"""Typer CLI scaffold matching the v2 command tree."""

from __future__ import annotations

import typer

from . import __version__

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
def auth_login() -> None:
    typer.echo("Not implemented yet: auth login.")


@auth_app.command("status")
def auth_status() -> None:
    typer.echo("Not implemented yet: auth status.")


@auth_app.command("logout")
def auth_logout() -> None:
    typer.echo("Not implemented yet: auth logout.")


@profiles_app.command("list")
def profiles_list() -> None:
    typer.echo("Not implemented yet: profiles list.")


@profiles_app.command("create")
def profiles_create(name: str) -> None:
    typer.echo(f"Not implemented yet: profiles create {name}.")


@profiles_app.command("delete")
def profiles_delete(name: str) -> None:
    typer.echo(f"Not implemented yet: profiles delete {name}.")


@profiles_app.command("switch")
def profiles_switch(name: str) -> None:
    typer.echo(f"Not implemented yet: profiles switch {name}.")


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
def config_init() -> None:
    typer.echo("Not implemented yet: config init.")


@config_app.command("show")
def config_show() -> None:
    typer.echo("Not implemented yet: config show.")


@app.command("read")
def read() -> None:
    typer.echo("Not implemented yet: read.")


@app.command("watch")
def watch() -> None:
    typer.echo("Not implemented yet: watch.")


@app.command("doctor")
def doctor() -> None:
    typer.echo("Not implemented yet: doctor.")


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
