"""Typer CLI entrypoint for the scaffold package."""

from __future__ import annotations

import typer

from . import __version__

app = typer.Typer(help="xui-reader scaffold CLI with stable entrypoint wiring.")

auth_app = typer.Typer(help="Authentication commands.")
profile_app = typer.Typer(help="Profile management commands.")
list_app = typer.Typer(help="List helpers.")
user_app = typer.Typer(help="User helpers.")
config_app = typer.Typer(help="Config commands.")

app.add_typer(auth_app, name="auth")
app.add_typer(profile_app, name="profile")
app.add_typer(list_app, name="list")
app.add_typer(user_app, name="user")
app.add_typer(config_app, name="config")

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show xui-reader version and exit."),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
