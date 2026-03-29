"""Best-effort launch agent install/update helpers for macOS desktop mode."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from xml.sax.saxutils import escape

from app.config import settings


@dataclass(frozen=True)
class LaunchAgentStatus:
    label: str
    plist_path: Path
    installed: bool
    loaded: bool
    warning: str | None = None


class DesktopLaunchAgentService:
    def __init__(self) -> None:
        self._label = settings.desktop_launch_agent_label

    @property
    def plist_path(self) -> Path:
        return Path.home() / "Library" / "LaunchAgents" / f"{self._label}.plist"

    def render(self, program_arguments: list[str]) -> str:
        interval_seconds = max(settings.desktop_launch_agent_interval_minutes, 5) * 60
        program_args_xml = "\n".join(
            f"    <string>{escape(argument)}</string>"
            for argument in program_arguments
        )
        return textwrap.dedent(
            f"""\
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
            <plist version="1.0">
            <dict>
              <key>Label</key>
              <string>{self._label}</string>
              <key>ProgramArguments</key>
              <array>
            {program_args_xml}
              </array>
              <key>RunAtLoad</key>
              <true/>
              <key>StartInterval</key>
              <integer>{interval_seconds}</integer>
              <key>StandardOutPath</key>
              <string>{settings.desktop_data_path / "logs" / "launch-agent.out.log"}</string>
              <key>StandardErrorPath</key>
              <string>{settings.desktop_data_path / "logs" / "launch-agent.err.log"}</string>
            </dict>
            </plist>
            """
        )

    def install_or_update(self, program_arguments: list[str]) -> LaunchAgentStatus:
        plist_path = self.plist_path
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        (settings.desktop_data_path / "logs").mkdir(parents=True, exist_ok=True)
        plist_path.write_text(self.render(program_arguments), encoding="utf-8")

        loaded = False
        warning = None
        if sys_platform_is_macos():
            uid = os.getuid()
            boot_target = f"gui/{uid}"
            subprocess.run(["launchctl", "bootout", boot_target, str(plist_path)], check=False, capture_output=True)
            result = subprocess.run(
                ["launchctl", "bootstrap", boot_target, str(plist_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                loaded = True
            else:
                warning = result.stderr.strip() or "launchctl bootstrap failed; agent will load on next login"

        return LaunchAgentStatus(
            label=self._label,
            plist_path=plist_path,
            installed=True,
            loaded=loaded,
            warning=warning,
        )


def sys_platform_is_macos() -> bool:
    return sys.platform == "darwin"
