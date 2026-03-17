"""Stop a running desktop launcher instance."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _pid_file_path() -> Path:
    from app.config import settings

    return settings.desktop_data_path / "stockscanner.pid"


def _terminate_pid(pid: int) -> None:
    if os.name == "nt":
        try:
            os.kill(pid, signal.SIGTERM)
            return
        except OSError:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
            )
            return

    os.kill(pid, signal.SIGTERM)


def main() -> None:
    os.environ.setdefault("DESKTOP_MODE", "true")
    pid_file = _pid_file_path()
    if not pid_file.exists():
        return

    try:
        payload = json.loads(pid_file.read_text(encoding="utf-8"))
        pid = int(payload["pid"])
        _terminate_pid(pid)
    finally:
        try:
            pid_file.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
