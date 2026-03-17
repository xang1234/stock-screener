"""Smoke test for the desktop runtime bundle."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def _request(method: str, url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_json(url: str, timeout_seconds: int = 120) -> dict:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            return _request("GET", url)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(1)
    detail = f" (last error: {last_error})" if last_error else ""
    raise RuntimeError(f"Timed out waiting for {url}{detail}")


def _wait_for_status(url: str, terminal_states: set[str], timeout_seconds: int = 240) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        payload = _request("GET", url)
        if payload.get("status") in terminal_states:
            return payload
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for terminal state from {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the Stock Scanner desktop runtime")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    _wait_for_json(f"{base_url}/livez")

    capabilities = _request("GET", f"{base_url}/api/v1/app-capabilities")
    if "desktop_mode" not in capabilities:
        raise RuntimeError("Capabilities response did not include desktop_mode")

    bootstrap = _request("POST", f"{base_url}/api/v1/app/bootstrap")
    if bootstrap.get("status") not in {"queued", "running", "completed"}:
        raise RuntimeError(f"Unexpected bootstrap status: {bootstrap}")

    bootstrap_status = _wait_for_status(
        f"{base_url}/api/v1/app/bootstrap/status",
        {"completed", "failed"},
        timeout_seconds=300,
    )
    if bootstrap_status["status"] == "failed":
        raise RuntimeError(f"Bootstrap failed: {bootstrap_status.get('error') or bootstrap_status}")

    scan = _request(
        "POST",
        f"{base_url}/api/v1/scans",
        payload={
            "universe": "custom",
            "symbols": ["AAPL", "MSFT", "NVDA"],
            "screeners": ["minervini"],
            "criteria": {},
        },
    )
    scan_id = scan["scan_id"]
    status = _wait_for_status(
        f"{base_url}/api/v1/scans/{scan_id}/status",
        {"completed", "failed", "cancelled"},
        timeout_seconds=300,
    )
    if status["status"] != "completed":
        raise RuntimeError(f"Scan did not complete successfully: {status}")

    results = _request("GET", f"{base_url}/api/v1/scans/{scan_id}/results")
    if "results" not in results:
        raise RuntimeError(f"Results payload missing results array: {results}")

    print(
        json.dumps(
            {
                "capabilities": capabilities,
                "bootstrap": bootstrap_status,
                "scan_status": status,
                "result_count": len(results.get("results", [])),
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        sys.exit(1)
