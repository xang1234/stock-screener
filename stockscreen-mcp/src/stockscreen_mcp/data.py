"""Read-only access to the static site's published JSON, cached per publish."""

from __future__ import annotations

import gzip
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError

from . import __version__

DEFAULT_BASE_URL = "https://xang1234.github.io/stock-screener/static-data/"
# GitHub Pages serves the data with Cache-Control: max-age=600.
MANIFEST_TTL_SECONDS = 600
# Chart series and per-field availability notes: large and unused by any tool.
HEAVY_FIELDS = frozenset({"price_sparkline_data", "rs_sparkline_data", "field_availability"})
# ponytail: pruned US rows are ~100 MB as Python objects (more while a chunk is parsed),
# so keep only the most recently used markets' files; raise it or switch to columnar rows.
MAX_CACHED_MARKETS = 2


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(
        url, headers={"Accept-Encoding": "gzip", "User-Agent": f"stockscreen-mcp/{__version__}"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
    return json.loads(body)


def prune(value: Any) -> Any:
    """Drop HEAVY_FIELDS at any depth (group files nest stock rows)."""
    if isinstance(value, dict):
        return {k: prune(v) for k, v in value.items() if k not in HEAVY_FIELDS}
    if isinstance(value, list):
        return [prune(v) for v in value]
    return value


class _GenerationReplaced(Exception):
    """The publish this load started from was replaced mid-load."""


class StaticData:
    def __init__(
        self,
        base_url: str | None = None,
        fetch: Callable[[str], Any] = fetch_json,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        base = base_url or os.environ.get("STOCKSCREEN_DATA_URL") or DEFAULT_BASE_URL
        self.base_url = base.rstrip("/") + "/"
        self._fetch = fetch
        self._clock = clock
        # ponytail: one lock for every load; per-file locks if concurrent calls ever matter.
        self._lock = threading.RLock()
        self._manifest: dict[str, Any] | None = None
        self._manifest_at = 0.0
        self._cache: dict[tuple[str, str], Any] = {}
        self._recent_markets: OrderedDict[str, None] = OrderedDict()

    def _get(self, path: str) -> Any:
        url = self.base_url + path
        try:
            return self._fetch(url)
        except urllib.error.HTTPError:
            raise
        # OSError covers URLError and timeouts; ValueError bad JSON; the rest truncated bodies.
        except (OSError, ValueError, EOFError, zlib.error, http.client.HTTPException) as exc:
            raise ToolError(f"Could not reach the published data at {url}: {exc!r}") from exc

    def manifest(self, *, refresh: bool = False) -> dict[str, Any]:
        with self._lock:
            stale = self._clock() - self._manifest_at > MANIFEST_TTL_SECONDS
            if refresh or self._manifest is None or stale:
                try:
                    manifest = self._get("manifest.json")
                except urllib.error.HTTPError as exc:
                    raise ToolError(f"The published data manifest is unavailable ({exc.code}).") from exc
                self._manifest, self._manifest_at = manifest, self._clock()
                root = manifest["data_root"]
                self._cache = {k: v for k, v in self._cache.items() if k[0] == root}
            return self._manifest

    def market(self, code: str) -> dict[str, Any]:
        code = code.strip().upper()
        manifest = self.manifest()
        entry = manifest["markets"].get(code)
        if entry is not None:
            return entry
        if code in manifest.get("unavailable_markets", []):
            raise ToolError(f"{code} is not available in the current publish; try again after the next one.")
        known = ", ".join(sorted(manifest["markets"]))
        raise ToolError(f"Unknown market {code!r}. Available markets: {known}.")

    def _touch_market(self, path: str) -> None:
        """Mark a market's files as recently used, evicting the least recent beyond the cap."""
        parts = path.split("/")
        if len(parts) < 2 or parts[0] != "markets":
            return
        self._recent_markets.pop(parts[1], None)
        self._recent_markets[parts[1]] = None
        while len(self._recent_markets) > MAX_CACHED_MARKETS:
            evicted, _ = self._recent_markets.popitem(last=False)
            prefix = f"markets/{evicted}/"
            self._cache = {k: v for k, v in self._cache.items() if not k[1].startswith(prefix)}

    def _store(self, root: str, key: tuple[str, str], value: Any) -> None:
        # A publish replaced mid-load is still read consistently, but never cached.
        if root == self.manifest()["data_root"]:
            self._cache[key] = value

    def _load(self, path: str, root: str, *, cache: bool = True) -> Any:
        """One file of the publish under `root` (pinned for the whole load)."""
        key = (root, path)
        self._touch_market(path)
        if key in self._cache:
            return self._cache[key]
        try:
            value = prune(self._get(root + path))
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and self.manifest(refresh=True)["data_root"] != root:
                raise _GenerationReplaced from exc
            raise ToolError(f"Published file {root}{path} is unavailable ({exc.code}).") from exc
        if cache:
            self._store(root, key, value)
        return value

    def _retrying(self, load: Callable[[str], Any]) -> Any:
        """Run a load against one publish; only the newest is deployed, so restart on a new one."""
        with self._lock:
            for _ in range(2):
                try:
                    return load(self.manifest()["data_root"])
                except _GenerationReplaced:
                    continue
            raise ToolError("The published data changed while loading; try again in a minute.")

    def load(self, path: str) -> Any:
        return self._retrying(lambda root: self._load(path, root))

    def page(self, code: str, page: str) -> Any | None:
        """A market page (home, breadth, groups, scan), or None if not published for it."""
        entry = self.page_entry(code, page)
        return None if entry is None else self.load(entry["path"])

    def asset(self, code: str, name: str) -> Any | None:
        """A market's optional asset, or None if the current publish has none."""

        def load_asset(root: str) -> Any | None:
            entry = self.market(code).get("assets", {}).get(name)
            return None if entry is None else self._load(entry["path"], root)

        return self._retrying(load_asset)

    def page_entry(self, code: str, page: str) -> dict[str, Any] | None:
        return self.market(code).get("pages", {}).get(page)

    def scan(self, code: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """The scan manifest and every row, initial rows and chunks merged by symbol."""
        if self.page_entry(code, "scan") is None:
            raise ToolError(f"Scan results are not published for {code.upper()}.")

        def load_all(root: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            scan_path = self.page_entry(code, "scan")["path"]
            scan = self._load(scan_path, root)
            # Lives under the market's prefix so eviction drops it with the market's files.
            key = (root, scan_path.rsplit("/", 1)[0] + "/rows")
            if key in self._cache:
                return scan, self._cache[key]
            merged = {row["symbol"]: row for row in scan.get("initial_rows", [])}
            for chunk in scan.get("chunks", []):
                for row in self._load(chunk["path"], root, cache=False)["rows"]:
                    merged[row["symbol"]] = row
            rows = list(merged.values())
            self._store(root, key, rows)
            return scan, rows

        return self._retrying(load_all)
