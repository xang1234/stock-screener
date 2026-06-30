"""Streaming reader helpers for daily price bundle payloads."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO


class StreamingJsonReader:
    """Incremental JSON decoder for bundle files with a very large rows array."""

    _CHUNK_SIZE = 1024 * 1024

    def __init__(self, handle: TextIO) -> None:
        self._handle = handle
        self._decoder = json.JSONDecoder()
        self._buffer = ""
        self._pos = 0
        self._eof = False

    def _compact(self) -> None:
        if self._pos > self._CHUNK_SIZE or self._pos > len(self._buffer) // 2:
            self._buffer = self._buffer[self._pos:]
            self._pos = 0

    def _read_more(self) -> None:
        self._compact()
        chunk = self._handle.read(self._CHUNK_SIZE)
        if chunk == "":
            self._eof = True
            return
        self._buffer += chunk

    def _ensure_char(self) -> str:
        while self._pos >= len(self._buffer):
            if self._eof:
                raise json.JSONDecodeError(
                    "Unexpected end of daily price bundle",
                    self._buffer,
                    self._pos,
                )
            self._read_more()
        return self._buffer[self._pos]

    def skip_whitespace(self) -> None:
        while True:
            char = self._ensure_char()
            if char not in " \t\r\n":
                return
            self._pos += 1

    def expect(self, expected: str) -> None:
        self.skip_whitespace()
        actual = self._ensure_char()
        if actual != expected:
            raise json.JSONDecodeError(
                f"Expected {expected!r}, got {actual!r}",
                self._buffer,
                self._pos,
            )
        self._pos += 1
        self._compact()

    def consume_if(self, expected: str) -> bool:
        self.skip_whitespace()
        if self._ensure_char() != expected:
            return False
        self._pos += 1
        self._compact()
        return True

    def decode_value(self) -> Any:
        self.skip_whitespace()
        while True:
            try:
                value, end = self._decoder.raw_decode(self._buffer, self._pos)
            except json.JSONDecodeError:
                if self._eof:
                    raise
                self._read_more()
                continue
            self._pos = end
            self._compact()
            return value

    def skip_value(self) -> None:
        self.skip_whitespace()
        char = self._ensure_char()
        if char in "[{":
            self._skip_compound_value()
            return
        self.decode_value()

    def _skip_compound_value(self) -> None:
        depth = 0
        in_string = False
        escaped = False
        while True:
            char = self._ensure_char()
            self._pos += 1
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char in "[{":
                    depth += 1
                elif char in "]}":
                    depth -= 1
                    if depth == 0:
                        self._compact()
                        return

    def iter_array(self) -> Iterator[Any]:
        self.expect("[")
        if self.consume_if("]"):
            return
        while True:
            yield self.decode_value()
            if self.consume_if(","):
                continue
            self.expect("]")
            return


def open_bundle_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def read_daily_price_bundle_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    with open_bundle_text(path) as handle:
        reader = StreamingJsonReader(handle)
        reader.expect("{")
        if reader.consume_if("}"):
            return metadata
        while True:
            key = reader.decode_value()
            if not isinstance(key, str):
                raise ValueError("Daily price bundle keys must be strings")
            reader.expect(":")
            if key == "rows":
                reader.skip_value()
            else:
                metadata[key] = reader.decode_value()
            if reader.consume_if(","):
                continue
            reader.expect("}")
            return metadata


def iter_daily_price_bundle_rows(path: Path) -> Iterator[dict[str, Any]]:
    with open_bundle_text(path) as handle:
        reader = StreamingJsonReader(handle)
        reader.expect("{")
        if reader.consume_if("}"):
            return
        while True:
            key = reader.decode_value()
            if not isinstance(key, str):
                raise ValueError("Daily price bundle keys must be strings")
            reader.expect(":")
            if key == "rows":
                for row_index, row in enumerate(reader.iter_array(), start=1):
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"Daily price bundle row {row_index} must be an object"
                        )
                    yield row
                if reader.consume_if(","):
                    continue
                reader.expect("}")
                return
            reader.skip_value()
            if reader.consume_if(","):
                continue
            reader.expect("}")
            return
