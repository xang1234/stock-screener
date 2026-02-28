"""Selector pack resolution and override validation behavior."""

from __future__ import annotations

import json
from pathlib import Path

from xui_reader.extract.selectors import (
    DEFAULT_SELECTOR_PACK,
    default_selector_pack,
    resolve_selector_pack,
)


def test_default_selector_pack_returns_copy() -> None:
    pack = default_selector_pack()
    assert pack == DEFAULT_SELECTOR_PACK
    pack["tweet.text"] = ("override",)
    assert DEFAULT_SELECTOR_PACK["tweet.text"] != ("override",)


def test_resolve_selector_pack_override_wins_for_valid_keys() -> None:
    result = resolve_selector_pack(
        override_data={
            "tweet.text": ['div[data-testid="tweetText"]', "article div.lang"],
            "tweet.time": "time[data-testid='tweet-time']",
        }
    )
    assert result.warnings == ()
    assert result.loaded_override is True
    assert result.selectors["tweet.text"] == ('div[data-testid="tweetText"]', "article div.lang")
    assert result.selectors["tweet.time"] == ("time[data-testid='tweet-time']",)


def test_resolve_selector_pack_reports_unknown_override_keys() -> None:
    result = resolve_selector_pack(
        override_data={
            "tweet.text": "div[data-testid='tweetText']",
            "tweet.unknown": ".bad",
            "not_a_selector": ".also-bad",
        }
    )
    joined = " ".join(result.warnings)
    assert "Unknown selector override key 'tweet.unknown'" in joined
    assert "Unknown selector override key 'not_a_selector'" in joined
    assert result.selectors["tweet.text"] == ("div[data-testid='tweetText']",)


def test_resolve_selector_pack_rejects_invalid_selector_value_types() -> None:
    result = resolve_selector_pack(
        override_data={
            "tweet.text": 123,
            "tweet.time": ["time", 1],
        }
    )
    joined = " ".join(result.warnings)
    assert "Selector override for 'tweet.text' must be" in joined
    assert "Selector override for 'tweet.time' must be" in joined
    assert result.selectors["tweet.text"] == DEFAULT_SELECTOR_PACK["tweet.text"]
    assert result.selectors["tweet.time"] == DEFAULT_SELECTOR_PACK["tweet.time"]


def test_resolve_selector_pack_reads_json_override_file(tmp_path: Path) -> None:
    override_path = tmp_path / "selectors.json"
    override_path.write_text(
        json.dumps({"selectors": {"tweet.text": "article div[data-testid='tweetText']"}}),
        encoding="utf-8",
    )

    result = resolve_selector_pack(override_path)

    assert result.warnings == ()
    assert result.loaded_override is True
    assert result.selectors["tweet.text"] == ("article div[data-testid='tweetText']",)


def test_resolve_selector_pack_reads_toml_override_file(tmp_path: Path) -> None:
    override_path = tmp_path / "selectors.toml"
    override_path.write_text(
        """
[selectors]
tweet.text = "article div[data-testid='tweetText']"
tweet.time = ["time", "article time"]
""".strip(),
        encoding="utf-8",
    )

    result = resolve_selector_pack(override_path)

    assert result.warnings == ()
    assert result.selectors["tweet.text"] == ("article div[data-testid='tweetText']",)
    assert result.selectors["tweet.time"] == ("time", "article time")


def test_resolve_selector_pack_falls_back_for_malformed_override(tmp_path: Path) -> None:
    override_path = tmp_path / "selectors.json"
    override_path.write_text("{not json", encoding="utf-8")

    result = resolve_selector_pack(override_path)

    assert result.selectors == DEFAULT_SELECTOR_PACK
    assert result.loaded_override is False
    assert any("invalid JSON" in warning for warning in result.warnings)


def test_resolve_selector_pack_falls_back_for_missing_override_file(tmp_path: Path) -> None:
    override_path = tmp_path / "missing.toml"

    result = resolve_selector_pack(override_path)

    assert result.selectors == DEFAULT_SELECTOR_PACK
    assert result.loaded_override is False
    assert any("was not found" in warning for warning in result.warnings)
