"""Fixture sanitizer behavior for snapshot-safe extractor fixtures."""

from __future__ import annotations

from pathlib import Path
import re

from xui_reader.extract.fixture_sanitizer import sanitize_fixture_file, sanitize_fixture_html
from xui_reader.extract.tweets import PrimaryFallbackTweetExtractor


RAW_HTML = """
<section>
  <article data-testid="tweet" data-user-id="99887766">
    <a href="/Alice/status/1234567890123456789">status</a>
    <a href="/Alice">profile</a>
    <time datetime="2026-03-02T01:02:03Z"></time>
    <div data-testid="tweetText">
      Reach me at alice@example.com or +1 (555) 123-9876.
      Authorization: Bearer super-secret-token.
      sessionid=real-cookie.
    </div>
    <div>@Alice replied to @Bob</div>
  </article>
</section>
""".strip()


def test_sanitize_fixture_html_redacts_secrets_and_pii() -> None:
    result = sanitize_fixture_html(RAW_HTML)
    sanitized = result.sanitized_html

    assert "super-secret-token" not in sanitized
    assert "real-cookie" not in sanitized
    assert "alice@example.com" not in sanitized
    assert "+1 (555) 123-9876" not in sanitized
    assert "@Alice" not in sanitized
    assert "@Bob" not in sanitized
    assert re.search(r"/user\d{2}/status/\d{19}", sanitized)
    assert "href=\"/user01\"" in sanitized
    assert "@user01 replied to @user02" in sanitized
    assert 'data-user-id="00000001"' in sanitized
    assert result.handle_replacements >= 4
    assert result.status_id_replacements == 1
    assert result.numeric_id_replacements == 1
    assert result.email_replacements == 1
    assert result.phone_replacements == 1


def test_sanitized_fixture_remains_representative_for_extractor_behavior() -> None:
    result = sanitize_fixture_html(RAW_HTML)
    extractor = PrimaryFallbackTweetExtractor()
    items = extractor.extract({"html": result.sanitized_html, "source_id": "list:demo"})

    assert len(items) == 1
    item = items[0]
    assert item.author_handle == "@user01"
    assert re.fullmatch(r"\d{19}", item.tweet_id)
    assert item.text is not None
    assert "Authorization" in item.text
    assert "super-secret-token" not in item.text


def test_sanitize_fixture_file_writes_default_output_path(tmp_path: Path) -> None:
    raw_path = tmp_path / "extractor_sample.html"
    raw_path.write_text(RAW_HTML, encoding="utf-8")

    result = sanitize_fixture_file(raw_path)

    assert result.output_path == tmp_path / "extractor_sample.sanitized.html"
    assert result.output_path is not None
    assert result.output_path.exists()
    assert result.output_path.read_text(encoding="utf-8") == result.sanitized_html


def test_sanitize_fixture_html_does_not_count_reserved_status_handle_as_handle_replacement() -> None:
    html = '<a href="/i/status/1234567890123456789">status</a>'

    result = sanitize_fixture_html(html)

    assert '/i/status/0000000000000000001' in result.sanitized_html
    assert result.status_id_replacements == 1
    assert result.handle_replacements == 0
