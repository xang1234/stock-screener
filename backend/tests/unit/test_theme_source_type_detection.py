"""Regression tests for source-type detection contract."""

from app.api.v1.themes import detect_source_type_from_url


def test_detect_source_type_treats_x_list_urls_as_twitter() -> None:
    assert detect_source_type_from_url("https://x.com/i/lists/84839422", "news") == "twitter"


def test_detect_source_type_treats_x_profile_urls_as_twitter() -> None:
    assert detect_source_type_from_url("https://x.com/somehandle", "substack") == "twitter"


def test_detect_source_type_does_not_treat_non_x_domains_as_twitter() -> None:
    assert detect_source_type_from_url("https://examplex.com/account", "news") == "news"


def test_detect_source_type_does_not_force_generic_rss_urls_to_substack() -> None:
    assert detect_source_type_from_url(
        "https://feeds.marketwatch.com/marketwatch/topstories/",
        "news",
    ) == "news"
