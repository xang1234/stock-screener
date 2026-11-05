"""scripts/render_seo_pages.py against a small publish with the real static-data layout."""

import importlib.util
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SITE = "https://example.github.io/stock-screener/"


def load_generator():
    spec = importlib.util.spec_from_file_location("render_seo_pages", REPO_ROOT / "scripts" / "render_seo_pages.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(symbol, **fields):
    base = {
        "symbol": symbol, "company_name": f"{symbol} Corp", "market": "US", "currency": "USD",
        "is_scannable": True, "data_status": "complete", "current_price": 50.0, "price_change_1d": 1.5,
        "rs_rating": 80.0, "rs_rating_1m": 70.0, "rs_rating_3m": 75.0, "rs_rating_12m": 85.0,
        "stage": 2, "stage_name": "Stage 2 - Advancing", "composite_score": 60.0, "minervini_score": 75.0,
        "canslim_score": 40.0, "ma_alignment": True, "ibd_industry_group": "Semiconductors", "ibd_group_rank": 5,
        "eps_growth_qq": 30.0, "sales_growth_qq": 20.0, "week_52_high_distance": -3.0, "market_cap_usd": 2.5e9,
        "price_sparkline_data": [1.0, 1.02, 0.99, 1.05, 1.08],
    }
    base.update(fields)
    return base


MINERVINI = {
    "id": "minervini", "name": "Minervini Trend Template", "description": "Stage 2 uptrend stocks",
    "sort_by": "minervini_score", "sort_order": "desc",
    "filter_expression": {"group_join": "any", "groups": [], "required": {"match": "all", "conditions": [
        {"field": "minervini_score", "kind": "range", "min": 70, "max": None},
        {"field": "rs_rating", "kind": "range", "min": 70, "max": None},
        {"field": "ma_alignment", "kind": "boolean", "value": True},
        {"field": "stage", "kind": "range", "min": 2, "max": 2},
    ]}},
}
CUP = {
    "id": "se_cup_handle", "name": "Cup with Handle", "description": "Cup-with-handle bases",
    "sort_by": "rs_rating", "sort_order": "desc",
    "filter_expression": {"group_join": "any", "groups": [], "required": {"match": "all", "conditions": [
        {"field": "se_pattern_primary", "kind": "categorical", "values": ["cup_with_handle"]},
    ]}},
}


def horizon(sessions, **fields):
    base = {
        "horizon_sessions": sessions, "sample_size": 0, "signal_dates": 0, "effective_dates": 0.0, "pending_windows": 0,
        "oldest_open_sessions": None, "thin": True, "positive_rate": None, "beat_benchmark_rate": None,
        "median_return_pct": None, "median_excess_return_pct": None,
        "excess_sample_size": 0, "skipped_missing_history": 0, "suppressed_overlapping": 0,
    }
    base.update(fields)
    return base


def cohort(preset, name, origin, events, horizons, retired=False):
    return {"preset": preset, "name": name, "retired": retired, "definition": "241a0f65f493c25e", "origin": origin,
            "first_date": "2026-10-22", "last_date": "2026-11-04", "events": events, "signal_dates": 6,
            "horizons": horizons}


SCOREBOARD = {
    "schema_version": "static-scoreboard-v1", "market": "US", "as_of_date": "2026-11-04",
    "generated_at": "2026-11-05T02:15:22Z", "benchmark": "SPY", "recording_since": "2026-10-05",
    "scored_since": "2026-10-05", "ledger_through": "2026-11-04", "missing_sessions": 2,
    "cohorts": [
        cohort("minervini", "Minervini", "live", 357,
               [horizon(5, pending_windows=357), horizon(20, pending_windows=357), horizon(60, pending_windows=357)]),
        cohort("new_highs", "<New> Highs", "live", 0, [horizon(5), horizon(20), horizon(60)]),
        cohort("vcp", "VCP", "live", 9, [
            horizon(5, skipped_missing_history=3, suppressed_overlapping=6), horizon(20), horizon(60)]),
        cohort("minervini", "Minervini", "artifact", 1441, [
            horizon(5, sample_size=1298, signal_dates=6, effective_dates=5.4, pending_windows=40, thin=False,
                    positive_rate=0.57, beat_benchmark_rate=0.48, median_return_pct=0.86, median_excess_return_pct=0.12,
                    excess_sample_size=1298, suppressed_overlapping=100),
            horizon(20, pending_windows=1341), horizon(60, pending_windows=1341)]),
        cohort("canslim", "CANSLIM", "artifact", 14, [
            horizon(5, sample_size=12, signal_dates=2, effective_dates=1.9, pending_windows=2, positive_rate=0.5,
                    beat_benchmark_rate=0.25, median_return_pct=-0.31, median_excess_return_pct=-0.95,
                    excess_sample_size=12),
            horizon(20, pending_windows=14), horizon(60, pending_windows=14)]),
        cohort("old_breakout", "Old Breakout", "artifact_pre_stage_fix", 8, [
            horizon(5, sample_size=8, signal_dates=2, effective_dates=2.0, positive_rate=0.6, median_return_pct=1.5),
            horizon(20, sample_size=3, signal_dates=1, effective_dates=1.0, pending_windows=5, positive_rate=0.6,
                    median_return_pct=1.5),
            horizon(60, pending_windows=8)], retired=True),
    ],
}


def write(root, rel, data):
    path = root / "static-data" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def dist(tmp_path):
    gen = "g/1/"
    us_rows = [
        row("AAA", minervini_score=90.0, se_pattern_primary="cup_with_handle"),
        row("BBB", minervini_score=72.0, company_name="<b>Evil & Co</b>"),
        row("CCC", minervini_score=72.0),  # ties BBB; site order breaks ties by symbol
        row("NEW", is_scannable=False, data_status="insufficient_history", rs_rating=None),
        row("BRK/B", minervini_score=10.0),
        # A short-history listing an IPO scan still marks scannable.
        row("IPO", data_status="insufficient_history", rs_rating=None),
    ]
    manifest = {
        "data_root": gen, "generated_at": "2026-10-12T05:25:03Z", "unavailable_markets": ["IN"],
        "markets": {
            "US": {"display_name": "United States", "as_of_date": "2026-10-09",
                   "assets": {"scoreboard": {"path": "markets/us/scoreboard.json"}}, "pages": {
                "scan": {"path": "markets/us/scan/manifest.json"}, "home": {"path": "markets/us/home.json"},
                "groups": {"path": "markets/us/groups.json"}, "breadth": {"path": "markets/us/breadth.json"}}},
            "AU": {"display_name": "Australia", "as_of_date": "2026-10-08", "pages": {
                "scan": {"path": "markets/au/scan/manifest.json"}, "home": {"path": "markets/au/home.json"}}},
        },
    }
    write(tmp_path, "manifest.json", manifest)
    write(tmp_path, gen + "markets/us/scan/manifest.json", {
        "as_of_date": "2026-10-09", "preset_screens": [MINERVINI, CUP], "initial_rows": us_rows[:2],
        "chunks": [{"path": "markets/us/scan/chunks/chunk-0001.json"}],
    })
    write(tmp_path, gen + "markets/us/scan/chunks/chunk-0001.json", {"rows": us_rows})
    write(tmp_path, gen + "markets/us/home.json", {"as_of_date": "2026-10-09", "market_health_exposure": {
        "exposure_score": 45.0, "stance": "Downtrend/Caution", "distribution_day_count": 8}})
    write(tmp_path, gen + "markets/us/groups.json", {"available": True, "rs_as_of_date": "2026-10-08", "payload": {
        "rankings": {"rankings": [
            {"rank": 1, "industry_group": "Semiconductors", "avg_rs_rating": 90.0, "rank_change_1w": 2,
             "rank_change_1m": -1, "rank_change_3m": 5, "num_stocks": 10, "top_symbol": "AAA"},
        ]}}})
    write(tmp_path, gen + "markets/us/breadth.json", {"available": True, "payload": {
        "current": {"date": "2026-10-07", "stocks_up_4pct": 182, "stocks_down_4pct": 116, "t2108_pct": 30.9,
                    "ratio_5day": 0.84, "ratio_10day": 0.85, "new_high_52week_count": 168, "new_low_52week_count": 239},
        "history_90d": [{"date": f"2026-10-0{d}", "stocks_up_4pct": 100 + d, "stocks_down_4pct": 50} for d in range(1, 10)],
    }})
    write(tmp_path, gen + "markets/us/scoreboard.json", SCOREBOARD)
    write(tmp_path, gen + "markets/au/scan/manifest.json", {
        "as_of_date": "2026-10-08", "preset_screens": [MINERVINI],
        "initial_rows": [row("BHP.AX", market="AU", currency="AUD")], "chunks": []})
    write(tmp_path, gen + "markets/au/home.json", {"as_of_date": "2026-10-08"})
    return tmp_path


def render(dist):
    assert load_generator().main(["--dist", str(dist), "--site-url", SITE]) == 0


def page(dist, rel):
    return (dist / rel / "index.html").read_text()


def test_stock_pages_cover_scannable_rows_only(dist):
    render(dist)
    assert (dist / "stocks/us/AAA/index.html").exists()
    assert (dist / "stocks/au/BHP.AX/index.html").exists()
    assert not (dist / "stocks/us/NEW").exists()
    assert not (dist / "stocks/us/IPO").exists()  # incomplete, no RS rating


def test_stock_page_content_is_escaped_canonical_and_linked(dist):
    render(dist)
    html = page(dist, "stocks/us/BBB")
    assert "&lt;b&gt;Evil &amp; Co&lt;/b&gt;" in html and "<b>Evil" not in html
    assert '<link rel="canonical" href="https://example.github.io/stock-screener/stocks/us/BBB/">' in html
    assert "RS rating 80" in html and "As of 2026-10-09" in html
    assert "/stock-screener/screens/us/minervini/" in html  # BBB passes Minervini
    assert "<svg" in html  # price sparkline
    assert "/stock-screener/#/scan?market=US" in html


def test_symbols_are_url_quoted_in_paths(dist):
    render(dist)
    assert (dist / "stocks/us/BRK_2FB/index.html").exists()
    assert "/stock-screener/stocks/us/BRK_2FB/" in page(dist, "stocks/us")
    # Dot-only symbols must not resolve to "." or ".." path segments.
    slug = load_generator().slug
    assert slug(".") == "_2E" and slug("..") == "_2E." and slug("BHP.AX") == "BHP.AX"


def test_screen_page_lists_matches_in_the_sites_order(dist):
    render(dist)
    html = page(dist, "screens/us/minervini")
    assert "3 stocks" in html
    positions = [html.index(f'/stock-screener/stocks/us/{s}/"') for s in ("AAA", "BBB", "CCC")]
    assert positions == sorted(positions)
    # The column the list is sorted by is shown, so the order is explainable.
    assert "<th class=n>Minervini score</th>" in html and "<td class=n>90</td>" in html
    assert '<link rel="icon" href="/stock-screener/fire.svg"' in html
    assert not (dist / "screens/us/canslim").exists()  # preset not in this publish


def test_market_without_breadth_has_no_breadth_page_or_link(dist):
    render(dist)
    assert (dist / "breadth/us/index.html").exists()
    assert not (dist / "breadth/au").exists()
    assert "/stock-screener/breadth/au/" not in page(dist, "markets/au")
    assert "182" in page(dist, "breadth/us")


def scoreboard_row(html, section_title, name):
    section = html.split(f"<h2>{section_title}</h2>", 1)[1].split("<h2>", 1)[0]
    return section.split(f"<td>{name}", 1)[1].split("</tr>", 1)[0]


def test_scoreboard_cells_say_too_few_pending_and_none_rather_than_a_number(dist):
    render(dist)
    html = page(dist, "scoreboard/us")
    live = "Recorded live"
    artifact = "Recovered from retained build artifacts, not recorded live"
    pre_fix = f"{artifact}, from before the Stage calculation fix (#671)"

    pending_row = scoreboard_row(html, live, "Minervini")
    assert "pending" in pending_row and "357 pending" in pending_row and "%" not in pending_row
    assert "No window has closed yet. Nothing here is a result." in html.split(f"<h2>{artifact}</h2>")[0]

    zero_match_row = scoreboard_row(html, live, "&lt;New&gt; Highs")
    assert "no picks" in zero_match_row and "pending" not in zero_match_row and "%" not in zero_match_row

    closed_row = scoreboard_row(html, artifact, "Minervini")
    assert "+0.9% median" in closed_row
    assert "1,298 closed · 6 dates, effective 5.4 · 40 pending · 100 overlapping" in closed_row

    thin_row = scoreboard_row(html, artifact, "CANSLIM")
    assert "too few" in thin_row and "12 closed · 2 dates, effective 1.9 · 2 pending" in thin_row and "%" not in thin_row

    retired_row = scoreboard_row(html, pre_fix, "Old Breakout")
    assert "Retired" in retired_row and "too few" in retired_row


def test_scoreboard_zero_sample_with_nothing_pending_is_not_called_pending(dist):
    render(dist)
    cells = scoreboard_row(page(dist, "scoreboard/us"), "Recorded live", "VCP").split("</td>")
    no_closed_cell = cells[1]

    assert no_closed_cell == ("<td>no closed windows<br><span class=muted>0 closed · 0 dates · 0 pending · "
                              "3 no price · 6 overlapping</span>")


def test_scoreboard_page_states_its_method_and_ledger(dist):
    render(dist)
    html = page(dist, "scoreboard/us")
    assert "Recording since 2026-10-05 · Ledger through 2026-11-04 · 2 missing sessions" in html
    assert "Entry is the first open after publication. Exit is the close of the Nth session." in html
    assert "Excess is against SPY." in html
    assert "Stocks that later delist drop out as no price, which can flatter results." in html
    assert "/stock-screener/#/scoreboard?market=US" in html
    assert "A pick is a stock newly entering the screen. Stocks on the list on the first recorded day, or on a day the screen&#x27;s definition or the Stage calculation changed, are not counted." in html
    assert "Effective dates count each date by how many picks it holds, so one busy day counts as about one." in html
    assert "Prices are as the provider reports them, so an unadjusted split can appear as a false move. Medians limit the effect." in html
    assert "Scored since" not in html


def test_scoreboard_meta_description_holds_counts_never_returns(dist):
    render(dist)
    html = page(dist, "scoreboard/us")
    description = html.split('<meta name="description" content="', 1)[1].split('"', 1)[0]
    assert description == ("United States preset screens tracked after publication: 5 screens, 1,720 picks, "
                           "3 windows closed and 1,717 pending at 20 sessions, as of 2026-11-04.")
    title = html.split("<title>", 1)[1].split("</title>", 1)[0]
    for text in (description, title):
        assert "%" not in text and "+" not in text


def test_scoreboard_picks_tracked_count_closed_plus_pending(dist):
    render(dist)
    assert "1,720 picks tracked" in page(dist, "markets/us")


def advertise_au_scoreboard(dist, content):
    manifest_path = dist / "static-data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["markets"]["AU"]["assets"] = {"scoreboard": {"path": "markets/au/scoreboard.json"}}
    manifest_path.write_text(json.dumps(manifest))
    write(dist, "g/1/markets/au/scoreboard.json", content)


def test_scoreboard_with_no_benchmark_prints_no_benchmark_sentence(dist):
    advertise_au_scoreboard(dist, {**SCOREBOARD, "market": "AU", "benchmark": None})
    render(dist)
    html = page(dist, "scoreboard/au")

    assert "<li>Returns are price only, in local currency, with no costs.</li>" in html
    assert "against" not in html and "None" not in html and "null" not in html


def test_scoreboard_shows_scored_since_when_it_differs_from_recording_since(dist):
    advertise_au_scoreboard(dist, {**SCOREBOARD, "recording_since": "2025-06-02", "scored_since": "2025-11-04"})
    render(dist)

    assert "Recording since 2025-06-02 · Scored since 2025-11-04 · Ledger through 2026-11-04" in page(dist, "scoreboard/au")


@pytest.mark.parametrize("content", [
    '{"cohorts": "not a list"}',
    "[]",
    "{}",
    "{not json",
    json.dumps({**SCOREBOARD, "cohorts": [{**SCOREBOARD["cohorts"][0], "horizons": []}]}),
], ids=["cohorts-string", "list", "empty-object", "unparseable", "no-horizons"])
def test_broken_scoreboard_skips_only_that_page_with_a_warning(dist, capsys, content):
    (dist / "static-data/g/1/markets/us/scoreboard.json").write_text(content)
    advertise_au_scoreboard(dist, {**SCOREBOARD, "market": "AU"})
    render(dist)

    assert not (dist / "scoreboard/us").exists()
    assert "/stock-screener/scoreboard/us/" not in page(dist, "markets/us")
    assert (dist / "stocks/us/AAA/index.html").exists()
    assert "1,720 picks tracked" in page(dist, "markets/au")
    assert "::warning::" in capsys.readouterr().out


def test_market_without_scoreboard_asset_has_no_page_or_link(dist):
    render(dist)
    assert (dist / "scoreboard/us/index.html").exists()
    assert "/stock-screener/scoreboard/us/" in page(dist, "markets/us")
    assert not (dist / "scoreboard/au").exists()
    assert "scoreboard" not in page(dist, "markets/au")


def test_advertised_scoreboard_with_no_file_has_no_page_or_link(dist):
    (dist / "static-data/g/1/markets/us/scoreboard.json").unlink()
    render(dist)
    assert not (dist / "scoreboard").exists()
    assert "/stock-screener/scoreboard/us/" not in page(dist, "markets/us")


def test_unavailable_markets_are_skipped(dist):
    render(dist)
    assert not (dist / "markets/in").exists()


def test_sitemap_lists_every_page_with_market_dates(dist):
    render(dist)
    tree = ET.parse(dist / "sitemap.xml")
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    entries = {u.find("s:loc", ns).text: u.find("s:lastmod", ns).text for u in tree.getroot()}
    assert entries[SITE + "stocks/us/AAA/"] == "2026-10-09"
    assert entries[SITE + "stocks/au/BHP.AX/"] == "2026-10-08"
    assert entries[SITE + "scoreboard/us/"] == "2026-11-04"
    assert SITE + "scoreboard/au/" not in entries
    assert SITE + "explore/" in entries and SITE + "groups/us/" in entries
    assert SITE + "stocks/us/NEW/" not in entries
    # Sections dated by their own snapshot when it trails the scan.
    assert entries[SITE + "breadth/us/"] == "2026-10-07"
    assert entries[SITE + "groups/us/"] == "2026-10-08"
    assert "(2026-10-07)" in page(dist, "breadth/us") and "(2026-10-08)" in page(dist, "groups/us")
    for url in entries:  # every listed URL was written
        assert (dist / url.removeprefix(SITE) / "index.html").exists()


def test_hub_and_stylesheet(dist):
    render(dist)
    hub = page(dist, "explore")
    assert "/stock-screener/markets/us/" in hub and "/stock-screener/markets/au/" in hub
    assert (dist / "seo.css").exists()


def test_flat_layout_without_data_generations(dist):
    # STATIC_DATA_GENERATIONS=false rollback: no data_root, pages live at the static-data root.
    data = dist / "static-data"
    shutil.copytree(data / "g/1", data, dirs_exist_ok=True)
    manifest = json.loads((data / "manifest.json").read_text())
    del manifest["data_root"]
    (data / "manifest.json").write_text(json.dumps(manifest))
    shutil.rmtree(data / "g")
    render(dist)
    assert (dist / "stocks/us/AAA/index.html").exists()
