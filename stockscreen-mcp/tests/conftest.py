"""A small fake publish with the same file layout and field names as the real static data."""

import copy
import io
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest

from stockscreen_mcp import data as data_module
from stockscreen_mcp import server
from stockscreen_mcp.history import History
from stockscreen_mcp.screens import evaluate_expression

BASE = "https://example.test/static-data/"
HISTORY = "https://example.test/releases/download/"
REPO = Path(__file__).resolve().parents[2]


def stock(symbol, **fields):
    row = {
        "symbol": symbol,
        "company_name": f"{symbol} Corp",
        "market": "US",
        "exchange": "XNAS",
        "current_price": 10.0,
        "price_sparkline_data": [1.0, 1.1],
        "rs_sparkline_data": [1.0, 0.9],
        "field_availability": {"short_interest": {"status": "missing"}},
        "scan_mode": "full",
        "data_status": "complete",
    }
    row.update(fields)
    return row


ROWS_INITIAL = [
    stock("AAA", rs_rating=95, stage=2, minervini_score=90, ma_alignment=True, composite_score=80,
          volume=2e8, ibd_industry_group="Semiconductors", ibd_group_rank=3, eps_growth_qq=40.0,
          se_pattern_primary="cup_handle", correction_survivor=True, pct_week=5.0),
    stock("BBB", exchange="XNYS", rs_rating=75, stage=2, minervini_score=72, ma_alignment=True, composite_score=60,
          volume=5e7, ibd_industry_group="Software", ibd_group_rank=40, eps_growth_qq=None),
    stock("CCC", rs_rating=99, stage=3, minervini_score=95, ma_alignment=True, composite_score=90,
          volume=3e8, ibd_industry_group="Semiconductors", ibd_group_rank=3),
]
ROWS_CHUNK = [
    # Duplicate of an initial row: the chunk copy is the full row.
    ROWS_INITIAL[0] | {"composite_score": 81},
    stock("DDD", rs_rating=None, stage=2, minervini_score=80, ma_alignment=True, composite_score=70),
    stock("EEE", rs_rating=88, stage=2, minervini_score=70, ma_alignment="true", composite_score=50),
]


def condition(field, kind, **spec):
    return {"field": field, "kind": kind, **spec}


def preset(preset_id, name, conditions, sort_by="composite_score"):
    return {
        "id": preset_id,
        "name": name,
        "short_name": name,
        "description": f"{name} description",
        "sort_by": sort_by,
        "sort_order": "desc",
        "tier": 1,
        "filter_schema_version": 2,
        "filters": {},
        "filter_expression": {
            "expression_version": 1,
            "group_join": "any",
            "groups": [],
            "required": {"id": "required", "name": "Always require", "enabled": True, "match": "all",
                         "conditions": conditions},
        },
    }


PRESETS = [
    preset("minervini", "Minervini Trend Template", [
        condition("minervini_score", "range", min=70, max=None),
        condition("rs_rating", "range", min=70, max=None),
        condition("ma_alignment", "boolean", value=True),
        condition("stage", "range", min=2, max=2),
    ], sort_by="minervini_score"),
    preset("correction_survivors", "Correction Survivors", [
        condition("correction_survivor", "boolean", value=True),
    ]),
    preset("se_cup_handle", "Cup with Handle", [
        condition("se_pattern_primary", "categorical", values=["cup_handle"], mode="include"),
    ]),
]


def market_entry(code, name, *, breadth=True, opportunity_state=True):
    slug = code.lower()
    pages = {
        "scan": {"path": f"markets/{slug}/scan/manifest.json"},
        "home": {"path": f"markets/{slug}/home.json"},
        "groups": {"path": f"markets/{slug}/groups.json"},
    }
    if breadth:
        pages["breadth"] = {"path": f"markets/{slug}/breadth.json"}
    return {
        "market": code,
        "display_name": name,
        "as_of_date": "2026-10-09",
        "features": {"scan": True, "breadth": breadth, "groups": True, "opportunity_state": opportunity_state},
        "publication": {"state": "current", "session_date": "2026-10-09", "late_after": "2099-01-01T00:00:00Z"},
        "pages": pages,
    }


def scan_manifest(code):
    return {
        "as_of_date": "2026-10-09",
        "rows_total": 5,
        "default_filters": {"minVolume": 100000000},
        "sort": {"field": "composite_score", "order": "desc"},
        "preset_screens": copy.deepcopy(PRESETS),
        "initial_rows": copy.deepcopy(ROWS_INITIAL),
        "chunks": [{"path": f"markets/{code.lower()}/scan/chunks/chunk-0001.json", "count": 3}],
    }


def group_row(rank, name, top_symbol, **extra):
    return {
        "rank": rank, "industry_group": name, "avg_rs_rating": 90.0 - rank, "median_rs_rating": 88.0,
        "num_stocks": 10, "num_stocks_rs_above_80": 5, "pct_rs_above_80": 50.0,
        "rank_change_1w": 2, "rank_change_1m": -1, "rank_change_3m": 5, "rank_change_6m": 9,
        "top_symbol": top_symbol, "top_symbol_name": f"{top_symbol} Corp", "top_rs_rating": 95.0,
        "rs_std_dev": 3.0, "date": "2026-10-09", **extra,
    }


def groups_file():
    semis_stocks = [
        {"symbol": "AAA", "company_name": "AAA Corp", "rs_rating": 95.0, "rs_rating_1m": 90.0, "stage": 2,
         "price": 50.0, "price_change_1d": 1.2, "price_sparkline_data": [1, 2], "rs_sparkline_data": [1, 2]},
        {"symbol": "CCC", "company_name": "CCC Corp", "rs_rating": 99.0, "rs_rating_1m": 97.0, "stage": 3,
         "price": 70.0, "price_change_1d": -0.5, "price_sparkline_data": [1, 2], "rs_sparkline_data": [1, 2]},
    ]
    return {
        "available": True,
        "market": "US",
        "rs_as_of_date": "2026-10-09",
        "payload": {
            "rankings": {"date": "2026-10-09", "total_groups": 2, "rankings": [
                group_row(1, "Semiconductors", "CCC"),
                group_row(2, "Software", "BBB"),
            ]},
            "movers": {"period": "1w", "gainers": [group_row(1, "Semiconductors", "CCC")],
                       "losers": [group_row(2, "Software", "BBB", rank_change_1w=-30)]},
            "group_details": {
                "Semiconductors": {
                    "industry_group": "Semiconductors", "current_rank": 1, "current_avg_rs": 89.0,
                    "current_median_rs": 88.0, "num_stocks": 2, "pct_rs_above_80": 100.0,
                    "rank_change_1w": 2, "rank_change_1m": -1, "rank_change_3m": 5, "rank_change_6m": 9,
                    "top_symbol": "CCC", "top_symbol_name": "CCC Corp",
                    "history": [{"date": f"2026-10-0{d}", "rank": d, "avg_rs_rating": 80.0 + d} for d in range(1, 10)],
                    "stocks": semis_stocks,
                },
            },
        },
    }


def home_file():
    return {
        "as_of_date": "2026-10-09",
        "market": "US",
        "market_health_exposure": {
            "date": "2026-10-09", "exposure_score": 45.0, "stance": "Downtrend/Caution", "trend": "bullish",
            "distribution_day_count": 8, "follow_through_day": False, "vix": 14.84, "benchmark_symbol": "SPY",
            "components": {"base": 50.0, "distribution_drag": -15.0},
            "history": [{"date": "2026-10-08", "exposure_score": 50.0}],
        },
        "key_markets": [{"symbol": "SPY", "display_name": "S&P 500 ETF", "latest_close": 778.57,
                         "change_1d": 0.6, "latest_date": "2026-10-09", "currency": "USD",
                         "history": [{"date": "2026-10-08", "close": 774.0}]}],
        "top_groups": [group_row(1, "Semiconductors", "CCC")],
        "scan_summary": {"rows_total": 5, "default_filtered_rows_total": 3, "top_results": [ROWS_INITIAL[0]]},
    }


def breadth_day(date, up4):
    return {
        "date": date, "stocks_up_4pct": up4, "stocks_down_4pct": 10, "ratio_5day": 1.2, "ratio_10day": 1.1,
        "advancing_count": 600, "declining_count": 300, "new_high_52week_count": 40, "new_low_52week_count": 5,
        "t2108_pct": 55.5, "stocks_up_25pct_quarter": 300, "stocks_down_25pct_quarter": 100,
        "stocks_up_25pct_month": 20, "stocks_down_25pct_month": 5, "stocks_up_50pct_month": 3,
        "stocks_down_50pct_month": 1, "stocks_up_13pct_34days": 150, "stocks_down_13pct_34days": 90,
        "eligibility_signature": "abc", "calculation_duration_seconds": 3.0,
    }


def breadth_file():
    history = [breadth_day(f"2026-10-0{d}", 20 + d) for d in range(9, 0, -1)]
    return {"available": True, "market": "US",
            "payload": {"current": history[0], "history_90d": history, "benchmark_symbol": "SPY"}}


def horizon_block(horizon_sessions, **fields):
    """A horizon block with zero counts and no closed window; pass fields to fill it in."""
    return {
        "horizon_sessions": horizon_sessions, "sample_size": 0, "signal_dates": 0, "effective_dates": 0.0,
        "pending_windows": 0, "oldest_open_sessions": None, "thin": True, "positive_rate": None,
        "beat_benchmark_rate": None, "median_return_pct": None, "median_excess_return_pct": None,
        "excess_sample_size": 0, "skipped_missing_history": 0, "suppressed_overlapping": 0, **fields,
    }


def scoreboard_cohort(preset_id, name, definition, origin, first_date, last_date, horizons, retired=False):
    return {
        "preset": preset_id, "name": name, "retired": retired, "definition": definition, "origin": origin,
        "first_date": first_date, "last_date": last_date, "events": 99, "signal_dates": 98, "horizons": horizons,
    }


def scoreboard_file():
    return {
        "schema_version": "static-scoreboard-v1",
        "market": "US",
        "as_of_date": "2026-10-09",
        "generated_at": "2026-10-10T02:00:00Z",
        "benchmark": "SPY",
        "recording_since": "2026-06-01",
        "scored_since": "2026-06-02",
        "ledger_through": "2026-10-09",
        "missing_sessions": 1,
        "cohorts": [
            scoreboard_cohort("minervini", "Minervini", "def-old", "artifact", "2026-06-01", "2026-09-30", [
                horizon_block(5, sample_size=120, signal_dates=30, effective_dates=24.5, thin=False,
                              positive_rate=0.6, beat_benchmark_rate=0.55, median_return_pct=2.1,
                              median_excess_return_pct=0.8, excess_sample_size=118, skipped_missing_history=2,
                              suppressed_overlapping=7),
                horizon_block(20, sample_size=80, signal_dates=22, effective_dates=18.2, thin=False,
                              pending_windows=40, oldest_open_sessions=19, positive_rate=0.5,
                              median_return_pct=-1.5),
                horizon_block(60, pending_windows=120, oldest_open_sessions=59),
            ]),
            scoreboard_cohort("minervini", "Minervini", "def-new", "live", "2026-10-01", "2026-10-09", [
                horizon_block(5, pending_windows=9, oldest_open_sessions=4),
                horizon_block(20, pending_windows=12, oldest_open_sessions=6),
                horizon_block(60, pending_windows=12, oldest_open_sessions=6),
            ]),
            scoreboard_cohort("se_cup_handle", "se_cup_handle", "def-cup", "artifact_pre_stage_fix", "2026-06-01", "2026-06-30", [
                horizon_block(5),
                horizon_block(20, sample_size=3, signal_dates=2, effective_dates=1.8, positive_rate=1.0,
                              beat_benchmark_rate=0.0, median_return_pct=4.0, median_excess_return_pct=-0.2,
                              excess_sample_size=3),
                horizon_block(60),
            ], retired=True),
        ],
    }


def publish(root, codes=("US", "AU", "HK")):
    names = {"US": "United States", "AU": "Australia", "HK": "Hong Kong"}
    files = {}
    markets = {}
    for code in codes:
        au = code == "AU"
        markets[code] = market_entry(code, names[code], breadth=not au, opportunity_state=not au)
        slug = code.lower()
        files[f"{root}markets/{slug}/scan/manifest.json"] = scan_manifest(code)
        files[f"{root}markets/{slug}/scan/chunks/chunk-0001.json"] = {"rows": copy.deepcopy(ROWS_CHUNK)}
        files[f"{root}markets/{slug}/home.json"] = home_file()
        files[f"{root}markets/{slug}/groups.json"] = groups_file()
        if not au:
            files[f"{root}markets/{slug}/breadth.json"] = breadth_file()
        if code == "US":
            markets[code]["assets"] = {"scoreboard": {"path": "markets/us/scoreboard.json"}}
            files[f"{root}markets/us/scoreboard.json"] = scoreboard_file()
    manifest = {
        "data_root": root,
        "generated_at": "2026-10-12T05:25:03Z",
        "default_market": "US",
        "supported_markets": list(codes),
        "unavailable_markets": ["IN"],
        "markets": markets,
    }
    return manifest, files


class FakePublish:
    """Serves a fake publish by URL and records every fetch."""

    def __init__(self):
        self.manifest, self.files = publish("g/1/")
        self.calls = []
        self.on_fetch = None  # optional side effect run before serving a URL

    def republish(self, root):
        """Simulate a new generation replacing the old one (#504: only one is deployed)."""
        self.manifest, self.files = publish(root)

    def __call__(self, url):
        self.calls.append(url)
        if self.on_fetch:
            self.on_fetch(url)
        path = url.removeprefix(BASE)
        if path == "manifest.json":
            return copy.deepcopy(self.manifest)
        if path not in self.files:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO())
        return copy.deepcopy(self.files[path])


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def fake_publish():
    return FakePublish()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def static_data(fake_publish, clock):
    return data_module.StaticData(base_url=BASE, fetch=fake_publish, clock=clock)


@pytest.fixture
def use_fake_data(monkeypatch, static_data):
    monkeypatch.setattr(server, "data", static_data)
    return static_data


@pytest.fixture
def anyio_backend():
    return "asyncio"


def archived(symbol, **fields):
    return stock(symbol, **{
        "rs_universe_size": 5000, "rs_formula_version": "balanced-horizon-percentile-v2", "stage": 2,
        "ma_alignment": True, **fields,
    })


# Minervini moves from SINCE_ROWS to UNTIL_ROWS: CCC and DDD enter, BBB leaves, NEW is new to the
# publish and ZZZ dropped from it.
SINCE_ROWS = [
    archived("AAA", rs_rating=95, minervini_score=90, composite_score=80, se_pattern_primary="cup_handle"),
    archived("BBB", rs_rating=75, minervini_score=72, composite_score=60),
    archived("CCC", rs_rating=99, minervini_score=95, stage=3),
    archived("DDD", rs_rating=65, minervini_score=80),
    archived("ZZZ", rs_rating=90, minervini_score=85),
]
UNTIL_ROWS = [
    archived("AAA", rs_rating=96, minervini_score=90, composite_score=80, se_pattern_primary="cup_handle",
             current_price=10.5),
    archived("BBB", rs_rating=60, minervini_score=72, composite_score=60),
    archived("CCC", rs_rating=99, minervini_score=95),
    archived("DDD", rs_rating=72, minervini_score=80),
    archived("NEW", rs_rating=91, minervini_score=88),
]


class FakeArchive:
    """Serves archive Releases under the names backend/app/scripts/market_history.py gives them, built with its
    own functions, and records every fetch."""

    def __init__(self, writer):
        self.writer = writer
        self.files = {}
        self.dates = {}
        self.errors = {}  # url -> HTTP status, or an exception, to fail with
        self.calls = []

    def url(self, market, asset):
        return f"{HISTORY}{self.writer.release_tag(market)}/{asset}"

    def add(self, market, as_of_date, rows, presets=PRESETS, origin="live"):
        scan = {"as_of_date": as_of_date, "preset_screens": copy.deepcopy(presets)}
        document = self.writer.build_document(market, scan, copy.deepcopy(rows), evaluate_expression)
        published_at = datetime.fromisoformat(f"{as_of_date}T22:00:00").replace(tzinfo=timezone.utc)
        self.writer.check_document(document, market, published_at)
        stamped = self.writer.stamped(document, published_at, self.writer.Origin(origin))
        self.files[self.url(market, self.writer.session_asset(market, as_of_date))] = self.writer.session_bytes(stamped)
        self.dates.setdefault(market, set()).add(as_of_date)
        index = self.writer.index_bytes(market, sorted(self.dates[market]))
        self.files[self.url(market, self.writer.INDEX_ASSET)] = index

    def __call__(self, url):
        self.calls.append(url)
        error = self.errors.get(url)
        if isinstance(error, BaseException):
            raise error
        if error is not None:
            raise urllib.error.HTTPError(url, error, "Error", {}, io.BytesIO())
        if url in self.files:
            return self.files[url]
        # GitHub's Release page, which exists once any asset does.
        tag = url.removeprefix("https://example.test/releases/tag/")
        if tag != url and any(name.startswith(f"{HISTORY}{tag}/") for name in self.files):
            return b"<!DOCTYPE html>"
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b"Not Found"))


@pytest.fixture(scope="session")
def writer():
    if not (REPO / "backend" / "app" / "scripts" / "market_history.py").exists():
        pytest.skip("not inside the repository checkout, so the archive writer is absent")
    sys.path.append(str(REPO / "backend"))
    from app.scripts import market_history

    return market_history


@pytest.fixture
def fake_archive(writer):
    archive = FakeArchive(writer)
    for as_of_date in ("2026-10-05", "2026-10-07", "2026-10-08"):
        archive.add("US", as_of_date, SINCE_ROWS)
    archive.add("US", "2026-10-09", UNTIL_ROWS)
    return archive


@pytest.fixture
def use_fake_history(monkeypatch, fake_archive, clock):
    history = History(base_url=HISTORY, fetch=fake_archive, clock=clock)
    monkeypatch.setattr(server, "history", history)
    return history
