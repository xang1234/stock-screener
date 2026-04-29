from __future__ import annotations

import json
from datetime import date
from pathlib import Path
import re
from unittest.mock import MagicMock

from celery.exceptions import Retry
import pytest
import requests

from app.services.official_market_universe_source_service import (
    OfficialMarketUniverseSourceService,
    OfficialMarketUniverseSnapshot,
    _FetchedSource,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "universe_sources"


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="cp950")


def test_parse_hk_rows_filters_equities_only():
    service = OfficialMarketUniverseSourceService()

    rows = service.parse_hk_rows(_fixture_bytes("hk_list_of_securities_fixture.xlsx"))

    assert [row["symbol"] for row in rows] == ["0001.HK", "0005.HK"]
    assert all(row["exchange"] == "XHKG" for row in rows)


def test_parse_jp_rows_filters_domestic_equities_and_sets_snapshot_date():
    service = OfficialMarketUniverseSourceService()

    snapshot = service.parse_jp_rows(_fixture_bytes("jp_data_j_fixture.xls"))

    assert snapshot.snapshot_as_of == "2026-03-31"
    assert snapshot.snapshot_id == "jpx-data-j-2026-03-31"
    assert [row["symbol"] for row in snapshot.rows] == ["1301.T", "7203.T"]
    assert [row["industry"] for row in snapshot.rows] == ["水産・農林業", "輸送用機器"]


def test_parse_nse_rows_filters_eq_series_and_canonicalizes_symbols():
    service = OfficialMarketUniverseSourceService()
    content = "\n".join(
        [
            "SYMBOL,NAME OF COMPANY,SERIES,DATE OF LISTING,PAID UP VALUE,MARKET LOT,ISIN NUMBER,FACE VALUE",
            "RELIANCE,Reliance Industries Limited,EQ,29-NOV-1995,10,1,INE002A01018,10",
            "RELBEES,Reliance ETF,BE,29-NOV-1995,10,1,INE999A01018,10",
            "TCS,Tata Consultancy Services Limited,EQ,25-AUG-2004,1,1,INE467B01029,1",
        ]
    ).encode("utf-8")

    rows = service.parse_nse_rows(content)

    assert [row["symbol"] for row in rows] == ["RELIANCE.NS", "TCS.NS"]
    assert all(row["exchange"] == "XNSE" for row in rows)
    assert rows[0]["isin"] == "INE002A01018"


def test_parse_bse_rows_requires_explicit_active_equity():
    service = OfficialMarketUniverseSourceService()
    content = json.dumps(
        [
            {
                "SCRIP_CD": "500325",
                "Issuer_Name": "Reliance Industries Ltd",
                "Status": "Active",
                "Segment": "Equity",
                "ISIN_NUMBER": "INE002A01018",
                "Mktcap": "1950000.0",
            },
            {
                "SCRIP_CD": "500002",
                "Issuer_Name": "ABB India Limited",
                "Status": "",
                "Segment": "Equity",
                "ISIN_NUMBER": "INE117A01022",
            },
            {
                "SCRIP_CD": "500112",
                "Issuer_Name": "State Trading Corporation",
                "Status": "Active",
                "Segment": "",
                "ISIN_NUMBER": "INE655A01013",
            },
        ]
    ).encode("utf-8")

    rows = service.parse_bse_rows(content)

    assert [row["symbol"] for row in rows] == ["500325.BO"]


def test_fetch_in_snapshot_prefers_nse_for_overlapping_isin(monkeypatch):
    service = OfficialMarketUniverseSourceService()

    monkeypatch.setattr(
        service,
        "fetch_nse_snapshot",
        lambda: OfficialMarketUniverseSnapshot(
            market="IN",
            source_name="nse_official",
            snapshot_id="nse-equity-2026-04-21",
            snapshot_as_of="2026-04-21",
            source_metadata={},
            rows=(
                {
                    "symbol": "RELIANCE.NS",
                    "name": "Reliance Industries Limited",
                    "exchange": "XNSE",
                    "sector": "",
                    "industry": "",
                    "market_cap": None,
                    "isin": "INE002A01018",
                },
                {
                    "symbol": "TCS.NS",
                    "name": "Tata Consultancy Services Limited",
                    "exchange": "XNSE",
                    "sector": "",
                    "industry": "",
                    "market_cap": None,
                    "isin": "INE467B01029",
                },
            ),
        ),
    )
    monkeypatch.setattr(
        service,
        "fetch_bse_snapshot",
        lambda: OfficialMarketUniverseSnapshot(
            market="IN",
            source_name="bse_official",
            snapshot_id="bse-equity-2026-04-21",
            snapshot_as_of="2026-04-21",
            source_metadata={},
            rows=(
                {
                    "symbol": "500325.BO",
                    "name": "Reliance Industries Ltd",
                    "exchange": "XBOM",
                    "sector": "",
                    "industry": "",
                    "market_cap": 1950000.0,
                    "isin": "INE002A01018",
                },
                {
                    "symbol": "506854.BO",
                    "name": "TANFAC Industries Ltd.",
                    "exchange": "XBOM",
                    "sector": "",
                    "industry": "",
                    "market_cap": 4816.33,
                    "isin": "INE639B01023",
                },
            ),
        ),
    )

    snapshot = service.fetch_in_snapshot()

    assert snapshot.market == "IN"
    assert [row["symbol"] for row in snapshot.rows] == ["RELIANCE.NS", "TCS.NS", "506854.BO"]
    assert snapshot.source_metadata["nse_count"] == 2
    assert snapshot.source_metadata["bse_count"] == 2
    assert snapshot.source_metadata["overlap_isin_count"] == 1


def test_fetch_kr_snapshot_uses_krx_provider_and_records_live_baseline_metadata(monkeypatch):
    class FakeKrxProvider:
        def listing_rows(self, *, boards, as_of=None):
            assert boards == ("KOSPI", "KOSDAQ")
            assert as_of is None
            kospi_rows = [
                {
                    "symbol": "005930.KS",
                    "local_code": "005930",
                    "name": "Samsung Electronics",
                    "exchange": "KOSPI",
                }
            ]
            kospi_rows.extend(
                {
                    "symbol": f"{index:06d}.KS",
                    "local_code": f"{index:06d}",
                    "name": f"KOSPI Co {index}",
                    "exchange": "KOSPI",
                }
                for index in range(1, 839)
            )
            kosdaq_rows = [
                {
                    "symbol": "091990.KQ",
                    "local_code": "091990",
                    "name": "Celltrion Healthcare",
                    "exchange": "KOSDAQ",
                }
            ]
            kosdaq_rows.extend(
                {
                    "symbol": f"{100000 + index:06d}.KQ",
                    "local_code": f"{100000 + index:06d}",
                    "name": f"KOSDAQ Co {index}",
                    "exchange": "KOSDAQ",
                }
                for index in range(1, 1819)
            )
            return [*kospi_rows, *kosdaq_rows]

    monkeypatch.setattr(
        OfficialMarketUniverseSourceService,
        "_seoul_today",
        staticmethod(lambda: date(2026, 4, 30)),
    )
    service = OfficialMarketUniverseSourceService(kr_provider=FakeKrxProvider())

    snapshot = service.fetch_kr_snapshot()

    assert snapshot.market == "KR"
    assert snapshot.source_name == "krx_official"
    assert snapshot.snapshot_id == "krx-listings-2026-04-30"
    assert snapshot.snapshot_as_of == "2026-04-30"
    assert len(snapshot.rows) == 2658
    rows_by_symbol = {row["symbol"]: row for row in snapshot.rows}
    assert rows_by_symbol["005930.KS"]["sector"] == "Information Technology"
    assert rows_by_symbol["091990.KQ"]["industry_group"] == "Biotechnology"
    assert snapshot.source_metadata["row_counts"] == {"kospi": 839, "kosdaq": 1819}
    assert snapshot.source_metadata["source_count"] == 2658
    assert snapshot.source_metadata["excluded_boards"] == ["KONEX"]
    assert snapshot.source_metadata["validated_krx_baseline"]["kospi"] == 839
    assert snapshot.source_metadata["validated_krx_baseline"]["kosdaq"] == 1819
    assert snapshot.source_metadata["validated_krx_baseline_tolerance"] == 0.02
    assert snapshot.source_metadata["validated_krx_baseline"]["source_url"].startswith(
        "https://global.krx.co.kr/"
    )


def test_enrich_kr_rows_returns_raw_rows_when_taxonomy_lazy_load_fails(monkeypatch):
    import app.services.market_taxonomy_service as taxonomy_module
    from app.services.market_taxonomy_service import TaxonomyLoadError

    rows = [
        {
            "symbol": "005930.KS",
            "local_code": "005930",
            "name": "Samsung Electronics",
            "exchange": "KOSPI",
            "sector": "",
        }
    ]

    class BrokenTaxonomy:
        @staticmethod
        def get(*args, **kwargs):
            raise TaxonomyLoadError("malformed korea-deep.csv")

    monkeypatch.setattr(taxonomy_module, "get_market_taxonomy_service", lambda: BrokenTaxonomy())

    enriched = OfficialMarketUniverseSourceService._enrich_kr_rows_with_taxonomy(rows)

    assert enriched == rows


def test_fetch_kr_snapshot_fails_when_live_counts_are_below_validated_baseline():
    class LowCountKrxProvider:
        def listing_rows(self, *, boards, as_of=None):
            return [
                {
                    "symbol": "005930.KS",
                    "local_code": "005930",
                    "name": "Samsung Electronics",
                    "exchange": "KOSPI",
                },
                {
                    "symbol": "091990.KQ",
                    "local_code": "091990",
                    "name": "Celltrion Healthcare",
                    "exchange": "KOSDAQ",
                },
            ]

    service = OfficialMarketUniverseSourceService(kr_provider=LowCountKrxProvider())

    with pytest.raises(ValueError, match="outside validated KRX baseline"):
        service.fetch_kr_snapshot()


def test_fetch_tw_snapshot_combines_twse_and_tpex_rows(monkeypatch):
    service = OfficialMarketUniverseSourceService()
    html_twse = _fixture_bytes("twse_stocks_fixture.html")
    html_tpex = _fixture_bytes("tpex_stocks_fixture.html")

    responses = {
        "https://isin.twse.com.tw/isin/e_C_public.jsp?strMode=2": _FetchedSource(
            url="https://isin.twse.com.tw/isin/e_C_public.jsp?strMode=2",
            content=html_twse,
            fetched_at="2026-04-16T01:00:00+00:00",
            last_modified="Wed, 16 Apr 2026 00:00:00 GMT",
            tls_verification_disabled=False,
        ),
        "https://isin.twse.com.tw/isin/e_C_public.jsp?strMode=4": _FetchedSource(
            url="https://isin.twse.com.tw/isin/e_C_public.jsp?strMode=4",
            content=html_tpex,
            fetched_at="2026-04-16T01:01:00+00:00",
            last_modified="Wed, 16 Apr 2026 00:00:00 GMT",
            tls_verification_disabled=False,
        ),
    }

    monkeypatch.setattr(service, "_http_get", lambda url, allow_insecure_fallback=False: responses[url])

    snapshot = service.fetch_tw_snapshot()

    assert snapshot.source_name == "tw_reference_bundle"
    assert snapshot.snapshot_id == "tw-reference-bundle-2026-04-16"
    assert snapshot.snapshot_as_of == "2026-04-16"
    assert [row["symbol"] for row in snapshot.rows] == ["1101.TW", "2330.TW", "6488.TWO"]
    assert snapshot.source_metadata["row_counts"] == {"twse": 2, "tpex": 1}


def test_fetch_tw_snapshot_fails_when_exchange_dates_do_not_match(monkeypatch):
    service = OfficialMarketUniverseSourceService()
    twse_html = _fixture_text("twse_stocks_fixture.html")
    tpex_html = _fixture_text("tpex_stocks_fixture.html").replace("2026/04/16", "2026/04/15")

    responses = iter(
        [
            _FetchedSource(
                url="twse",
                content=twse_html.encode("cp950"),
                fetched_at="2026-04-16T01:00:00+00:00",
                last_modified=None,
                tls_verification_disabled=False,
            ),
            _FetchedSource(
                url="tpex",
                content=tpex_html.encode("cp950"),
                fetched_at="2026-04-16T01:01:00+00:00",
                last_modified=None,
                tls_verification_disabled=False,
            ),
        ]
    )
    monkeypatch.setattr(service, "_http_get", lambda url, allow_insecure_fallback=False: next(responses))

    with pytest.raises(ValueError, match="date mismatch"):
        service.fetch_tw_snapshot()


def test_fetch_tw_snapshot_requires_explicit_opt_in_for_insecure_tls(monkeypatch):
    service = OfficialMarketUniverseSourceService()
    html_twse = _fixture_bytes("twse_stocks_fixture.html")
    html_tpex = _fixture_bytes("tpex_stocks_fixture.html")
    calls = []

    def fake_get(url, allow_insecure_fallback=False):
        calls.append((url, allow_insecure_fallback))
        if "strMode=2" in url:
            return _FetchedSource(
                url=url,
                content=html_twse,
                fetched_at="2026-04-16T01:00:00+00:00",
                last_modified=None,
                tls_verification_disabled=False,
            )
        return _FetchedSource(
            url=url,
            content=html_tpex,
            fetched_at="2026-04-16T01:01:00+00:00",
            last_modified=None,
            tls_verification_disabled=False,
        )

    monkeypatch.setattr(service, "_http_get", fake_get)
    monkeypatch.setattr(
        "app.services.official_market_universe_source_service.settings.tw_universe_allow_insecure_fallback",
        False,
    )

    service.fetch_tw_snapshot()

    assert calls == [
        ("https://isin.twse.com.tw/isin/e_C_public.jsp?strMode=2", False),
        ("https://isin.twse.com.tw/isin/e_C_public.jsp?strMode=4", False),
    ]


def test_http_get_retries_timeouts_before_succeeding(monkeypatch):
    service = OfficialMarketUniverseSourceService(timeout_seconds=7)
    calls: list[dict[str, object]] = []
    sleeps: list[float] = []

    class _Response:
        url = "https://example.com/final"
        content = b"ok"
        headers = {"Last-Modified": "Wed, 16 Apr 2026 00:00:00 GMT"}

        def raise_for_status(self) -> None:
            return None

    responses = iter(
        [
            requests.exceptions.ReadTimeout("slow upstream"),
            _Response(),
        ]
    )

    def fake_get(url, headers, timeout, verify=True):
        calls.append(
            {
                "url": url,
                "headers": headers,
                "timeout": timeout,
                "verify": verify,
            }
        )
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(
        "app.services.official_market_universe_source_service.requests.get",
        fake_get,
    )
    monkeypatch.setattr(
        "app.services.official_market_universe_source_service.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    fetched = service._http_get("https://example.com/source.csv")

    assert fetched.url == "https://example.com/final"
    assert fetched.content == b"ok"
    assert len(calls) == 2
    assert calls[0]["headers"]["User-Agent"] == service._user_agent
    assert calls[0]["timeout"] == 7
    assert calls[0]["verify"] is True
    assert calls[1]["verify"] is True
    assert sleeps == [1.0]


def test_http_get_raises_after_exhausting_timeout_retries(monkeypatch):
    service = OfficialMarketUniverseSourceService(timeout_seconds=7)
    calls: list[dict[str, object]] = []
    sleeps: list[float] = []

    def fake_get(url, headers, timeout, verify=True):
        calls.append(
            {
                "url": url,
                "headers": headers,
                "timeout": timeout,
                "verify": verify,
            }
        )
        raise requests.exceptions.ReadTimeout("still timing out")

    monkeypatch.setattr(
        "app.services.official_market_universe_source_service.requests.get",
        fake_get,
    )
    monkeypatch.setattr(
        "app.services.official_market_universe_source_service.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    with pytest.raises(requests.exceptions.ReadTimeout, match="still timing out"):
        service._http_get("https://example.com/source.csv")

    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


def test_fetch_nse_snapshot_uses_browser_like_headers(monkeypatch):
    service = OfficialMarketUniverseSourceService()
    calls: list[dict[str, object]] = []

    def fake_get(url, *, allow_insecure_fallback=False, extra_headers=None):
        calls.append(
            {
                "url": url,
                "allow_insecure_fallback": allow_insecure_fallback,
                "extra_headers": dict(extra_headers or {}),
            }
        )
        return _FetchedSource(
            url=url,
            content="\n".join(
                [
                    "SYMBOL,NAME OF COMPANY,SERIES,DATE OF LISTING,PAID UP VALUE,MARKET LOT,ISIN NUMBER,FACE VALUE",
                    "RELIANCE,Reliance Industries Limited,EQ,29-NOV-1995,10,1,INE002A01018,10",
                ]
            ).encode("utf-8"),
            fetched_at="2026-04-22T00:00:00+00:00",
            last_modified="Tue, 21 Apr 2026 21:35:02 GMT",
            tls_verification_disabled=False,
        )

    monkeypatch.setattr(service, "_http_get", fake_get)

    snapshot = service.fetch_nse_snapshot()

    assert snapshot.snapshot_id == "nse-equity-2026-04-21"
    assert len(calls) == 1
    assert calls[0]["url"] == "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    assert calls[0]["allow_insecure_fallback"] is False
    assert calls[0]["extra_headers"]["Accept"] == "text/csv,*/*"
    assert calls[0]["extra_headers"]["Accept-Language"] == "en-US,en;q=0.9"
    assert calls[0]["extra_headers"]["Referer"] == "https://www.nseindia.com/"
    assert re.match(r"^Mozilla/5\.0", calls[0]["extra_headers"]["User-Agent"])


def test_fetch_nse_snapshot_falls_back_to_archives_after_timeout(monkeypatch):
    service = OfficialMarketUniverseSourceService()
    calls: list[dict[str, object]] = []

    def fake_get(url, *, allow_insecure_fallback=False, extra_headers=None):
        calls.append(
            {
                "url": url,
                "allow_insecure_fallback": allow_insecure_fallback,
                "extra_headers": dict(extra_headers or {}),
            }
        )
        if "nsearchives" in url:
            raise requests.exceptions.ReadTimeout("nsearchives timed out")
        return _FetchedSource(
            url=url,
            content="\n".join(
                [
                    "SYMBOL,NAME OF COMPANY,SERIES,DATE OF LISTING,PAID UP VALUE,MARKET LOT,ISIN NUMBER,FACE VALUE",
                    "TCS,Tata Consultancy Services Limited,EQ,25-AUG-2004,1,1,INE467B01029,1",
                ]
            ).encode("utf-8"),
            fetched_at="2026-04-22T00:00:00+00:00",
            last_modified="Tue, 21 Apr 2026 21:35:02 GMT",
            tls_verification_disabled=False,
        )

    monkeypatch.setattr(service, "_http_get", fake_get)

    snapshot = service.fetch_nse_snapshot()

    assert [row["symbol"] for row in snapshot.rows] == ["TCS.NS"]
    assert [call["url"] for call in calls] == [
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
    ]
    assert snapshot.source_metadata["source_urls"] == [
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
    ]


def test_refresh_official_market_universe_ingests_snapshot(monkeypatch):
    import app.tasks.universe_tasks as module
    import app.wiring.bootstrap as bootstrap

    snapshot = OfficialMarketUniverseSnapshot(
        market="HK",
        source_name="hkex_official",
        snapshot_id="hkex-listofsecurities-2026-04-16",
        snapshot_as_of="2026-04-16",
        source_metadata={"filters": {"category_equals": "Equity"}},
        rows=(
            {
                "symbol": "0001.HK",
                "name": "CKH HOLDINGS",
                "exchange": "XHKG",
                "sector": "",
                "industry": "",
                "market_cap": None,
            },
        ),
    )

    fake_lock = MagicMock()
    fake_lock.acquire.return_value = (True, False)
    monkeypatch.setattr(bootstrap, "get_data_fetch_lock", lambda: fake_lock)
    monkeypatch.setattr("app.services.runtime_preferences_service.is_market_enabled_now", lambda _market: True)
    monkeypatch.setattr(module, "_count_active_universe", lambda market: 10)
    emitted = []
    monkeypatch.setattr(module, "_emit_universe_drift", lambda market, prior: emitted.append((market, prior)))
    no_github_sync = MagicMock()
    no_github_sync.sync_weekly_reference_from_github.return_value = {"status": "missing"}
    monkeypatch.setattr(module, "get_provider_snapshot_service", lambda: no_github_sync)
    monkeypatch.setattr(
        "app.services.official_market_universe_source_service.OfficialMarketUniverseSourceService.fetch_market_snapshot",
        lambda self, market: snapshot,
    )
    monkeypatch.setattr(
        module,
        "_ingest_official_snapshot",
        lambda snap: {"total": len(snap.rows), "added": 1, "updated": 0, "rejected": 0},
    )
    started = []
    completed = []
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: started.append(kwargs))
    monkeypatch.setattr(module, "mark_market_activity_completed", lambda *args, **kwargs: completed.append(kwargs))

    module.refresh_official_market_universe.request.id = "task-123"
    module.refresh_official_market_universe.request.retries = 0
    result = module.refresh_official_market_universe.run(market="HK")

    assert result["status"] == "success"
    assert result["snapshot_id"] == snapshot.snapshot_id
    assert result["total"] == 1
    assert emitted == [("HK", 10)]
    assert started[0]["stage_key"] == "universe"
    assert started[0]["lifecycle"] == "weekly_refresh"
    assert completed[0]["stage_key"] == "universe"
    fake_lock.release.assert_called_once_with("task-123", market="HK")


def test_refresh_official_market_universe_retries_when_market_lock_is_busy(monkeypatch):
    import app.tasks.universe_tasks as module
    import app.wiring.bootstrap as bootstrap

    fake_lock = MagicMock()
    fake_lock.acquire.return_value = (False, False)
    fake_lock.get_current_holder.return_value = {"task_name": "weekly_full_refresh", "task_id": "abc"}
    monkeypatch.setattr(bootstrap, "get_data_fetch_lock", lambda: fake_lock)
    monkeypatch.setattr("app.services.runtime_preferences_service.is_market_enabled_now", lambda _market: True)

    retry_calls = []

    def fake_retry(*args, **kwargs):
        retry_calls.append(kwargs)
        raise Retry("retry")

    monkeypatch.setattr(module.refresh_official_market_universe, "retry", fake_retry)
    module.refresh_official_market_universe.request.id = "task-123"
    module.refresh_official_market_universe.request.retries = 1

    with pytest.raises(Retry):
        module.refresh_official_market_universe.run(market="JP")

    assert retry_calls[0]["countdown"] == 600
    assert retry_calls[0]["max_retries"] == 12
    fake_lock.release.assert_not_called()


def test_refresh_official_market_universe_does_not_ingest_on_fetch_failure(monkeypatch):
    import app.tasks.universe_tasks as module
    import app.wiring.bootstrap as bootstrap

    fake_lock = MagicMock()
    fake_lock.acquire.return_value = (True, False)
    monkeypatch.setattr(bootstrap, "get_data_fetch_lock", lambda: fake_lock)
    monkeypatch.setattr("app.services.runtime_preferences_service.is_market_enabled_now", lambda _market: True)
    monkeypatch.setattr(module, "_count_active_universe", lambda market: 10)
    monkeypatch.setattr(module, "_emit_universe_drift", lambda market, prior: None)
    no_github_sync = MagicMock()
    no_github_sync.sync_weekly_reference_from_github.return_value = {"status": "missing"}
    monkeypatch.setattr(module, "get_provider_snapshot_service", lambda: no_github_sync)
    monkeypatch.setattr(
        "app.services.official_market_universe_source_service.OfficialMarketUniverseSourceService.fetch_market_snapshot",
        MagicMock(side_effect=RuntimeError("upstream unavailable")),
    )
    ingest_mock = MagicMock()
    monkeypatch.setattr(module, "_ingest_official_snapshot", ingest_mock)

    module.refresh_official_market_universe.request.id = "task-123"
    module.refresh_official_market_universe.request.retries = 0

    with pytest.raises(RuntimeError, match="upstream unavailable"):
        module.refresh_official_market_universe.run(market="TW")

    ingest_mock.assert_not_called()
    fake_lock.release.assert_called_once_with("task-123", market="TW")
