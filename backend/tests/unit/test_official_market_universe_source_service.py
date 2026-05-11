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
    provider_as_of_dates = []

    class FakeKrxProvider:
        def listing_rows(self, *, boards, as_of=None):
            assert boards == ("KOSPI", "KOSDAQ")
            provider_as_of_dates.append(as_of)
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

    market_calendar = MagicMock()
    market_calendar.last_completed_trading_day.return_value = date(2026, 4, 30)
    service = OfficialMarketUniverseSourceService(
        kr_provider=FakeKrxProvider(),
        market_calendar=market_calendar,
    )

    snapshot = service.fetch_kr_snapshot()

    assert provider_as_of_dates == [date(2026, 4, 30)]
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
    assert snapshot.source_metadata["listing_as_of"] == "2026-04-30"
    assert snapshot.source_metadata["excluded_boards"] == ["KONEX"]
    assert snapshot.source_metadata["validated_krx_baseline"]["kospi"] == 839
    assert snapshot.source_metadata["validated_krx_baseline"]["kosdaq"] == 1819
    assert snapshot.source_metadata["validated_krx_baseline_tolerance"] == 0.02
    assert snapshot.source_metadata["validated_krx_baseline"]["source_url"].startswith(
        "https://global.krx.co.kr/"
    )


def test_fetch_kr_snapshot_falls_back_to_previous_seoul_weekday_when_calendar_unavailable(monkeypatch):
    provider_as_of_dates = []

    class FakeKrxProvider:
        def listing_rows(self, *, boards, as_of=None):
            provider_as_of_dates.append(as_of)
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

    market_calendar = MagicMock()
    market_calendar.last_completed_trading_day.side_effect = RuntimeError("calendar unavailable")
    monkeypatch.setattr(
        OfficialMarketUniverseSourceService,
        "_seoul_today",
        staticmethod(lambda: date(2026, 5, 2)),
    )
    service = OfficialMarketUniverseSourceService(
        kr_provider=FakeKrxProvider(),
        market_calendar=market_calendar,
    )

    snapshot = service.fetch_kr_snapshot()

    assert provider_as_of_dates == [date(2026, 5, 1)]
    assert snapshot.snapshot_id == "krx-listings-2026-05-01"
    assert snapshot.snapshot_as_of == "2026-05-01"
    assert snapshot.source_metadata["listing_as_of"] == "2026-05-01"


def test_fetch_kr_snapshot_falls_back_to_current_listing_rows_when_historical_empty(monkeypatch):
    provider_calls = []

    class FakeKrxProvider:
        def listing_rows(self, *, boards, as_of=None):
            provider_calls.append((boards, as_of))
            if as_of is not None:
                return []
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

    market_calendar = MagicMock()
    market_calendar.last_completed_trading_day.return_value = date(2026, 4, 30)
    monkeypatch.setattr(
        OfficialMarketUniverseSourceService,
        "_seoul_today",
        staticmethod(lambda: date(2026, 5, 2)),
    )
    service = OfficialMarketUniverseSourceService(
        kr_provider=FakeKrxProvider(),
        market_calendar=market_calendar,
    )

    snapshot = service.fetch_kr_snapshot()

    assert provider_calls == [
        (("KOSPI", "KOSDAQ"), date(2026, 4, 30)),
        (("KOSPI", "KOSDAQ"), None),
    ]
    assert [row["symbol"] for row in snapshot.rows] == ["005930.KS", "091990.KQ"]
    assert snapshot.snapshot_id == "krx-listings-2026-04-30"
    assert snapshot.snapshot_as_of == "2026-04-30"
    assert snapshot.source_metadata["requested_listing_as_of"] == "2026-04-30"
    assert snapshot.source_metadata["listing_as_of"] == "2026-04-30"
    assert snapshot.source_metadata["krx_listing_mode"] == "current_listing_fallback"
    assert snapshot.source_metadata["historical_listing_empty"] is True


def test_fetch_kr_snapshot_raises_when_historical_and_current_listing_rows_are_empty():
    class EmptyKrxProvider:
        def listing_rows(self, *, boards, as_of=None):
            return []

    market_calendar = MagicMock()
    market_calendar.last_completed_trading_day.return_value = date(2026, 4, 30)
    service = OfficialMarketUniverseSourceService(
        kr_provider=EmptyKrxProvider(),
        market_calendar=market_calendar,
    )

    with pytest.raises(ValueError, match="KR official universe fetch returned no KOSPI/KOSDAQ rows"):
        service.fetch_kr_snapshot()


def test_fetch_kr_snapshot_falls_back_when_historical_rows_have_no_supported_boards(monkeypatch):
    provider_calls = []

    class FakeKrxProvider:
        def listing_rows(self, *, boards, as_of=None):
            provider_calls.append((boards, as_of))
            if as_of is not None:
                return [
                    {
                        "symbol": "000001.KN",
                        "local_code": "000001",
                        "name": "Konex Only",
                        "exchange": "KONEX",
                    }
                ]
            return [
                {
                    "symbol": "005930.KS",
                    "local_code": "005930",
                    "name": "Samsung Electronics",
                    "exchange": "KOSPI",
                }
            ]

    market_calendar = MagicMock()
    market_calendar.last_completed_trading_day.return_value = date(2026, 4, 30)
    monkeypatch.setattr(
        OfficialMarketUniverseSourceService,
        "_seoul_today",
        staticmethod(lambda: date(2026, 5, 1)),
    )
    service = OfficialMarketUniverseSourceService(
        kr_provider=FakeKrxProvider(),
        market_calendar=market_calendar,
    )

    snapshot = service.fetch_kr_snapshot()

    assert provider_calls == [
        (("KOSPI", "KOSDAQ"), date(2026, 4, 30)),
        (("KOSPI", "KOSDAQ"), None),
    ]
    assert [row["symbol"] for row in snapshot.rows] == ["005930.KS"]
    assert snapshot.source_metadata["row_counts"] == {"kospi": 1, "kosdaq": 0}
    assert snapshot.source_metadata["krx_listing_mode"] == "current_listing_fallback"


def test_fetch_kr_snapshot_filters_unsupported_boards_from_mixed_rows():
    class MixedKrxProvider:
        def listing_rows(self, *, boards, as_of=None):
            return [
                {
                    "symbol": "005930.KS",
                    "local_code": "005930",
                    "name": "Samsung Electronics",
                    "exchange": "KOSPI",
                },
                {
                    "symbol": "000001.KN",
                    "local_code": "000001",
                    "name": "Konex Only",
                    "exchange": "KONEX",
                },
            ]

    market_calendar = MagicMock()
    market_calendar.last_completed_trading_day.return_value = date(2026, 4, 30)
    service = OfficialMarketUniverseSourceService(
        kr_provider=MixedKrxProvider(),
        market_calendar=market_calendar,
    )

    snapshot = service.fetch_kr_snapshot()

    assert [row["symbol"] for row in snapshot.rows] == ["005930.KS"]
    assert snapshot.source_metadata["row_counts"] == {"kospi": 1, "kosdaq": 0}
    assert snapshot.source_metadata["source_count"] == 1


def test_fetch_kr_snapshot_raises_when_fallback_has_no_supported_boards():
    class UnsupportedKrxProvider:
        def listing_rows(self, *, boards, as_of=None):
            return [
                {
                    "symbol": "000001.KN",
                    "local_code": "000001",
                    "name": "Konex Only",
                    "exchange": "KONEX",
                }
            ]

    market_calendar = MagicMock()
    market_calendar.last_completed_trading_day.return_value = date(2026, 4, 30)
    service = OfficialMarketUniverseSourceService(
        kr_provider=UnsupportedKrxProvider(),
        market_calendar=market_calendar,
    )

    with pytest.raises(ValueError, match="KR official universe fetch returned no KOSPI/KOSDAQ rows"):
        service.fetch_kr_snapshot()


@pytest.mark.parametrize(
    ("today", "expected_previous"),
    [
        (date(2026, 5, 4), date(2026, 5, 1)),
        (date(2026, 5, 5), date(2026, 5, 4)),
        (date(2026, 5, 9), date(2026, 5, 8)),
        (date(2026, 5, 10), date(2026, 5, 8)),
    ],
)
def test_previous_seoul_business_day_returns_last_completed_weekday(today, expected_previous):
    assert OfficialMarketUniverseSourceService._previous_seoul_business_day(today) == expected_previous


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


def test_fetch_kr_snapshot_records_baseline_drift_without_rejecting_source():
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

    snapshot = service.fetch_kr_snapshot()

    assert [row["symbol"] for row in snapshot.rows] == ["005930.KS", "091990.KQ"]
    assert snapshot.source_metadata["krx_baseline_status"] == "outside_static_baseline"
    assert snapshot.source_metadata["validated_krx_baseline_breaches"] == [
        {"board": "kospi", "actual": 1, "expected": 839, "min": 823, "max": 855},
        {"board": "kosdaq", "actual": 1, "expected": 1819, "min": 1783, "max": 1855},
    ]


def test_fetch_cn_snapshot_uses_akshare_provider_and_records_validated_baseline():
    class FakeCnProvider:
        def listing_rows(self, *, as_of=None):
            return [
                {
                    "symbol": "600519.SS",
                    "local_code": "600519",
                    "name": "Kweichow Moutai",
                    "exchange": "SSE",
                    "board": "SSE_MAIN",
                    "industry": "Beverage Manufacturing",
                },
                {
                    "symbol": "000001.SZ",
                    "local_code": "000001",
                    "name": "Ping An Bank",
                    "exchange": "SZSE",
                    "board": "SZSE_MAIN",
                    "industry": "Banking",
                },
                {
                    "symbol": "920118.BJ",
                    "local_code": "920118",
                    "name": "Taihu Snow",
                    "exchange": "BJSE",
                    "board": "BSE",
                    "industry": "Textile Manufacturing",
                },
            ]

    service = OfficialMarketUniverseSourceService(cn_provider=FakeCnProvider())

    snapshot = service.fetch_cn_snapshot()

    assert snapshot.market == "CN"
    assert snapshot.source_name == "cn_akshare_eastmoney"
    assert snapshot.snapshot_id.startswith("cn-a-share-")
    assert [row["symbol"] for row in snapshot.rows] == ["600519.SS", "000001.SZ", "920118.BJ"]
    assert snapshot.source_metadata["row_counts"] == {"sse": 1, "szse": 1, "bse": 1}
    assert snapshot.source_metadata["source_count"] == 3
    assert snapshot.source_metadata["validated_cn_baseline"]["total"] == 5492
    assert snapshot.source_metadata["validated_cn_baseline_tolerance"] == 0.02
    assert snapshot.source_metadata["cn_baseline_status"] == "outside_static_baseline"


def test_get_cn_provider_uses_cn_listing_default_when_timeout_not_overridden(monkeypatch):
    from app.config import settings as settings_module

    monkeypatch.setattr(settings_module, "universe_source_timeout_seconds", 60)
    monkeypatch.setattr(settings_module, "universe_source_timeout_seconds_cn", 300)

    service = OfficialMarketUniverseSourceService()
    cn_provider = service._get_cn_provider()

    assert cn_provider._timeout_seconds == 60
    assert cn_provider._listing_timeout_seconds == 300


def test_get_cn_provider_propagates_explicit_service_level_timeout(monkeypatch):
    from app.config import settings as settings_module

    monkeypatch.setattr(settings_module, "universe_source_timeout_seconds", 60)
    monkeypatch.setattr(settings_module, "universe_source_timeout_seconds_cn", 300)

    service = OfficialMarketUniverseSourceService(timeout_seconds=2)
    cn_provider = service._get_cn_provider()

    assert cn_provider._timeout_seconds == 2
    assert cn_provider._listing_timeout_seconds == 2


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


# ---------------------------------------------------------------------------
# CA / TMX
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instrument_type,expected_excluded",
    [
        ("Common Shares", False),
        ("Class A Subordinate Voting Shares", False),
        ("Trust Units", False),
        ("Limited Partnership Units", False),
        ("Preferred Shares", False),
        ("ETF", True),
        ("Exchange-Traded Fund", True),
        ("Closed-End Fund", True),
        ("Mutual Fund", True),
        ("Notes 6.5%", True),
        ("Convertible Debentures", True),
        ("Subscription Receipt", True),
        ("Warrants", True),
        ("Rights", True),
        # False-positive guards: substrings that include excluded tokens but
        # are NOT excluded instrument types.
        ("Footnotes Reference", False),
        ("Bondsmith Common Shares", False),
        ("Notable Common Shares", False),
    ],
)
def test_ca_instrument_exclusion_matches_whole_words_only(instrument_type, expected_excluded):
    service = OfficialMarketUniverseSourceService()

    assert service._ca_is_excluded_instrument(instrument_type.lower()) is expected_excluded


def test_parse_ca_rows_filters_excluded_instruments_and_normalizes_symbols():
    service = OfficialMarketUniverseSourceService()
    payload = json.dumps(
        {
            "results": [
                {"symbol": "RY", "name": "Royal Bank of Canada", "instrumentType": "Common Shares"},
                {"symbol": "BIP.UN", "name": "Brookfield Infra", "instrumentType": "Trust Units"},
                {"symbol": "XIU", "name": "iShares S&P/TSX 60", "instrumentType": "ETF"},
                {"symbol": "XYZ", "name": "Some Note Issuer", "instrumentType": "Notes 6.5%"},
                {"symbol": "", "name": "No symbol", "instrumentType": "Common Shares"},
                "not a dict",
            ]
        }
    ).encode("utf-8")

    rows = service.parse_ca_rows(payload, exchange="TSX")

    assert [row["symbol"] for row in rows] == ["RY", "BIP.UN"]
    assert all(row["exchange"] == "TSX" for row in rows)


def test_parse_ca_rows_handles_alternate_payload_shapes():
    service = OfficialMarketUniverseSourceService()

    list_payload = json.dumps(
        [{"ticker": "SHOP", "issuerName": "Shopify", "type": "Common Shares"}]
    ).encode("utf-8")
    rows = service.parse_ca_rows(list_payload, exchange="TSX")
    assert rows[0]["symbol"] == "SHOP"
    assert rows[0]["name"] == "Shopify"

    companies_payload = json.dumps(
        {"companies": [{"rootTicker": "NVA", "companyName": "Nova", "securityType": "Common Shares"}]}
    ).encode("utf-8")
    rows = service.parse_ca_rows(companies_payload, exchange="TSXV")
    assert rows[0]["symbol"] == "NVA"
    assert rows[0]["exchange"] == "TSXV"


def test_parse_ca_rows_rejects_invalid_payload():
    service = OfficialMarketUniverseSourceService()

    with pytest.raises(ValueError, match="Invalid TMX directory payload"):
        service.parse_ca_rows(b"not json{", exchange="TSX")

    with pytest.raises(ValueError, match="Unsupported CA exchange"):
        service.parse_ca_rows(b"[]", exchange="NYSE")


def _ca_letter_payload(symbols_by_letter: dict[str, list[dict]]) -> dict[str, _FetchedSource]:
    """Build a {url: _FetchedSource} map for the TMX letter-bucket fetcher."""
    base_tsx = "https://www.tsx.com/json/company-directory/search/tsx/"
    base_tsxv = "https://www.tsx.com/json/company-directory/search/tsxv/"
    out: dict[str, _FetchedSource] = {}
    for letter in (chr(c) for c in range(ord("A"), ord("Z") + 1)):
        for board, base in (("TSX", base_tsx), ("TSXV", base_tsxv)):
            results = symbols_by_letter.get(f"{board}:{letter}", [])
            payload = json.dumps({"results": results}).encode("utf-8")
            out[f"{base}{letter}"] = _FetchedSource(
                url=f"{base}{letter}",
                content=payload,
                fetched_at=f"2026-05-09T0{0 if letter < 'M' else 1}:00:00+00:00",
                last_modified="Fri, 09 May 2026 12:00:00 GMT",
                tls_verification_disabled=False,
            )
    return out


def test_fetch_ca_snapshot_iterates_letter_buckets_and_dedupes(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(
        app_settings,
        "ca_universe_source_tsx_url",
        "https://www.tsx.com/json/company-directory/search/tsx/{initial}",
    )
    monkeypatch.setattr(
        app_settings,
        "ca_universe_source_tsxv_url",
        "https://www.tsx.com/json/company-directory/search/tsxv/{initial}",
    )

    fixtures = _ca_letter_payload(
        {
            "TSX:R": [
                {"symbol": "RY", "name": "Royal Bank", "instrumentType": "Common Shares"},
                {"symbol": "RCI.B", "name": "Rogers Class B", "instrumentType": "Class B Shares"},
                # Duplicate within the same bucket (different fields) — first wins.
                {"symbol": "RY", "name": "Royal Bank duplicate", "instrumentType": "Common Shares"},
            ],
            "TSX:S": [
                {"symbol": "SHOP", "name": "Shopify", "instrumentType": "Common Shares"},
                # ETF — should be filtered out by parse_ca_rows.
                {"symbol": "XIU", "name": "iShares S&P/TSX 60", "instrumentType": "ETF"},
            ],
            "TSXV:N": [
                {"symbol": "NVA", "name": "Nova Mining", "instrumentType": "Common Shares"},
            ],
        }
    )

    service = OfficialMarketUniverseSourceService()
    monkeypatch.setattr(
        service,
        "_http_get",
        lambda url, allow_insecure_fallback=False: fixtures[url],
    )

    snapshot = service.fetch_ca_snapshot()

    symbols = [row["symbol"] for row in snapshot.rows]
    assert symbols == ["RCI.B", "RY", "SHOP", "NVA"]
    assert snapshot.source_metadata["row_counts"] == {"tsx": 3, "tsxv": 1}
    assert snapshot.source_metadata["fetch_mode"] == {"tsx": "letter_buckets", "tsxv": "letter_buckets"}
    assert snapshot.source_metadata["fetch_attempts"]["tsx"] == [
        chr(c) for c in range(ord("A"), ord("Z") + 1)
    ]
    assert snapshot.source_metadata["fetch_errors"]["tsx"] == {}


def test_fetch_ca_snapshot_tolerates_per_letter_failures(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(
        app_settings,
        "ca_universe_source_tsx_url",
        "https://www.tsx.com/json/company-directory/search/tsx/{initial}",
    )
    monkeypatch.setattr(
        app_settings,
        "ca_universe_source_tsxv_url",
        "https://www.tsx.com/json/company-directory/search/tsxv/{initial}",
    )

    fixtures = _ca_letter_payload(
        {
            "TSX:R": [
                {"symbol": "RY", "name": "Royal Bank", "instrumentType": "Common Shares"},
            ],
            "TSXV:N": [
                {"symbol": "NVA", "name": "Nova Mining", "instrumentType": "Common Shares"},
            ],
        }
    )

    failing_letter_url = "https://www.tsx.com/json/company-directory/search/tsx/Q"

    def fake_http_get(url, allow_insecure_fallback=False):
        if url == failing_letter_url:
            raise requests.exceptions.ConnectionError("synthetic failure")
        return fixtures[url]

    service = OfficialMarketUniverseSourceService()
    monkeypatch.setattr(service, "_http_get", fake_http_get)

    snapshot = service.fetch_ca_snapshot()

    assert "Q" in snapshot.source_metadata["fetch_errors"]["tsx"]
    assert "RY" in [row["symbol"] for row in snapshot.rows]
    assert "NVA" in [row["symbol"] for row in snapshot.rows]


def test_fetch_ca_snapshot_raises_when_no_rows_returned_anywhere(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(
        app_settings,
        "ca_universe_source_tsx_url",
        "https://www.tsx.com/json/company-directory/search/tsx/{initial}",
    )
    monkeypatch.setattr(
        app_settings,
        "ca_universe_source_tsxv_url",
        "https://www.tsx.com/json/company-directory/search/tsxv/{initial}",
    )

    empty_payload = json.dumps({"results": []}).encode("utf-8")

    def fake_http_get(url, allow_insecure_fallback=False):
        return _FetchedSource(
            url=url,
            content=empty_payload,
            fetched_at="2026-05-09T00:00:00+00:00",
            last_modified=None,
            tls_verification_disabled=False,
        )

    service = OfficialMarketUniverseSourceService()
    monkeypatch.setattr(service, "_http_get", fake_http_get)

    with pytest.raises(ValueError, match="no equity rows for TSX, TSXV"):
        service.fetch_ca_snapshot()


def test_fetch_ca_snapshot_raises_when_one_board_returns_no_rows(monkeypatch):
    """A full TSX or TSXV outage must surface as a hard error rather than
    publishing a half-empty CA snapshot."""
    from app.config import settings as app_settings

    monkeypatch.setattr(
        app_settings,
        "ca_universe_source_tsx_url",
        "https://www.tsx.com/json/company-directory/search/tsx/{initial}",
    )
    monkeypatch.setattr(
        app_settings,
        "ca_universe_source_tsxv_url",
        "https://www.tsx.com/json/company-directory/search/tsxv/{initial}",
    )

    fixtures = _ca_letter_payload(
        {
            # TSX has rows on letter R, TSXV has rows on no letter.
            "TSX:R": [
                {"symbol": "RY", "name": "Royal Bank", "instrumentType": "Common Shares"},
            ],
        }
    )

    service = OfficialMarketUniverseSourceService()
    monkeypatch.setattr(
        service,
        "_http_get",
        lambda url, allow_insecure_fallback=False: fixtures[url],
    )

    with pytest.raises(ValueError, match="no equity rows for TSXV"):
        service.fetch_ca_snapshot()


def test_fetch_ca_snapshot_uses_single_url_when_template_lacks_placeholder(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(
        app_settings,
        "ca_universe_source_tsx_url",
        "https://example.invalid/tsx-mirror.json",
    )
    monkeypatch.setattr(
        app_settings,
        "ca_universe_source_tsxv_url",
        "https://example.invalid/tsxv-mirror.json",
    )

    payloads = {
        "https://example.invalid/tsx-mirror.json": _FetchedSource(
            url="tsx",
            content=json.dumps(
                {"results": [{"symbol": "RY", "name": "Royal Bank", "instrumentType": "Common Shares"}]}
            ).encode("utf-8"),
            fetched_at="2026-05-09T00:00:00+00:00",
            last_modified="Fri, 09 May 2026 00:00:00 GMT",
            tls_verification_disabled=False,
        ),
        "https://example.invalid/tsxv-mirror.json": _FetchedSource(
            url="tsxv",
            content=json.dumps(
                {"results": [{"symbol": "NVA", "name": "Nova", "instrumentType": "Common Shares"}]}
            ).encode("utf-8"),
            fetched_at="2026-05-09T00:00:00+00:00",
            last_modified="Fri, 09 May 2026 00:00:00 GMT",
            tls_verification_disabled=False,
        ),
    }

    service = OfficialMarketUniverseSourceService()
    monkeypatch.setattr(
        service,
        "_http_get",
        lambda url, allow_insecure_fallback=False: payloads[url],
    )

    snapshot = service.fetch_ca_snapshot()

    assert snapshot.source_metadata["fetch_mode"] == {
        "tsx": "single_url",
        "tsxv": "single_url",
    }
    assert snapshot.source_metadata["fetch_attempts"] == {
        "tsx": ["single"],
        "tsxv": ["single"],
    }
    assert {row["symbol"] for row in snapshot.rows} == {"RY", "NVA"}


# ---------------------------------------------------------------------------
# DE (Deutsche Boerse / Xetra) snapshot tests
# ---------------------------------------------------------------------------


_DE_BASE_URL = "https://api.boerse-frankfurt.de/v1/search/equity_search"


def _de_page_payload(
    rows: list[dict[str, object]], *, total: int | None = None
) -> bytes:
    """Build a Boerse Frankfurt search-API page response."""
    body: dict[str, object] = {"data": rows}
    if total is not None:
        body["totalCount"] = total
    return json.dumps(body).encode("utf-8")


def _de_fetched_source(content: bytes, *, fetched_at: str = "2026-05-09T01:00:00+00:00") -> _FetchedSource:
    return _FetchedSource(
        url=_DE_BASE_URL,
        content=content,
        fetched_at=fetched_at,
        last_modified="Sat, 09 May 2026 00:00:00 GMT",
        tls_verification_disabled=False,
    )


def _write_de_isin_map_csv(tmp_path) -> str:
    """Write a CSV that the fetcher can use as both fallback and ISIN→ticker map."""
    csv_path = tmp_path / "de_dax40_constituents.csv"
    csv_path.write_text(
        "symbol,name,exchange,index,isin\n"
        "SAP.DE,SAP SE,XETR,DAX,DE0007164600\n"
        "ALV.DE,Allianz SE,XETR,DAX,DE0008404005\n",
        encoding="utf-8",
    )
    return str(csv_path)


def test_fetch_de_snapshot_paginates_xetra_equities(monkeypatch, tmp_path):
    from app.config import settings as app_settings
    from app.services import official_market_universe_source_service as svc_module

    monkeypatch.setattr(app_settings, "de_universe_source_url", _DE_BASE_URL)
    monkeypatch.setattr(
        app_settings, "de_universe_fallback_csv_path", _write_de_isin_map_csv(tmp_path)
    )
    # Shrink the page size so the test can exercise multi-page pagination
    # without constructing 100-row fixtures. The pagination loop stops on
    # partial-page responses, so each non-final page must be exactly the
    # configured size to trigger another fetch.
    monkeypatch.setattr(svc_module, "_BOERSE_FRANKFURT_PAGE_SIZE", 2)

    page_one = _de_page_payload(
        [
            # Resolved via explicit tickerSymbol field.
            {
                "isin": "DE0007164600",
                "wkn": "716460",
                "tickerSymbol": "SAP",
                "name": {"originalValue": "SAP SE"},
                "isXetraTradable": True,
            },
            # Resolved via the ISIN→ticker map (no explicit ticker field).
            {
                "isin": "DE0008404005",
                "wkn": "840400",
                "name": "Allianz SE",
                "isXetraTradable": True,
            },
        ],
        total=3,
    )
    page_two = _de_page_payload(
        [
            # Resolved via ``ticker`` (alternate explicit field).
            {
                "isin": "DE000A1EWWW0",
                "wkn": "A1EWWW",
                "ticker": "ADS",
                "name": "Adidas AG",
                "xetraTradable": True,
            },
        ],
        total=3,
    )

    pages = iter([page_one, page_two])

    def fake_http_get(url, allow_insecure_fallback=False, extra_headers=None):
        assert extra_headers is not None, "DE fetch must send signed headers"
        assert "X-Security" in extra_headers
        assert "Client-Date" in extra_headers
        return _de_fetched_source(next(pages))

    service = OfficialMarketUniverseSourceService()
    monkeypatch.setattr(service, "_http_get", fake_http_get)

    snapshot = service.fetch_de_snapshot()

    assert snapshot.market == "DE"
    assert snapshot.source_name == "dbg_official"
    assert snapshot.source_metadata["fetch_mode"] == "live_http"
    symbols = [row["symbol"] for row in snapshot.rows]
    assert symbols == sorted(["SAP.DE", "ALV.DE", "ADS.DE"])
    assert snapshot.source_metadata["row_counts"]["xetra"] == 3
    assert snapshot.source_metadata["row_counts"]["frankfurt"] == 0
    assert snapshot.source_metadata["row_counts"]["total"] == 3
    assert snapshot.source_metadata["page_attempts"] == [0, 2]
    assert snapshot.source_metadata["ticker_resolution"] == {
        "explicit_field": 2,
        "isin_map": 1,
        "unresolved": 0,
    }


def test_fetch_de_snapshot_drops_rows_without_resolvable_ticker(monkeypatch, tmp_path):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "de_universe_source_url", _DE_BASE_URL)
    monkeypatch.setattr(
        app_settings, "de_universe_fallback_csv_path", _write_de_isin_map_csv(tmp_path)
    )

    page_one = _de_page_payload(
        [
            # Resolvable via ISIN map (in the bundled CSV).
            {
                "isin": "DE0007164600",
                "wkn": "716460",
                "name": "SAP SE",
                "isXetraTradable": True,
            },
            # Unresolvable: WKN is the only identifier and we never use WKN
            # as a ticker. No tickerSymbol, no slug, ISIN not in the map.
            # Expect this to be dropped with an `unresolved` resolution.
            {
                "isin": "DE000UNK0001",
                "wkn": "UNK001",
                "name": "Unknown Company",
                "isXetraTradable": True,
            },
        ],
        total=2,
    )
    pages = iter([page_one, _de_page_payload([], total=2)])

    def fake_http_get(url, allow_insecure_fallback=False, extra_headers=None):
        return _de_fetched_source(next(pages))

    service = OfficialMarketUniverseSourceService()
    monkeypatch.setattr(service, "_http_get", fake_http_get)

    snapshot = service.fetch_de_snapshot()

    assert [row["symbol"] for row in snapshot.rows] == ["SAP.DE"]
    assert snapshot.source_metadata["ticker_resolution"]["unresolved"] == 1


def test_fetch_de_snapshot_tolerates_per_page_failures(monkeypatch, tmp_path):
    from app.config import settings as app_settings
    from app.services import official_market_universe_source_service as svc_module

    monkeypatch.setattr(app_settings, "de_universe_source_url", _DE_BASE_URL)
    monkeypatch.setattr(
        app_settings, "de_universe_fallback_csv_path", _write_de_isin_map_csv(tmp_path)
    )
    monkeypatch.setattr(svc_module, "_BOERSE_FRANKFURT_PAGE_SIZE", 2)

    page_one = _de_page_payload(
        [
            {
                "isin": "DE0007164600",
                "wkn": "716460",
                "tickerSymbol": "SAP",
                "name": "SAP SE",
                "isXetraTradable": True,
            },
            {
                "isin": "DE0008404005",
                "wkn": "840400",
                "tickerSymbol": "ALV",
                "name": "Allianz SE",
                "isXetraTradable": True,
            },
        ],
        total=4,
    )
    page_three = _de_page_payload(
        [
            {
                "isin": "DE000A1EWWW0",
                "wkn": "A1EWWW",
                "tickerSymbol": "ADS",
                "name": "Adidas AG",
                "xetraTradable": True,
            },
        ],
        total=4,
    )

    call_count = {"n": 0}

    def fake_http_get(url, allow_insecure_fallback=False, extra_headers=None):
        call_count["n"] += 1
        if call_count["n"] == 2:
            # Second page (offset=2) blows up — but the live path must still
            # succeed because pages 1 and 3 each return rows.
            raise requests.exceptions.ConnectionError("synthetic page-2 failure")
        if call_count["n"] == 1:
            return _de_fetched_source(page_one)
        return _de_fetched_source(page_three)

    service = OfficialMarketUniverseSourceService()
    monkeypatch.setattr(service, "_http_get", fake_http_get)

    snapshot = service.fetch_de_snapshot()

    assert snapshot.source_metadata["fetch_mode"] == "live_http"
    assert {row["symbol"] for row in snapshot.rows} == {"SAP.DE", "ALV.DE", "ADS.DE"}
    # Per-page error surfaces in fetch_errors so operators can see what failed
    # without having to dig into worker logs.
    assert "page_2" in snapshot.source_metadata["fetch_errors"]
    assert "synthetic page-2 failure" in snapshot.source_metadata["fetch_errors"]["page_2"]


def test_fetch_de_snapshot_short_circuits_on_consecutive_failures(monkeypatch, tmp_path):
    """A salt rotation / DNS outage hits every page; the fetcher must abort
    quickly rather than retrying all 50 offsets before falling back."""
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "de_universe_source_url", _DE_BASE_URL)
    monkeypatch.setattr(
        app_settings, "de_universe_fallback_csv_path", _write_de_isin_map_csv(tmp_path)
    )

    call_count = {"n": 0}

    def fake_http_get(url, allow_insecure_fallback=False, extra_headers=None):
        call_count["n"] += 1
        raise requests.exceptions.ConnectionError(f"synthetic outage #{call_count['n']}")

    service = OfficialMarketUniverseSourceService()
    monkeypatch.setattr(service, "_http_get", fake_http_get)

    snapshot = service.fetch_de_snapshot()

    assert snapshot.source_metadata["fetch_mode"] == "csv_fallback"
    # Live path aborted after 3 consecutive failures, not the full 50-page cap.
    assert call_count["n"] == 3
    assert snapshot.source_metadata["fetch_errors"]["live_http"].startswith(
        "DE live universe fetch failed for every page (3 attempts)"
    )


def test_fetch_de_snapshot_falls_back_to_csv_on_http_failure(monkeypatch, tmp_path):
    from app.config import settings as app_settings

    csv_path = tmp_path / "de_dax40_constituents.csv"
    csv_path.write_text(
        "symbol,name,exchange,index\n"
        "SAP.DE,SAP SE,XETR,DAX\n"
        "ALV.DE,Allianz SE,XETR,DAX\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app_settings, "de_universe_source_url", _DE_BASE_URL)
    monkeypatch.setattr(app_settings, "de_universe_fallback_csv_path", str(csv_path))

    def fake_http_get(url, allow_insecure_fallback=False, extra_headers=None):
        raise requests.exceptions.ConnectionError("synthetic upstream outage")

    service = OfficialMarketUniverseSourceService()
    monkeypatch.setattr(service, "_http_get", fake_http_get)

    snapshot = service.fetch_de_snapshot()

    assert snapshot.market == "DE"
    assert snapshot.source_name == "de_manual_csv"
    assert snapshot.source_metadata["fetch_mode"] == "csv_fallback"
    assert snapshot.source_metadata["fetch_errors"]["live_http"].startswith(
        "DE live universe fetch failed for every page (3 attempts)"
    )
    assert snapshot.source_metadata["fallback_csv_path"] == str(csv_path)
    assert {row["symbol"] for row in snapshot.rows} == {"SAP.DE", "ALV.DE"}
    assert snapshot.snapshot_id.startswith("de-csv-fallback-")


@pytest.mark.parametrize("xetra_first", [True, False])
def test_fetch_de_snapshot_dedupes_by_isin_preferring_xetra(
    monkeypatch, tmp_path, xetra_first
):
    """The Xetra preference must hold regardless of payload ordering — even
    when the API returns the Frankfurt-floor venue before the Xetra one for
    the same ISIN."""
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "de_universe_source_url", _DE_BASE_URL)
    monkeypatch.setattr(
        app_settings, "de_universe_fallback_csv_path", _write_de_isin_map_csv(tmp_path)
    )

    xetra_row = {
        "isin": "DE0007164600",
        "wkn": "716460",
        "tickerSymbol": "SAP",
        "name": "SAP SE",
        "isXetraTradable": True,
    }
    frankfurt_row = {
        "isin": "DE0007164600",
        "wkn": "716460",
        "tickerSymbol": "SAP",
        "name": "SAP SE Frankfurt",
        "isXetraTradable": False,
    }
    ordered_rows = [xetra_row, frankfurt_row] if xetra_first else [frankfurt_row, xetra_row]
    page_one = _de_page_payload(ordered_rows, total=2)

    pages = iter([page_one, _de_page_payload([], total=2)])

    def fake_http_get(url, allow_insecure_fallback=False, extra_headers=None):
        return _de_fetched_source(next(pages))

    service = OfficialMarketUniverseSourceService()
    monkeypatch.setattr(service, "_http_get", fake_http_get)

    snapshot = service.fetch_de_snapshot()

    assert [row["symbol"] for row in snapshot.rows] == ["SAP.DE"]
    assert snapshot.rows[0]["exchange"] == "XETR"


def test_fetch_de_snapshot_paginates_on_raw_row_count_not_filtered(monkeypatch, tmp_path):
    """Pagination must use the raw upstream page size, not the post-filter row
    count. A page full of unresolved tickers (filtered out by ticker derivation)
    must not be mistaken for a partial page that ends pagination early."""
    from app.config import settings as app_settings
    from app.services import official_market_universe_source_service as svc_module

    monkeypatch.setattr(app_settings, "de_universe_source_url", _DE_BASE_URL)
    monkeypatch.setattr(
        app_settings, "de_universe_fallback_csv_path", _write_de_isin_map_csv(tmp_path)
    )
    monkeypatch.setattr(svc_module, "_BOERSE_FRANKFURT_PAGE_SIZE", 2)

    # Page 1: 2 raw rows, but only 1 resolves (the other has no ticker fields
    # and is dropped). If pagination used post-filter count (1 < 2 → "partial
    # page"), it would stop here and never request page 2.
    page_one = _de_page_payload(
        [
            {
                "isin": "DE0007164600",
                "wkn": "716460",
                "tickerSymbol": "SAP",
                "name": "SAP SE",
                "isXetraTradable": True,
            },
            {
                "isin": "DE000UNK0001",
                "wkn": "UNK001",
                "name": "Unknown",  # no tickerSymbol/ticker → dropped
                "isXetraTradable": True,
            },
        ],
        total=3,
    )
    page_two = _de_page_payload(
        [
            {
                "isin": "DE000A1EWWW0",
                "wkn": "A1EWWW",
                "tickerSymbol": "ADS",
                "name": "Adidas AG",
                "isXetraTradable": True,
            },
        ],
        total=3,
    )

    pages = iter([page_one, page_two])

    def fake_http_get(url, allow_insecure_fallback=False, extra_headers=None):
        return _de_fetched_source(next(pages))

    service = OfficialMarketUniverseSourceService()
    monkeypatch.setattr(service, "_http_get", fake_http_get)

    snapshot = service.fetch_de_snapshot()

    # Both pages were fetched; both resolvable symbols ended up in the bundle.
    assert {row["symbol"] for row in snapshot.rows} == {"SAP.DE", "ADS.DE"}
    assert snapshot.source_metadata["page_attempts"] == [0, 2]
    assert snapshot.source_metadata["ticker_resolution"]["unresolved"] == 1


def test_fetch_de_snapshot_raises_when_live_fails_and_csv_missing(monkeypatch, tmp_path):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "de_universe_source_url", _DE_BASE_URL)
    monkeypatch.setattr(
        app_settings,
        "de_universe_fallback_csv_path",
        str(tmp_path / "missing.csv"),
    )

    def fake_http_get(url, allow_insecure_fallback=False, extra_headers=None):
        raise requests.exceptions.ConnectionError("synthetic upstream outage")

    service = OfficialMarketUniverseSourceService()
    monkeypatch.setattr(service, "_http_get", fake_http_get)

    with pytest.raises(ValueError, match="no fallback CSV rows"):
        service.fetch_de_snapshot()


def test_parse_de_results_handles_alternate_payload_keys():
    service = OfficialMarketUniverseSourceService()
    payload = json.dumps(
        {
            "results": [
                {
                    "isin": "DE0007164600",
                    "wkn": "716460",
                    "tickerSymbol": "SAP",
                    "name": "SAP SE",
                    "isXetraTradable": True,
                },
                # Frankfurt-only listing, ticker via alternate explicit field.
                {
                    "isin": "DE0006062144",
                    "wkn": "606214",
                    "ticker": "1COV",
                    "name": "Covestro",
                    "isXetraTradable": False,
                },
                # Missing wkn → skipped
                {"isin": "DE000XXX0000", "tickerSymbol": "ANON", "name": "Anonymous"},
            ],
            "recordsTotal": 3,
        }
    ).encode("utf-8")

    rows, total, resolution, raw_row_count = service._parse_de_results(payload)

    assert total == 3
    # raw_row_count reflects the upstream payload size *before* filtering out
    # rows without WKN (the third entry is missing wkn and gets dropped).
    assert raw_row_count == 3
    symbols = [row["symbol"] for row in rows]
    assert symbols == ["SAP.DE", "1COV.F"]
    assert rows[1]["exchange"] == "XFRA"
    assert resolution["explicit_field"] == 2
    assert "slug_derived" not in resolution


def test_parse_de_results_rejects_invalid_payload():
    service = OfficialMarketUniverseSourceService()
    with pytest.raises(ValueError, match="Invalid Boerse Frankfurt response"):
        service._parse_de_results(b"not-json")
    with pytest.raises(ValueError, match="Unexpected Boerse Frankfurt payload shape"):
        service._parse_de_results(b"[]")


def test_load_de_csv_fallback_returns_empty_when_path_missing(monkeypatch, tmp_path):
    from app.config import settings as app_settings

    monkeypatch.setattr(
        app_settings,
        "de_universe_fallback_csv_path",
        str(tmp_path / "does-not-exist.csv"),
    )
    service = OfficialMarketUniverseSourceService()
    assert service._load_de_csv_fallback() == []


def test_derive_de_ticker_prefers_explicit_field_then_isin_map():
    """Slug-based derivation was removed in review: slugs encode the company
    name, not the ticker (e.g. ``adidas-ag`` → ``ADIDAS`` instead of the Yahoo
    symbol ``ADS``). Live resolution is now explicit ticker field → ISIN map,
    and any row that falls through both is dropped."""
    derive = OfficialMarketUniverseSourceService._derive_de_ticker
    isin_map = {"DE0007164600": "SAP", "DE0008404005": "ALV"}

    # Explicit field wins even when the ISIN is in the map.
    ticker, source = derive(
        {"tickerSymbol": "EXP", "slug": "sap-se"},
        isin="DE0007164600",
        isin_to_ticker=isin_map,
    )
    assert (ticker, source) == ("EXP", "explicit_field")

    # Falls through to ISIN map when no explicit field — slug is ignored.
    ticker, source = derive(
        {"slug": "ignore-this-slug"},
        isin="DE0008404005",
        isin_to_ticker=isin_map,
    )
    assert (ticker, source) == ("ALV", "isin_map")

    # Slug is no longer a resolution path: this row drops even with a slug.
    ticker, source = derive(
        {"slug": "covestro-ag"},
        isin="DE0006062144",
        isin_to_ticker=isin_map,
    )
    assert (ticker, source) == (None, "unresolved")

    # Returns None when nothing resolves — caller drops the row.
    ticker, source = derive(
        {"wkn": "716460"},
        isin="DE000UNK0001",
        isin_to_ticker=isin_map,
    )
    assert (ticker, source) == (None, "unresolved")


def test_load_de_isin_to_ticker_map(monkeypatch, tmp_path):
    from app.config import settings as app_settings

    csv_path = tmp_path / "de.csv"
    csv_path.write_text(
        "symbol,name,exchange,index,isin\n"
        "SAP.DE,SAP SE,XETR,DAX,DE0007164600\n"
        "ALV.DE,Allianz SE,XETR,DAX,DE0008404005\n"
        # Row missing isin column should be skipped silently.
        "ADS.DE,Adidas AG,XETR,DAX,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app_settings, "de_universe_fallback_csv_path", str(csv_path))

    isin_map = OfficialMarketUniverseSourceService._load_de_isin_to_ticker_map()
    assert isin_map == {
        "DE0007164600": "SAP",
        "DE0008404005": "ALV",
    }


def test_load_de_isin_to_ticker_map_returns_empty_when_csv_lacks_isin_column(
    monkeypatch, tmp_path
):
    from app.config import settings as app_settings

    csv_path = tmp_path / "de_no_isin.csv"
    csv_path.write_text(
        "symbol,name,exchange,index\nSAP.DE,SAP SE,XETR,DAX\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app_settings, "de_universe_fallback_csv_path", str(csv_path))

    assert OfficialMarketUniverseSourceService._load_de_isin_to_ticker_map() == {}
