from __future__ import annotations

import io
import zipfile

from app.services.kr_market_data_service import OpenDartFundamentalsService


class _FakeResponse:
    def __init__(self, *, content: bytes = b"", payload: dict | None = None) -> None:
        self.content = content
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, *, params: dict, timeout: int) -> _FakeResponse:
        self.calls.append((url, params))
        if url.endswith("corpCode.xml"):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, mode="w") as archive:
                archive.writestr(
                    "CORPCODE.xml",
                    """
                    <result>
                      <list>
                        <corp_code>00126380</corp_code>
                        <corp_name>Samsung Electronics</corp_name>
                        <stock_code>005930</stock_code>
                      </list>
                    </result>
                    """,
                )
            return _FakeResponse(content=buffer.getvalue())
        return _FakeResponse(
            payload={
                "status": "000",
                "list": [
                    {"account_nm": "Revenue", "thstrm_amount": "100,000"},
                    {"account_nm": "Operating income", "thstrm_amount": "12,000"},
                    {"account_nm": "Net income", "thstrm_amount": "9,000"},
                    {"account_nm": "Gross profit", "thstrm_amount": "20,000"},
                    {"account_nm": "Total assets", "thstrm_amount": "300,000"},
                    {"account_nm": "Total liabilities", "thstrm_amount": "120,000"},
                    {"account_nm": "Total equity", "thstrm_amount": "180,000"},
                    {"account_nm": "Current assets", "thstrm_amount": "80,000"},
                    {"account_nm": "Current liabilities", "thstrm_amount": "40,000"},
                ],
            }
        )


def test_opendart_returns_empty_when_api_key_missing() -> None:
    service = OpenDartFundamentalsService(api_key="")

    assert service.get_statement_fundamentals("005930") == {}


def test_opendart_maps_statement_rows_to_existing_fundamental_fields() -> None:
    session = _FakeSession()
    service = OpenDartFundamentalsService(api_key="token", session=session)

    fields = service.get_statement_fundamentals("5930", business_year=2025)

    assert fields["revenue_current"] == 100000
    assert fields["operating_margin"] == 12.0
    assert fields["profit_margin"] == 9.0
    assert fields["gross_margin"] == 20.0
    assert fields["roe"] == 5.0
    assert fields["roa"] == 3.0
    assert fields["debt_to_equity"] == 66.66666666666666
    assert fields["current_ratio"] == 2.0
    assert fields["recent_quarter_date"] == "2025-FY"
    assert session.calls[1][1]["corp_code"] == "00126380"
