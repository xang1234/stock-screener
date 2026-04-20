from __future__ import annotations

from app.scripts.build_india_taxonomy import build_india_taxonomy_rows


def test_build_india_taxonomy_rows_emits_dual_listed_alias_rows():
    rows, counts = build_india_taxonomy_rows(
        nexus_rows=[
            {
                "Symbol": "RELIANCE",
                "Company Name": "Reliance Industries",
                "Industry (Sector)": "ENERGY (STOCKS)",
                "Subgroup (Theme)": "Oil & Gas",
                "Sub-industry": "Integrated Oil & Gas",
            }
        ],
        nse_rows=[
            {
                "symbol": "RELIANCE.NS",
                "exchange": "XNSE",
                "isin": "INE002A01018",
                "name": "Reliance Industries Limited",
            }
        ],
        bse_rows=[
            {
                "symbol": "500325.BO",
                "exchange": "XBOM",
                "isin": "INE002A01018",
                "name": "Reliance Industries Ltd",
                "security_id": "RELIANCE",
            }
        ],
    )

    assert [row["Symbol"] for row in rows] == ["500325.BO", "RELIANCE.NS"]
    assert all(row["ISIN"] == "INE002A01018" for row in rows)
    assert all(row["Subgroup (Theme)"] == "Oil & Gas" for row in rows)
    assert counts["primary_local"] == 1
