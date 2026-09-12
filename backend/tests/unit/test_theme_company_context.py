from datetime import datetime, timedelta, timezone

from app.models.stock import StockFundamental
from app.models.stock_universe import StockUniverse
from app.services.theme_company_context import build_company_context

AS_OF = datetime(2026, 9, 11, tzinfo=timezone.utc)


def _add_universe(db, symbol, name, *, active=True, source="official", **fields):
    db.add(
        StockUniverse(
            symbol=symbol,
            name=name,
            market="US",
            is_active=active,
            source=source,
            last_seen_in_source_at=AS_OF - timedelta(days=1),
            **fields,
        )
    )


def test_keeps_only_active_explicit_cashtags_and_validated_name_resolutions(universe_session):
    _add_universe(universe_session, "NBIS", "Nebius Group N.V.")
    _add_universe(universe_session, "MSFT", "Microsoft Corporation")
    _add_universe(universe_session, "POST", "Post Holdings")
    _add_universe(universe_session, "DEAD", "Inactive Corp", active=False)
    universe_session.commit()

    context = build_company_context(
        universe_session,
        "$nbis announced capacity. POST and KEEP are ordinary prose.",
        as_of=AS_OF,
        identity_only=True,
        resolved_symbols=("msft", "dead"),
    )

    assert [company["symbol"] for company in context["companies"]] == ["NBIS", "MSFT"]
    assert [company["name"] for company in context["companies"]] == [
        "Nebius Group N.V.",
        "Microsoft Corporation",
    ]
    assert context["companies"][0]["identity_source"] == (
        "local_stock_universe:official:2026-09-10T00:00:00+00:00"
    )
    assert context["companies"][0]["profile_status"] == "identity_only"
    assert "inactive_resolved_symbol:DEAD" in context["warnings"]


def test_rejects_unknown_and_ambiguous_numeric_cashtags(universe_session):
    _add_universe(universe_session, "0700.HK", "Tencent Holdings")
    universe_session.commit()

    context = build_company_context(
        universe_session,
        "$MISSING and $700 are unresolvable; $700.HK is explicit.",
        as_of=AS_OF,
        identity_only=True,
    )

    assert [company["symbol"] for company in context["companies"]] == ["0700.HK"]
    assert "unknown_explicit_symbol:MISSING" in context["warnings"]
    assert "ambiguous_numeric_cashtag:700" in context["warnings"]


def test_publishes_only_fresh_attributed_profile_fields(universe_session):
    _add_universe(
        universe_session,
        "NBIS",
        "Nebius Group N.V.",
        sector="Universe sector must not leak",
        industry="Universe industry must not leak",
    )
    refreshed = AS_OF - timedelta(days=10)
    universe_session.add(
        StockFundamental(
            symbol="NBIS",
            sector="Technology",
            industry="Cloud Infrastructure",
            description_yfinance="A cloud infrastructure company.",
            yahoo_profile_refreshed_at=refreshed,
            field_provenance={"sector": "yfinance", "industry": "yfinance"},
        )
    )
    universe_session.commit()

    company = build_company_context(
        universe_session, "$NBIS", as_of=AS_OF
    )["companies"][0]

    assert company["name"] == "Nebius Group N.V."
    assert company["sector"] == "Technology"
    assert company["industry"] == "Cloud Infrastructure"
    assert company["business_description"] == "A cloud infrastructure company."
    assert company["profile_source"] == {
        "business_description": "yfinance",
        "industry": "yfinance",
        "sector": "yfinance",
    }
    assert company["profile_as_of"] == {
        "business_description": "2026-09-01T00:00:00+00:00",
        "industry": "2026-09-01T00:00:00+00:00",
        "sector": "2026-09-01T00:00:00+00:00",
    }
    assert company["profile_status"] == "available"


def test_suppresses_stale_and_unattributed_business_fields(universe_session):
    _add_universe(universe_session, "OLD", "Old Profile Inc.")
    _add_universe(universe_session, "RAW", "Raw Profile Inc.")
    universe_session.add_all(
        [
            StockFundamental(
                symbol="OLD",
                sector="Technology",
                industry="Software",
                description_yfinance="This stale description must not be published.",
                yahoo_profile_refreshed_at=AS_OF - timedelta(days=181),
                field_provenance={"sector": "yfinance", "industry": "yfinance"},
            ),
            StockFundamental(
                symbol="RAW",
                sector="Technology",
                industry="Software",
                description_yfinance="This unattributed description must not be published.",
            ),
        ]
    )
    universe_session.commit()

    companies = build_company_context(
        universe_session, "$OLD $RAW", as_of=AS_OF
    )["companies"]

    assert [(company["symbol"], company["profile_status"]) for company in companies] == [
        ("OLD", "stale"),
        ("RAW", "unattributed"),
    ]
    for company in companies:
        assert company["sector"] is None
        assert company["industry"] is None
        assert company["business_description"] is None
        assert company["profile_source"] is None
        assert company["profile_as_of"] is None


def test_marks_conflicting_description_provenance_unusable(universe_session):
    _add_universe(universe_session, "NBIS", "Nebius Group N.V.")
    universe_session.add(
        StockFundamental(
            symbol="NBIS",
            description_yfinance="A cloud company.",
            yahoo_profile_refreshed_at=AS_OF - timedelta(days=2),
            field_provenance={"description_yfinance": "finviz"},
        )
    )
    universe_session.commit()

    context = build_company_context(universe_session, "$NBIS", as_of=AS_OF)
    company = context["companies"][0]

    assert company["profile_status"] == "conflicting"
    assert company["business_description"] is None
    assert company["profile_source"] is None
    assert "conflicting_profile_provenance:NBIS" in context["warnings"]


def test_withholds_profile_data_not_available_at_requested_as_of(universe_session):
    _add_universe(universe_session, "NBIS", "Nebius Group N.V.")
    universe_session.add(
        StockFundamental(
            symbol="NBIS",
            description_yfinance="This profile was not available yet.",
            yahoo_profile_refreshed_at=AS_OF + timedelta(days=1),
        )
    )
    universe_session.commit()

    context = build_company_context(universe_session, "$NBIS", as_of=AS_OF)
    company = context["companies"][0]

    assert company["profile_status"] == "not_yet_available"
    assert company["business_description"] is None
    assert "future_profile_fields:NBIS" in context["warnings"]


def test_marks_bounded_business_description_as_truncated(universe_session):
    _add_universe(universe_session, "NBIS", "Nebius Group N.V.")
    universe_session.add(
        StockFundamental(
            symbol="NBIS",
            description_yfinance="d" * 1_501,
            yahoo_profile_refreshed_at=AS_OF - timedelta(days=1),
        )
    )
    universe_session.commit()

    context = build_company_context(universe_session, "$NBIS", as_of=AS_OF)
    company = context["companies"][0]

    assert len(company["business_description"]) == 1_500
    assert "description_truncated:NBIS" in context["warnings"]


def test_rejects_unrecognized_explicit_description_provenance(universe_session):
    _add_universe(universe_session, "NBIS", "Nebius Group N.V.")
    universe_session.add(
        StockFundamental(
            symbol="NBIS",
            description_yfinance="A cloud company.",
            yahoo_profile_refreshed_at=AS_OF - timedelta(days=1),
            field_provenance={"description_yfinance": "unknown_provider"},
        )
    )
    universe_session.commit()

    context = build_company_context(universe_session, "$NBIS", as_of=AS_OF)

    assert context["companies"][0]["profile_status"] == "conflicting"
    assert context["companies"][0]["business_description"] is None
    assert "conflicting_profile_provenance:NBIS" in context["warnings"]


def test_does_not_autoflush_pending_rows_during_read_only_resolution(universe_session):
    pending = StockUniverse(
        symbol="NBIS",
        name="Nebius Group N.V.",
        market="US",
        is_active=True,
    )
    universe_session.add(pending)

    context = build_company_context(universe_session, "$NBIS", as_of=AS_OF)

    assert context["companies"] == []
    assert pending in universe_session.new


def test_omits_commodity_futures_cashtag_from_equity_company_context(universe_session):
    """A commodity contract root must not load an unrelated equity profile."""
    _add_universe(universe_session, "HG", "Hamilton Insurance Group Ltd")
    universe_session.commit()

    context = build_company_context(
        universe_session,
        "Do or die time for $HG $copper. Trying to complete the blowoff move.",
        as_of=AS_OF,
    )

    assert context["companies"] == []
    assert "commodity_futures_symbol:HG" in context["warnings"]


def test_keeps_named_equity_identity_when_commodity_word_is_nearby(universe_session):
    _add_universe(universe_session, "HG", "Hamilton Insurance Group Ltd")
    universe_session.commit()

    context = build_company_context(
        universe_session,
        "Hamilton Insurance ($HG) reported results. Copper prices also rose.",
        as_of=AS_OF,
        identity_only=True,
    )

    assert [company["symbol"] for company in context["companies"]] == ["HG"]
    assert "commodity_futures_symbol:HG" not in context["warnings"]
