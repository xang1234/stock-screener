from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.analysis.factor_attribution import (
    FACTOR_MODELS,
    Holding,
    calendar_time_returns,
    fit_factor_model,
    parse_ken_french_daily,
)

FIVE_FACTOR_TEXT = (
    "This file was created by using the 202609 CRSP database.\r\n"
    "The Tbill return is the simple daily rate that, over the number of trading days\r\n"
    "\r\n"
    ",Mkt-RF,SMB,HML,RMW,CMA,RF\r\n"
    "20260928,   -0.82,    0.27,   -0.50,    0.68,    0.39,    0.01\r\n"
    "20260929,   -0.21,   -0.03,  -99.99,   -0.26,   -0.43,    0.01\r\n"
    "20260930,   -0.26,    0.10,   -0.88,   -0.35,    0.19,    0.01\r\n"
    "\r\n"
    " Annual Factors: January-December\r\n"
    ",Mkt-RF,SMB,HML,RMW,CMA,RF\r\n"
    "2025,   12.00,    1.00,    2.00,    3.00,    4.00,    5.00\r\n"
    "\r\n"
    "Copyright 2026 Eugene F. Fama and Kenneth R. French\r\n"
)

MOMENTUM_TEXT = (
    "This file was created by using the 202609 CRSP database.  It\r\n"
    "\r\n"
    "Missing data are indicated by -99.99 or -999.\r\n"
    "\r\n"
    ",Mom\r\n"
    "20260929,  -0.31\r\n"
    "20260930,    -999\r\n"
    "\r\n"
    "Copyright 2026 Eugene F. Fama and Kenneth R. French\r\n"
)


def test_parse_reads_first_block_as_decimal_returns():
    factors = parse_ken_french_daily(FIVE_FACTOR_TEXT)

    assert list(factors.columns) == ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    assert list(factors.index) == list(pd.to_datetime(["2026-09-28", "2026-09-29", "2026-09-30"]))
    assert factors.loc["2026-09-28", "mkt_rf"] == pytest.approx(-0.0082)
    assert factors.loc["2026-09-30", "rf"] == pytest.approx(0.0001)
    assert np.isnan(factors.loc["2026-09-29", "hml"])
    assert factors.loc["2026-09-29", "rmw"] == pytest.approx(-0.0026)


def test_parse_momentum_header_and_missing_code():
    factors = parse_ken_french_daily(MOMENTUM_TEXT)

    assert list(factors.columns) == ["mom"]
    assert factors.loc["2026-09-29", "mom"] == pytest.approx(-0.0031)
    assert np.isnan(factors.loc["2026-09-30", "mom"])


def test_parse_rejects_text_without_a_header_row():
    with pytest.raises(ValueError):
        parse_ken_french_daily("not a Ken French file\r\n20260930, 1.0\r\n")


SESSIONS = [date(2026, 3, day) for day in (2, 3, 4, 5, 6, 9)]


def _history(bars: dict[int, tuple[float, float]]) -> pd.DataFrame:
    """Daily bars keyed by March 2026 day: (Open, Close)."""
    return pd.DataFrame(
        [{"Open": open_, "Close": close} for open_, close in bars.values()],
        index=pd.DatetimeIndex([pd.Timestamp(2026, 3, day) for day in bars]),
    )


def test_overlapping_holdings_are_equal_weighted_and_enter_at_the_open():
    histories = {
        # The 3/2 close of 90 would give 110/90 - 1 if the entry day used the prior close.
        "AAA": _history({2: (90, 90), 3: (100, 110), 4: (110, 99), 5: (99, 99), 6: (99, 120)}),
        "BBB": _history({3: (40, 45), 4: (50, 55), 5: (55, 66), 6: (66, 33)}),
    }
    holdings = [
        Holding("AAA", date(2026, 3, 3), date(2026, 3, 5)),
        Holding("BBB", date(2026, 3, 4), date(2026, 3, 6)),
    ]

    portfolio = calendar_time_returns(holdings, histories, sessions=SESSIONS)

    assert list(portfolio.index) == list(
        pd.to_datetime(["2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06"])
    )
    assert portfolio["ret"].tolist() == pytest.approx([0.10, 0.0, 0.10, -0.50])
    assert portfolio["n_holdings"].tolist() == [1, 2, 2, 1]


@pytest.mark.parametrize(
    "broken_bar",
    [None, (20.0, 0.0), (20.0, float("nan"))],
    ids=["missing", "zero-close", "nan-close"],
)
def test_holding_leaves_the_portfolio_at_its_first_bad_bar(broken_bar):
    gappy = {3: (10.0, 11.0), 4: (11.0, 12.1), 6: (20.0, 40.0), 9: (40.0, 80.0)}
    if broken_bar is not None:
        gappy[5] = broken_bar
    histories = {
        "GAP": _history(dict(sorted(gappy.items()))),
        "FULL": _history({3: (10, 10), 4: (10, 10), 5: (10, 12), 6: (12, 12), 9: (12, 12)}),
    }
    holdings = [
        Holding("GAP", date(2026, 3, 3), date(2026, 3, 9)),
        Holding("FULL", date(2026, 3, 3), date(2026, 3, 9)),
    ]

    portfolio = calendar_time_returns(holdings, histories, sessions=SESSIONS)

    # GAP counts 3/3 (+10%) and 3/4 (+10%); from 3/5 on only FULL is held. GAP neither
    # bridges the gap (40/12.1 - 1) nor comes back once bars resume (80/40 - 1).
    assert portfolio["ret"].tolist() == pytest.approx([0.05, 0.05, 0.20, 0.0, 0.0])
    assert portfolio["n_holdings"].tolist() == [2, 2, 1, 1, 1]


def test_days_without_a_holding_are_absent():
    histories = {"AAA": _history({day: (10, 10) for day in (2, 3, 4, 5, 6, 9)})}
    holdings = [
        Holding("AAA", date(2026, 3, 2), date(2026, 3, 3)),
        Holding("AAA", date(2026, 3, 6), date(2026, 3, 9)),
    ]

    portfolio = calendar_time_returns(holdings, histories, sessions=SESSIONS)

    assert list(portfolio.index) == list(
        pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-06", "2026-03-09"])
    )


def _synthetic(n_days: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2024-01-01", periods=n_days)
    factors = pd.DataFrame(
        {
            "mkt_rf": rng.normal(0.0004, 0.010, n_days),
            "smb": rng.normal(0.0001, 0.005, n_days),
            "hml": rng.normal(-0.0001, 0.005, n_days),
            "rmw": rng.normal(0.0, 0.004, n_days),
            "cma": rng.normal(0.0, 0.004, n_days),
            # A positive momentum premium, so leaving it out lands in alpha.
            "mom": rng.normal(0.0010, 0.005, n_days),
            "rf": np.full(n_days, 0.0001),
        },
        index=index,
    )
    ret = (
        0.0004
        + 1.2 * factors["mkt_rf"]
        + 0.5 * factors["smb"]
        - 0.3 * factors["hml"]
        + 0.8 * factors["mom"]
        + factors["rf"]
        + rng.normal(0.0, 0.002, n_days)
    )
    return pd.DataFrame({"ret": ret, "n_holdings": 10}, index=index), factors


def test_fit_recovers_loadings_and_alpha():
    portfolio, factors = _synthetic()

    fit = fit_factor_model(portfolio, factors, "ff5_mom")

    expected = {"mkt_rf": 1.2, "smb": 0.5, "hml": -0.3, "rmw": 0.0, "cma": 0.0, "mom": 0.8}
    assert fit.loadings == pytest.approx(expected, abs=0.05)
    assert fit.alpha_daily == pytest.approx(0.0004, abs=0.0002)
    assert fit.alpha_t > 2
    assert abs(fit.t_stats["rmw"]) < 3 < fit.t_stats["mom"]
    assert fit.alpha_annualized_pct == pytest.approx(((1 + fit.alpha_daily) ** 252 - 1) * 100)
    assert fit.r_squared > 0.9
    assert fit.n_days == 500
    assert fit.hac_lags == 5  # floor(4 * (500 / 100) ** (2 / 9))


def test_capm_folds_omitted_momentum_into_alpha():
    portfolio, factors = _synthetic()

    carhart = fit_factor_model(portfolio, factors, "carhart4")
    capm = fit_factor_model(portfolio, factors, "capm")

    assert set(capm.loadings) == {"mkt_rf"}
    # About 0.8 * 0.0010 of daily momentum premium moves into the CAPM alpha.
    assert capm.alpha_daily > carhart.alpha_daily + 0.0002


def test_overlapping_holdings_shrink_the_alpha_t_stat():
    portfolio, factors = _synthetic()
    rng = np.random.default_rng(11)
    # A 20-day moving sum of shocks: the serial correlation 20-session holdings leave behind.
    overlap_noise = np.convolve(rng.normal(0.0, 0.001, len(portfolio) + 19), np.ones(20), "valid")
    portfolio = portfolio.assign(ret=portfolio["ret"] + overlap_noise)

    newey_west = fit_factor_model(portfolio, factors, "carhart4", hac_lags=20)
    white = fit_factor_model(portfolio, factors, "carhart4", hac_lags=0)

    assert newey_west.alpha_daily == white.alpha_daily
    assert abs(newey_west.alpha_t) < 0.6 * abs(white.alpha_t)


def test_fit_uses_only_days_with_factor_data():
    portfolio, factors = _synthetic()

    fit = fit_factor_model(portfolio, factors.iloc[:-20], "ff3", hac_lags=2)

    assert fit.n_days == 480
    assert fit.hac_lags == 2


def test_fit_needs_ten_more_days_than_regressors():
    portfolio, factors = _synthetic(15)

    # carhart4 has 5 regressors with the intercept: 15 days fit, 14 do not.
    assert fit_factor_model(portfolio, factors, "carhart4").n_days == 15
    with pytest.raises(ValueError):
        fit_factor_model(portfolio.iloc[:14], factors, "carhart4")


def test_returns_after_entry_include_dividends_through_adj_close():
    # 3/4 is ex-dividend: Close drops 1% by the payout while Adj Close stays flat.
    history = pd.DataFrame(
        {"Open": [10.0, 11.0], "Close": [11.0, 10.89], "Adj Close": [10.89, 10.89]},
        index=pd.DatetimeIndex([pd.Timestamp(2026, 3, 3), pd.Timestamp(2026, 3, 4)]),
    )
    holdings = [Holding("DIV", date(2026, 3, 3), date(2026, 3, 4))]

    portfolio = calendar_time_returns(holdings, {"DIV": history}, sessions=SESSIONS)

    # The entry day stays Close/Open - 1, both unadjusted; Adj Close is on another scale.
    assert portfolio["ret"].tolist() == pytest.approx([0.10, 0.0])


def _fixed_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """40 days of patterned factors and serially correlated residuals, no RNG."""
    i = np.arange(40)
    index = pd.bdate_range("2025-01-01", periods=40)
    mkt = ((i * 7) % 11 - 5) / 1000
    mom = ((i * 5) % 9 - 4) / 2000
    shock = ((i * 3) % 7 - 3) / 2000
    noise = shock + 0.8 * np.concatenate([[0.0], shock[:-1]])
    factors = pd.DataFrame({"mkt_rf": mkt, "mom": mom, "rf": 0.0001}, index=index)
    ret = 0.0003 + 1.1 * mkt + 0.4 * mom + 0.0001 + noise
    return pd.DataFrame({"ret": ret, "n_holdings": 5}, index=index), factors


def test_newey_west_t_stats_match_an_independent_hac_computation(monkeypatch):
    portfolio, factors = _fixed_fixture()
    monkeypatch.setitem(FACTOR_MODELS, "mkt_mom", ("mkt_rf", "mom"))

    fit = fit_factor_model(portfolio, factors, "mkt_mom", hac_lags=3)

    # statsmodels 0.14.6 OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": 3,
    # "use_correction": False}) on the same data, computed outside the repo.
    assert fit.alpha_t == pytest.approx(2.198856099436389, rel=1e-9)
    assert fit.t_stats["mkt_rf"] == pytest.approx(23.17672266047837, rel=1e-9)
    assert fit.t_stats["mom"] == pytest.approx(3.6517310089543535, rel=1e-9)


def test_a_pick_whose_window_is_still_open_is_held_through_the_last_session():
    from types import SimpleNamespace

    from scripts.factor_attribution_report import held_picks

    def pick(symbol, exits, suppressed=frozenset()):
        return SimpleNamespace(
            raw=SimpleNamespace(symbol=symbol),
            entry_at=date(2026, 3, 2),
            exits=exits,
            suppressed=suppressed,
        )

    events = [
        pick("CLOSED", {5: date(2026, 3, 6)}),
        pick("OPEN", {}),
        pick("REPEAT", {}, suppressed=frozenset({5})),
    ]

    holdings = held_picks(events, horizon=5, last_session=date(2026, 3, 4))

    assert holdings == [
        Holding("CLOSED", date(2026, 3, 2), date(2026, 3, 6)),
        Holding("OPEN", date(2026, 3, 2), date(2026, 3, 4)),
    ]
