"""Calendar-time factor attribution: do picks earn alpha beyond the Fama-French factors?

Each day's portfolio return is the equal-weight mean of the daily returns of the picks
held that day. Its excess over the risk-free rate is regressed on Ken French daily
factors, with Newey-West standard errors since holdings overlap across days.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

FACTOR_MODELS: dict[str, tuple[str, ...]] = {
    "capm": ("mkt_rf",),
    "ff3": ("mkt_rf", "smb", "hml"),
    "carhart4": ("mkt_rf", "smb", "hml", "mom"),
    "ff5_mom": ("mkt_rf", "smb", "hml", "rmw", "cma", "mom"),
}

TRADING_DAYS_PER_YEAR = 252
# Regressions need this many observations beyond their regressor count.
MIN_SPARE_OBSERVATIONS = 10
_MISSING_CODES = (-99.99, -999.0)


@dataclass(frozen=True)
class Holding:
    """One counted pick held from the entry session's open to the exit session's close."""

    symbol: str
    entry_at: date
    exit_at: date


@dataclass(frozen=True)
class FactorFit:
    model: str
    alpha_daily: float
    alpha_annualized_pct: float
    alpha_t: float
    loadings: dict[str, float]
    t_stats: dict[str, float]
    r_squared: float
    n_days: int
    hac_lags: int


def calendar_time_returns(
    holdings: Sequence[Holding],
    histories: Mapping[str, pd.DataFrame],
    *,
    sessions: Sequence[date],
) -> pd.DataFrame:
    """Daily equal-weight portfolio returns over the market's ``sessions``.

    A holding earns Close/Open - 1 on its entry session and Close_t / Close_t-1 - 1 after.
    It leaves the portfolio from the first held session its history lacks or prices at or
    below zero, so no return ever spans a gap. Days with no holding are absent. Returns
    are not stopped out: a stop makes the payoff non-linear and biases the loadings.
    """
    calendar = pd.DatetimeIndex(sessions)
    held_returns = []
    for holding in holdings:
        held = calendar[
            (calendar >= pd.Timestamp(holding.entry_at))
            & (calendar <= pd.Timestamp(holding.exit_at))
        ]
        bars = histories.get(holding.symbol, pd.DataFrame(columns=["Open", "Close"])).reindex(held)
        opens = bars["Open"].to_numpy(dtype=float)
        closes = bars["Close"].to_numpy(dtype=float)
        previous = np.concatenate([opens[:1], closes[:-1]])
        # NaN compares False, so a missing bar ends the holding like a bad price does.
        valid = np.logical_and.accumulate((closes > 0) & (previous > 0))
        held_returns.append(pd.Series(closes[valid] / previous[valid] - 1.0, index=held[valid]))

    if not held_returns:
        return pd.DataFrame({"ret": [], "n_holdings": []}, index=pd.DatetimeIndex([]))
    wide = pd.concat(held_returns, axis=1).sort_index()
    return pd.DataFrame({"ret": wide.mean(axis=1), "n_holdings": wide.count(axis=1)})


def fit_factor_model(
    portfolio: pd.DataFrame,
    factors: pd.DataFrame,
    model: str,
    hac_lags: int | None = None,
) -> FactorFit:
    """OLS of the portfolio's excess return on ``model``'s factors, with Newey-West t-stats."""
    names = FACTOR_MODELS[model]
    data = portfolio[["ret"]].join(factors[[*names, "rf"]], how="inner").dropna()
    n_days = len(data)
    n_regressors = len(names) + 1
    if n_days < n_regressors + MIN_SPARE_OBSERVATIONS:
        raise ValueError(
            f"{model} needs {n_regressors + MIN_SPARE_OBSERVATIONS} days with factor data,"
            f" got {n_days}"
        )

    y = (data["ret"] - data["rf"]).to_numpy()
    x = np.column_stack([np.ones(n_days), data[list(names)].to_numpy()])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    lags = int(4 * (n_days / 100) ** (2 / 9)) if hac_lags is None else hac_lags

    scores = x * resid[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        autocov = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (lags + 1)) * (autocov + autocov.T)
    bread = np.linalg.inv(x.T @ x)
    t = coef / np.sqrt(np.diag(bread @ meat @ bread))

    centered = y - y.mean()
    return FactorFit(
        model=model,
        alpha_daily=float(coef[0]),
        alpha_annualized_pct=float(((1 + coef[0]) ** TRADING_DAYS_PER_YEAR - 1) * 100),
        alpha_t=float(t[0]),
        loadings=dict(zip(names, map(float, coef[1:]))),
        t_stats=dict(zip(names, map(float, t[1:]))),
        r_squared=float(1 - resid @ resid / (centered @ centered)),
        n_days=n_days,
        hac_lags=lags,
    )


def parse_ken_french_daily(text: str) -> pd.DataFrame:
    """The first data block of a Ken French daily factor CSV, in decimal returns.

    Columns are lowercased with ``-`` as ``_`` (``Mkt-RF`` is ``mkt_rf``); the missing-value
    codes -99.99 and -999 become NaN.
    """
    lines = iter(text.splitlines())
    header = next((line for line in lines if line.startswith(",")), None)
    if header is None:
        raise ValueError("no Ken French header row (a line starting with ',')")
    columns = [name.strip().lower().replace("-", "_") for name in header.split(",")[1:]]
    rows = [line.split(",") for line in itertools.takewhile(str.strip, lines)]
    values = np.array([[float(cell) for cell in row[1:]] for row in rows])
    values[np.isin(values, _MISSING_CODES)] = np.nan
    index = pd.to_datetime([row[0].strip() for row in rows], format="%Y%m%d")
    return pd.DataFrame(values / 100, index=index, columns=columns)
