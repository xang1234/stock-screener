"""Factor attribution of the replay picks against Ken French US daily factors.

Holds every replay pick counted at one validation horizon from its entry session's open to
its exit session's close, averages the held picks' daily returns into a calendar-time
portfolio, and regresses its excess return on each model in FACTOR_MODELS. A significant
alpha that survives momentum and size is skill; one that vanishes was factor exposure.

Mid-session: it reads cached prices through the market's last completed session only, so
today's partial bar is never used, and it triggers no price fetch (cache-only reads). The
cut-off is fixed at start, so a close during the run changes nothing. The Ken French
download is its only network call; --factors-dir skips it. It writes nothing.

Usage:
    python scripts/factor_attribution_report.py --horizon 20 --factors-dir ~/ken-french
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import zipfile
from dataclasses import asdict
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import requests

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.analysis.factor_attribution import (  # noqa: E402
    FACTOR_MODELS,
    Holding,
    calendar_time_returns,
    fit_factor_model,
    parse_ken_french_daily,
)
from app.database import SessionLocal  # noqa: E402
from app.services.validation_service import (  # noqa: E402
    EXTENSION_LOOKBEHIND,
    HORIZONS,
    PRICE_CACHE_PERIOD,
    PriceOutcomeCalculator,
    ReplayPickValidationSource,
    _normalize_price_history,
    mark_overlapping_windows,
)

KEN_FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FACTOR_FILES = ("F-F_Research_Data_5_Factors_2x3_daily", "F-F_Momentum_Factor_daily")
FACTOR_NAMES = tuple(dict.fromkeys(name for names in FACTOR_MODELS.values() for name in names))


def load_factors(factors_dir: Path | None) -> pd.DataFrame:
    five, momentum = (parse_ken_french_daily(_factor_text(name, factors_dir)) for name in FACTOR_FILES)
    return five.join(momentum, how="inner")


def _factor_text(name: str, factors_dir: Path | None) -> str:
    if factors_dir is None:
        response = requests.get(KEN_FRENCH_URL.format(name=name), timeout=60)
        response.raise_for_status()
        archive = response.content
    elif (factors_dir / f"{name}.csv").exists():
        return (factors_dir / f"{name}.csv").read_text(encoding="latin-1")
    else:
        archive = (factors_dir / f"{name}_CSV.zip").read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        return zipped.read(zipped.namelist()[0]).decode("latin-1")


def collect_portfolio(db, calendar, price_cache, market: str, horizon: int) -> pd.DataFrame:
    """Calendar-time returns of the picks counted at ``horizon``, as the validation page counts them."""
    through = calendar.last_completed_trading_day(market)
    events, _ = ReplayPickValidationSource().collect(
        db, cutoff_date=date.min, market=market, until_date=through
    )
    if not events:
        return calendar_time_returns([], {}, sessions=[])
    sessions = calendar.trading_days(
        market, min(event.event_at for event in events) - EXTENSION_LOOKBEHIND, through
    )
    frames = price_cache.get_many_cached_only(
        sorted({event.symbol for event in events}), period=PRICE_CACHE_PERIOD, minimum_rows=1
    )
    # One read serves the evaluation and the returns, so both see the same bars.
    cached = SimpleNamespace(get_many_cached_only=lambda *_args, **_kwargs: frames)
    evaluated, _ = PriceOutcomeCalculator(cached).evaluate_many(events, sessions=sessions)
    holdings = [
        Holding(event.raw.symbol, event.entry_at, event.exits[horizon])
        for event in mark_overlapping_windows(evaluated)
        if event.entry_at is not None and horizon in event.exits and horizon not in event.suppressed
    ]
    histories = {
        symbol: history
        for symbol, frame in frames.items()
        if (history := _normalize_price_history(frame, through)) is not None
    }
    return calendar_time_returns(holdings, histories, sessions=sessions)


def build_report(
    portfolio: pd.DataFrame, factors: pd.DataFrame, *, market: str, horizon: int, market_open: bool
) -> dict:
    return {
        "market": market,
        "horizon": horizon,
        "market_open_during_run": market_open,
        "start": portfolio.index[0].date().isoformat(),
        "end": portfolio.index[-1].date().isoformat(),
        "n_days": len(portfolio),
        "avg_holdings": round(float(portfolio["n_holdings"].mean()), 1),
        "factors_through": factors.index[-1].date().isoformat(),
        "days_after_factors": int((portfolio.index > factors.index[-1]).sum()),
        "fits": [asdict(fit_factor_model(portfolio, factors, model)) for model in FACTOR_MODELS],
    }


def print_report(report: dict) -> None:
    print(
        f"Replay picks, {report['market']}, {report['horizon']}-session holds"
        f" (market open during run: {'yes' if report['market_open_during_run'] else 'no'})"
    )
    print(
        f"Portfolio {report['start']}..{report['end']}: {report['n_days']} days,"
        f" {report['avg_holdings']} holdings/day on average"
    )
    print(
        f"Ken French factors through {report['factors_through']}:"
        f" {report['days_after_factors']} portfolio days after it dropped"
    )
    print()
    print(
        f"{'model':<9}{'days':>5}{'alpha %/yr':>12}{'alpha t':>9}"
        + "".join(f"{name:>15}" for name in FACTOR_NAMES)
        + f"{'R2':>7}"
    )
    for fit in report["fits"]:
        loadings = "".join(
            f"{fit['loadings'][name]:>8.2f} ({fit['t_stats'][name]:>4.1f})"
            if name in fit["loadings"]
            else " " * 15
            for name in FACTOR_NAMES
        )
        print(
            f"{fit['model']:<9}{fit['n_days']:>5}{fit['alpha_annualized_pct']:>12.1f}"
            f"{fit['alpha_t']:>9.2f}{loadings}{fit['r_squared']:>7.2f}"
        )
    print(f"\nEach cell is loading (Newey-West t, {report['fits'][0]['hac_lags']} lags).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--market", default="US")
    parser.add_argument("--horizon", type=int, default=20, choices=HORIZONS)
    parser.add_argument("--factors-dir", type=Path, help="folder with the two Ken French CSVs or zips")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args()
    market = args.market.strip().upper()
    if market != "US":
        parser.error(f"only US factors are wired up; {market} needs regional Ken French files")

    logging.basicConfig(level=logging.WARNING)
    from app.wiring.bootstrap import (
        get_market_calendar_service,
        get_price_cache,
        initialize_process_runtime_services,
    )

    initialize_process_runtime_services()
    calendar = get_market_calendar_service()
    factors = load_factors(args.factors_dir)
    db = SessionLocal()
    try:
        market_open = calendar.is_market_open(market)
        portfolio = collect_portfolio(db, calendar, get_price_cache(), market, args.horizon)
    finally:
        db.close()
    if portfolio.empty:
        print(f"No {market} replay pick was held at the {args.horizon}-session horizon.")
        return 1

    report = build_report(
        portfolio, factors, market=market, horizon=args.horizon, market_open=market_open
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
