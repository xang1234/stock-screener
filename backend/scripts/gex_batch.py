"""Batch GEX computation for a ticker universe.

Reads a JSON config with ticker entries and writes per-ticker gamma exposure
snapshots to a JSON output file for downstream DB persistence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# gex_tasks.py invokes this as `python scripts/gex_batch.py` (a real script
# file, not `-c`), so sys.path[0] is this file's own directory
# (.../backend/scripts) -- the backend root (.../backend, which `import app`
# needs) is never on sys.path on its own, and nothing else (no PYTHONPATH,
# no .pth/editable install) provides it. Same bug, same fix as
# max_pain_batch.py: every deferred `from app.services import ...` below
# fails deterministically -- 100% of the time, every ticker -- unless the
# backend root is added here before those run.
_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

_ET = ZoneInfo("America/New_York")


class PermanentFetchError(Exception):
    """Raised when data for a ticker is structurally unavailable."""


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        cfg = json.load(handle)

    tickers = []
    for entry in cfg.get("tickers", []):
        if isinstance(entry, str):
            tickers.append({"symbol": entry, "yahoo_ticker": entry, "company_name": None})
        else:
            symbol = entry.get("symbol")
            if not symbol:
                continue
            tickers.append(
                {
                    "symbol": symbol,
                    "yahoo_ticker": entry.get("yahoo_ticker", symbol),
                    "company_name": entry.get("company_name"),
                }
            )

    cfg["tickers"] = tickers
    cfg.setdefault("strike_range_pct", 20.0)
    cfg.setdefault("max_strikes", None)
    return cfg


# ---------------------------------------------------------------------------
# Rate limiting -- shared Redis budget so concurrent fetch threads stay under
# one aggregate cap instead of each hammering Yahoo independently. Uses its
# own key ("yfinance:gex") separate from max_pain's budget since the two jobs
# now run as distinct chained steps rather than overlapping.
# ---------------------------------------------------------------------------

_rate_limiter = None
_rate_limiter_lock = threading.Lock()


def _get_rate_limiter():
    global _rate_limiter
    if _rate_limiter is None:
        with _rate_limiter_lock:
            if _rate_limiter is None:
                from app.services.rate_limiter import RedisRateLimiter
                _rate_limiter = RedisRateLimiter()
    return _rate_limiter


def _throttle(rate_interval_s: float) -> None:
    if rate_interval_s > 0:
        _get_rate_limiter().wait("yfinance:gex", min_interval_s=rate_interval_s)


def _get_yf_session():
    """Shared curl_cffi session (browser TLS fingerprint) for yfinance calls.

    Yahoo's bot detection routinely 429s plain `requests`-based clients --
    this is the same mitigation app/services/yf_session.py already provides
    for the rest of the backend's yfinance traffic; a per-ticker plain
    yf.Ticker() here would have skipped it entirely. Returns None (yfinance
    falls back to its default session) if curl_cffi/settings aren't available.
    """
    try:
        from app.services.yf_session import get_session
        return get_session()
    except Exception:
        return None


def _is_rate_limit_error(error: str | None) -> bool:
    try:
        from app.services.price_fetch_failures import is_rate_limit_error
        return is_rate_limit_error(error)
    except Exception:
        lower = (error or "").lower()
        return any(s in lower for s in ("rate", "429", "too many", "limit", "throttl"))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_gamma(spot: float, strike: float, time_years: float, vol: float) -> float:
    if spot <= 0 or strike <= 0 or time_years <= 0 or vol <= 0:
        return 0.0
    numerator = math.log(spot / strike) + 0.5 * vol * vol * time_years
    denominator = vol * math.sqrt(time_years)
    if denominator <= 0:
        return 0.0
    d1 = numerator / denominator
    return _norm_pdf(d1) / (spot * vol * math.sqrt(time_years))


def _infer_flip_level(gex_by_strike: list[tuple[float, float]]) -> float | None:
    if not gex_by_strike:
        return None

    cumulative = 0.0
    previous = None
    for strike, gex in sorted(gex_by_strike, key=lambda item: item[0]):
        cumulative += gex
        if previous is not None and previous != 0 and ((previous < 0 <= cumulative) or (previous > 0 >= cumulative)):
            return strike
        previous = cumulative

    nearest = min(gex_by_strike, key=lambda item: abs(item[1]))
    return nearest[0]


def fetch_one(symbol_cfg: dict, strike_range_pct: float, max_strikes: int | None,
              rate_interval_s: float = 0.0) -> dict:
    import yfinance as yf

    symbol = symbol_cfg["symbol"]
    yahoo_ticker = symbol_cfg.get("yahoo_ticker") or symbol

    session = _get_yf_session()
    if session is not None:
        try:
            ticker = yf.Ticker(yahoo_ticker, session=session)
        except TypeError:
            ticker = yf.Ticker(yahoo_ticker)
    else:
        ticker = yf.Ticker(yahoo_ticker)

    _throttle(rate_interval_s)
    expirations = ticker.options
    if not expirations:
        raise PermanentFetchError("no options chain available")

    expiration = expirations[0]
    _throttle(rate_interval_s)
    chain = ticker.option_chain(expiration)

    try:
        spot = float(ticker.fast_info["last_price"])
    except Exception:
        _throttle(rate_interval_s)
        hist = ticker.history(period="1d")
        if hist.empty:
            raise PermanentFetchError("unable to resolve spot price")
        spot = float(hist["Close"].iloc[-1])

    calls = chain.calls
    puts = chain.puts
    if calls is None or puts is None or calls.empty or puts.empty:
        raise PermanentFetchError("empty options chain")

    if "impliedVolatility" not in calls.columns or "impliedVolatility" not in puts.columns:
        raise PermanentFetchError("implied volatility not available in options chain")

    # Options expire based on US market hours, not UTC midnight -- using
    # datetime.utcnow().date() as "today" under-counts days-to-expiry by one
    # for near-dated contracts once it's past ~8pm ET (already the next UTC
    # calendar date), distorting gamma for 0-1 DTE options. Same class of fix
    # as trading_date (migration 20260805_0028).
    expiry_dt = datetime.strptime(expiration, "%Y-%m-%d")
    today_et = datetime.now(_ET).date()
    days_to_expiry = max((expiry_dt.date() - today_et).days, 1)
    time_years = max(days_to_expiry / 365.0, 1.0 / 365.0)

    lo = spot * (1 - strike_range_pct / 100.0)
    hi = spot * (1 + strike_range_pct / 100.0)

    strike_pool = sorted(set(calls["strike"].tolist()) | set(puts["strike"].tolist()))
    strikes = [value for value in strike_pool if lo <= value <= hi]
    if max_strikes is not None and max_strikes > 0 and len(strikes) > max_strikes:
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
        half = max_strikes // 2
        start = max(0, atm_idx - half)
        end = min(len(strikes), start + max_strikes)
        start = max(0, end - max_strikes)
        strikes = strikes[start:end]

    call_rows = calls[calls["strike"].isin(strikes)]
    put_rows = puts[puts["strike"].isin(strikes)]

    call_oi_total = 0
    put_oi_total = 0
    call_gex_total = 0.0
    put_gex_total = 0.0
    by_strike = {}

    for _, row in call_rows.iterrows():
        strike = float(row["strike"])
        oi = int(row.get("openInterest", 0) or 0)
        iv = float(row.get("impliedVolatility", 0) or 0)
        if oi <= 0 or iv <= 0:
            continue
        gamma = _bs_gamma(spot, strike, time_years, iv)
        gex = gamma * oi * 100.0 * spot * spot * 0.01
        call_oi_total += oi
        call_gex_total += gex
        by_strike[strike] = by_strike.get(strike, 0.0) + gex

    for _, row in put_rows.iterrows():
        strike = float(row["strike"])
        oi = int(row.get("openInterest", 0) or 0)
        iv = float(row.get("impliedVolatility", 0) or 0)
        if oi <= 0 or iv <= 0:
            continue
        gamma = _bs_gamma(spot, strike, time_years, iv)
        gex = -gamma * oi * 100.0 * spot * spot * 0.01
        put_oi_total += oi
        put_gex_total += gex
        by_strike[strike] = by_strike.get(strike, 0.0) + gex

    total_gex = call_gex_total + put_gex_total
    flip_level = _infer_flip_level(list(by_strike.items()))
    distance_pct = None
    if flip_level not in (None, 0):
        distance_pct = ((spot - flip_level) / flip_level) * 100.0

    company_name = symbol_cfg.get("company_name")
    if not company_name:
        _throttle(rate_interval_s)
        try:
            company_name = ticker.info.get("shortName") or symbol
        except Exception:
            company_name = symbol

    return {
        "symbol": symbol,
        "company_name": company_name,
        "status": "OK",
        "expiration": expiration,
        "spot_price": spot,
        "call_oi": call_oi_total,
        "put_oi": put_oi_total,
        "call_gex": call_gex_total,
        "put_gex": put_gex_total,
        "total_gex": total_gex,
        "flip_level": flip_level,
        "distance_to_flip_pct": distance_pct,
        "fetched_at": datetime.utcnow().isoformat(),
    }


def fetch_with_retry(
    symbol_cfg: dict,
    strike_range_pct: float,
    max_strikes: int | None,
    max_retry_minutes: float,
    retry_interval_sec: float,
    rate_interval_s: float = 0.0,
) -> dict:
    symbol = symbol_cfg["symbol"]
    deadline = time.monotonic() + (max_retry_minutes * 60.0)
    attempt = 0
    last_error = "unknown"

    while True:
        attempt += 1
        try:
            result = fetch_one(symbol_cfg, strike_range_pct, max_strikes, rate_interval_s=rate_interval_s)
            print(f"[{symbol}] OK on attempt {attempt}", flush=True)
            return result
        except PermanentFetchError as exc:
            print(f"[{symbol}] permanent failure: {exc}", flush=True)
            return {
                "symbol": symbol,
                "company_name": symbol_cfg.get("company_name"),
                "status": "FAILED",
                "error": str(exc),
                "fetched_at": datetime.utcnow().isoformat(),
            }
        except ImportError as exc:
            # A broken/missing module will never fix itself by retrying --
            # retrying anyway just burns the full max_retry_minutes window
            # on every single symbol for nothing. Treat it like a permanent
            # failure.
            print(f"[{symbol}] import failure, not retrying: {exc}", flush=True)
            return {
                "symbol": symbol,
                "company_name": symbol_cfg.get("company_name"),
                "status": "FAILED",
                "error": str(exc),
                "fetched_at": datetime.utcnow().isoformat(),
            }
        except Exception as exc:
            last_error = str(exc)

        if time.monotonic() >= deadline:
            print(f"[{symbol}] FAILED after {attempt} attempts: {last_error}", flush=True)
            return {
                "symbol": symbol,
                "company_name": symbol_cfg.get("company_name"),
                "status": "FAILED",
                "error": last_error,
                "fetched_at": datetime.utcnow().isoformat(),
            }

        if _is_rate_limit_error(last_error):
            # Yahoo is actively throttling us -- retrying at the normal pace
            # would just hammer straight back into the same throttle window.
            # Back off harder than a plain transient failure.
            penalty = max(retry_interval_sec * 3, 30.0)
            print(f"[{symbol}] rate-limit signal detected ({last_error}); backing off {penalty:.0f}s...", flush=True)
            time.sleep(penalty)
        else:
            time.sleep(retry_interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute full-universe GEX snapshots")
    parser.add_argument("config", help="Path to input JSON config")
    parser.add_argument("--output", default="gex_output.json", help="Path to output JSON")
    parser.add_argument("--max-retry-minutes", type=float, default=0.75)
    parser.add_argument("--retry-interval-sec", type=float, default=3.0)
    parser.add_argument(
        "--fetch-concurrency", type=int,
        default=int(os.environ.get("GEX_FETCH_CONCURRENCY", "6")),
        help="Number of tickers to fetch concurrently (threads)",
    )
    parser.add_argument(
        "--rate-limit-per-sec", type=float,
        default=float(os.environ.get("GEX_YFINANCE_RATE_LIMIT", "3.5")),
        help="Aggregate Yahoo requests/sec across all concurrent fetch threads (shared Redis budget). 0 disables throttling.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    strike_range_pct = float(cfg.get("strike_range_pct", 20.0))
    raw_max_strikes = cfg.get("max_strikes")
    max_strikes = int(raw_max_strikes) if isinstance(raw_max_strikes, int) and raw_max_strikes > 0 else None

    rate_interval_s = (1.0 / args.rate_limit_per_sec) if args.rate_limit_per_sec > 0 else 0.0
    concurrency = max(1, args.fetch_concurrency)

    tickers = cfg.get("tickers", [])
    results_by_symbol = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_symbol = {
            pool.submit(
                fetch_with_retry, ticker_cfg,
                strike_range_pct, max_strikes,
                args.max_retry_minutes, args.retry_interval_sec,
                rate_interval_s,
            ): ticker_cfg["symbol"]
            for ticker_cfg in tickers
        }
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            results_by_symbol[symbol] = future.result()

    # Restore original config order in the output.
    rows = [results_by_symbol[t["symbol"]] for t in tickers]

    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "strike_range_pct": strike_range_pct,
        "max_strikes": max_strikes,
        "rows": rows,
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    ok = len([row for row in rows if row.get("status") == "OK"])
    failed = len(rows) - ok
    print(f"Done. {ok} OK, {failed} failed, total {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
