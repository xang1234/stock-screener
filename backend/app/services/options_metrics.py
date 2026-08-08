"""Options chain exposure and volatility metrics utilities.

Expected input: options_chain: list[dict] where each dict contains:
  - strike: float
  - type: 'call'|'put'
  - gamma: float (per-contract gamma)
  - delta: float (per-contract delta; calls positive, puts negative)
  - vanna: float
  - charm: float
  - open_interest: int
  - iv: float (implied volatility, e.g., 0.45)

These helpers compute per-strike aggregate exposures (DEX/VEX/CEX), key
gamma levels, IVR and skew.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Tuple
from zoneinfo import ZoneInfo
import math

_ET = ZoneInfo("America/New_York")

# Bumped whenever calculate_options_metrics()'s payload shape changes (a
# field added/removed/renamed). POST /v1/options/metrics' cache-hit check
# compares this against a cached entry's own "schema_version" to decide
# whether to trust it -- checking for one specific field's presence (the
# previous approach) only catches the batch-vs-live distinction, not "this
# was cached by an OLDER version of this function and is missing whatever
# field got added most recently." Every past instance of that exact bug
# (missing iv_smile/unusual_volume in entries cached before this field
# existed) silently sat in Redis for up to its full 7-day TTL. Bump this
# any time the payload shape changes so old entries stop being trusted
# immediately instead of waiting out their TTL.
# v3: added next_earnings_date.
OPTIONS_METRICS_SCHEMA_VERSION = 3

import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


def _coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_valid_iv(value: Any) -> bool:
    if value is None:
        return False
    try:
        iv = float(value)
    except (TypeError, ValueError):
        return False
    return iv > 0.0


def _interpolate_zero_crossing(prev_strike: float, prev_cum: float, cur_strike: float, cur_cum: float) -> float:

    if cur_cum == prev_cum:
        return cur_strike
    if prev_cum == 0:
        return prev_strike
    if cur_cum == 0:
        return cur_strike
    fraction = (0 - prev_cum) / (cur_cum - prev_cum)
    return prev_strike + fraction * (cur_strike - prev_strike)


def aggregate_by_strike(options_chain: List[Dict[str, Any]], spot: float) -> Dict[float, Dict[str, Any]]:
    """Aggregate dollar exposures by strike.

    Returns a dict keyed by strike with aggregated fields:
      call_gex, put_gex, total_gex, dex, vex, cex, oi, iv_sample_count, iv_avg

    `spot` is required to express these in dollar terms -- a bare
    `gamma * oi * 100` (no spot factor) is dimensionally a shares-per-$1-move
    rate, not a dollar exposure, and doesn't match the convention used
    elsewhere (see gex_batch.py / _bs_gamma_delta's GEX, which is dollar
    gamma per 1% underlying move: gamma * oi * 100 * spot**2 * 0.01). This
    function previously omitted spot entirely, so its GEX/DEX/VEX/CEX were
    off from gex_batch.py's numbers by a factor of the stock price -- the
    two panels on the Options Analytics dashboard were showing
    non-comparable magnitudes for the same underlying quantity.
    """
    strikes: Dict[float, Dict[str, Any]] = {}
    for opt in options_chain:
        strike = float(opt["strike"])
        typ = opt.get("type", "call").lower()
        oi = int(opt.get("open_interest") or 0)
        delta = _coerce_float(opt.get("delta"), 0.0)
        gamma = _coerce_float(opt.get("gamma"), 0.0)
        vanna = _coerce_float(opt.get("vanna"))
        charm = _coerce_float(opt.get("charm"))
        iv = _coerce_float(opt.get("iv"))

        entry = strikes.setdefault(strike, {
            "strike": strike,
            "call_gex": 0.0,
            "put_gex": 0.0,
            "total_gex": 0.0,
            "dex": 0.0,
            "vex": 0.0,
            "cex": 0.0,
            "oi": 0,
            "iv_sum": 0.0,
            "iv_count": 0,
            "has_vex": False,
            "has_cex": False,
        })

        # Dollar gamma exposure per 1% move in the underlying, with put-side
        # gamma represented as negative exposure -- matches gex_batch.py's
        # convention (the daily GEX batch feeding /v1/gex/history).
        raw_gex = (gamma or 0.0) * oi * 100 * spot * spot * 0.01
        gex = raw_gex if typ == "call" else -raw_gex
        if typ == "call":
            entry["call_gex"] += max(gex, 0.0)
        else:
            entry["put_gex"] += min(gex, 0.0)

        entry["total_gex"] += gex

        # Dollar delta/vanna/charm exposure per strike.
        if delta is not None:
            entry["dex"] += delta * oi * 100 * spot
        if vanna is not None:
            entry["vex"] += vanna * oi * 100 * spot
            entry["has_vex"] = True
        if charm is not None:
            entry["cex"] += charm * oi * 100 * spot
            entry["has_cex"] = True
        entry["oi"] += oi

        if iv is not None:
            try:
                entry["iv_sum"] += float(iv)
                entry["iv_count"] += 1
            except Exception:
                pass

    # finalize iv avg
    for s in strikes.values():
        s["iv_avg"] = (s["iv_sum"] / s["iv_count"]) if s["iv_count"] else None
        if not s.pop("has_vex", False):
            s["vex"] = None
        if not s.pop("has_cex", False):
            s["cex"] = None
        # remove internal sums
        s.pop("iv_sum", None)
        s.pop("iv_count", None)

    return dict(sorted(strikes.items()))


def compute_max_pain(rows: List[Tuple[float, int, int]]) -> Optional[float]:
    """Max pain strike: the strike where total OI-weighted intrinsic value
    paid out to option holders is minimized. `rows` is (strike, call_oi,
    put_oi) tuples -- same algorithm as scripts/max_pain_batch.py's
    compute_max_pain(), duplicated here (rather than imported) since that
    module is a standalone subprocess script, not a package this can import.
    """
    if not rows:
        return None
    best_strike, best_pain = None, None
    for p, _, _ in rows:
        total = 0.0
        for k, c, pu in rows:
            total += c * max(p - k, 0)
            total += pu * max(k - p, 0)
        if best_pain is None or total < best_pain:
            best_pain, best_strike = total, p
    return best_strike


def compute_key_gamma_levels(strike_agg: Dict[float, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Compute Call Wall, Put Wall, and Zero Gamma (gamma flip) price levels.

    Call Wall is the strike with the largest positive call-side GEX. Put Wall is
    the strike with the most negative put-side GEX (largest absolute negative
    value). Zero Gamma is estimated by cumulative total_gex crossing from
    negative to positive with linear interpolation between surrounding strikes.
    """
    if not strike_agg:
        return {"call_wall": None, "put_wall": None, "zero_gamma": None}

    strikes = sorted(strike_agg.keys())
    call_wall = None
    max_call_gex = float('-inf')
    put_wall = None
    min_put_gex = float('inf')

    cum = 0.0
    cum_list: List[Tuple[float, float]] = []  # (strike, cumulative)
    for k in strikes:
        entry = strike_agg[k]
        call_gex = float(entry.get("call_gex", 0.0) or 0.0)
        put_gex = float(entry.get("put_gex", 0.0) or 0.0)
        total = float(entry.get("total_gex", 0.0) or 0.0)

        if call_gex > 0.0 and call_gex > max_call_gex:
            max_call_gex = call_gex
            call_wall = k

        if put_gex < 0.0 and put_gex < min_put_gex:
            min_put_gex = put_gex
            put_wall = k

        cum += total
        cum_list.append((k, cum))

    zero_gamma = None
    for i in range(1, len(cum_list)):
        prev_strike, prev_cum = cum_list[i - 1]
        cur_strike, cur_cum = cum_list[i]
        if prev_cum < 0 and cur_cum >= 0:
            zero_gamma = _interpolate_zero_crossing(prev_strike, prev_cum, cur_strike, cur_cum)
            break

    return {"call_wall": call_wall, "put_wall": put_wall, "zero_gamma": zero_gamma}


def compute_net_exposures(strike_agg: Dict[float, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute total net DEX/VEX/CEX across all strikes."""
    net_dex = 0.0
    net_vex = 0.0
    net_cex = 0.0
    for s in strike_agg.values():
        net_dex += s.get("dex", 0.0) or 0.0
        net_vex += s.get("vex", 0.0) or 0.0
        net_cex += s.get("cex", 0.0) or 0.0

    return {"net_dex": net_dex, "net_vex": net_vex, "net_cex": net_cex}


def compute_ivr(current_iv: Optional[float], iv_52w_low: Optional[float], iv_52w_high: Optional[float]) -> Optional[float]:
    if current_iv is None:
        return None
    if iv_52w_low is None or iv_52w_high is None:
        return None

    try:
        iv_low = float(iv_52w_low)
        iv_high = float(iv_52w_high)
        denom = iv_high - iv_low
        if denom <= 0:
            return None
        return (float(current_iv) - iv_low) / denom * 100.0
    except Exception:
        return None


def find_nearest_iv_for_delta(options_chain: List[Dict[str, Any]], target_delta: float, typ: str) -> Optional[float]:

    """Find IV of option with delta closest to target_delta for given side ('call' or 'put').
    target_delta should be a positive number representing absolute delta (eg 0.25).
    For puts, delta values are typically negative; we compare absolute values.
    """
    best = None
    best_diff = None
    for opt in options_chain:
        if opt.get("type", "call").lower() != typ:
            continue
        if not _is_valid_iv(opt.get("iv")):
            continue
        d = float(opt.get("delta") or 0.0)
        absd = abs(d)
        diff = abs(absd - abs(target_delta))
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = opt

    if best is None:
        return None
    return best.get("iv")


def compute_skew(options_chain: List[Dict[str, Any]], target_delta: float = 0.25) -> Optional[float]:

    """Skew = IV(25-delta put) - IV(25-delta call) for same expiration approximation."""
    iv_put = find_nearest_iv_for_delta(options_chain, target_delta, "put")
    iv_call = find_nearest_iv_for_delta(options_chain, target_delta, "call")
    if iv_put is None or iv_call is None:
        return None
    try:
        return float(iv_put) - float(iv_call)
    except Exception:
        return None


def extract_iv_smile(calls: pd.DataFrame, puts: pd.DataFrame) -> Dict[str, List[Dict[str, float]]]:
    """Per-strike IV for calls and puts, kept as two separate series.

    Unlike aggregate_by_strike()'s `iv_avg` (which blends call+put IV at each
    strike into one number for the GEX/DEX aggregation), a smile/skew chart
    needs the two sides kept apart -- that gap between them *is* the skew.

    Rows with IV <= 0 are dropped: yfinance reports impliedVolatility=0 for
    stale/no-quote contracts, not a real zero-volatility market, and plotting
    them would show a fake spike to zero at illiquid strikes.
    """
    def _rows(df: pd.DataFrame) -> List[Dict[str, float]]:
        if df is None or df.empty or "strike" not in df.columns or "impliedVolatility" not in df.columns:
            return []
        valid = df[(df["impliedVolatility"] > 0) & df["strike"].notna()]
        return [
            {"strike": float(row["strike"]), "iv": float(row["impliedVolatility"])}
            for _, row in valid.sort_values("strike").iterrows()
        ]

    return {"calls": _rows(calls), "puts": _rows(puts)}


def find_unusual_volume(
    calls: pd.DataFrame, puts: pd.DataFrame, min_ratio: float = 1.5
) -> List[Dict[str, Any]]:
    """Contracts trading at an unusual multiple of their existing open
    interest today (volume / open_interest > min_ratio) -- a rough "someone
    is doing something today, not just holding an existing position" flag.

    Requires open_interest > 0: a contract with volume today but zero prior
    OI has no existing position size to compare against (and would otherwise
    divide by zero), so it's excluded rather than treated as "infinitely
    unusual".
    """
    rows: List[Dict[str, Any]] = []
    for df, option_type in ((calls, "call"), (puts, "put")):
        if df is None or df.empty or "strike" not in df.columns:
            continue
        for _, row in df.iterrows():
            oi = _safe_int(row.get("openInterest"))
            volume = _safe_float(row.get("volume"))
            if oi <= 0:
                continue
            ratio = volume / oi
            if ratio > min_ratio:
                rows.append({
                    "strike": float(row["strike"]),
                    "type": option_type,
                    "volume": int(volume),
                    "open_interest": oi,
                    "ratio": round(ratio, 2),
                })

    rows.sort(key=lambda r: r["ratio"], reverse=True)
    return rows


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _nearest_option_row(df: pd.DataFrame, underlying_price: float) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    if "strike" not in df.columns:
        return None
    df = df.copy()
    df = df[df["strike"].notna()]
    if df.empty:
        return None
    idx = (df["strike"].sub(underlying_price).abs()).idxmin()
    return df.loc[idx]


def _compute_historical_volatility(history: pd.DataFrame) -> Optional[float]:
    if history is None or history.empty or "Close" not in history.columns:
        return None

    closes = history["Close"].dropna()
    if closes.shape[0] < 2:
        return None

    closes = closes.tail(21)
    returns = closes.pct_change().dropna().apply(math.log1p)
    if returns.empty:
        return None

    return float(returns.std(ddof=0) * math.sqrt(252))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_gamma_delta(spot: float, strike: float, time_years: float, vol: float, option_type: str) -> Tuple[float, float]:
    """Black-Scholes gamma and delta.

    yfinance option chains do not reliably include gamma/delta columns, so
    walls/skew must be derived from strike + implied volatility instead of a
    (usually absent or stale) provider-supplied greek.
    """
    if spot <= 0 or strike <= 0 or time_years <= 0 or vol <= 0:
        return 0.0, 0.0
    denominator = vol * math.sqrt(time_years)
    if denominator <= 0:
        return 0.0, 0.0
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * time_years) / denominator
    gamma = _norm_pdf(d1) / (spot * denominator)
    delta = _norm_cdf(d1) if option_type == "call" else _norm_cdf(d1) - 1.0
    return gamma, delta


def _bs_vanna_charm(spot: float, strike: float, time_years: float, vol: float) -> Tuple[float, float]:
    """Black-Scholes vanna and charm, zero-rate/zero-dividend (matches
    _bs_gamma_delta's simplified d1, which carries no r/q term).

    Yahoo's public options data doesn't supply vanna or charm at all (unlike
    gamma/delta, which are at least sometimes present as stale/absent
    columns) -- there's no provider value to fall back to, so these are
    model estimates, not market-observed figures. Callers should flag that
    (see `greeks_methodology` on the API response) rather than presenting
    them as if Yahoo reported them directly.

    vanna = d(Delta)/d(vol) = -phi(d1) * d2 / vol
    charm = d(Delta)/d(T)   = -phi(d1) * d2 / (2T)
    Both are identical for calls and puts under r=q=0 (the "-1" delta offset
    for puts is a constant, so it drops out of both derivatives). `charm`
    here is the mathematical derivative w.r.t. time-to-expiry T -- positive
    means delta rises as T increases (further from expiry), not a
    "per calendar day" decay figure.
    """
    if spot <= 0 or strike <= 0 or time_years <= 0 or vol <= 0:
        return 0.0, 0.0
    sqrt_t = math.sqrt(time_years)
    denominator = vol * sqrt_t
    if denominator <= 0:
        return 0.0, 0.0
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * time_years) / denominator
    d2 = d1 - denominator
    pdf_d1 = _norm_pdf(d1)
    vanna = -pdf_d1 * d2 / vol
    charm = -pdf_d1 * d2 / (2.0 * time_years)
    return vanna, charm


def _time_to_expiry_years(expiration: str) -> float:
    """Time to expiry in years, using the US/Eastern trading day as "today".

    Options expire based on US market hours, not UTC midnight. Using
    datetime.utcnow().date() as "today" under-counts days-to-expiry by one
    for any 0-1 DTE contract once it's past ~8pm ET (already the next UTC
    calendar date), which meaningfully distorts gamma/vanna/charm for
    near-dated options -- the same class of bug fixed for trading_date in
    migration 20260805_0028.
    """
    try:
        expiry_dt = datetime.strptime(expiration, "%Y-%m-%d").date()
        today_et = datetime.now(_ET).date()
        days_to_expiry = max((expiry_dt - today_et).days, 1)
    except (TypeError, ValueError):
        days_to_expiry = 1
    return max(days_to_expiry / 365.0, 1.0 / 365.0)


# Below this, a reported IV is treated as a degenerate/placeholder reading
# rather than a real market quote. Raised from an earlier 0.01 (1%) floor
# after live diagnosis on 2026-08-06 showed yfinance serving a flat ~1.5-3%
# "IV" across an entire chain -- for SPY and QQQ simultaneously, with open
# interest otherwise genuinely populated (so the zero-OI staleness check
# elsewhere doesn't catch it) -- which the old floor happily accepted as
# "valid" once the walk-outward loop below reached it. No liquid large-cap
# or index ETF has legitimately printed single-digit-percent ATM IV; 8% is
# comfortably below any real quiet-market reading while well above that
# degenerate band.
_MIN_PLAUSIBLE_IV = 0.08


def _find_valid_atm_option(df: pd.DataFrame, underlying_price: float, min_iv: float = _MIN_PLAUSIBLE_IV) -> Optional[pd.Series]:
    """Find the strike closest to the money that has a usable (non-degenerate) IV.

    yfinance sometimes reports impliedVolatility as 0, near-zero, or a flat
    implausibly-low placeholder (see _MIN_PLAUSIBLE_IV) for a stale/illiquid
    contract even when its strike is otherwise ATM. Using that value directly
    collapses VRP to roughly `-historical_volatility`. Walk outward from the
    nearest strike until a contract with a plausible IV is found.
    """
    if df is None or df.empty or "strike" not in df.columns or "impliedVolatility" not in df.columns:
        return None
    ordered = df.assign(_dist=(df["strike"] - underlying_price).abs()).sort_values("_dist")
    for _, row in ordered.iterrows():
        iv = row.get("impliedVolatility")
        if iv is not None and iv >= min_iv:
            return row
    return None


def _chain_iv_is_degenerate(iv_values: List[float], min_distinct: int = 10) -> bool:
    """True when a chain's impliedVolatility readings look like a synthetic
    placeholder rather than real market quotes.

    Live-diagnosed on 2026-08-06: yfinance served a SPY chain where 96 calls
    across a wide, actively-traded strike range (thousands of contracts of
    volume on many strikes) collapsed onto just 8 distinct IV values, each
    almost exactly double the previous one (~0.00001, 0.0020, 0.0078, 0.0156,
    0.0313, 0.0625, 0.1250, 0.2500) -- a geometric step function, not a
    smile/skew curve, and unrelated to the real volume sitting on those
    strikes. QQQ served the same pattern simultaneously. A _MIN_PLAUSIBLE_IV
    floor alone can't catch this: the sequence spans from near-zero up
    through values ordinarily plausible on their own (0.125, 0.25), so
    whatever floor is set, some rung of the ladder clears it.

    A real IV surface varies near-continuously strike to strike (skew), so
    collapsing onto a handful of distinct values across dozens of contracts
    is the tell, independent of the specific numbers involved.
    """
    if len(iv_values) < 20:
        return False
    distinct = len({round(v, 6) for v in iv_values if v is not None})
    return distinct < min_distinct


def find_current_atm_iv_from_chain(
    options_chain: List[Dict[str, Any]], underlying_price: float, min_iv: float = _MIN_PLAUSIBLE_IV
) -> Optional[float]:
    """List-of-dicts equivalent of `_find_valid_atm_option` for callers (like the
    nightly batch task) that build a plain `options_chain` instead of a
    yfinance DataFrame. Walks outward from the nearest strike per side (calls
    preferred, then puts) to avoid a stale/near-zero IV on an illiquid ATM
    contract, unless the whole chain's IV looks synthetic (see
    _chain_iv_is_degenerate), in which case there's nothing trustworthy to
    walk toward.
    """
    if _chain_iv_is_degenerate([opt.get("iv") for opt in options_chain]):
        return None
    for option_type in ("call", "put"):
        candidates = sorted(
            (
                opt for opt in options_chain
                if opt.get("type") == option_type and opt.get("iv") is not None
            ),
            key=lambda opt: abs(opt["strike"] - underlying_price),
        )
        for opt in candidates:
            if opt["iv"] >= min_iv:
                return float(opt["iv"])
    return None


def _iv_history_range_from_rows(rows) -> Tuple[Optional[float], Optional[float]]:
    values: List[float] = []
    for (v,) in rows:
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            continue

    if len(values) < 2:
        return None, None
    return min(values), max(values)


def _get_iv_history_range(db: Optional["Session"], ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """Read-only lookup of the trailing ~252-row (low, high) ATM IV range for
    a ticker, without adding a new point.

    Used for on-demand, non-default-expiration lookups (term structure) --
    see _update_iv_history_and_get_range's docstring for why those must not
    write to this table.
    """
    if db is None:
        return None, None

    from ..models.iv_history import IvHistory

    ticker = ticker.upper()
    try:
        rows = (
            db.query(IvHistory.atm_iv)
            .filter(IvHistory.ticker == ticker)
            .order_by(IvHistory.trading_date.desc())
            .limit(252)
            .all()
        )
    except Exception:
        return None, None

    return _iv_history_range_from_rows(rows)


def _update_iv_history_and_get_range(
    db: Optional["Session"], ticker: str, current_iv: Optional[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Persist today's ATM IV in the iv_history table (one row per ticker per
    trading day) and return the observed (low, high) over the trailing ~252
    rows so IV Rank can be computed.

    No dedicated historical-IV data source exists in this codebase, so the
    range is bootstrapped incrementally from live metric calculations rather
    than mixing in stock-price 52-week highs/lows (a different quantity).

    Previously this lived in a Redis hash (7d-TTL keys elsewhere in this
    codebase made a Redis flush/cache-clear silently reset every ticker's IV
    Rank back to "building up history again"). Postgres makes it durable
    across cache clears and restarts, consistent with how max_pain_snapshots
    and gex_snapshots already persist their daily options data.

    IMPORTANT: only call this for a ticker's NEAREST/default expiration (the
    nightly batch, and the default no-expiration /v1/options/metrics fetch).
    There's no `expiration` column here -- one row per ticker per day, full
    stop -- so writing a far-dated expiration's IV here would overwrite that
    day's near-term value with an unrelated one (different expirations have
    structurally different IV levels), corrupting the rolling series everyone
    else's IV Rank is computed against. On-demand term-structure lookups for
    a user-picked expiration must use the read-only _get_iv_history_range
    instead.
    """
    if current_iv is None or current_iv <= 0 or db is None:
        return None, None

    from .options_snapshot_upsert import trading_date_for
    from ..models.iv_history import IvHistory

    ticker = ticker.upper()
    trading_date = trading_date_for(datetime.utcnow())
    try:
        existing = (
            db.query(IvHistory)
            .filter(IvHistory.ticker == ticker, IvHistory.trading_date == trading_date)
            .first()
        )
        if existing is not None:
            existing.atm_iv = current_iv
        else:
            db.add(IvHistory(ticker=ticker, trading_date=trading_date, atm_iv=current_iv))
        db.commit()

        rows = (
            db.query(IvHistory.atm_iv)
            .filter(IvHistory.ticker == ticker)
            .order_by(IvHistory.trading_date.desc())
            .limit(252)
            .all()
        )
    except Exception:
        db.rollback()
        return None, None

    return _iv_history_range_from_rows(rows)


def _coerce_date(value: Any) -> Optional["date"]:
    """Best-effort coercion of whatever yfinance hands back (datetime.date,
    datetime.datetime, pandas.Timestamp, or an ISO date string) into a plain
    date, for comparison purposes only."""
    if value is None:
        return None
    if hasattr(value, "date") and callable(value.date):
        try:
            return value.date()
        except Exception:
            pass
    try:
        from datetime import date as _date_cls
        if isinstance(value, _date_cls):
            return value
    except Exception:
        pass
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _get_next_earnings_date(yf_ticker) -> Optional[str]:
    """Nearest future earnings date for this ticker, as an ISO date string,
    or None if Yahoo doesn't have one listed for it.

    Not having an earnings date is the common case (most tickers most of the
    time, and some tickers -- ETFs, small/thin names -- never have one), not
    a failure -- every branch below is defensive and swallows its own
    errors rather than letting a missing/malformed earnings calendar break
    the whole options-metrics payload.

    yfinance's calendar API has changed shape across versions (a dict in
    recent releases, a DataFrame in older ones), so `.calendar` is tried
    first (cheap, already-cached with the Ticker object) and
    `.get_earnings_dates()` (an extra network call) is a fallback rather
    than the primary path.
    """
    today = datetime.now(_ET).date()

    try:
        cal = yf_ticker.calendar
        raw_dates = None
        if isinstance(cal, dict):
            raw_dates = cal.get("Earnings Date")
        elif cal is not None and hasattr(cal, "empty") and not cal.empty and "Earnings Date" in getattr(cal, "index", []):
            raw_dates = list(cal.loc["Earnings Date"].dropna())
        if raw_dates:
            if not isinstance(raw_dates, (list, tuple)):
                raw_dates = [raw_dates]
            future = sorted(d for d in (_coerce_date(v) for v in raw_dates) if d is not None and d >= today)
            if future:
                return future[0].isoformat()
    except Exception:
        pass

    try:
        earnings_df = yf_ticker.get_earnings_dates(limit=8)
        if earnings_df is not None and not earnings_df.empty:
            future = sorted(
                d for d in (_coerce_date(idx) for idx in earnings_df.index) if d is not None and d >= today
            )
            if future:
                return future[0].isoformat()
    except Exception:
        pass

    return None


def calculate_options_metrics(
    ticker: str, expiration: str, db: Optional["Session"] = None, record_iv_history: bool = True
) -> Dict[str, Any]:
    """Fetch an options chain from yfinance and compute institutional options metrics.

    The returned payload includes GEX, walls, PCR, premium, HV, VRP, and expected move.

    `db` is used to persist/read today's ATM IV for IV Rank (see
    _update_iv_history_and_get_range); when omitted, IVR is left null rather
    than raising, since not every caller has a DB session to hand.

    `record_iv_history` must be False for any caller passing a non-default
    expiration (e.g. the term-structure endpoint, where `expiration` is
    whatever the user picked in the dropdown) -- iv_history has no
    `expiration` column, so writing a far-dated expiration's IV into it would
    corrupt the single per-ticker-per-day series IV Rank is computed from.
    Only the nightly batch and the default/nearest-expiration fetch should
    pass True. When False, the 52w range is still read (read-only) so IVR
    can be displayed for that expiration's own compute.
    """
    from .yfinance_service import YFinanceService
    import yfinance as yf

    svc = YFinanceService()
    svc._wait_for_yfinance_rate_limit()

    # Shared curl_cffi session (see yf_session.py) -- Yahoo's bot detection
    # routinely 429s plain requests-based clients. This is the highest-traffic
    # live yfinance call path (every /v1/options/metrics cache miss and every
    # /v1/options/term-structure selection goes through it), so it's the one
    # most worth this mitigation; YFinanceService itself doesn't apply it
    # anywhere yet -- a broader gap across that class, out of scope here.
    try:
        from .yf_session import get_session
        session = get_session()
    except Exception:
        session = None
    yf_ticker = yf.Ticker(ticker, session=session) if session is not None else yf.Ticker(ticker)
    try:
        option_chain = yf_ticker.option_chain(expiration)
    except Exception as exc:
        raise ValueError(f"Could not fetch option chain for {ticker} {expiration}: {exc}") from exc

    next_earnings_date = _get_next_earnings_date(yf_ticker)

    calls = option_chain.calls if hasattr(option_chain, "calls") else pd.DataFrame()
    puts = option_chain.puts if hasattr(option_chain, "puts") else pd.DataFrame()

    history = svc.get_historical_data(ticker, period="1mo", interval="1d", use_cache=False)
    underlying_price = None
    historical_volatility = None
    if history is not None and not history.empty and "Close" in history.columns:
        closes = history["Close"].dropna()
        if not closes.empty:
            underlying_price = _safe_float(closes.iloc[-1])
            historical_volatility = _compute_historical_volatility(history)

    if underlying_price is None or underlying_price == 0.0:
        info = getattr(yf_ticker, "info", {}) or {}
        underlying_price = _safe_float(
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

    if underlying_price == 0.0:
        raise ValueError(f"Unable to determine underlying price for {ticker}")

    calls = calls.copy() if not calls.empty else pd.DataFrame(columns=["strike", "openInterest", "volume", "lastPrice", "impliedVolatility", "gamma"])
    puts = puts.copy() if not puts.empty else pd.DataFrame(columns=["strike", "openInterest", "volume", "lastPrice", "impliedVolatility", "gamma"])

    time_years = _time_to_expiry_years(expiration)

    for df, option_type in ((calls, "call"), (puts, "put")):
        if "volume" not in df.columns:
            df["volume"] = 0
        if "lastPrice" not in df.columns:
            df["lastPrice"] = 0.0
        if "impliedVolatility" not in df.columns:
            df["impliedVolatility"] = df.get("impliedVol", 0.0)
        if "strike" not in df.columns:
            df["strike"] = 0.0
        df["volume"] = df["volume"].fillna(0).apply(_safe_float)
        df["lastPrice"] = df["lastPrice"].fillna(0).apply(_safe_float)
        df["openInterest"] = df["openInterest"].fillna(0).apply(_safe_int)
        df["impliedVolatility"] = df["impliedVolatility"].fillna(0).apply(_safe_float)
        df["strike"] = df["strike"].fillna(0).apply(_safe_float)

        # yfinance option chains do not reliably supply gamma/delta, so derive
        # both via Black-Scholes from strike + implied volatility rather than
        # trusting a (usually absent) provider-supplied greek column. vanna/
        # charm aren't provided by yfinance at all -- these are pure model
        # estimates (see _bs_vanna_charm docstring); flagged via
        # `greeks_methodology` in the response rather than presented as
        # market-observed values.
        greeks = [
            _bs_gamma_delta(underlying_price, strike, time_years, iv, option_type)
            for strike, iv in zip(df["strike"], df["impliedVolatility"])
        ]
        df["gamma"] = [g for g, _ in greeks]
        df["delta"] = [d for _, d in greeks]

        vanna_charm = [
            _bs_vanna_charm(underlying_price, strike, time_years, iv)
            for strike, iv in zip(df["strike"], df["impliedVolatility"])
        ]
        df["vanna"] = [v for v, _ in vanna_charm]
        df["charm"] = [c for _, c in vanna_charm]

    # Max pain, restricted to +/-20% of spot (same window scripts/max_pain_batch.py
    # uses) -- keeps the O(n^2) pain calc bounded and the result comparable to
    # the daily batch's nearest-expiration max_pain_strike.
    strike_lo, strike_hi = underlying_price * 0.8, underlying_price * 1.2
    call_oi_by_strike = dict(zip(calls["strike"], calls["openInterest"]))
    put_oi_by_strike = dict(zip(puts["strike"], puts["openInterest"]))
    pain_strikes = sorted(
        k for k in (set(call_oi_by_strike) | set(put_oi_by_strike))
        if strike_lo <= k <= strike_hi
    )
    pain_rows = [
        (k, int(call_oi_by_strike.get(k, 0)), int(put_oi_by_strike.get(k, 0)))
        for k in pain_strikes
    ]
    max_pain_strike = compute_max_pain(pain_rows)

    # Dollar gamma exposure per 1% move in the underlying -- must match
    # aggregate_by_strike()'s convention below (and gex_batch.py's), or this
    # function's own top-level total_gex disagrees with its own per-strike
    # `strikes` list for the same chain.
    calls["call_gex"] = calls["gamma"] * calls["openInterest"] * 100 * underlying_price * underlying_price * 0.01
    puts["put_gex"] = puts["gamma"] * puts["openInterest"] * 100 * underlying_price * underlying_price * 0.01 * -1

    total_call_gex = float(calls["call_gex"].sum())
    total_put_gex = float(puts["put_gex"].sum())
    total_gex = total_call_gex + total_put_gex

    call_wall = None
    call_wall_gex = None
    if not calls.empty and calls["call_gex"].gt(0).any():
        call_wall_idx = calls["call_gex"].idxmax()
        call_wall = float(calls.loc[call_wall_idx, "strike"])
        call_wall_gex = float(calls.loc[call_wall_idx, "call_gex"])

    put_wall = None
    put_wall_gex = None
    if not puts.empty and puts["put_gex"].lt(0).any():
        put_wall_idx = puts["put_gex"].idxmin()
        put_wall = float(puts.loc[put_wall_idx, "strike"])
        put_wall_gex = float(puts.loc[put_wall_idx, "put_gex"])

    total_call_volume = float(calls["volume"].sum())
    total_put_volume = float(puts["volume"].sum())
    total_call_oi = float(calls["openInterest"].sum())
    total_put_oi = float(puts["openInterest"].sum())

    volume_pcr = None
    if total_call_volume > 0:
        volume_pcr = total_put_volume / total_call_volume

    oi_pcr = None
    if total_call_oi > 0:
        oi_pcr = total_put_oi / total_call_oi

    call_premium_notional = float((calls["volume"] * calls["lastPrice"] * 100).sum())
    put_premium_notional = float((puts["volume"] * puts["lastPrice"] * 100).sum())

    iv_smile = extract_iv_smile(calls, puts)
    unusual_volume = find_unusual_volume(calls, puts)

    options_chain = []
    for _, row in calls.iterrows():
        options_chain.append({
            "strike": float(row["strike"]),
            "type": "call",
            "delta": float(row.get("delta", 0.0) or 0.0),
            "gamma": float(row["gamma"]),
            "vanna": float(row.get("vanna", 0.0) or 0.0),
            "charm": float(row.get("charm", 0.0) or 0.0),
            "open_interest": int(row["openInterest"]),
            "iv": float(row["impliedVolatility"]) if row["impliedVolatility"] > 0 else None,
        })
    for _, row in puts.iterrows():
        options_chain.append({
            "strike": float(row["strike"]),
            "type": "put",
            "delta": float(row.get("delta", 0.0) or 0.0),
            "gamma": float(row["gamma"]),
            "vanna": float(row.get("vanna", 0.0) or 0.0),
            "charm": float(row.get("charm", 0.0) or 0.0),
            "open_interest": int(row["openInterest"]),
            "iv": float(row["impliedVolatility"]) if row["impliedVolatility"] > 0 else None,
        })

    atm_call = _nearest_option_row(calls, underlying_price)
    atm_put = _nearest_option_row(puts, underlying_price)

    atm_strike = None
    if atm_call is not None:
        atm_strike = float(atm_call["strike"])
    elif atm_put is not None:
        atm_strike = float(atm_put["strike"])

    atm_call_last_price = float(atm_call["lastPrice"]) if atm_call is not None else None
    atm_put_last_price = float(atm_put["lastPrice"]) if atm_put is not None else None

    # A strictly-nearest-strike contract can report a stale/near-zero IV for
    # illiquid names (e.g. ACN), which collapses VRP to ~ -historical_volatility.
    # Walk outward to the nearest strike with a usable IV instead.
    atm_call_iv_row = _find_valid_atm_option(calls, underlying_price)
    atm_put_iv_row = _find_valid_atm_option(puts, underlying_price)
    atm_call_iv = float(atm_call_iv_row["impliedVolatility"]) if atm_call_iv_row is not None else None
    atm_put_iv = float(atm_put_iv_row["impliedVolatility"]) if atm_put_iv_row is not None else None
    current_atm_iv = atm_call_iv if atm_call_iv is not None else atm_put_iv

    # See _chain_iv_is_degenerate: reject the whole chain's IV rather than
    # trust a value that only looks plausible in isolation.
    all_ivs = list(calls.get("impliedVolatility", [])) + list(puts.get("impliedVolatility", []))
    if _chain_iv_is_degenerate(all_ivs):
        current_atm_iv = None
        atm_call_iv = None
        atm_put_iv = None

    expected_move = None
    if atm_call_last_price is not None and atm_put_last_price is not None:
        expected_move = atm_call_last_price + atm_put_last_price

    volatility_risk_premium = None
    if current_atm_iv is not None and historical_volatility is not None:
        volatility_risk_premium = current_atm_iv - historical_volatility

    if record_iv_history:
        iv_52w_low, iv_52w_high = _update_iv_history_and_get_range(db, ticker, current_atm_iv)
    else:
        iv_52w_low, iv_52w_high = _get_iv_history_range(db, ticker)
    result = compute_options_metrics(options_chain, spot=underlying_price, current_iv=current_atm_iv,
                                      iv_52w_low=iv_52w_low, iv_52w_high=iv_52w_high)
    result.update({
        "ticker": ticker,
        "expiration": expiration,
        "underlying_price": underlying_price,
        "historical_volatility": historical_volatility,
        "current_atm_iv": current_atm_iv,
        "volatility_risk_premium": volatility_risk_premium,
        "expected_move": expected_move,
        "atm_strike": atm_strike,
        "volume_put_call_ratio": volume_pcr,
        "open_interest_put_call_ratio": oi_pcr,
        "total_call_oi": int(total_call_oi),
        "total_put_oi": int(total_put_oi),
        "call_premium_notional": call_premium_notional,
        "put_premium_notional": put_premium_notional,
        "total_call_gex": total_call_gex,
        "total_put_gex": total_put_gex,
        "total_gex": total_gex,
        "call_wall": call_wall,
        "put_wall": put_wall,
        # GEX magnitude at each wall strike -- lets a caller (e.g. the
        # per-expiration Structural Levels card) render the same "Call Wall
        # GEX: N" detail the nightly batch analysis provides, without a
        # second request. There's no equivalent cumulative-GEX-at-flip-level
        # figure here (see key_levels.zero_gamma, which is often an
        # interpolated price between two strikes rather than a real one, so
        # "cumulative GEX at that exact level" isn't well-defined the way it
        # is for a wall).
        "call_wall_gex": call_wall_gex,
        "put_wall_gex": put_wall_gex,
        "iv_smile": iv_smile,
        "unusual_volume": unusual_volume,
        "next_earnings_date": next_earnings_date,
        "max_pain_strike": max_pain_strike,
        "max_pain_distance_pct": (
            ((underlying_price - max_pain_strike) / max_pain_strike) * 100.0
            if max_pain_strike not in (None, 0)
            else None
        ),
        # gamma/delta/vanna/charm are all Black-Scholes estimates from
        # strike + IV + time-to-expiry, not values reported by the options
        # exchange -- Yahoo's public chain doesn't reliably supply any of
        # them. Surfaced explicitly so a real-vs-modeled distinction is never
        # silently lost between backend and UI.
        "greeks_methodology": "black_scholes_derived",
        # When this was actually computed -- distinct from how fresh the
        # underlying data is (this fetch could be live or a 7-day-old Redis
        # cache hit at the API layer). Naive UTC, same convention as
        # fetched_at elsewhere in this codebase.
        "computed_at": datetime.utcnow().isoformat(),
        "schema_version": OPTIONS_METRICS_SCHEMA_VERSION,
    })
    return result


def get_iv_term_structure(ticker: str, max_expirations: int = 8) -> List[Dict[str, Any]]:
    """ATM IV across the nearest `max_expirations` expirations -- a live
    cross-section of IV vs time-to-expiry (contango/backwardation), not a
    per-ticker history the way iv_history is. Each expiration needs its own
    yfinance option_chain() fetch, so this costs up to `max_expirations`
    live round-trips (rate-limited between each, same as everywhere else in
    this module) -- meaningfully more expensive than every other function
    here, which do one fetch. Callers should treat this as an explicitly
    on-demand chart, not something bundled into the default metrics payload.

    Deliberately does not touch iv_history (see
    calculate_options_metrics' record_iv_history docstring) -- writing N
    different expirations' IV into that single per-ticker-per-day series
    would corrupt it far worse than a single non-default expiration would.
    """
    from .yfinance_service import YFinanceService
    import yfinance as yf

    svc = YFinanceService()
    try:
        from .yf_session import get_session
        session = get_session()
    except Exception:
        session = None
    yf_ticker = yf.Ticker(ticker, session=session) if session is not None else yf.Ticker(ticker)

    expirations = list(getattr(yf_ticker, "options", []) or [])[:max_expirations]
    if not expirations:
        return []

    svc._wait_for_yfinance_rate_limit()
    history = svc.get_historical_data(ticker, period="1mo", interval="1d", use_cache=False)
    underlying_price = None
    if history is not None and not history.empty and "Close" in history.columns:
        closes = history["Close"].dropna()
        if not closes.empty:
            underlying_price = _safe_float(closes.iloc[-1])
    if not underlying_price:
        info = getattr(yf_ticker, "info", {}) or {}
        underlying_price = _safe_float(
            info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        )
    if not underlying_price:
        raise ValueError(f"Unable to determine underlying price for {ticker}")

    today = datetime.now(_ET).date()
    curve: List[Dict[str, Any]] = []
    for expiration in expirations:
        svc._wait_for_yfinance_rate_limit()
        try:
            chain = yf_ticker.option_chain(expiration)
        except Exception:
            continue

        calls = chain.calls if hasattr(chain, "calls") else pd.DataFrame()
        puts = chain.puts if hasattr(chain, "puts") else pd.DataFrame()
        call_row = _find_valid_atm_option(calls, underlying_price)
        put_row = _find_valid_atm_option(puts, underlying_price)
        call_iv = _safe_float(call_row["impliedVolatility"]) if call_row is not None else None
        put_iv = _safe_float(put_row["impliedVolatility"]) if put_row is not None else None
        atm_iv = call_iv if call_iv is not None else put_iv
        if atm_iv is None:
            continue

        try:
            exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
            days_to_expiry = max(0, (exp_date - today).days)
        except ValueError:
            days_to_expiry = None

        curve.append({
            "expiration": expiration,
            "atm_iv": atm_iv,
            "days_to_expiry": days_to_expiry,
        })

    return curve


def compute_options_metrics(options_chain: List[Dict[str, Any]], spot: float, current_iv: Optional[float] = None,
                            iv_52w_low: Optional[float] = None, iv_52w_high: Optional[float] = None) -> Dict[str, Any]:
    """High-level composer that returns aggregated metrics and per-strike exposures."""
    strike_agg = aggregate_by_strike(options_chain, spot)
    key_levels = compute_key_gamma_levels(strike_agg)
    net = compute_net_exposures(strike_agg)
    ivr = None
    if current_iv is not None:
        ivr = compute_ivr(
            float(current_iv),
            float(iv_52w_low) if iv_52w_low is not None else None,
            float(iv_52w_high) if iv_52w_high is not None else None,
        )

    skew = compute_skew(options_chain, target_delta=0.25)

    return {
        "key_levels": key_levels,
        "net": net,
        "ivr": ivr,
        "skew": skew,
        "strikes": list(strike_agg.values()),
    }
