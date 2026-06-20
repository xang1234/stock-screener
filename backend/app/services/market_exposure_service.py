"""Market exposure ("when to be aggressive") computation.

A transparent, rules-based 0-100 recommended-exposure score for a market,
blended from inputs already in the DB:

  * index OHLCV (distribution days, follow-through day, 50/200-DMA, trend)
  * market breadth (net 4% movers)
  * VIX (US only)

The rubric is intentionally a set of module-level constants — it is a tuning
problem, not an architecture one. Each score contribution is recorded in
``components`` so the UI can show *why* (vs IBD's black box).

Compute-once / store / read-many: the pipeline task calls ``compute_and_store``
daily; the Daily Snapshot payloads (live + static) call ``build_exposure_payload``.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from ..models.market_breadth import MarketBreadth
from ..models.market_exposure import MarketExposure
from ..models.stock import StockPrice

# --- Distribution day detection -------------------------------------------
DISTRIBUTION_LOOKBACK = 25       # rolling sessions
DISTRIBUTION_DOWN_THRESHOLD = 0.002  # index down >= 0.2%; volume > prior session

# --- Follow-through day (v1 heuristic — see detect_follow_through_day) ------
FTD_GAIN_PCT = 0.015     # >= +1.5% up day
FTD_WINDOW = 15          # look back this many sessions for a qualifying day
FTD_LOW_LOOKBACK = 20    # window used to locate the "recent low"
FTD_MIN_RALLY_DAY = 4    # day 4+ off the recent low (heuristic proxy)

# --- Moving averages / trend ----------------------------------------------
MA_FAST = 50
MA_SLOW = 200

# --- Score rubric -----------------------------------------------------------
# Trend-led and continuous: the score is driven by how far the index sits above
# (or below) its 50/200-DMAs, so it tracks the index. Distribution days / VIX /
# breadth are bounded modifiers, not the primary gate. A couple of principled
# overlays keep the risk floor. Tuned + validated on SPY 2024-2026 (the new
# score correlates ~0.93 with the index's % distance above its 200-DMA).
TREND_BASE = 50.0               # neutral starting point

# Continuous distance-from-MA terms: points = (price/ma - 1) * GAIN, clamped.
DIST_200_GAIN = 250.0           # ~ +25 pts at 10% above the 200-DMA
DIST_200_CAP = 30.0
DIST_50_GAIN = 250.0
DIST_50_CAP = 12.0
MA_ALIGN_BONUS = 8.0            # +/- for 50-DMA above/below 200-DMA (golden/death cross)

# Distribution days: a DRAG above a baseline (the rolling-25 count sits ~5-6 on
# a normal index, so only the excess signals real institutional selling).
DIST_BASELINE = 3
DIST_DRAG_PER_DAY = 3.0
DIST_DRAG_MAX = 20.0

NET_4PCT_NEG_PENALTY = 6.0      # net 4% movers negative
VIX_ELEVATED = 20.0
VIX_ELEVATED_PENALTY = 8.0
VIX_HIGH = 30.0
VIX_HIGH_PENALTY = 10.0

# Principled risk overlays (hard ceilings — rare, not the main driver).
CAP_BELOW_200DMA = 50.0         # never "aggressive" below the 200-DMA
DIST_HEAVY = 8                  # genuinely heavy distribution
CAP_HEAVY_DISTRIBUTION = 45.0

FTD_FLOOR = 45.0                # recent FTD after a correction raises the floor

# --- History seeding (one-time, so the timeline renders on launch) ----------
EXPOSURE_HISTORY_MIN_ROWS = 60   # below this many stored rows -> seed history
EXPOSURE_BACKFILL_DAYS = 220     # trailing calendar days to seed (~150 sessions)

# Stance bands, highest lower-bound first.
STANCE_BANDS = [
    (85.0, "Power Trend"),
    (65.0, "Confirmed Uptrend"),
    (50.0, "Uptrend Under Pressure"),
    (30.0, "Downtrend/Caution"),
    (0.0, "Correction — In Cash"),
]


def _f(value) -> Optional[float]:
    """Coerce a numpy/pandas scalar to a plain float, or None when missing."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def count_distribution_days(
    ohlcv_df: pd.DataFrame,
    lookback: int = DISTRIBUTION_LOOKBACK,
    down_threshold: float = DISTRIBUTION_DOWN_THRESHOLD,
) -> int:
    """Count distribution days in the trailing ``lookback`` window.

    A distribution day is a session where Close fell >= ``down_threshold`` vs
    the prior Close AND Volume exceeded the prior session's Volume. Pure
    function over an OHLCV DataFrame (columns Open/High/Low/Close/Volume).
    Returns 0 on empty/insufficient data.
    """
    if ohlcv_df is None or len(ohlcv_df) < 2:
        return 0
    pct = ohlcv_df["Close"].pct_change()
    vol_up = ohlcv_df["Volume"] > ohlcv_df["Volume"].shift(1)
    mask = (pct <= -down_threshold) & vol_up
    return int(mask.tail(lookback).sum())


def detect_follow_through_day(
    ohlcv_df: pd.DataFrame,
    gain_pct: float = FTD_GAIN_PCT,
    window: int = FTD_WINDOW,
    low_lookback: int = FTD_LOW_LOOKBACK,
    min_rally_day: int = FTD_MIN_RALLY_DAY,
) -> tuple[bool, Optional[date]]:
    """v1 HEURISTIC follow-through-day flag. Returns (found, date_of_ftd).

    CEILING: this is NOT a true IBD follow-through detector — it does not track
    a confirmed rally attempt with a state machine. It flags the most recent
    session within ``window`` where Close rose >= ``gain_pct`` on rising volume,
    occurring at least ``min_rally_day`` sessions after the trailing
    ``low_lookback`` low. It is used only to *raise the score floor* after a
    correction, never to lift the score on its own. Full rally-attempt FSM is v2.
    """
    if ohlcv_df is None or len(ohlcv_df) < min_rally_day + 1:
        return False, None
    close = ohlcv_df["Close"]
    pct = close.pct_change()
    vol_up = ohlcv_df["Volume"] > ohlcv_df["Volume"].shift(1)
    strong_up = (pct >= gain_pct) & vol_up

    low_window = close.tail(low_lookback)
    if low_window.empty:
        return False, None
    low_label = low_window.index[int(low_window.values.argmin())]
    low_pos = ohlcv_df.index.get_loc(low_label)

    candidates = strong_up.tail(window)
    for ts in reversed(list(candidates.index)):
        if not bool(candidates.loc[ts]):
            continue
        if ohlcv_df.index.get_loc(ts) - low_pos >= min_rally_day:
            return True, ts.date()
    return False, None


def compute_trend(ohlcv_df: pd.DataFrame) -> dict:
    """Return {price, ma50, ma200, trend} from the index OHLCV.

    MAs are None when there is insufficient history (<200 rows for the slow MA),
    in which case the MA-based score caps are skipped (neutral).
    """
    if ohlcv_df is None or ohlcv_df.empty:
        return {"price": None, "ma50": None, "ma200": None, "trend": None}
    close = ohlcv_df["Close"]
    price = _f(close.iloc[-1])
    ma50 = _f(close.rolling(MA_FAST).mean().iloc[-1]) if len(close) >= MA_FAST else None
    ma200 = _f(close.rolling(MA_SLOW).mean().iloc[-1]) if len(close) >= MA_SLOW else None

    trend = "neutral"
    if None not in (price, ma50, ma200):
        if price > ma50 > ma200:
            trend = "bullish"
        elif price < ma50 < ma200:
            trend = "bearish"
    return {"price": price, "ma50": ma50, "ma200": ma200, "trend": trend}


def _score(trend: dict, dist_count: int, ftd: bool,
           vix: Optional[float], net_4pct: Optional[int]) -> tuple[float, dict]:
    """Blend inputs into a 0-100 score. Returns (score, components).

    Trend-led: a continuous core from the index's distance above/below its
    50/200-DMAs (so the score tracks the index), then bounded modifiers
    (distribution drag, VIX, breadth), then principled risk overlays and the
    FTD floor. Every contribution is recorded in ``components`` for the
    transparent "why". MAs are None when history is short (<200 rows for the
    slow MA); those terms are skipped, leaving a neutral core.
    """
    price, ma50, ma200 = trend.get("price"), trend.get("ma50"), trend.get("ma200")
    core = TREND_BASE
    components: dict = {"base": TREND_BASE}

    # Trend core — continuous distance from the moving averages (tracks the index)
    if price is not None and ma200:
        d200 = max(-DIST_200_CAP, min(DIST_200_CAP, (price / ma200 - 1) * DIST_200_GAIN))
        core += d200
        components["dist_from_200dma"] = round(d200, 1)
    if price is not None and ma50:
        d50 = max(-DIST_50_CAP, min(DIST_50_CAP, (price / ma50 - 1) * DIST_50_GAIN))
        core += d50
        components["dist_from_50dma"] = round(d50, 1)
    if ma50 is not None and ma200 is not None:
        align = MA_ALIGN_BONUS if ma50 >= ma200 else -MA_ALIGN_BONUS
        core += align
        components["ma_alignment"] = align

    # Bounded modifiers — texture within the trend regime, never the main gate
    if dist_count > DIST_BASELINE:
        drag = -min(DIST_DRAG_MAX, DIST_DRAG_PER_DAY * (dist_count - DIST_BASELINE))
        core += drag
        components["distribution_drag"] = drag
    if net_4pct is not None and net_4pct < 0:
        core -= NET_4PCT_NEG_PENALTY
        components["net4pct_penalty"] = -NET_4PCT_NEG_PENALTY
    if vix is not None and vix > VIX_ELEVATED:
        core -= VIX_ELEVATED_PENALTY
        components["vix_elevated_penalty"] = -VIX_ELEVATED_PENALTY
    if vix is not None and vix > VIX_HIGH:
        core -= VIX_HIGH_PENALTY
        components["vix_high_penalty"] = -VIX_HIGH_PENALTY

    # Principled risk overlays (rare hard ceilings)
    if price is not None and ma200 is not None and price < ma200:
        core = min(core, CAP_BELOW_200DMA)
        components["below_200dma_cap"] = CAP_BELOW_200DMA
    if dist_count >= DIST_HEAVY:
        core = min(core, CAP_HEAVY_DISTRIBUTION)
        components["heavy_distribution_cap"] = CAP_HEAVY_DISTRIBUTION

    # FTD recovery floor — applied last
    if ftd and price is not None and ma50 is not None and price <= ma50:
        if core < FTD_FLOOR:
            components["ftd_floor"] = FTD_FLOOR
        core = max(core, FTD_FLOOR)

    return round(max(0.0, min(100.0, core)), 1), components


def _stance(score: float) -> str:
    for lower, label in STANCE_BANDS:
        if score >= lower:
            return label
    return STANCE_BANDS[-1][1]


def _vix_on_date(db: Session, as_of_date: date) -> Optional[float]:
    """VIX close for exactly ``as_of_date`` (None if no ^VIX bar that day).

    Exact-date only: a stale ^VIX close must not drive the VIX penalties for a
    different, later session.
    """
    row = (
        db.query(StockPrice)
        .filter(StockPrice.symbol == "^VIX", StockPrice.date == as_of_date)
        .first()
    )
    return _f(row.close) if row is not None else None


def _net_4pct_on_date(db: Session, market: str, as_of_date: date) -> Optional[int]:
    """Net 4% movers for exactly ``as_of_date`` (None if no breadth row that day).

    Exact-date only: the breadth penalty must reflect ``as_of_date``, not an
    earlier breadth regime — otherwise an exposure backfill run without a
    matching breadth backfill stamps every historical row with today's breadth.
    """
    row = (
        db.query(MarketBreadth)
        .filter(MarketBreadth.market == market, MarketBreadth.date == as_of_date)
        .first()
    )
    if row is None:
        return None
    return int((row.stocks_up_4pct or 0) - (row.stocks_down_4pct or 0))


def compute_exposure(market: str, as_of_date: date, db: Session) -> dict:
    """Compute the exposure dict for one market as of ``as_of_date``.

    Returns ``{"error": ...}`` (no row written) when index OHLCV is unavailable,
    so the pipeline guard treats it as a failure (mirrors the breadth task).
    """
    market = (market or "US").upper()

    # Benchmark service uses its own SessionLocal/Redis — do NOT pass the task's
    # db (it closes the session in a finally block).
    from .benchmark_cache_service import BenchmarkCacheService

    bundle = BenchmarkCacheService().get_benchmark_bundle(market=market, period="2y")
    if bundle is None or bundle.data is None or bundle.data.empty:
        return {"error": "no_benchmark_data", "market": market, "date": as_of_date.isoformat()}

    # Slice "as of" — tz-agnostic date mask (handles naive + tz-aware indexes).
    df = bundle.data[bundle.data.index.date <= as_of_date]
    if df.empty:
        return {"error": "no_benchmark_data", "market": market, "date": as_of_date.isoformat()}
    # The most recent bar must BE as_of_date. Otherwise we'd write a row dated
    # as_of_date scored from a prior session's close/volume — a fresh date backed
    # by stale index inputs (happens when the benchmark refresh lags the date).
    if df.index[-1].date() != as_of_date:
        return {"error": "benchmark_not_current", "market": market, "date": as_of_date.isoformat()}

    trend = compute_trend(df)
    dist = count_distribution_days(df)
    ftd, ftd_date = detect_follow_through_day(df)
    vix = _vix_on_date(db, as_of_date) if market == "US" else None
    net_4pct = _net_4pct_on_date(db, market, as_of_date)

    score, components = _score(trend, dist, ftd, vix, net_4pct)

    return {
        "market": market,
        "date": as_of_date,
        "exposure_score": score,
        "stance": _stance(score),
        "benchmark_price": trend["price"],
        "benchmark_ma50": trend["ma50"],
        "benchmark_ma200": trend["ma200"],
        "trend": trend["trend"],
        "distribution_day_count": dist,
        "follow_through_day": ftd,
        "follow_through_date": ftd_date,
        "vix": vix,
        "net_4pct": net_4pct,
        "components": components,
        "benchmark_symbol": bundle.benchmark_symbol,
    }


def compute_and_store(market: str, as_of_date: date, db: Session) -> dict:
    """Compute and upsert one MarketExposure row by (date, market).

    ``compute_exposure`` returns a dict whose keys are exactly MarketExposure
    columns (including market/date), so it is applied directly — there is no
    separate field mapping to keep in sync as the model grows.
    """
    result = compute_exposure(market, as_of_date, db)
    if result.get("error"):
        return result

    row = (
        db.query(MarketExposure)
        .filter(MarketExposure.date == as_of_date, MarketExposure.market == result["market"])
        .first()
    )
    if row is not None:
        for key, value in result.items():
            setattr(row, key, value)
    else:
        db.add(MarketExposure(**result))
    db.commit()
    return result


def build_exposure_payload(
    db: Session,
    market: str,
    history_days: int = 180,
    as_of_date: Optional[date] = None,
) -> Optional[dict]:
    """Shared reader for the Daily Snapshot payloads (live + static).

    Returns the latest stored row's headline + a ``history`` list of
    {date, exposure_score, stance} over the trailing ``history_days``. None when
    no rows exist yet (the UI renders a muted placeholder).

    ``as_of_date`` pins the payload to a date: the live snapshot omits it (uses
    the absolute latest row), while the static export passes the published run's
    date so the exposure section stays coherent with the rest of ``home.json``.
    """
    market = (market or "US").upper()
    latest_q = db.query(MarketExposure).filter(MarketExposure.market == market)
    if as_of_date is not None:
        # Pinned (static) mode: require the EXACT export date so the section stays
        # coherent with the exact-date-gated breadth/groups — omit the section
        # (None) rather than fall back to a stale earlier row.
        latest_q = latest_q.filter(MarketExposure.date == as_of_date)
    latest = latest_q.order_by(MarketExposure.date.desc()).first()
    if latest is None:
        return None

    start = latest.date - timedelta(days=history_days)
    rows = (
        db.query(MarketExposure)
        .filter(
            MarketExposure.market == market,
            MarketExposure.date >= start,
            MarketExposure.date <= latest.date,
        )
        .order_by(MarketExposure.date.asc())
        .all()
    )
    # follow_through marks the actual FTD *event* day: the row whose detected
    # follow_through_date is its own date (later rows still see that FTD in their
    # trailing window, but only the event day has date == follow_through_date).
    history = [
        {
            "date": r.date.isoformat(),
            "exposure_score": r.exposure_score,
            "stance": r.stance,
            "follow_through": bool(r.follow_through_day and r.follow_through_date == r.date),
        }
        for r in rows
    ]
    return {
        "market": market,
        "date": latest.date.isoformat(),
        "exposure_score": latest.exposure_score,
        "stance": latest.stance,
        "distribution_day_count": latest.distribution_day_count,
        "follow_through_day": latest.follow_through_day,
        "trend": latest.trend,
        "vix": latest.vix,
        "benchmark_symbol": latest.benchmark_symbol,
        "components": latest.components,
        "history": history,
    }


def backfill_exposure(db: Session, market: str, start: date, end: date) -> dict:
    """Compute + store exposure for every trading day in [start, end].

    The single source of truth for range backfills (used by the Celery backfill
    task and the daily self-heal). Idempotent — compute_and_store upserts.
    Trading-day enumeration lives in MarketCalendarService, not here.
    """
    from .market_calendar_service import MarketCalendarService

    market = (market or "US").upper()
    seeded = failed = 0
    for day in MarketCalendarService().trading_days(market, start, end):
        try:
            if compute_and_store(market, day, db).get("error"):
                failed += 1
            else:
                seeded += 1
        except Exception:
            db.rollback()
            failed += 1
    return {"seeded": seeded, "failed": failed}


def ensure_exposure_history(
    db: Session,
    market: str,
    *,
    min_rows: int = EXPOSURE_HISTORY_MIN_ROWS,
    days: int = EXPOSURE_BACKFILL_DAYS,
) -> dict:
    """Seed history once so the timeline isn't empty on launch.

    If the market has fewer than ``min_rows`` stored rows, backfill the trailing
    ``days`` window; otherwise no-op. Idempotent, so the daily pipeline calls
    this on every run and it self-heals after the first deploy with no manual
    step. ponytail: re-fetches the benchmark frame per day, but it's a one-time
    seed over a Redis-cached frame.
    """
    from .market_calendar_service import MarketCalendarService

    market = (market or "US").upper()
    existing = db.query(MarketExposure).filter(MarketExposure.market == market).count()
    if existing >= min_rows:
        return {"seeded": 0, "skipped": True}

    end = MarketCalendarService().last_completed_trading_day(market)
    return backfill_exposure(db, market, end - timedelta(days=days), end)
