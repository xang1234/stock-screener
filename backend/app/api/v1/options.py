from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

from ...database import get_db
from ...models.max_pain import MaxPainSnapshot
from ...models.gex import GexSnapshot
from ...models.options_metrics_snapshot import OptionsMetricsSnapshot
from ...services.options_metrics import (
    calculate_options_metrics,
    compute_options_metrics,
    compute_key_gamma_levels,
    get_iv_term_structure,
    OPTIONS_METRICS_SCHEMA_VERSION,
)
from ...services.options_snapshot_upsert import upsert_snapshot, trading_date_for
from ...services.symbol_format import normalize_symbol
from ...schemas.options_metrics import OptionsMetricsResponse

from ...services.yfinance_service import YFinanceService
import yfinance as yf
from ...wiring.bootstrap import get_redis_client
import json

logger = logging.getLogger(__name__)
router = APIRouter()

_ET = ZoneInfo("America/New_York")

# Bucket boundaries in calendar days from today (ET). An expiration falls
# into the first bucket whose upper bound it's <= to.
_EXPIRATION_BUCKETS = [
    ("this_week", 7),
    ("this_month", 35),
    ("this_quarter", 100),
    ("later", None),
]


def _is_zero_open_interest(total_call_oi: Optional[int], total_put_oi: Optional[int]) -> bool:
    """True when a freshly computed payload has zero open interest on both
    sides of the chain -- a known yfinance/Yahoo off-hours data-staleness
    pattern (OI reported as 0 before their daily snapshot refreshes), not a
    real market condition for a chain that otherwise has live volume/price
    data. Used to gate both the persisted-snapshot fallback and the
    "don't clobber a good historical row" persistence guard below.
    """
    return (total_call_oi or 0) == 0 and (total_put_oi or 0) == 0


def _options_metrics_from_snapshot(row: OptionsMetricsSnapshot) -> Dict[str, Any]:
    """Reconstruct an OptionsMetricsResponse-shaped payload from a persisted
    OptionsMetricsSnapshot row, for serving when live yfinance data is
    stale (see _is_zero_open_interest). Marked explicitly as fallback data
    -- callers must not treat this as a live reading."""
    return {
        "ticker": row.ticker,
        "expiration": row.expiration.isoformat() if row.expiration else None,
        "key_levels": {
            "call_wall": row.call_wall,
            "put_wall": row.put_wall,
            "zero_gamma": row.zero_gamma,
        },
        # NetExposures requires non-optional floats -- coerce, since the
        # columns themselves are nullable (e.g. an older/incomplete row).
        # An unhandled ValidationError here would 500 the whole request,
        # which is worse than the zeros this fallback exists to avoid.
        "net": {
            "net_dex": row.net_dex or 0.0,
            "net_vex": row.net_vex or 0.0,
            "net_cex": row.net_cex or 0.0,
        },
        "ivr": row.ivr,
        "skew": row.skew,
        "strikes": row.strikes_json or [],
        "volume_put_call_ratio": row.volume_put_call_ratio,
        "open_interest_put_call_ratio": row.open_interest_put_call_ratio,
        "total_call_oi": row.total_call_oi,
        "total_put_oi": row.total_put_oi,
        "call_premium_notional": row.call_premium_notional,
        "put_premium_notional": row.put_premium_notional,
        "underlying_price": row.underlying_price,
        "historical_volatility": row.historical_volatility,
        "current_atm_iv": row.current_atm_iv,
        "volatility_risk_premium": row.volatility_risk_premium,
        "expected_move": row.expected_move,
        "atm_strike": row.atm_strike,
        "total_call_gex": row.total_call_gex,
        "total_put_gex": row.total_put_gex,
        "total_gex": row.total_gex,
        "call_wall_gex": row.call_wall_gex,
        "put_wall_gex": row.put_wall_gex,
        "iv_smile": row.iv_smile_json,
        "unusual_volume": row.unusual_volume_json or [],
        "next_earnings_date": row.next_earnings_date.isoformat() if row.next_earnings_date else None,
        "max_pain_strike": row.max_pain_strike,
        "max_pain_distance_pct": row.max_pain_distance_pct,
        "greeks_methodology": row.greeks_methodology,
        "computed_at": row.fetched_at.isoformat() if row.fetched_at else None,
        "schema_version": None,  # not a live payload -- never satisfies the cache-hit gate above
        "data_source": "persisted_fallback",
        "is_stale_fallback": True,
        "snapshot_trading_date": row.trading_date.isoformat() if row.trading_date else None,
        "snapshot_fetched_at": row.fetched_at.isoformat() if row.fetched_at else None,
    }


def _bucket_expirations(expirations: List[str], today_et) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {name: [] for name, _ in _EXPIRATION_BUCKETS}
    for exp in expirations:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        except ValueError:
            continue
        days = (exp_date - today_et).days
        for name, upper_bound in _EXPIRATION_BUCKETS:
            if upper_bound is None or days <= upper_bound:
                buckets[name].append(exp)
                break
    return buckets


@router.get("/expirations/{symbol}")
async def get_options_expirations(symbol: str) -> dict:
    """List available expirations for a symbol, bucketed into This Week /
    This Month / This Quarter / Later (by calendar days from today, ET).

    Cheap relative to the analysis endpoints -- yfinance's `ticker.options`
    is just the expiration date list, no chain data. Used to populate an
    expiration selector before the caller fetches the (expensive) chain for
    whichever specific date they pick.
    """
    symbol = symbol.upper()
    cache_key = f"options_expirations:{symbol}"

    try:
        redis_client = get_redis_client()
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    try:
        svc = YFinanceService()
        try:
            svc._wait_for_yfinance_rate_limit()
        except Exception:
            pass

        try:
            from ...services.yf_session import get_session
            session = get_session()
        except Exception:
            session = None

        ticker = yf.Ticker(symbol, session=session) if session is not None else yf.Ticker(symbol)
        expirations = list(getattr(ticker, "options", []) or [])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not expirations:
        raise HTTPException(status_code=404, detail=f"No options data found for {symbol}")

    today_et = datetime.now(_ET).date()
    result = {
        "symbol": symbol,
        "as_of": datetime.utcnow().isoformat(),
        "all_expirations": expirations,
        "buckets": _bucket_expirations(expirations, today_et),
    }

    try:
        redis_client = get_redis_client()
        redis_client.setex(cache_key, 3600, json.dumps(result))
    except Exception:
        pass

    return result


@router.get("/iv-term-structure/{symbol}")
async def get_iv_term_structure_endpoint(
    symbol: str,
    max_expirations: int = Query(8, ge=2, le=12, description="Number of nearest expirations to sample"),
) -> dict:
    """ATM IV across the nearest expirations -- a live cross-section
    (contango/backwardation), not a history. See get_iv_term_structure()'s
    docstring for why this isn't cheap: it's `max_expirations` live yfinance
    option-chain fetches, not one.

    Cached in Redis for 30 minutes (short vs. e.g. /expirations' 1h TTL,
    since this reflects live positioning rather than a static expiration
    list) so repeat views of the same ticker don't re-pay that cost.
    """
    normalized_symbol = normalize_symbol(symbol)
    if normalized_symbol is None:
        raise HTTPException(status_code=422, detail=f"Invalid symbol format: {symbol!r}")

    cache_key = f"iv_term_structure:{normalized_symbol}:{max_expirations}"
    try:
        redis_client = get_redis_client()
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    try:
        curve = get_iv_term_structure(normalized_symbol, max_expirations=max_expirations)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result = {
        "symbol": normalized_symbol,
        "as_of": datetime.utcnow().isoformat(),
        "curve": curve,
    }

    try:
        redis_client = get_redis_client()
        redis_client.setex(cache_key, 1800, json.dumps(result))
    except Exception:
        pass

    return result


@router.get("/term-structure/{symbol}")
async def get_options_term_structure(
    symbol: str,
    expiration: str = Query(..., description="Expiration date, YYYY-MM-DD, from /expirations/{symbol}"),
    db: Session = Depends(get_db),
) -> dict:
    """Max Pain + GEX for one specific expiration, computed live from today's
    chain (not the daily batch's nearest-expiration snapshot).

    This is the "term structure" building block: call it once per expiration
    a user selects (This Week / This Month / This Quarter / a specific date)
    to compare positioning across expirations, all computed from today's
    data -- see calculate_options_metrics() for why this is a cross-section,
    not a forecast.

    Every call persists its result to max_pain_snapshots/gex_snapshots
    (upserted on ticker+trading_date+expiration), so /v1/max-pain/history and
    /v1/gex/history filtered to this expiration accumulate real history for
    whichever ticker+expiration combinations get viewed -- there's no
    universe-wide daily batch fetching every expiration for every ticker,
    which would multiply yfinance load by however many expirations are
    tracked. History for an expiration only starts from the first time it's
    viewed here.
    """
    normalized_symbol = normalize_symbol(symbol)
    if normalized_symbol is None:
        raise HTTPException(status_code=422, detail=f"Invalid symbol format: {symbol!r}")

    try:
        expiration_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid expiration format: {expiration!r}, expected YYYY-MM-DD")

    try:
        # expiration here is always an explicit, user-picked date (never the
        # nearest-expiration default) -- must not feed iv_history, or a
        # far-dated expiration's IV would overwrite today's IV Rank data
        # point with an unrelated value. See calculate_options_metrics'
        # record_iv_history docstring.
        metrics = calculate_options_metrics(normalized_symbol, expiration, db=db, record_iv_history=False)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    fetched_at = datetime.utcnow()
    trading_date = trading_date_for(fetched_at)
    spot = metrics.get("underlying_price")

    try:
        upsert_snapshot(
            db,
            MaxPainSnapshot,
            ticker=normalized_symbol,
            trading_date=trading_date,
            expiration=expiration_date,
            values={
                "company_name": None,
                "status": "OK",
                "error": None,
                "max_pain_strike": metrics.get("max_pain_strike"),
                "call_oi": metrics.get("total_call_oi"),
                "put_oi": metrics.get("total_put_oi"),
                "put_call_ratio": metrics.get("open_interest_put_call_ratio"),
                "last_price": spot,
                "distance_pct": metrics.get("max_pain_distance_pct"),
                "batch_id": None,
                "fetched_at": fetched_at,
            },
        )
        flip_level = metrics.get("key_levels", {}).get("zero_gamma")
        distance_to_flip_pct = (
            ((spot - flip_level) / flip_level) * 100.0
            if spot is not None and flip_level not in (None, 0)
            else None
        )
        upsert_snapshot(
            db,
            GexSnapshot,
            ticker=normalized_symbol,
            trading_date=trading_date,
            expiration=expiration_date,
            values={
                "company_name": None,
                "status": "OK",
                "error": None,
                "spot_price": spot,
                "call_oi": metrics.get("total_call_oi"),
                "put_oi": metrics.get("total_put_oi"),
                "call_gex": metrics.get("total_call_gex"),
                "put_gex": metrics.get("total_put_gex"),
                "total_gex": metrics.get("total_gex"),
                "flip_level": flip_level,
                "distance_to_flip_pct": distance_to_flip_pct,
                "batch_id": None,
                "fetched_at": fetched_at,
            },
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception(
            "Failed to persist term-structure snapshot for %s %s", normalized_symbol, expiration
        )

    return {
        "symbol": normalized_symbol,
        "expiration": expiration,
        "trading_date": trading_date.isoformat(),
        **metrics,
    }


@router.post("/metrics", response_model=OptionsMetricsResponse)
async def post_options_metrics(payload: Dict[str, Any], db: Session = Depends(get_db)):
  """Compute options exposure metrics.

  Payload may either include `options_chain` (list) or a `symbol` to fetch
  options from yfinance (nearest expiry). Optional `current_iv`,
  `iv_52w_low`, `iv_52w_high` may be provided; when missing and `symbol`
  is present, the endpoint will attempt to infer them from yfinance.

  When supplying `options_chain` directly (no `symbol`), also supply
  `underlying_price` -- GEX/DEX/VEX/CEX are dollar exposures and require the
  spot price to compute; there's no other way to derive it from a
  caller-supplied chain.
  """
  try:
    options_chain: Optional[List[Dict[str, Any]]] = payload.get("options_chain")
    symbol: Optional[str] = payload.get("symbol")

    if not options_chain and not symbol:
      raise HTTPException(status_code=422, detail="Missing options_chain or symbol in payload")

    # If symbol provided and options_chain absent, try cache then compute live.
    # (Previously this also called a now-removed _options_from_yfinance()
    # pre-fetch here that read a `gamma`/`delta` column yfinance doesn't
    # actually provide -- it always came back zeroed -- and whose result was
    # then thrown away anyway once `expiration` resolved below and
    # calculate_options_metrics() took over. That was two extra live yfinance
    # round-trips wasted on every cache miss for no effect.)
    if not options_chain and symbol:
      try:
        redis_client = get_redis_client()
        key = f"options_metrics:{symbol.upper()}"
        cached = redis_client.get(key)
        if cached:
          cached_data = json.loads(cached)
          # This same key is also written by the nightly batch task
          # (options_analysis_tasks.py::analyze_options_exposure), which
          # only calls compute_options_metrics() -- key_levels/net/ivr/skew/
          # strikes -- never the fuller calculate_options_metrics() below,
          # since it doesn't fetch the 1mo price history or per-strike
          # lastPrice/volume that HV/VRP/expected_move/premium notionals
          # need. That abbreviated payload has no `schema_version` at all,
          # so the check below already rejects it same as any outdated one.
          #
          # Comparing the whole schema_version (not just checking for one
          # field's presence, the previous approach) also catches a FULL
          # payload cached by an older version of calculate_options_metrics
          # -- e.g. entries cached before iv_smile/unusual_volume existed
          # had underlying_price and would have passed a presence check,
          # silently missing those fields for up to their full 7-day TTL.
          # Fall through to a live compute instead; that live compute
          # re-caches a current payload below, self-healing this key for
          # the next 7 days.
          #
          # A cached entry with zero OI on both sides is never trusted here
          # either, schema_version match or not -- it's the same
          # off-hours yfinance staleness pattern (see
          # _is_zero_open_interest), and simply returning it would bypass
          # the persisted-snapshot fallback entirely for up to the full
          # 7-day TTL. Falling through re-runs the live-compute branch
          # below, which applies that fallback itself.
          if cached_data.get("schema_version") == OPTIONS_METRICS_SCHEMA_VERSION and not _is_zero_open_interest(
            cached_data.get("total_call_oi"), cached_data.get("total_put_oi")
          ):
            return cached_data
      except Exception:
        # cache miss or error — fall back to live fetch
        pass

    expiration = payload.get("expiration")
    # Whether the caller asked for a specific expiration vs. let this default
    # to the nearest one -- iv_history has no expiration dimension, so only
    # the nearest-expiration case should feed it (see calculate_options_metrics'
    # record_iv_history docstring); an explicit non-default expiration must
    # not overwrite that day's IV Rank data point with an unrelated value.
    explicit_expiration = expiration is not None
    current_iv = payload.get("current_iv")
    iv_52w_low = payload.get("iv_52w_low")
    iv_52w_high = payload.get("iv_52w_high")

    if symbol and not options_chain and not expiration:
      try:
        ticker = yf.Ticker(symbol)
        expirations = getattr(ticker, 'options', []) or []
        if expirations:
          expiration = expirations[0]
      except Exception:
        expiration = None
      if not expiration:
        raise HTTPException(status_code=404, detail=f"No options data found for symbol {symbol}")

    if symbol and not options_chain and expiration:
      result = calculate_options_metrics(symbol, expiration, db=db, record_iv_history=not explicit_expiration)
      if not explicit_expiration:
        is_stale = _is_zero_open_interest(result.get("total_call_oi"), result.get("total_put_oi"))

        if is_stale:
          # Live yfinance data has zero OI on both sides -- almost always an
          # off-hours/pre-market data-staleness artifact (see
          # _is_zero_open_interest), not a real market condition. Serve the
          # most recent persisted snapshot that actually had real OI instead
          # of a wall of zeros, and deliberately skip the Redis re-cache and
          # DB persist below for this reading -- caching or persisting it
          # would either poison the 7-day cache with garbage or clobber a
          # good historical row for this same (ticker, trading_date,
          # expiration).
          fallback_row = (
            db.query(OptionsMetricsSnapshot)
            .filter(
              OptionsMetricsSnapshot.ticker == symbol.upper(),
              or_(OptionsMetricsSnapshot.total_call_oi > 0, OptionsMetricsSnapshot.total_put_oi > 0),
            )
            .order_by(OptionsMetricsSnapshot.fetched_at.desc())
            .first()
          )
          if fallback_row is not None:
            fallback_payload = _options_metrics_from_snapshot(fallback_row)
            # Price itself isn't subject to the same OI-staleness pattern --
            # prefer the fresh live quote over the snapshot's if we have one.
            if result.get("underlying_price") is not None:
              fallback_payload["underlying_price"] = result["underlying_price"]
            return fallback_payload
          # No usable snapshot to fall back to -- return the honest (zero)
          # live reading rather than fabricate data, but still flag it so
          # the frontend can explain why the numbers look off instead of
          # presenting them as normal.
          result["data_source"] = "live_zero_oi"
          result["is_stale_fallback"] = False
          return result

        # Re-cache the complete payload under the same key the batch task
        # writes an abbreviated version to (see the cache-hit block above)
        # -- this is what actually self-heals a batch-only ticker's cache
        # to the full payload for the rest of its 7-day TTL. Only for the
        # default/nearest-expiration view, matching that key's lack of an
        # expiration dimension.
        try:
          redis_client = get_redis_client()
          redis_client.set(f"options_metrics:{symbol.upper()}", json.dumps(result), ex=7 * 24 * 3600)
        except Exception:
          pass

        # Durable history for the FULL payload -- this is the "live_full"
        # write path (see OptionsMetricsSnapshot's docstring), distinct from
        # the abbreviated rows the nightly batch writes. Only fires for the
        # default/nearest-expiration view, same scope as the Redis re-cache
        # above, so history stays aligned with what schema_version-gated
        # cache reads actually serve.
        try:
          fetched_at = datetime.utcnow()
          key_levels = result.get("key_levels") or {}
          net = result.get("net") or {}
          upsert_snapshot(
            db,
            OptionsMetricsSnapshot,
            ticker=symbol.upper(),
            trading_date=trading_date_for(fetched_at),
            expiration=datetime.strptime(expiration, "%Y-%m-%d").date() if expiration else None,
            values={
              "status": "OK",
              "error": None,
              "source": "live_full",
              "schema_version": result.get("schema_version"),
              "underlying_price": result.get("underlying_price"),
              "call_wall": result.get("call_wall") or key_levels.get("call_wall"),
              "call_wall_gex": result.get("call_wall_gex"),
              "put_wall": result.get("put_wall") or key_levels.get("put_wall"),
              "put_wall_gex": result.get("put_wall_gex"),
              "zero_gamma": key_levels.get("zero_gamma"),
              "total_call_gex": result.get("total_call_gex"),
              "total_put_gex": result.get("total_put_gex"),
              "total_gex": result.get("total_gex"),
              "net_dex": net.get("net_dex"),
              "net_vex": net.get("net_vex"),
              "net_cex": net.get("net_cex"),
              "ivr": result.get("ivr"),
              "skew": result.get("skew"),
              "historical_volatility": result.get("historical_volatility"),
              "current_atm_iv": result.get("current_atm_iv"),
              "volatility_risk_premium": result.get("volatility_risk_premium"),
              "expected_move": result.get("expected_move"),
              "atm_strike": result.get("atm_strike"),
              "volume_put_call_ratio": result.get("volume_put_call_ratio"),
              "open_interest_put_call_ratio": result.get("open_interest_put_call_ratio"),
              "total_call_oi": result.get("total_call_oi"),
              "total_put_oi": result.get("total_put_oi"),
              "call_premium_notional": result.get("call_premium_notional"),
              "put_premium_notional": result.get("put_premium_notional"),
              "max_pain_strike": result.get("max_pain_strike"),
              "max_pain_distance_pct": result.get("max_pain_distance_pct"),
              "next_earnings_date": (
                datetime.strptime(result["next_earnings_date"], "%Y-%m-%d").date()
                if result.get("next_earnings_date")
                else None
              ),
              "greeks_methodology": result.get("greeks_methodology"),
              "strikes_json": result.get("strikes"),
              "iv_smile_json": result.get("iv_smile"),
              "unusual_volume_json": result.get("unusual_volume"),
              "fetched_at": fetched_at,
            },
          )
          db.commit()
        except Exception:
          db.rollback()
          logger.exception("Failed to persist OptionsMetricsSnapshot (live_full) for %s", symbol)
      return result

    # If we have a symbol and any IV pieces are missing, try to fetch price range
    if symbol and (current_iv is None or iv_52w_low is None or iv_52w_high is None):
      try:
        svc = YFinanceService()
        pr = svc.get_price_range(symbol)
        if pr:
          if current_iv is None:
            # current_iv from impliedVol of near-the-money option is preferable,
            # but fall back to simple heuristic using last close volatility proxy
            current_iv = payload.get('current_iv') or None
          if iv_52w_low is None and pr.get('low_52w') is not None:
            iv_52w_low = pr.get('low_52w')
          if iv_52w_high is None and pr.get('high_52w') is not None:
            iv_52w_high = pr.get('high_52w')
      except Exception:
        pass

    underlying_price = payload.get("underlying_price")
    if underlying_price is None:
      raise HTTPException(
        status_code=422,
        detail="underlying_price is required when supplying options_chain directly",
      )

    result = compute_options_metrics(options_chain, spot=float(underlying_price), current_iv=current_iv,
                     iv_52w_low=iv_52w_low, iv_52w_high=iv_52w_high)
    # pydantic model will validate/serialize
    return result
  except HTTPException:
    raise
  except Exception as exc:
    raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/analysis")
async def post_options_analysis(payload: Dict[str, Any]):
  """Comprehensive options exposure analysis including structural levels.

  Returns Call Wall, Put Wall, Flip Level, spot price, and strike-by-strike GEX/VEX/CEX.

  Not called by the frontend (it uses GET /analysis/{symbol}, backed by
  analyze_options_exposure in options_analysis_tasks.py, which derives gamma
  via Black-Scholes). This POST variant instead reads `gamma`/`vanna`/`charm`
  straight off the raw yfinance option chain rows, columns yfinance doesn't
  actually populate -- they come back 0.0 here. Left unfixed since nothing
  currently exercises this path; if you wire something to it, route gamma
  through the same _bs_gamma_delta helper the GET path uses instead of
  trusting these columns.
  """
  import pandas as pd
  from datetime import datetime
  
  try:
    symbol: Optional[str] = payload.get("symbol")
    if not symbol:
      raise HTTPException(status_code=422, detail="Missing symbol in payload")
    
    # Fetch options for next 3 expirations
    svc = YFinanceService()
    try:
      svc._wait_for_yfinance_rate_limit()
    except Exception:
      pass
    
    ticker = yf.Ticker(symbol)
    expirations = getattr(ticker, 'options', []) or []
    if not expirations:
      raise HTTPException(status_code=404, detail=f"No options data found for {symbol}")
    
    # Get spot price
    hist = ticker.history(period="1d")
    if hist.empty:
      raise HTTPException(status_code=404, detail=f"No market data found for {symbol}")
    spot_price = float(hist["Close"].iloc[-1])
    
    # Fetch next 3 expirations
    all_options = []
    for exp_date in expirations[:3]:
      try:
        oc = ticker.option_chain(exp_date)
        
        # Process calls
        for _, r in oc.calls.iterrows():
          try:
            oi = int(r.get('openInterest', 0) or 0)
            if oi == 0:
              continue
            
            all_options.append({
              "strike": float(r.get('strike', 0.0)),
              "type": "call",
              "gamma": float(r.get('gamma', 0.0) or 0.0),
              "vanna": float(r.get('vanna', 0.0) or 0.0),
              "charm": float(r.get('charm', 0.0) or 0.0),
              "openInterest": oi,
            })
          except (ValueError, TypeError):
            continue
        
        # Process puts
        for _, r in oc.puts.iterrows():
          try:
            oi = int(r.get('openInterest', 0) or 0)
            if oi == 0:
              continue
            
            all_options.append({
              "strike": float(r.get('strike', 0.0)),
              "type": "put",
              "gamma": float(r.get('gamma', 0.0) or 0.0),
              "vanna": float(r.get('vanna', 0.0) or 0.0),
              "charm": float(r.get('charm', 0.0) or 0.0),
              "openInterest": oi,
            })
          except (ValueError, TypeError):
            continue
      except Exception:
        continue
    
    if not all_options:
      raise HTTPException(status_code=404, detail=f"No options with OI found for {symbol}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_options)
    
    # Calculate dealer exposures (Short Calls, Long Puts)
    S = spot_price
    exposures = []
    
    for _, row in df.iterrows():
      option_type = row["type"]
      gamma = row["gamma"]
      vanna = row["vanna"]
      charm = row["charm"]
      oi = row["openInterest"]
      strike = row["strike"]
      
      # Market maker assumption: short calls, long puts
      if option_type == "call":
        sign = 1
      else:
        sign = -1
      
      call_gex = -sign * gamma * oi * 100 * S
      vex = -sign * vanna * oi * 100 * S if pd.notna(row.get("vanna")) else None
      cex = -sign * charm * oi * 100 * S if pd.notna(row.get("charm")) else None
      total_gex = call_gex
      
      exposures.append({
        "strike": strike,
        "type": option_type,
        "GEX": call_gex,
        "VEX": vex,
        "CEX": cex,
        "Total_GEX": total_gex,
      })
    
    exp_df = pd.DataFrame(exposures)
    
    # Aggregate by strike
    agg_df = exp_df.groupby("strike").agg({
      "GEX": "sum",
      "Total_GEX": "sum",
      "VEX": "sum",
      "CEX": "sum",
    }).reset_index()
    
    agg_df.columns = ["strike", "Call_GEX", "Total_GEX", "Total_VEX", "Total_CEX"]
    agg_df = agg_df.sort_values("strike").reset_index(drop=True)
    agg_df["Put_GEX"] = agg_df["Total_GEX"] - agg_df["Call_GEX"]

    strike_agg = {
      float(row["strike"]): {
        "call_gex": float(row["Call_GEX"]),
        "put_gex": float(row["Put_GEX"]),
        "total_gex": float(row["Total_GEX"]),
      }
      for _, row in agg_df.iterrows()
    }
    key_levels = compute_key_gamma_levels(strike_agg)
    call_wall = float(key_levels["call_wall"])
    call_wall_gex = float(strike_agg[call_wall]["call_gex"])
    put_wall = float(key_levels["put_wall"])
    put_wall_gex = float(strike_agg[put_wall]["put_gex"])

    flip_level = None
    flip_cumgex = None
    if key_levels["zero_gamma"] is not None:
      flip_level = float(key_levels["zero_gamma"])
      cum_total = agg_df["Total_GEX"].cumsum()
      flip_idx = agg_df.index[agg_df["strike"] == flip_level]
      if len(flip_idx) > 0:
        flip_cumgex = float(cum_total.iloc[flip_idx[0]])
    
    # Build strike-by-strike data for chart
    strikes_data = []
    for _, row in agg_df.iterrows():
      strikes_data.append({
        "strike": float(row["strike"]),
        "Total_GEX": float(row["Total_GEX"]),
        "Total_VEX": float(row["Total_VEX"]),
        "Total_CEX": float(row["Total_CEX"]),
      })
    
    return {
      "symbol": symbol.upper(),
      "spot_price": spot_price,
      "call_wall": {
        "strike": call_wall,
        "gex": call_wall_gex,
      },
      "put_wall": {
        "strike": put_wall,
        "gex": put_wall_gex,
      },
      "flip_level": {
        "strike": flip_level,
        "cumulative_gex": flip_cumgex,
      } if flip_level else None,
      "strikes": strikes_data,
    }
  
  except HTTPException:
    raise
  except Exception as exc:
    raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/analysis/{symbol}")
async def get_options_analysis(symbol: str):
  """Retrieve pre-computed options exposure analysis for a ticker.
  
  This endpoint returns cached results from the nightly batch job.
  If no cached results exist, analysis is computed on-demand (slower, 5-10s).
  """
  try:
    symbol = symbol.upper()
    
    # Try cache first
    try:
      redis_client = get_redis_client()
      cache_key = f"options_analysis:{symbol}"
      cached = redis_client.get(cache_key)
      if cached:
        return json.loads(cached)
    except Exception:
      pass  # Cache miss or error
    
    # If no cache, trigger on-demand analysis via Celery.
    # Must route to a queue an actual worker consumes -- the bare 'data_fetch'
    # name has no subscriber (see start_celery.sh / docker-compose.yml, both
    # only listen on the per-market data_fetch_<market> queues), so every
    # cache miss here previously dispatched into a void and reliably burned
    # the full 30s timeout below before returning a 504. This is the same
    # class of bug batch_analyze_options_exposure was already fixed for
    # (see options_analysis_tasks.py) but this on-demand path was missed.
    from ...tasks.market_queues import data_fetch_queue_for_market
    from ...tasks.options_analysis_tasks import analyze_options_exposure
    task_result = analyze_options_exposure.apply_async(
        args=[symbol], queue=data_fetch_queue_for_market("US")
    )
    
    # Wait up to 30 seconds for result
    try:
      analysis_result = task_result.get(timeout=30)
      return analysis_result
    except Exception as e:
      raise HTTPException(status_code=504, detail=f"Analysis timed out: {str(e)}")
  
  except HTTPException:
    raise
  except Exception as exc:
    raise HTTPException(status_code=500, detail=str(exc)) from exc
