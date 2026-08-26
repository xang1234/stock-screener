"""
Celery tasks for options dealer exposure analysis.
Analyzes Gamma, Vanna, and Charm exposures for any stock ticker.
Includes batch job to precompute analysis for all active stocks.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from celery import shared_task

from ..services.yfinance_service import YFinanceService
from ..services.rate_budget_policy import RateBudgetPolicy
from ..services.options_metrics import (
    compute_key_gamma_levels,
    compute_options_metrics,
    find_current_atm_iv_from_chain,
    _bs_gamma_delta,
    _bs_vanna_charm,
    _time_to_expiry_years,
    _update_iv_history_and_get_range,
)
from ..services.options_snapshot_upsert import trading_date_for, upsert_snapshot
from ..wiring.bootstrap import get_redis_client
from ..models import StockUniverse
from ..models.options_metrics_snapshot import OptionsMetricsSnapshot
from ..database import SessionLocal
from .market_queues import data_fetch_queue_for_market

logger = logging.getLogger(__name__)


def _parse_expiration_date(expiration_str):
    """Parse a yfinance 'YYYY-MM-DD' expiration string into a date, or None."""
    if not expiration_str:
        return None
    try:
        return datetime.strptime(expiration_str, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


@shared_task(bind=True, max_retries=2)
def batch_analyze_options_exposure(self, market: str = "US", limit: int = 2000) -> Dict[str, Any]:
    """
    Batch job to analyze options exposure for all active stocks in a market.
    
    Runs nightly to precompute structural levels (Call Wall, Put Wall, Flip Level)
    for all active tickers and cache results for 24+ hours.
    
    Args:
        market: Market code (default "US")
        limit: Maximum number of stocks to analyze (default 2000)
    
    Returns:
        Job summary: {analyzed: count, cached: count, errors: count, market: market}
    """
    try:
        session = SessionLocal()
        analyzed = 0
        errors = 0
        
        try:
            # Get all active symbols for the market
            active_symbols = (
                session.query(StockUniverse.symbol)
                .filter(StockUniverse.is_active == True, StockUniverse.market == market)
                .limit(limit)
                .all()
            )
            
            symbols = [s[0] for s in active_symbols]
            total = len(symbols)
            logger.info(f"Starting batch analysis for {total} {market} stocks")
            
            # Get batch size from rate budget policy
            batch_size = RateBudgetPolicy.get_batch_size("yfinance", market)
            logger.info(f"Using batch size: {batch_size}")

            # Route to the per-market data_fetch queue (a real, consumed queue) —
            # the previous hardcoded 'data_fetch' queue name has no worker
            # subscribed to it, so subtasks silently never ran.
            queue_name = data_fetch_queue_for_market(market)

            # Process in batches with rate limiting between batches
            for i in range(0, total, batch_size):
                batch = symbols[i : i + batch_size]
                
                for symbol in batch:
                    try:
                        # Run analysis for this symbol
                        analyze_options_exposure.apply_async(args=[symbol], queue=queue_name)
                        analyzed += 1
                    except Exception as e:
                        logger.error(f"Error queuing analysis for {symbol}: {e}")
                        errors += 1

                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': analyzed,
                        'total': total,
                        'percent': round((analyzed / total) * 100, 1) if total else 100.0,
                        'message': f"Queued {analyzed}/{total} symbols for options analysis",
                        'errors': errors,
                    },
                )
                
                # Wait between batches to respect rate limits
                try:
                    svc = YFinanceService()
                    svc.wait_for_market(market)
                    logger.info(f"Batch {i // batch_size + 1} complete, waiting before next batch...")
                except Exception:
                    pass
        
        finally:
            session.close()
        
        result = {
            "analyzed": analyzed,
            "errors": errors,
            "market": market,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        logger.info(f"Batch analysis complete: {result}")
        return result
    
    except Exception as exc:
        logger.error(f"Batch analysis failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@shared_task(bind=True, max_retries=2)
def analyze_options_exposure(self, symbol: str) -> Dict[str, Any]:
    """
    Comprehensive options exposure analysis including structural levels.
    
    Returns Call Wall, Put Wall, Flip Level, spot price, and strike-by-strike GEX/VEX/CEX.
    Results are cached in Redis for 24 hours.
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")
    
    Returns:
        Analysis results dict with structural levels and strike data
    """
    try:
        symbol = symbol.upper()
        
        # Check cache first
        try:
            redis_client = get_redis_client()
            cache_key = f"options_analysis:{symbol}"
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass  # Cache miss or error, proceed with analysis
        
        # Fetch options for next 3 expirations
        svc = YFinanceService()
        try:
            svc._wait_for_yfinance_rate_limit()
        except Exception:
            pass

        # Shared curl_cffi session (see services/yf_session.py) -- Yahoo's bot
        # detection routinely 429s plain requests-based clients. This is now
        # the actual on-demand fallback for GET /v1/options/analysis/{symbol}
        # (the orphan-queue bug that used to make this dispatch never run at
        # all is fixed elsewhere -- see api/v1/options.py), so it's worth not
        # undermining that fix by hitting easily-avoidable 429s here.
        try:
            from ..services.yf_session import get_session
            session = get_session()
        except Exception:
            session = None
        ticker = yf.Ticker(symbol, session=session) if session is not None else yf.Ticker(symbol)
        expirations = getattr(ticker, 'options', []) or []
        if not expirations:
            raise ValueError(f"No options data available for {symbol}")
        
        # Get spot price
        hist = ticker.history(period="1d")
        if hist.empty:
            raise ValueError(f"No market data found for {symbol}")
        spot_price = float(hist["Close"].iloc[-1])
        
        # Fetch next 3 expirations. yfinance option chains don't reliably include
        # a `gamma` column, so it's derived via Black-Scholes from strike + IV +
        # time-to-expiry (see services/options_metrics.py::_bs_gamma_delta) instead
        # of trusting a column that's usually absent and silently fills with 0.
        # Vanna/charm aren't provided by yfinance at all -- also Black-Scholes
        # model estimates (see _bs_vanna_charm), not market-observed values.
        all_options = []
        nearest_chain: List[Dict[str, Any]] = []
        for exp_index, exp_date in enumerate(expirations[:3]):
            try:
                oc = ticker.option_chain(exp_date)
                time_years = _time_to_expiry_years(exp_date)
                for df_source, option_type in ((oc.calls, 'call'), (oc.puts, 'put')):
                    if df_source is None or df_source.empty:
                        continue

                    df_source = df_source.copy()
                    df_source['openInterest'] = pd.to_numeric(df_source.get('openInterest', 0), errors='coerce').fillna(0).astype(int)
                    df_source['strike'] = pd.to_numeric(df_source.get('strike', 0.0), errors='coerce').fillna(0.0)
                    df_source['impliedVolatility'] = pd.to_numeric(df_source.get('impliedVolatility', 0.0), errors='coerce').fillna(0.0)

                    greeks = [
                        _bs_gamma_delta(spot_price, strike, time_years, iv, option_type)
                        for strike, iv in zip(df_source['strike'], df_source['impliedVolatility'])
                    ]
                    df_source['gamma'] = [g for g, _ in greeks]
                    df_source['delta'] = [d for _, d in greeks]

                    vanna_charm = [
                        _bs_vanna_charm(spot_price, strike, time_years, iv)
                        for strike, iv in zip(df_source['strike'], df_source['impliedVolatility'])
                    ]
                    df_source['vanna'] = [v for v, _ in vanna_charm]
                    df_source['charm'] = [c for _, c in vanna_charm]
                    df_source['type'] = option_type

                    if exp_index == 0:
                        # Reused below to derive options_metrics:{symbol} (GEX walls,
                        # skew, IVR-ready payload) from this same fetched chain —
                        # options_tasks.schedule_daily_update used to re-fetch this
                        # independently, doubling yfinance calls for every symbol.
                        for _, row in df_source.iterrows():
                            nearest_chain.append({
                                "strike": float(row["strike"]),
                                "type": option_type,
                                "delta": float(row["delta"]),
                                "gamma": float(row["gamma"]),
                                "vanna": float(row["vanna"]),
                                "charm": float(row["charm"]),
                                "open_interest": int(row["openInterest"]),
                                "iv": float(row["impliedVolatility"]) if row["impliedVolatility"] > 0 else None,
                            })

                    df_source = df_source[df_source['openInterest'] > 0]
                    if df_source.empty:
                        continue

                    all_options.append(df_source[['strike', 'type', 'gamma', 'vanna', 'charm', 'openInterest']])
            except Exception:
                continue

        if not all_options:
            raise ValueError(f"No options with OI found for {symbol}")

        df = pd.concat(all_options, ignore_index=True)

        S = spot_price
        df['sign'] = 1
        df.loc[df['type'] == 'put', 'sign'] = -1
        # Dollar gamma exposure per 1% move in the underlying -- must match
        # options_metrics.py's aggregate_by_strike()/calculate_options_metrics()
        # and gex_batch.py's convention. This previously used a bare `* S`
        # (missing the second S factor and the 0.01 scale), which put Call
        # Wall/Put Wall/Flip Level's *magnitude* out of step with every other
        # GEX number shown elsewhere on the dashboard (wall/flip *selection*
        # was unaffected, since S is the same constant across every strike of
        # one ticker's chain -- only the displayed dollar values were wrong).
        df['raw_gex'] = df['gamma'] * df['openInterest'] * 100 * S * S * 0.01
        df['call_gex'] = df['raw_gex'].where(df['type'] == 'call', 0.0)
        df['put_gex'] = -df['raw_gex'].where(df['type'] == 'put', 0.0)
        df['total_gex'] = df['call_gex'] + df['put_gex']
        df['VEX'] = -df['sign'] * df['vanna'] * df['openInterest'] * 100 * S
        df['CEX'] = -df['sign'] * df['charm'] * df['openInterest'] * 100 * S

        exp_df = df[['strike', 'call_gex', 'put_gex', 'total_gex', 'VEX', 'CEX']].copy()
        
        # Aggregate by strike
        agg_df = exp_df.groupby("strike").agg({
            "call_gex": "sum",
            "put_gex": "sum",
            "total_gex": "sum",
            "VEX": "sum",
            "CEX": "sum",
        }).reset_index()
        
        agg_df.columns = ["strike", "Call_GEX", "Put_GEX", "Total_GEX", "Total_VEX", "Total_CEX"]
        agg_df = agg_df.sort_values("strike").reset_index(drop=True)
        
        strike_agg = {
            float(row["strike"]): {
                "call_gex": float(row["Call_GEX"]),
                "put_gex": float(row["Put_GEX"]),
                "total_gex": float(row["Total_GEX"]),
            }
            for _, row in agg_df.iterrows()
        }
        key_levels = compute_key_gamma_levels(strike_agg)
        call_wall = key_levels["call_wall"]
        call_wall_gex = float(strike_agg[call_wall]["call_gex"]) if call_wall is not None else None
        put_wall = key_levels["put_wall"]
        put_wall_gex = float(strike_agg[put_wall]["put_gex"]) if put_wall is not None else None

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
        
        result = {
            "symbol": symbol,
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
            "timestamp": datetime.utcnow().isoformat(),
            # gamma/delta/vanna/charm behind these walls/strikes are all
            # Black-Scholes estimates (strike + IV + time-to-expiry), not
            # values reported by the options exchange -- Yahoo's public
            # chain doesn't reliably supply any of them.
            "greeks_methodology": "black_scholes_derived",
        }
        
        # Cache results for 24 hours
        try:
            redis_client = get_redis_client()
            cache_key = f"options_analysis:{symbol}"
            redis_client.setex(cache_key, 86400, json.dumps(result))
        except Exception:
            pass  # Cache error, but result is still returned

        # Derive the simpler options-metrics payload (GEX walls, skew) used by
        # the Options Analytics Dashboard SummaryCards from the SAME nearest-
        # expiry chain fetched above, instead of a second independent sweep.
        if nearest_chain:
            try:
                # IVR requires a current ATM IV + a 52w range derived from the
                # same self-bootstrapped history the live /metrics endpoint
                # uses. Without this, the cached payload always has ivr=null,
                # which is what the dashboard was showing for every symbol
                # served from cache (i.e. almost all page loads).
                current_atm_iv = find_current_atm_iv_from_chain(nearest_chain, spot_price)
                iv_session = SessionLocal()
                try:
                    iv_52w_low, iv_52w_high = _update_iv_history_and_get_range(
                        iv_session, symbol, current_atm_iv
                    )
                finally:
                    iv_session.close()
                metrics = compute_options_metrics(
                    nearest_chain,
                    spot=spot_price,
                    current_iv=current_atm_iv,
                    iv_52w_low=iv_52w_low,
                    iv_52w_high=iv_52w_high,
                )
                metrics["greeks_methodology"] = "black_scholes_derived"
                metrics["computed_at"] = result["timestamp"]
                redis_client = get_redis_client()
                redis_client.set(f"options_metrics:{symbol}", json.dumps(metrics), ex=7 * 24 * 3600)

                # Durable history alongside the 7-day Redis cache. This is the
                # "batch_abbreviated" write path (see OptionsMetricsSnapshot's
                # docstring) -- only what's derivable from the chain already
                # fetched above, not the fuller payload calculate_options_metrics()
                # produces on-demand. schema_version stays null so downstream
                # readers can tell this row apart from a live_full one.
                #
                # Skipped entirely when every strike shows zero open interest
                # -- a known yfinance off-hours data-staleness pattern (see
                # _is_zero_open_interest in api/v1/options.py), not a real
                # market condition. Persisting it would clobber a good
                # historical row for this same (ticker, trading_date,
                # expiration) with garbage.
                total_batch_oi = sum((s.get("oi") or 0) for s in (metrics.get("strikes") or []))
                if total_batch_oi <= 0:
                    logger.debug("Skipping OptionsMetricsSnapshot persist for %s: zero OI across all strikes", symbol)
                else:
                    try:
                        fetched_at = datetime.utcnow()
                        snapshot_db = SessionLocal()
                        try:
                            key_levels = metrics.get("key_levels") or {}
                            net = metrics.get("net") or {}
                            upsert_snapshot(
                                snapshot_db,
                                OptionsMetricsSnapshot,
                                ticker=symbol,
                                trading_date=trading_date_for(fetched_at),
                                expiration=_parse_expiration_date(expirations[0] if expirations else None),
                                values={
                                    "status": "OK",
                                    "error": None,
                                    "source": "batch_abbreviated",
                                    "schema_version": None,
                                    "underlying_price": spot_price,
                                    "call_wall": key_levels.get("call_wall"),
                                    "put_wall": key_levels.get("put_wall"),
                                    "zero_gamma": key_levels.get("zero_gamma"),
                                    "net_dex": net.get("net_dex"),
                                    "net_vex": net.get("net_vex"),
                                    "net_cex": net.get("net_cex"),
                                    "ivr": metrics.get("ivr"),
                                    "skew": metrics.get("skew"),
                                    "greeks_methodology": metrics.get("greeks_methodology"),
                                    "strikes_json": metrics.get("strikes"),
                                    "fetched_at": fetched_at,
                                },
                            )
                            snapshot_db.commit()
                        finally:
                            snapshot_db.close()
                    except Exception:
                        logger.debug("Failed to persist OptionsMetricsSnapshot for %s", symbol)
            except Exception:
                logger.debug("Failed to derive/cache options_metrics for %s", symbol)

        return result
    
    except Exception as exc:
        # Retry up to max_retries times
        raise self.retry(exc=exc, countdown=30)
