"""Universe-wide options scanner backing the Options Command Center page.

Every ranking here reads the LATEST persisted OptionsMetricsSnapshot row per
active US-market ticker (see app/models/options_metrics_snapshot.py) -- a
per-request live yfinance fetch across the whole universe would be far too
expensive, so this is intentionally a read of whatever has already been
captured by the nightly batch or by users viewing the single-symbol
dashboard. Coverage grows organically over time and can be sparse (e.g.
during the yfinance open-interest data gaps documented on
OptionsMetricsSnapshot) -- every ranking list may legitimately come back
shorter than its nominal "top 10", or empty, when too few symbols have the
fields that ranking needs.

The macro SPY/QQQ bar is the one exception: it's just two symbols, so
_fetch_live_macro_index does a live on-demand fetch when there's no usable
persisted snapshot, so the top bar stays populated regardless of universe
coverage.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...database import get_db
from ...models.options_metrics_snapshot import OptionsMetricsSnapshot
from ...models.stock_universe import StockUniverse
from ...services.options_market_signal import evaluate_snapshot_signal

logger = logging.getLogger(__name__)

router = APIRouter()

_TOP_N = 10
_FLIP_PROXIMITY_PCT = 1.5
_FLIP_PROXIMITY_FALLBACK_N = 3


def _is_degenerate_snapshot(row: OptionsMetricsSnapshot) -> bool:
    """True for a snapshot written while yfinance was serving its known
    off-hours zero-OI garbage (see _is_zero_open_interest in
    app/api/v1/options.py) -- EXPLICIT zero OI on both sides (a live_full
    fetch that got zeros back), which also collapses total_gex to 0 and can
    drag current_atm_iv down to a near-floor value that isn't a real market
    reading. The write-path guard added in eac281c3/f7401a0e stops *new*
    rows like this from being persisted, but rows written before that fix
    landed can still be sitting in the table, and would otherwise surface as
    "the latest snapshot" for their ticker since there's nothing newer to
    supersede them yet. Filtering here means those tickers correctly drop
    out of every ranking (and out of the macro bar) until a real fetch
    replaces them, rather than showing corrupted numbers.

    Deliberately NOT triggered by NULL OI: the batch_abbreviated write path
    (analyze_options_exposure) never populates total_call_oi/total_put_oi at
    all -- it's a lighter payload than the live_full fetch, not a degenerate
    one -- so those columns are always NULL there, on every legitimately
    good row. Treating NULL the same as explicit 0 (an earlier version of
    this check did, via `(x or 0) == 0`) silently dropped every
    batch_abbreviated row out of the universe, which is most of what the
    nightly/manual batch sweep actually produces.
    """
    return row.total_call_oi == 0 and row.total_put_oi == 0


def _latest_snapshots_for_active_universe(db: Session) -> List[OptionsMetricsSnapshot]:
    """One row per active US-market ticker: whichever persisted snapshot is
    most recent for that ticker, regardless of source (live_full vs.
    batch_abbreviated) -- callers filter further by whichever fields their
    specific ranking actually needs. Degenerate zero-OI rows (see
    _is_degenerate_snapshot) are dropped entirely; a ticker with only a
    degenerate row on file simply has no data yet, same as a ticker with no
    row at all."""
    subq = (
        db.query(OptionsMetricsSnapshot)
        .join(StockUniverse, StockUniverse.symbol == OptionsMetricsSnapshot.ticker)
        .filter(StockUniverse.active_filter(), StockUniverse.market == "US")
        .distinct(OptionsMetricsSnapshot.ticker)
        .order_by(OptionsMetricsSnapshot.ticker, OptionsMetricsSnapshot.fetched_at.desc())
    )
    return [r for r in subq.all() if not _is_degenerate_snapshot(r)]


def _symbol_row(symbol: str, **fields: Any) -> Dict[str, Any]:
    return {"symbol": symbol, **fields}


def _rank_volatility_acceleration(rows: List[OptionsMetricsSnapshot]) -> List[Dict[str, Any]]:
    # total_gex == 0 is excluded alongside None: a real chain essentially
    # never nets to exactly zero, so 0 means "not actually computed" (an
    # empty/degenerate chain read) rather than a genuine flat GEX reading.
    eligible = [r for r in rows if r.total_gex]
    eligible.sort(key=lambda r: r.total_gex)
    return [
        _symbol_row(
            r.ticker,
            price=r.underlying_price,
            totalGex=r.total_gex,
            distanceToFlipPct=_pct_distance(r.underlying_price, r.zero_gamma),
            regime="short_gamma" if r.total_gex < 0 else "long_gamma",
        )
        for r in eligible[:_TOP_N]
    ]


def _pct_distance(spot: Optional[float], level: Optional[float]) -> Optional[float]:
    if spot is None or not level:
        return None
    return round((spot - level) / level * 100.0, 2)


def _rank_gamma_flip_proximity(rows: List[OptionsMetricsSnapshot]) -> Dict[str, Any]:
    """Tickers trading closest to their zero-gamma flip level. Normally
    restricted to within _FLIP_PROXIMITY_PCT (1.5%) of the flip, but that
    can legitimately come back empty on a quiet day -- rather than showing
    a bare "no matches" table, widen to the _FLIP_PROXIMITY_FALLBACK_N (3)
    closest tickers regardless of distance, flagged via `widened` so the
    frontend can label it as an outside-threshold fallback."""
    all_candidates = []
    for r in rows:
        distance = _pct_distance(r.underlying_price, r.zero_gamma)
        if distance is None:
            continue
        all_candidates.append((abs(distance), r, distance))
    all_candidates.sort(key=lambda t: t[0])

    within_threshold = [c for c in all_candidates if c[0] <= _FLIP_PROXIMITY_PCT]
    widened = len(within_threshold) == 0
    source = all_candidates[:_FLIP_PROXIMITY_FALLBACK_N] if widened else within_threshold[:_TOP_N]

    return {
        "widened": widened,
        "rows": [
            _symbol_row(r.ticker, spot=r.underlying_price, flipLevel=r.zero_gamma, distancePct=distance)
            for _, r, distance in source
        ],
    }


def _rank_vrp(rows: List[OptionsMetricsSnapshot], *, rich: bool) -> List[Dict[str, Any]]:
    """Shared ranking for both VRP tables. `rich=True` -> "Top Rich VRP"
    (IV > HV, premium-selling candidates), sorted most-positive VRP first.
    `rich=False` -> "Top Cheap VRP" (IV < HV, premium-buying candidates),
    sorted most-negative VRP first. Each side strictly excludes the other's
    sign -- a symbol with ~0 VRP appears in neither table rather than both."""
    eligible = []
    for r in rows:
        if r.current_atm_iv is None or r.historical_volatility is None:
            continue
        vrp = r.current_atm_iv - r.historical_volatility
        if rich and vrp > 0:
            eligible.append((vrp, r))
        elif not rich and vrp < 0:
            eligible.append((vrp, r))
    eligible.sort(key=lambda t: t[0], reverse=rich)
    return [
        _symbol_row(r.ticker, iv=r.current_atm_iv, hv=r.historical_volatility, vrpPct=round(vrp * 100.0, 1))
        for vrp, r in eligible[:_TOP_N]
    ]


def _rank_extreme_skew(rows: List[OptionsMetricsSnapshot]) -> List[Dict[str, Any]]:
    # Most negative skew = strongest call skew, matching the frontend's
    # "call IV exceeds put IV" bullish convention (see options_market_signal).
    eligible = [r for r in rows if r.skew is not None]
    eligible.sort(key=lambda r: r.skew)
    return [_symbol_row(r.ticker, skew=r.skew) for r in eligible[:_TOP_N]]


def _rank_net_premium_inflows(rows: List[OptionsMetricsSnapshot]) -> List[Dict[str, Any]]:
    eligible = [r for r in rows if r.call_premium_notional is not None and r.put_premium_notional is not None]
    eligible.sort(key=lambda r: (r.call_premium_notional - r.put_premium_notional), reverse=True)
    return [
        _symbol_row(
            r.ticker,
            callPremium=r.call_premium_notional,
            putPremium=r.put_premium_notional,
            netPremium=round(r.call_premium_notional - r.put_premium_notional, 2),
        )
        for r in eligible[:_TOP_N]
    ]


def _rank_unusual_volume_oi(rows: List[OptionsMetricsSnapshot]) -> List[Dict[str, Any]]:
    # Only live_full rows populate unusual_volume_json -- flatten every
    # ticker's flagged contracts into one list and take the highest ratios
    # across the whole universe, not per-symbol.
    contracts: List[Dict[str, Any]] = []
    for r in rows:
        for contract in (r.unusual_volume_json or []):
            ratio = contract.get("ratio")
            if ratio is None:
                continue
            contracts.append({
                "symbol": r.ticker,
                "strike": contract.get("strike"),
                "type": contract.get("type"),
                "volume": contract.get("volume"),
                "openInterest": contract.get("open_interest"),
                "ratio": ratio,
            })
    contracts.sort(key=lambda c: c["ratio"], reverse=True)
    return contracts[:_TOP_N]


def _generate_alerts(rows: List[OptionsMetricsSnapshot]) -> List[Dict[str, Any]]:
    """One alert per ticker whose latest snapshot has a strong enough
    Executive Signal score, plus a dedicated wall-breach alert regardless of
    the aggregate score -- see mockData.js's documented convention this
    mirrors (>= 4 critical, >= 1.5 warning, structural breach always at
    least warning)."""
    alerts: List[Dict[str, Any]] = []
    next_id = 1

    for r in rows:
        signal = evaluate_snapshot_signal(r)
        breached_call = r.underlying_price is not None and r.call_wall is not None and r.underlying_price >= r.call_wall
        breached_put = r.underlying_price is not None and r.put_wall is not None and r.underlying_price <= r.put_wall

        if breached_call or breached_put:
            wall = r.call_wall if breached_call else r.put_wall
            direction = "Call Wall" if breached_call else "Put Wall"
            severity = "critical" if abs(signal.score) >= 4 else "warning"
            alerts.append({
                "id": next_id,
                "severity": severity,
                "text": f"Gamma Squeeze Alert: ${r.ticker} breached {direction} (${wall:.2f})",
            })
            next_id += 1
            continue

        if abs(signal.score) >= 4:
            alerts.append({
                "id": next_id,
                "severity": "critical",
                "text": f"${r.ticker} Executive Signal: {signal.label} (score {signal.score:+.1f})",
            })
            next_id += 1
        elif abs(signal.score) >= 1.5:
            alerts.append({
                "id": next_id,
                "severity": "warning",
                "text": f"${r.ticker} Executive Signal: {signal.label} (score {signal.score:+.1f})",
            })
            next_id += 1

    return alerts[:20]


def _macro_index_from_row(row: OptionsMetricsSnapshot, symbol: str) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "spot": row.underlying_price,
        "flipLevel": row.zero_gamma,
        "callWall": row.call_wall,
        "putWall": row.put_wall,
        "regime": "long_gamma" if (row.total_gex or 0) >= 0 else "short_gamma",
    }


def _macro_index_from_live_result(result: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    key_levels = result.get("key_levels") or {}
    total_gex = result.get("total_gex")
    return {
        "symbol": symbol,
        "spot": result.get("underlying_price"),
        "flipLevel": key_levels.get("zero_gamma"),
        "callWall": result.get("call_wall") or key_levels.get("call_wall"),
        "putWall": result.get("put_wall") or key_levels.get("put_wall"),
        "regime": "long_gamma" if (total_gex or 0) >= 0 else "short_gamma",
    }


_EMPTY_MACRO_INDEX_TEMPLATE = {"spot": None, "flipLevel": None, "callWall": None, "putWall": None, "regime": None}


def _empty_macro_index(symbol: str) -> Dict[str, Any]:
    return {"symbol": symbol, **_EMPTY_MACRO_INDEX_TEMPLATE}


def _fetch_live_macro_index(db: Session, symbol: str) -> Dict[str, Any]:
    """Live on-demand fetch + persist for a single macro index symbol
    (SPY/QQQ), used only when there's no usable persisted snapshot to read.
    The macro bar exists to always answer "is the tape calm or dangerous"
    regardless of which single-stock pages anyone has viewed or whether the
    nightly batch has reached these two symbols yet -- unlike the ranking
    tables (which read persisted data only, since a live fetch per universe
    symbol per request would be far too expensive), a live fetch for just
    these two symbols is cheap enough to do inline.

    Never fabricates a number: if yfinance itself has no expiration list, or
    the live read comes back with the same off-hours zero-OI garbage this
    module already filters out of persisted rows (_is_degenerate_snapshot),
    this returns the empty/"no data" shape instead -- honest absence, not a
    mocked placeholder.
    """
    import yfinance as yf

    from ...services.options_metrics import calculate_options_metrics
    from ...services.options_snapshot_upsert import trading_date_for, upsert_snapshot

    try:
        expirations = getattr(yf.Ticker(symbol), "options", []) or []
    except Exception:
        expirations = []
    if not expirations:
        return _empty_macro_index(symbol)

    try:
        result = calculate_options_metrics(symbol, expirations[0], db=db, record_iv_history=True)
    except Exception:
        logger.exception("Live macro fetch failed for %s", symbol)
        return _empty_macro_index(symbol)

    if (result.get("total_call_oi") or 0) == 0 and (result.get("total_put_oi") or 0) == 0:
        return _empty_macro_index(symbol)

    try:
        fetched_at = datetime.utcnow()
        key_levels = result.get("key_levels") or {}
        net = result.get("net") or {}
        upsert_snapshot(
            db,
            OptionsMetricsSnapshot,
            ticker=symbol.upper(),
            trading_date=trading_date_for(fetched_at),
            expiration=datetime.strptime(expirations[0], "%Y-%m-%d").date(),
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
        logger.exception("Failed to persist live macro fetch for %s", symbol)

    return _macro_index_from_live_result(result, symbol)


def _macro_index(db: Session, row: Optional[OptionsMetricsSnapshot], symbol: str) -> Dict[str, Any]:
    if row is not None:
        return _macro_index_from_row(row, symbol)
    return _fetch_live_macro_index(db, symbol)


@router.get("/")
def get_command_center_snapshot(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Everything the Options Command Center page needs in one call: macro
    SPY/QQQ levels, the six ranking tables, and generated alerts. All
    derived from persisted OptionsMetricsSnapshot rows -- see module
    docstring for why this can legitimately return sparse/empty lists.

    Note: there is no genuine $SPX gamma-regime figure here -- SPX index
    options aren't tracked separately, so SPY's own regime is used as a
    proxy rather than fabricating an aggregate. VIX term structure is
    intentionally absent: this app has no VIX futures data source at all
    (see MarketExposure.vix, a single spot value only), so there is nothing
    real to report.
    """
    rows = _latest_snapshots_for_active_universe(db)
    by_ticker = {r.ticker: r for r in rows}

    # Resolved once each (not per-field) -- _macro_index does a live fetch
    # when there's no persisted row, and that fetch is not cheap enough to
    # repeat for the same symbol within one request.
    spy_index = _macro_index(db, by_ticker.get("SPY"), "SPY")
    qqq_index = _macro_index(db, by_ticker.get("QQQ"), "QQQ")

    return {
        "macro": {
            "spxProxy": {
                "label": "$SPX (SPY proxy)",
                "regime": spy_index["regime"],
                "flipLevel": spy_index["flipLevel"],
                "spot": spy_index["spot"],
            },
            "indices": [spy_index, qqq_index],
        },
        "volatilityAcceleration": _rank_volatility_acceleration(rows),
        "gammaFlipProximity": _rank_gamma_flip_proximity(rows),
        "richVrp": _rank_vrp(rows, rich=True),
        "cheapVrp": _rank_vrp(rows, rich=False),
        "extremeSkew": _rank_extreme_skew(rows),
        "netPremiumInflows": _rank_net_premium_inflows(rows),
        "unusualVolumeOi": _rank_unusual_volume_oi(rows),
        "alerts": _generate_alerts(rows),
        "coverage": {
            "activeUniverseSymbolsWithData": len(rows),
        },
    }
