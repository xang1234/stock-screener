"""Relative Rotation Graph (RRG) service.

Turns the stored daily ``ibd_group_ranks.avg_rs_rating`` time series into the
RS-Ratio (x) / RS-Momentum (y) coordinate pair plus a weekly "tail" that traces
each group's (or sector's) path through the four RRG quadrants:

    Leading (x>=100, y>=100)   — strong and getting stronger
    Weakening (x>=100, y<100)  — strong but momentum rolling over
    Lagging (x<100, y<100)     — weak and getting weaker
    Improving (x<100, y>=100)  — weak but momentum turning up

Design notes
------------
* ``avg_rs_rating`` is already a 0-100 *cross-sectional* percentile (RS vs the
  benchmark universe). We therefore normalize **temporally** — z-scoring each
  group against its *own* trailing history — rather than cross-sectionally again,
  which would double-normalize and largely re-encode the existing rank column.
  A ``frame`` hook is left for a future cross-sectional A/B without an API break.
* All the heavy lifting here is **pure** float math (no DB, no pandas, no RNG),
  so it is golden-snapshot testable under ``make gate-5``. The DB-aware
  orchestrator (``RRGService``) imports sqlalchemy/models lazily inside its
  methods so this module stays importable without the full app bootstrap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Tunable constants (module-level so they can be adjusted without touching the
# call sites and so tests can reference them by name).
# ---------------------------------------------------------------------------

SMOOTH_RATIO_SPAN = 5       # EMA span (weekly points) applied to the raw RS series
Z_WINDOW = 26               # trailing window (weeks) for the RS-Ratio z-score
RATIO_SCALE = 5.0           # ±2σ -> roughly 90..110 (conventional RRG band)
MOM_WINDOW = 4              # lookback (weeks) for the RS-Ratio rate-of-change
SMOOTH_MOM_SPAN = 3         # EMA span (weekly points) applied to the momentum ROC
MOM_Z_WINDOW = 13           # trailing window (weeks) for the RS-Momentum z-score
MOM_SCALE = 5.0             # momentum display scale (matches ratio for a square plot)
DISPLAY_CLAMP: Tuple[float, float] = (80.0, 120.0)  # guardrail against outliers

MIN_TAIL_WEEKS = 12         # below this, a group is omitted entirely
MIN_WEEKS = Z_WINDOW + MOM_WINDOW   # full-confidence threshold (~30 weeks)
MAX_FILL_WEEKS = 2          # carry-forward at most this many empty weeks

EPS = 1e-6

DEFAULT_TAIL_WEEKS = 8
DEFAULT_LOOKBACK_DAYS = 400  # enough daily rows to build Z_WINDOW + tail weeks

RRG_GROUPS_ENABLED_MARKETS = frozenset({"US", "HK", "JP", "IN", "TW"})
RRG_SECTORS_ENABLED_MARKETS = frozenset({"US", "HK", "JP", "IN"})


@dataclass(frozen=True)
class RRGParams:
    """Parameters controlling the RRG transform."""

    tail_weeks: int = DEFAULT_TAIL_WEEKS
    smooth_ratio_span: int = SMOOTH_RATIO_SPAN
    z_window: int = Z_WINDOW
    mom_window: int = MOM_WINDOW


def _week_start(d: date) -> date:
    """UTC Sunday-origin start of the ISO-ish week containing ``d``.

    Matches the frontend ``aggregateToWeekly`` rule (JS ``getUTCDay()`` where
    Sunday==0): ``dow = (weekday()+1) % 7`` maps Mon..Sun -> 1..0, so we step
    back to the preceding Sunday.
    """
    dow = (d.weekday() + 1) % 7
    return d - timedelta(days=dow)


def _bucket_weekly(
    daily: Sequence[Tuple[date, float]],
) -> List[Tuple[date, float]]:
    """Collapse a daily ``(date, value)`` series to one close-of-week point.

    Each week is keyed by its UTC Sunday start; the value kept is the latest
    trading day's value within that week. Output is ascending by week start.
    Tolerant of unsorted input.
    """
    latest: dict[date, Tuple[date, float]] = {}
    for d, value in daily:
        wk = _week_start(d)
        prev = latest.get(wk)
        if prev is None or d >= prev[0]:
            latest[wk] = (d, value)
    return [(wk, latest[wk][1]) for wk in sorted(latest)]


def _ema(values: Sequence[float], span: int) -> List[float]:
    """Exponential moving average, ``adjust=False`` (seed = first value).

    Equivalent to ``pandas.Series(values).ewm(span=span, adjust=False).mean()``
    but dependency-free and deterministic.
    """
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    out = [float(values[0])]
    for v in values[1:]:
        out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
    return out


def classify_quadrant(x: float, y: float) -> str:
    """Map an (RS-Ratio, RS-Momentum) point to its RRG quadrant.

    The 100/100 cross is the origin; boundary values (exactly 100) belong to the
    strong/rising side so a brand-new "at-typical" series reads as Leading.
    """
    strong = x >= 100.0
    rising = y >= 100.0
    if strong and rising:
        return "Leading"
    if strong and not rising:
        return "Weakening"
    if not strong and not rising:
        return "Lagging"
    return "Improving"  # weak but rising


def _clamp(v: float) -> float:
    lo, hi = DISPLAY_CLAMP
    return max(lo, min(hi, v))


def _centered_zscore_series(
    values: Sequence[float], window: int, scale: float
) -> List[float]:
    """Per-point trailing temporal z-score, re-centered at 100 and clamped.

    For each index ``i`` the z-score uses the trailing ``window`` values ending
    at ``i`` (population std). A flat/zero-variance window yields z=0 -> 100 (EPS
    guard). This is the shared normalization behind both RRG axes — z-scoring
    each axis against its *own* recent history is what makes points orbit the
    100/100 center rather than saturate."""
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        w = values[start : i + 1]
        n = len(w)
        mu = sum(w) / n
        sd = math.sqrt(sum((v - mu) ** 2 for v in w) / n)
        z = 0.0 if sd < EPS else (values[i] - mu) / sd
        out.append(_clamp(100.0 + scale * z))
    return out


def _rs_ratio_series(weekly_vals: Sequence[float], params: RRGParams) -> List[float]:
    """RS-Ratio (x): EMA-smooth the raw RS series, then a trailing temporal
    z-score centered at 100 over up to ``z_window`` weeks."""
    smooth = _ema(weekly_vals, params.smooth_ratio_span)
    return _centered_zscore_series(smooth, params.z_window, RATIO_SCALE)


def _rs_momentum_series(rs_ratio: Sequence[float], params: RRGParams) -> List[float]:
    """RS-Momentum (y): EMA-smoothed rate-of-change of the RS-Ratio, then its
    OWN trailing temporal z-score centered at 100.

    Returns a list aligned to absolute indices ``[mom_window, len)`` — the j-th
    entry corresponds to weekly index ``mom_window + j``. Already centered at 100
    (the caller plots it directly)."""
    mw = params.mom_window
    roc = [rs_ratio[i] - rs_ratio[i - mw] for i in range(mw, len(rs_ratio))]
    roc_smooth = _ema(roc, SMOOTH_MOM_SPAN)
    return _centered_zscore_series(roc_smooth, MOM_Z_WINDOW, MOM_SCALE)


def compute_group_rrg(
    daily_series: Sequence[Tuple[date, float]],
    params: Optional[RRGParams] = None,
) -> Optional[Dict[str, Any]]:
    """Full RRG pipeline for ONE series (a group or a sector).

    Pure: no DB, no RNG. This is the golden-snapshot surface. Returns a dict
    ``{current, tail, is_provisional}`` or ``None`` when there is not enough
    history (< ``MIN_TAIL_WEEKS`` weekly points) to plot a meaningful tail.

    * ``current`` / ``tail`` points are ``{"date", "x", "y"}`` (x=RS-Ratio,
      y=RS-Momentum, both centered at 100), ascending oldest->newest.
    * ``is_provisional`` is True when the series is shorter than ``MIN_WEEKS``,
      so the most recent point used a shortened (sub-``z_window``) window.
    """
    params = params or RRGParams()
    weekly = _bucket_weekly(daily_series)
    if len(weekly) < MIN_TAIL_WEEKS:
        return None

    weeks = [w for w, _ in weekly]
    vals = [v for _, v in weekly]

    rs_ratio = _rs_ratio_series(vals, params)
    roc_smooth = _rs_momentum_series(rs_ratio, params)

    mw = params.mom_window
    points: List[Dict[str, Any]] = []
    for j, y in enumerate(roc_smooth):
        i = mw + j
        points.append(
            {
                "date": weeks[i].isoformat(),
                "x": round(rs_ratio[i], 4),
                "y": round(y, 4),
            }
        )

    if not points:
        return None

    tail = points[-params.tail_weeks :]
    return {
        "current": tail[-1],
        "tail": tail,
        "is_provisional": len(weekly) < MIN_WEEKS,
    }


# ---------------------------------------------------------------------------
# DB-aware orchestrator. All sqlalchemy/model imports are LAZY (inside methods)
# so the pure-math surface above stays importable without the app bootstrap.
# ---------------------------------------------------------------------------


class RRGService:
    """Builds RRG payloads from the stored daily group-rank history.

    The math lives in the pure functions above; this class only sources the
    ``avg_rs_rating`` series (one batched query) and decorates each result with
    rank/num_stocks/quadrant metadata. Sector scope rolls groups up to their
    dominant GICS sector and aggregates the same series.
    """

    def __init__(
        self,
        *,
        group_rank_service: Any,
        market_group_ranking_service: Any | None = None,
        taxonomy_service: Any | None = None,
    ) -> None:
        self._group_rank_service = group_rank_service
        self._market_group_ranking_service = market_group_ranking_service
        self._taxonomy_service = taxonomy_service

    def _get_market_group_ranking_service(self) -> Any:
        if self._market_group_ranking_service is None:
            from .market_group_ranking_service import get_market_group_ranking_service

            self._market_group_ranking_service = get_market_group_ranking_service()
        return self._market_group_ranking_service

    def _get_taxonomy_service(self) -> Any:
        if self._taxonomy_service is None:
            from .market_taxonomy_service import get_market_taxonomy_service

            self._taxonomy_service = get_market_taxonomy_service()
        return self._taxonomy_service

    def get_group_sector_map(self, db: Any, market: str = "US") -> Dict[str, str]:
        """Map each IBD industry group -> its constituents' dominant GICS sector.

        Derived from ``StockUniverse.sector`` (fully populated) via a majority
        vote, since the codebase has no IBD-native sector taxonomy.
        """
        market = (market or "US").upper()
        if market != "US" and market in RRG_SECTORS_ENABLED_MARKETS:
            sector_map = self._get_taxonomy_service().sector_map_for_market(market)
            if sector_map:
                return sector_map

        from collections import Counter, defaultdict

        from ..models.industry import IBDIndustryGroup
        from ..models.stock_universe import StockUniverse

        rows = (
            db.query(IBDIndustryGroup.industry_group, StockUniverse.sector)
            .join(StockUniverse, StockUniverse.symbol == IBDIndustryGroup.symbol)
            .filter(
                IBDIndustryGroup.market == market,
                StockUniverse.market == market,
                StockUniverse.sector.isnot(None),
            )
            .all()
        )
        votes: dict[str, Counter] = defaultdict(Counter)
        for group, sector in rows:
            if sector:
                votes[group][sector] += 1
        return {group: ctr.most_common(1)[0][0] for group, ctr in votes.items()}

    def get_rrg(
        self,
        db: Any,
        *,
        market: str = "US",
        scope: str = "groups",
        tail_weeks: int = DEFAULT_TAIL_WEEKS,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ) -> Dict[str, Any]:
        """Assemble the RRG payload for one market + scope (``groups``/``sectors``)."""
        return self.get_rrg_scopes(
            db,
            market=market,
            scopes=(scope,),
            tail_weeks=tail_weeks,
            lookback_days=lookback_days,
        )[scope]

    def get_rrg_scopes(
        self,
        db: Any,
        *,
        market: str = "US",
        scopes: Sequence[str] = ("groups",),
        tail_weeks: int = DEFAULT_TAIL_WEEKS,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute several scopes from a SINGLE input fetch.

        The current rankings + batched history are read once and every requested
        scope is derived from them, so the static exporter can emit groups and
        sectors without re-querying. ``get_rrg`` is the single-scope shorthand.
        """
        market = (market or "US").upper()
        if market not in RRG_GROUPS_ENABLED_MARKETS:
            return self._empty_scopes(market, scopes)

        if all(scope == "sectors" and market not in RRG_SECTORS_ENABLED_MARKETS for scope in scopes):
            return self._empty_scopes(market, scopes)

        params = RRGParams(tail_weeks=tail_weeks)
        latest_date, meta, group_series = self._fetch_inputs(db, market, lookback_days)
        if latest_date is None:
            return self._empty_scopes(market, scopes)
        return {
            scope: self._build_scope_payload(
                db, market, scope, latest_date, meta, group_series, params
            )
            for scope in scopes
        }

    @staticmethod
    def _empty_scopes(
        market: str,
        scopes: Sequence[str],
        *,
        latest_date: str | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        return {
            scope: {"date": latest_date, "market": market, "scope": scope, "groups": []}
            for scope in scopes
        }

    def _fetch_inputs(
        self, db: Any, market: str, lookback_days: int
    ) -> Tuple[
        Optional[str],
        Dict[str, Dict[str, Any]],
        Dict[str, List[Tuple[date, float, int]]],
    ]:
        """Read the current rankings (metadata) + batched group history (one query)."""
        if market != "US":
            return self._get_market_group_ranking_service().get_all_groups_history(
                db,
                market=market,
                days=lookback_days,
            )

        return self._fetch_us_inputs(db, market, lookback_days)

    def _fetch_us_inputs(
        self, db: Any, market: str, lookback_days: int
    ) -> Tuple[
        Optional[str],
        Dict[str, Dict[str, Any]],
        Dict[str, List[Tuple[date, float, int]]],
    ]:
        from datetime import date as _date

        from ..models.industry import IBDGroupRank

        current = self._group_rank_service.get_current_rankings(
            db, limit=197, market=market
        )
        if not current:
            return None, {}, {}

        latest_date = current[0]["date"]  # ISO string
        meta = {row["industry_group"]: row for row in current}

        cutoff = _date.fromisoformat(latest_date) - timedelta(days=lookback_days)
        rows = (
            db.query(
                IBDGroupRank.industry_group,
                IBDGroupRank.date,
                IBDGroupRank.avg_rs_rating,
                IBDGroupRank.num_stocks,
            )
            .filter(IBDGroupRank.market == market, IBDGroupRank.date >= cutoff)
            .order_by(IBDGroupRank.industry_group, IBDGroupRank.date)
            .all()
        )
        return latest_date, meta, self._collect_group_series(rows)

    def _build_scope_payload(
        self,
        db: Any,
        market: str,
        scope: str,
        latest_date: str,
        meta: Dict[str, Dict[str, Any]],
        group_series: Dict[str, List[Tuple[date, float, int]]],
        params: RRGParams,
    ) -> Dict[str, Any]:
        if scope == "sectors":
            if market not in RRG_SECTORS_ENABLED_MARKETS:
                return {"date": latest_date, "market": market, "scope": scope, "groups": []}
            series, scope_meta = self._aggregate_sectors(db, market, group_series)
        else:
            series = {
                g: [(d, rs) for (d, rs, _ns) in pts]
                for g, pts in group_series.items()
            }
            scope_meta = meta

        groups_out: List[Dict[str, Any]] = []
        for name, daily in series.items():
            rrg = compute_group_rrg(daily, params)
            if rrg is None:
                continue
            info = scope_meta.get(name, {})
            cur = rrg["current"]
            groups_out.append(
                {
                    "industry_group": name,
                    "rank": info.get("rank"),
                    "num_stocks": info.get("num_stocks"),
                    "avg_rs_rating": info.get("avg_rs_rating"),
                    "quadrant": classify_quadrant(cur["x"], cur["y"]),
                    "is_provisional": rrg["is_provisional"],
                    "current": cur,
                    "tail": rrg["tail"],
                }
            )

        groups_out.sort(key=lambda g: (g["rank"] is None, g["rank"] or 0))
        return {
            "date": latest_date,
            "market": market,
            "scope": scope,
            "groups": groups_out,
        }

    @staticmethod
    def _collect_group_series(
        rows: Sequence[Tuple[str, date, float, Optional[int]]],
    ) -> Dict[str, List[Tuple[date, float, int]]]:
        from collections import defaultdict

        series: dict[str, List[Tuple[date, float, int]]] = defaultdict(list)
        for group, d, rs, ns in rows:
            series[group].append((d, float(rs), int(ns or 0)))
        return series

    def _aggregate_sectors(
        self,
        db: Any,
        market: str,
        group_series: Dict[str, List[Tuple[date, float, int]]],
    ) -> Tuple[Dict[str, List[Tuple[date, float]]], Dict[str, Dict[str, Any]]]:
        """Roll group series up to num_stocks-weighted sector series + sector meta."""
        from collections import defaultdict

        sector_of = self.get_group_sector_map(db, market)
        # sector -> date -> [weighted_sum, weight_sum]
        agg: dict[str, dict[date, List[float]]] = defaultdict(
            lambda: defaultdict(lambda: [0.0, 0.0])
        )
        for group, points in group_series.items():
            sector = sector_of.get(group)
            if not sector:
                continue
            for d, rs, ns in points:
                cell = agg[sector][d]
                cell[0] += rs * ns
                cell[1] += ns

        series: Dict[str, List[Tuple[date, float]]] = {}
        meta: Dict[str, Dict[str, Any]] = {}
        ranked: List[Tuple[str, float]] = []
        for sector, by_date in agg.items():
            pts: List[Tuple[date, float]] = []
            last_weight = 0.0
            for d in sorted(by_date):
                wsum, nsum = by_date[d]
                if nsum > 0:
                    pts.append((d, wsum / nsum))
                    last_weight = nsum
            if not pts:
                continue
            series[sector] = pts
            meta[sector] = {
                "avg_rs_rating": round(pts[-1][1], 4),
                "num_stocks": int(last_weight),
            }
            ranked.append((sector, pts[-1][1]))

        ranked.sort(key=lambda t: t[1], reverse=True)
        for rank, (sector, _rs) in enumerate(ranked, start=1):
            meta[sector]["rank"] = rank
        return series, meta
