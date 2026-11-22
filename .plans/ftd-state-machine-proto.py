"""Prototype: v1 FTD heuristic (copied from origin/main) vs a rally-attempt state machine.

Usage: python ftd_proto.py SPY [correction_pct]
"""
import sys
import pandas as pd
import yfinance as yf

# ---- v1, verbatim logic from services/market_exposure_service.py -------------
def v1(df, gain=0.015, window=15, low_lookback=20, min_day=4):
    close = df["Close"]; pct = close.pct_change(fill_method=None)
    strong = (pct >= gain) & (df["Volume"] > df["Volume"].shift(1))
    lw = close.tail(low_lookback); low_pos = df.index.get_loc(lw.index[int(lw.values.argmin())])
    cand = strong.tail(window)
    for ts in reversed(list(cand.index)):
        if cand.loc[ts] and df.index.get_loc(ts) - low_pos >= min_day:
            return ts.date()
    return None

# ---- FSM ----------------------------------------------------------------------
def fsm(df, corr=0.08, gain=0.015, min_day=4):
    state, peak, low, day, ftd_low = "uptrend", None, None, 0, None
    events = []
    o = df.reset_index()
    for i, r in o.iterrows():
        d, c, lo, v = r["Date"].date(), r["Close"], r["Low"], r["Volume"]
        pc = o.at[i - 1, "Close"] if i else c; pv = o.at[i - 1, "Volume"] if i else v
        if state == "uptrend":
            peak = c if peak is None else max(peak, c)
            if ftd_low is not None and c < ftd_low:
                events.append((d, "ftd_failed")); state, low, day, ftd_low = "correction", lo, 0, None
            elif c <= peak * (1 - corr):
                events.append((d, "correction")); state, low, day, ftd_low = "correction", lo, 0, None
            continue
        if lo < low:                       # undercut resets the attempt
            if state == "rally_attempt": events.append((d, "attempt_reset"))
            state, low, day = "correction", lo, 0
            continue
        if state == "correction":
            if c > pc: state, day = "rally_attempt", 1
            continue
        day += 1                           # rally_attempt
        if day >= min_day and c / pc - 1 >= gain and v > pv:
            events.append((d, f"FTD day{day}")); state, peak, ftd_low = "uptrend", c, low
    return events, state, day

sym = sys.argv[1]; corr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
df = yf.download(sym, start="2021-06-01", auto_adjust=True, progress=False)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
df.index.name = "Date"
ev, st, day = fsm(df, corr=corr)
v1_days = sorted({x for i in range(220, len(df) + 1) if (x := v1(df.iloc[:i])) is not None and x == df.index[i - 1].date()})
print(f"{sym} corr={corr:.0%} through {df.index[-1].date()} | FSM now: {st} day {day}")
print("FSM events:", [(str(d), e) for d, e in ev if str(d) >= "2022-01-01"])
print(f"v1 FTD event days ({len(v1_days)}):", [str(d) for d in v1_days if str(d) >= "2022-01-01"])
