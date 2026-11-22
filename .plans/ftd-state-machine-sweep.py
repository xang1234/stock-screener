"""Sweep cycle-FSM parameters over every market benchmark. Read-only, yfinance."""
import sys
import numpy as np
import pandas as pd
import yfinance as yf

BENCH = {"US": "SPY", "HK": "^HSI", "IN": "^NSEI", "JP": "^N225", "KR": "^KS11", "TW": "^TWII",
         "CN": "000300.SS", "CA": "^GSPTSE", "DE": "^GDAXI", "SG": "^STI", "AU": "^AXJO", "MY": "^KLSE"}


def fsm(df, corr, gain, min_day=4, expiry=None, ma_entry=False):
    c, lo, v = df["Close"].to_numpy(), df["Low"].to_numpy(), df["Volume"].to_numpy()
    ma50 = df["Close"].rolling(50).mean().to_numpy()
    dist = df["dist"].to_numpy()
    state, peak, low, day, ftd_low = "up", c[0], None, 0, None
    ev, longest = [], 0
    for i in range(1, len(c)):
        if state == "up":
            peak = max(peak, c[i])
            broke = ftd_low is not None and c[i] < ftd_low
            entry = c[i] <= peak * (1 - corr) or (ma_entry and c[i] < ma50[i] and dist[i] >= 8)
            if broke or entry:
                ev.append((df.index[i].date(), "failed" if broke else "correction"))
                state, low, day, ftd_low = "corr", lo[i], 0, None
            continue
        if c[i] > peak:                      # recovered the pre-correction high without an FTD
            ev.append((df.index[i].date(), "recovered")); state, peak, ftd_low = "up", c[i], None
            continue
        if lo[i] < low:
            state, low, day = "corr", lo[i], 0
            continue
        if state == "corr":
            if c[i] > c[i - 1]:
                state, day = "att", 1
            continue
        day += 1
        longest = max(longest, day)
        g = gain[i] if hasattr(gain, "__len__") else gain
        if day >= min_day and c[i] / c[i - 1] - 1 >= g and v[i] > v[i - 1]:
            ev.append((df.index[i].date(), f"FTD d{day}"))
            state, peak, ftd_low = "up", c[i], low
        elif expiry and day >= expiry:
            state, day = "corr", 0  # attempt expired; same low, next up close is day 1
    return ev, longest


rows = []
for mkt, sym in BENCH.items():
    df = yf.download(sym, start="2021-06-01", auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.dropna(subset=["Close"])
    vol_ok = float((df["Volume"].fillna(0) > 0).mean())
    pct = df["Close"].pct_change()
    df["dist"] = ((pct <= -0.002) & (df["Volume"] > df["Volume"].shift(1))).rolling(25).sum()
    atr = ((df["High"] - df["Low"]) / df["Close"]).rolling(20).mean().to_numpy()
    df = df[df.index >= "2022-01-01"]
    atr = atr[-len(df):]
    for name, kw in {
        "8%/1.5%": dict(corr=0.08, gain=0.015),
        "8%/1.5%/exp25+recover": dict(corr=0.08, gain=0.015, expiry=25),
    }.items():
        ev, longest = fsm(df, **kw)
        ftd = [d for d, e in ev if e.startswith("FTD")]
        fails = sum(e == "failed" for _, e in ev); rec = sum(e == "recovered" for _, e in ev)
        rows.append((mkt, sym, f"{vol_ok:.0%}", name, sum(e == "correction" for _, e in ev), len(ftd), fails, rec, longest,
                     ",".join(str(d)[2:] for d in ftd[-4:])))
out = pd.DataFrame(rows, columns=["mkt", "sym", "vol>0", "params", "corr", "ftd", "fail", "recov", "max_att", "last_ftds"])
pd.set_option("display.width", 250)
print(out.to_string(index=False))
