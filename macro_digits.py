import os
import time
import json
import argparse
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import pytz
from dotenv import load_dotenv

from tools.io_utils import ensure_out_dir, now_et_iso, write_json

# Yahoo Finance
import yfinance as yf

# FRED
from fredapi import Fred

#python crawl_news.py --date 2026-01-23 --asof 09:25  

NY_TZ = pytz.timezone("America/New_York")
UTC = pytz.UTC

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True, parents=True)


# -----------------------------
# Helpers
# -----------------------------
def now_ny() -> datetime:
    return datetime.now(NY_TZ)


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "NA"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def safe_float(x) -> Optional[float]:
    """Convert scalars / 1-element Series / numpy scalars -> float, else None."""
    if x is None:
        return None

    # Pandas Series / Index (grab first element)
    if isinstance(x, (pd.Series, pd.Index)):
        if len(x) == 0:
            return None
        x = x.iloc[0]

    # Numpy scalar
    if isinstance(x, (np.generic,)):
        x = x.item()

    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def filter_window(df, t0, t1):
    """Filter dataframe to time window [t0, t1] inclusive."""
    if df is None or df.empty:
        return df
    return df[(df.index >= t0) & (df.index <= t1)]


def get_premarket_window(ref_dt: datetime, asof: Optional[str] = None) -> Tuple[datetime, datetime, str]:
    """
    Returns (t0, t_end, mode)
    - ref_dt: The target date (today or historical)
    - t0 = 04:00 ET on ref_dt
    - t_end:
        - If ref_dt is TODAY: now (clamped to 09:30)
        - If ref_dt is PAST: 09:30 (full replay)
        - If asof is set: specific time on ref_dt
    """
    real_now = now_ny()
    
    t0 = ref_dt.replace(hour=4, minute=0, second=0, microsecond=0)
    open_time = ref_dt.replace(hour=9, minute=30, second=0, microsecond=0)

    # Determine default t_end based on whether we are looking at today or history
    if ref_dt.date() < real_now.date():
        # Historical: Default to full premarket (up to open)
        t_end = open_time
        mode = f"HISTORICAL {ref_dt.strftime('%Y-%m-%d')}"
    else:
        # Today: Default to "now", capped at open
        t_end = real_now if real_now < open_time else open_time
        mode = "LIVE PREMARKET" if real_now < open_time else "REPLAY (CAPPED TO 09:30)"

    if asof:
        try:
            hh, mm = [int(x) for x in asof.split(":")]
            t_req = ref_dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
            
            # Clamp logic
            if t_req < t0: t_req = t0
            if t_req > open_time: t_req = open_time
            
            t_end = t_req
            mode += f" (ASOF {t_end.strftime('%H:%M')})"
        except ValueError:
            print(f"⚠️ Invalid format for --asof '{asof}'. Expected HH:MM. Using default.")

    return t0, t_end, mode


def premarket_start_dt(ref: datetime) -> datetime:
    # 04:00 ET for the same day as ref
    return ref.replace(hour=4, minute=0, second=0, microsecond=0)


def to_ny_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance timestamps are often UTC. Convert to NY if tz-aware; if naive, assume UTC then convert.
    """
    if df is None or df.empty:
        return df
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(UTC)
    df = df.copy()
    df.index = idx.tz_convert(NY_TZ)
    return df


def get_prev_close_daily(symbols: List[str], ref_date: datetime) -> Dict[str, Optional[float]]:
    """
    Fetch previous regular close relative to ref_date.
    If ref_date is today, fetch recent days.
    If ref_date is historical, fetch days leading up to it.
    """
    if not symbols:
        return {}

    # For historical, we need an end date that includes the day BEFORE ref_date
    # We'll just grab a chunk of time ending at ref_date
    start_dt = ref_date - timedelta(days=10)
    
    try:
        data = yf.download(
            tickers=" ".join(symbols),
            start=start_dt.strftime("%Y-%m-%d"),
            end=ref_date.strftime("%Y-%m-%d"), # yf end is exclusive, so this gets us data strictly before ref_date
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
    except Exception:
        return {}

    if data is None or data.empty:
        return {}

    prev_close: Dict[str, Optional[float]] = {}
    
    # Check if we have a single ticker (columns are single level or just OHLC)
    is_multi = isinstance(data.columns, pd.MultiIndex)
    
    # If single symbol, force it to be handled consistently if possible, or just handle direct
    if not is_multi and len(symbols) == 1:
        sym = symbols[0]
        if "Close" in data.columns:
            closes = data["Close"].dropna()
            if not closes.empty:
                prev_close[sym] = safe_float(closes.iloc[-1])
            else:
                prev_close[sym] = None
        return prev_close

    # MultiIndex columns for multi-ticker
    for sym in symbols:
        try:
            if is_multi:
                if sym in data.columns.get_level_values(0):
                    closes = data[(sym, "Close")].dropna()
                else:
                    prev_close[sym] = None
                    continue
            else:
                prev_close[sym] = None
                continue

            if not closes.empty:
                prev_close[sym] = safe_float(closes.iloc[-1])
            else:
                prev_close[sym] = None
        except Exception:
            prev_close[sym] = None

    return prev_close


def get_intraday_series(sym: str, ref_date: datetime, interval: str = "1m", retries: int = 2) -> pd.DataFrame:
    """
    Pull intraday bars with retries.
    If ref_date is today, use period="1d".
    If ref_date is past, use start/end.
    """
    is_today = (ref_date.date() == now_ny().date())
    
    # Setup args
    kwargs = {
        "tickers": sym,
        "interval": interval,
        "prepost": True,
        "auto_adjust": False,
        "threads": False,
        "progress": False,
    }

    if is_today:
        kwargs["period"] = "1d"
    else:
        # yfinance expected start/end string YYYY-MM-DD
        # end is exclusive, so we add 1 day to cover the full ref_date
        kwargs["start"] = ref_date.strftime("%Y-%m-%d")
        kwargs["end"] = (ref_date + timedelta(days=1)).strftime("%Y-%m-%d")

    for i in range(retries + 1):
        try:
            df = yf.download(**kwargs)
            
            if df is None or df.empty:
                if i < retries:
                    time.sleep(1.0)
                    continue
                return pd.DataFrame()
            
            df = to_ny_index(df)
            if df.empty: 
                 if i < retries:
                    time.sleep(1.0)
                    continue
                 return pd.DataFrame()
                 
            return df
        except Exception:
            if i < retries:
                time.sleep(1.0)
                continue
            return pd.DataFrame()
            
    return pd.DataFrame()


def compute_premarket_return(sym: str, t_start: datetime, t_end: datetime) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Return (start_price, last_price, pct_return) within the window [t_start, t_end].
    """
    # Performance optimization: Use 1m only for liquid proxies, 5m for sectors/others
    liquid_proxies = ["ES=F", "NQ=F", "YM=F", "RTY=F", "SPY", "QQQ", "DIA", "IWM"]
    primary_interval = "1m" if sym in liquid_proxies else "5m"

    # We determine ref_date from t_start
    ref_date = t_start

    df = get_intraday_series(sym, ref_date=ref_date, interval=primary_interval)
    
    # If 1m failed or was empty (and we used 1m), try 5m fallback
    # Often 1m data is only available for 7 days, so historical > 7d needs 5m
    if df.empty and primary_interval == "1m":
        df = get_intraday_series(sym, ref_date=ref_date, interval="5m")
        
    if df.empty:
        return None, None, None

    # Cap the data to the requested window (e.g. 04:00 to 09:30)
    # We filter purely by time.
    mask = (df.index >= t_start) & (df.index <= t_end)
    df_window = df[mask]

    if df_window.empty:
        return None, None, None

    if "Open" not in df_window.columns or "Close" not in df_window.columns:
        return None, None, None

    # Get first Open in window and last Close in window
    p0 = safe_float(df_window["Open"].iloc[0])
    plast = safe_float(df_window["Close"].iloc[-1])

    if p0 is None or plast is None or p0 == 0:
        return p0, plast, None
        
    pct = (plast / p0 - 1.0) * 100.0
    return p0, plast, pct


# -----------------------------
# New Technical Context Logic
# -----------------------------

def compute_range_position(sym: str, t0: datetime, t_end: datetime):
    """
    Returns dict with premarket high/low/last and range position.
    range_pos = (last - low) / (high - low)
    """
    # Try 1m then 5m, using t0 as reference date
    df = get_intraday_series(sym, ref_date=t0, interval="1m")
    if df is None or df.empty:
        df = get_intraday_series(sym, ref_date=t0, interval="5m")
        
    if df is None or df.empty:
        return {"sym": sym, "high": None, "low": None, "last": None, "range_pos": None, "label": "NA (no data)"}

    df = filter_window(df, t0, t_end)
    if df is None or df.empty:
        return {"sym": sym, "high": None, "low": None, "last": None, "range_pos": None, "label": "NA (empty window)"}

    hi = safe_float(df["High"].max()) if "High" in df.columns else None
    lo = safe_float(df["Low"].min()) if "Low" in df.columns else None
    last = safe_float(df["Close"].iloc[-1]) if "Close" in df.columns else None

    if hi is None or lo is None or last is None or hi == lo:
        return {"sym": sym, "high": hi, "low": lo, "last": last, "range_pos": None, "label": "NA (bad range)"}

    range_pos = (last - lo) / (hi - lo)

    # Interpret
    if range_pos <= 0.20:
        label = "Pressing lows (bearish pressure)"
    elif range_pos >= 0.80:
        label = "Pressing highs (bullish pressure)"
    else:
        label = "Mid-range (chop / indecision)"

    return {"sym": sym, "high": hi, "low": lo, "last": last, "range_pos": range_pos, "label": label}


def print_range_position(symbols, t0, t_end, results=None):
    print("\n[PREMARKET RANGE POSITION]  (04:00 -> asof)")
    if results is None:
        results = [compute_range_position(sym, t0, t_end) for sym in symbols]
    for r in results:
        rp = r["range_pos"]
        rp_str = f"{rp:.3f}" if rp is not None else "NA"
        sym = r.get("sym", "")
        print(
            f"  {sym:7s} | High={fmt(r['high'],4)} Low={fmt(r['low'],4)} Last={fmt(r['last'],4)} "
            f"| RangePos={rp_str:>5s} | {r['label']}"
        )


def compute_return_in_window(sym: str, t_start: datetime, t_end: datetime):
    """
    Return % from first bar Open in [t_start,t_end] to last bar Close in [t_start,t_end].
    Distinct from compute_premarket_return as it focuses strictly on bar data within bounds.
    """
    df = get_intraday_series(sym, ref_date=t_start, interval="1m")
    if df is None or df.empty:
        df = get_intraday_series(sym, ref_date=t_start, interval="5m")
    if df is None or df.empty:
        return None

    df = filter_window(df, t_start, t_end)
    if df is None or df.empty:
        return None

    o0 = safe_float(df["Open"].iloc[0]) if "Open" in df.columns else None
    c1 = safe_float(df["Close"].iloc[-1]) if "Close" in df.columns else None
    if o0 is None or c1 is None or o0 == 0:
        return None

    return (c1 / o0 - 1.0) * 100.0


def compute_shock(sym: str, t0: datetime, asof: datetime,
                  shock_start_hm=(8,55), shock_end_hm=(9,25),
                  thresh=0.20):
    """
    Compares:
      early_ret: 04:00 -> 08:55
      late_ret : 08:55 -> 09:25  (but capped by asof)
    If asof < 08:55, returns NA.
    thresh is percent move threshold for SHOCK labels.
    """
    # Build shock window times on the same date as asof
    shock_start = asof.replace(hour=shock_start_hm[0], minute=shock_start_hm[1], second=0, microsecond=0)
    shock_end   = asof.replace(hour=shock_end_hm[0],   minute=shock_end_hm[1],   second=0, microsecond=0)

    # Clamp shock_end to asof (so replay at 09:10 doesn’t try to go past it)
    if asof < shock_start:
        return {"sym": sym, "early_ret": None, "late_ret": None, "label": "NA (asof < 08:55)"}

    late_end = shock_end if asof >= shock_end else asof

    early_ret = compute_return_in_window(sym, t0, shock_start)
    late_ret  = compute_return_in_window(sym, shock_start, late_end)

    if late_ret is None:
        return {"sym": sym, "early_ret": early_ret, "late_ret": None, "label": "NA (late window missing)"}

    # Label shock direction using late_ret
    if late_ret <= -thresh:
        label = "SHOCK DOWN (accelerating selling into open)"
    elif late_ret >= +thresh:
        label = "SHOCK UP (reversal/squeeze risk into open)"
    else:
        label = "NO SHOCK (late window calm / drift)"

    return {"sym": sym, "early_ret": early_ret, "late_ret": late_ret, "label": label,
            "late_window": f"{shock_start.strftime('%H:%M')}->{late_end.strftime('%H:%M')}"}


def print_shock(symbols, t0, asof, results=None):
    print("\n[OPEN SHOCK DETECTOR]  (compare 04:00->08:55 vs 08:55->09:25)")
    if results is None:
        results = [compute_shock(sym, t0, asof) for sym in symbols]
    for s in results:
        er = fmt(s.get("early_ret"), 3) if s.get("early_ret") is not None else "NA"
        lr = fmt(s.get("late_ret"), 3) if s.get("late_ret") is not None else "NA"
        lw = s.get("late_window", "08:55->09:25")
        sym = s.get("sym", "")
        print(f"  {sym:7s} | EarlyRet(04:00->08:55)={er:>7s}% | LateRet({lw})={lr:>7s}% | {s['label']}")


# -----------------------------
# Data Structures
# -----------------------------

@dataclass
class QuoteRow:
    symbol: str
    prev_close: Optional[float]
    last: Optional[float]
    chg: Optional[float]
    pct: Optional[float]


def build_quote_rows(symbols: List[str], t_start: datetime, t_end: datetime) -> List[QuoteRow]:
    ref_date = t_start
    prevs = get_prev_close_daily(symbols, ref_date)
    rows: List[QuoteRow] = []

    for sym in symbols:
        # returns calc uses capped time
        p0, plast, _prem_ret = compute_premarket_return(sym, t_start, t_end)
        prev = prevs.get(sym)
        
        # For 'last', we use the last price found in the premarket window
        last = plast

        chg = (last - prev) if (last is not None and prev is not None) else None
        pct = ((last / prev - 1.0) * 100.0) if (last is not None and prev not in (None, 0)) else None

        rows.append(QuoteRow(sym, prev, last, chg, pct))

    return rows


# -----------------------------
# FRED
# -----------------------------
def fred_latest(fred: Fred, series_id: str) -> Optional[float]:
    try:
        s = fred.get_series(series_id)
        s = s.dropna()
        if len(s) == 0:
            return None
        return safe_float(s.iloc[-1])
    except Exception:
        return None


def fred_change(fred: Fred, series_id: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (latest, change_vs_prev)
    """
    try:
        s = fred.get_series(series_id).dropna()
        if len(s) == 0:
            return None, None
        latest = safe_float(s.iloc[-1])
        prev = safe_float(s.iloc[-2]) if len(s) >= 2 else None
        chg = (latest - prev) if (latest is not None and prev is not None) else None
        return latest, chg
    except Exception:
        return None, None


# -----------------------------
# Scoring
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def score_from_threshold(value: Optional[float], lo: float, hi: float) -> float:
    """
    Map value into [-1, +1] via thresholds.
    <= lo => -1, >= hi => +1, linear in between.
    """
    if value is None:
        return 0.0
    if value <= lo:
        return -1.0
    if value >= hi:
        return +1.0
    return (value - lo) / (hi - lo) * 2.0 - 1.0


def compute_risk_score(features: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, float]]:
    """
    Returns (score_0_to_100, component_scores)
    Convention: higher score = MORE risk-off.
    """
    # Each component yields risk-off contribution in [0..1] roughly
    comps: Dict[str, float] = {}

    # Futures / proxies: negative returns => risk-off
    # We'll treat returns in % (premarket 04:00->now). -0.6% is meaningful, +0.6% is meaningful.
    for key in ["ES_prem_ret", "NQ_prem_ret", "SPY_prem_ret", "QQQ_prem_ret", "DIA_prem_ret", "IWM_prem_ret"]:
        v = features.get(key)
        # Convert to risk-off: negative => +, positive => -
        # score_from_threshold outputs -1..+1, where + means positive return.
        s = score_from_threshold(v, lo=-0.6, hi=+0.6)
        # risk-off contribution: invert, map to 0..1
        comps[key] = clamp((1.0 - s) / 2.0, 0.0, 1.0)

    # Sector rotation: XLK - XLP and XLK - XLI (premarket)
    # Negative => defensive outperforms tech => risk-off
    for key in ["XLK_minus_XLP", "XLK_minus_XLI"]:
        v = features.get(key)
        s = score_from_threshold(v, lo=-0.4, hi=+0.4)  # percent points
        comps[key] = clamp((1.0 - s) / 2.0, 0.0, 1.0)

    # Vol: VIX change vs prev close (%). Rising VIX = risk-off.
    vix_pct = features.get("VIX_pct_vs_prev")
    if vix_pct is None:
        vix_pct = features.get("VX1_pct_vs_prev")  # fallback proxy if you have it
    s = score_from_threshold(vix_pct, lo=-3.0, hi=+6.0)
    comps["VOL"] = clamp((s + 1.0) / 2.0, 0.0, 1.0)  # rising => more risk-off

    # Rates: DGS10 change (daily, percentage points). Falling yields often risk-off.
    dgs10_chg = features.get("DGS10_chg")
    # If yields drop (negative), that's risk-off => higher contribution
    # Map -0.07 => strong risk-off, +0.07 => risk-on
    s = score_from_threshold(dgs10_chg, lo=-0.07, hi=+0.07)
    comps["RATES"] = clamp((1.0 - s) / 2.0, 0.0, 1.0)

    # Credit: HY OAS and BAA10YM changes (daily). Widening spreads => risk-off.
    hy_chg = features.get("HY_OAS_chg")
    baa_chg = features.get("BAA10YM_chg")
    # HY OAS in percentage points typically; widening positive = risk-off
    s_hy = score_from_threshold(hy_chg, lo=-0.05, hi=+0.08)
    comps["CREDIT_HY"] = clamp((s_hy + 1.0) / 2.0, 0.0, 1.0)
    s_baa = score_from_threshold(baa_chg, lo=-0.05, hi=+0.08)
    comps["CREDIT_BAA"] = clamp((s_baa + 1.0) / 2.0, 0.0, 1.0)

    # Breadth: adv - decl ratio in [-1..+1]. Negative => risk-off.
    breadth = features.get("BREADTH_ADVDECL_RATIO")
    s = score_from_threshold(breadth, lo=-0.35, hi=+0.35)
    comps["BREADTH"] = clamp((1.0 - s) / 2.0, 0.0, 1.0)

    # Weights (tweakable)
    weights = {
        # Futures/proxies
        "ES_prem_ret": 1.6,
        "NQ_prem_ret": 1.6,
        "SPY_prem_ret": 1.0,
        "QQQ_prem_ret": 1.0,
        "DIA_prem_ret": 0.6,
        "IWM_prem_ret": 0.8,
        # Rotation
        "XLK_minus_XLP": 1.2,
        "XLK_minus_XLI": 0.8,
        # Vol / macro / breadth
        "VOL": 1.2,
        "RATES": 0.8,
        "CREDIT_HY": 0.9,
        "CREDIT_BAA": 0.5,
        "BREADTH": 1.2,
    }

    # Weighted average -> 0..100
    total_w = 0.0
    total = 0.0
    for k, w in weights.items():
        v = comps.get(k, 0.5)
        total += w * v
        total_w += w

    raw = total / total_w if total_w else 0.5
    score_0_100 = clamp(raw * 100.0, 0.0, 100.0)
    return score_0_100, comps


# -----------------------------
# Breadth basket
# -----------------------------
DEFAULT_BREADTH_BASKET = [
    # Mega/large (liquid, good for a "quick breadth" proxy)
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","V",
    "UNH","XOM","LLY","AVGO","MA","HD","COST","KO","PEP","BAC",
    "WMT","TMO","ABBV","CRM","NFLX","ADBE","CSCO","ORCL","INTC","AMD",
    "QCOM","TXN","NOW","AMAT","MU","GE","CAT","BA","HON","GS",
    "MS","IBM","DIS","NKE","UPS","SBUX","PFE","CVX","PM"
]


def _get_series_for_symbol(df: pd.DataFrame, sym: str, field: str = "Close") -> pd.Series:
    """
    Works with yfinance output where columns can be:
      - MultiIndex (sym, field)  OR  (field, sym)
      - Single index (field) for single ticker
    Returns an empty Series if not found.
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    cols = df.columns

    # MultiIndex case
    if isinstance(cols, pd.MultiIndex):
        if (sym, field) in cols:
            return df[(sym, field)]
        if (field, sym) in cols:
            return df[(field, sym)]
        return pd.Series(dtype="float64")

    # Single ticker case
    if field in cols:
        return df[field]

    return pd.Series(dtype="float64")


def compute_breadth(basket: List[str], t_cap: datetime) -> Dict[str, Optional[float]]:
    """
    Breadth proxy:
    - prev close from daily bars
    - last from intraday (prepost), capped at t_cap
    - count adv/dec
    """
    out = {
        "ADV": 0, "DECL": 0, "UNCH": 0,
        "ADVDECL_RATIO": None,
        "BASKET_N": len(basket),
        "COVERAGE": 0.0,
        "USED": 0,
        "MISSING_PREV": 0,
        "MISSING_LAST": 0
    }
    if not basket:
        return out

    ref_date = t_cap
    prevs = get_prev_close_daily(basket, ref_date)

    # Setup download args
    is_today = (ref_date.date() == now_ny().date())
    kwargs = {
        "tickers": " ".join(basket),
        "interval": "5m", # 5m is safer for historical breadth
        "prepost": True,
        "group_by": "ticker",
        "auto_adjust": False,
        "threads": True,
        "progress": False,
    }
    if is_today:
        kwargs["period"] = "1d"
    else:
        kwargs["start"] = ref_date.strftime("%Y-%m-%d")
        kwargs["end"] = (ref_date + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        intr = yf.download(**kwargs)
    except Exception:
        intr = pd.DataFrame()
    
    intr = to_ny_index(intr)
    
    # Cap to window end
    intr = intr[intr.index <= t_cap]

    used = 0
    adv = decl = unch = 0
    
    for sym in basket:
        prev = prevs.get(sym)
        if prev is None:
            out["MISSING_PREV"] += 1
            continue
        
        # Robust extraction
        s_close = _get_series_for_symbol(intr, sym, "Close").dropna()
        
        if s_close.empty:
            out["MISSING_LAST"] += 1
            continue
            
        last = float(s_close.iloc[-1])
        used += 1

        if last > prev:
            adv += 1
        elif last < prev:
            decl += 1
        else:
            unch += 1

    denom = adv + decl
    # Ratio in [-1, +1], but keep None if no data
    if denom > 0:
        ratio = (adv - decl) / denom
    else:
        # If USED==0 or just no adv/decl (all unch), we technically have a ratio of 0 if unch > 0
        # But if used > 0 and denom == 0 (all unch), ratio is 0.
        # If used == 0, ratio is None.
        ratio = 0.0 if used > 0 else None
    
    # Calculate coverage
    coverage = (used / len(basket)) * 100.0 if basket else 0.0

    out["ADV"] = adv
    out["DECL"] = decl
    out["UNCH"] = unch
    out["ADVDECL_RATIO"] = ratio
    out["USED"] = used
    out["COVERAGE"] = coverage
    
    print(f"   [Breadth Debug] Basket={len(basket)} Used={used} MissingPrev={out['MISSING_PREV']} MissingLast={out['MISSING_LAST']} Coverage={coverage:.1f}%")
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Premarket Risk-On/Risk-Off Context (Yahoo Finance + FRED)")
    ap.add_argument("--basket-size", type=int, default=50, help="Breadth basket size (default 50)")
    ap.add_argument("--no-breadth", action="store_true", help="Skip breadth computation (faster)")
    ap.add_argument("--json", action="store_true", help="Also write JSON output to out/")
    ap.add_argument("--out_dir", type=str, default="out", help="Output directory (default: out)")
    ap.add_argument("--out_json", type=str, default=None, help="Write structured JSON output to a specific path")
    ap.add_argument("--out_jsonl", type=str, default=None, help="Write JSONL output (unused; reserved)")
    ap.add_argument("--asof", type=str, default=None, help="Replay as-of time in ET (HH:MM). Example: --asof 09:25")
    ap.add_argument("--date", type=str, default=None, help="Historical date YYYY-MM-DD (defaults to today)")
    args = ap.parse_args()

    load_dotenv()
    out_dir = ensure_out_dir(args.out_dir)

    # Capture run time immediately
    t_run = now_ny()

    fred_key = os.getenv("FRED_API_KEY", "").strip()
    if not fred_key:
        print("❌ Missing FRED_API_KEY in environment. Add it to your .env or export it.")
        print("   You can still run without FRED by setting dummy key, but FRED features will be NA.")
    fred = Fred(api_key=fred_key) if fred_key else None

    # Determine reference date (today or historical)
    if args.date:
        try:
            d_parsed = datetime.strptime(args.date, "%Y-%m-%d")
            # Create a localized datetime for that morning
            # We use 09:00 as a generic anchor, but get_premarket_window fixes the exact window
            ref_dt = NY_TZ.localize(d_parsed.replace(hour=9, minute=0, second=0))
        except ValueError:
            print(f"❌ Invalid date format: {args.date}. Use YYYY-MM-DD.")
            return
    else:
        ref_dt = t_run
    
    # Calculate window using the new helper
    t_start, t_end, mode = get_premarket_window(ref_dt, args.asof)

    # ---- Symbols ----
    futures = ["ES=F", "NQ=F", "YM=F", "RTY=F"]
    proxies = ["SPY", "QQQ", "DIA", "IWM"]
    sectors = ["XLK", "XLP", "XLI"]
    vol = ["^VIX", "VXX"]  # or UVXY/UVIX if you prefer


    all_syms = futures + proxies + sectors + vol

    # ---- Quotes ----
    # Pass t_end so quotes reflect the premarket window, not post-market prices if run late.
    quote_rows = build_quote_rows(all_syms, t_start, t_end)

    # Premarket returns for selected
    prem_rets: Dict[str, Optional[float]] = {}
    for sym in futures + proxies + sectors:
        _p0, _plast, pct = compute_premarket_return(sym, t_start, t_end)
        prem_rets[f"{sym}_prem_ret"] = pct

    # Sector rotation features
    XLK = prem_rets.get("XLK_prem_ret")
    XLP = prem_rets.get("XLP_prem_ret")
    XLI = prem_rets.get("XLI_prem_ret")
    XLK_minus_XLP = (XLK - XLP) if (XLK is not None and XLP is not None) else None
    XLK_minus_XLI = (XLK - XLI) if (XLK is not None and XLI is not None) else None

    # VIX/VX1 pct vs prev close
    # Note: QuoteRow.last is already capped at t_end inside build_quote_rows
    vix_prev = next((r.prev_close for r in quote_rows if r.symbol == "^VIX"), None)
    vix_last = next((r.last for r in quote_rows if r.symbol == "^VIX"), None)
    vix_pct_vs_prev = ((vix_last / vix_prev - 1) * 100) if (vix_last is not None and vix_prev not in (None, 0)) else None

    vx_prev = next((r.prev_close for r in quote_rows if r.symbol == "^VX1"), None)
    vx_last = next((r.last for r in quote_rows if r.symbol == "^VX1"), None)
    vx_pct_vs_prev = ((vx_last / vx_prev - 1) * 100) if (vx_last is not None and vx_prev not in (None, 0)) else None

    # ---- FRED ----
    DGS10_latest = DGS10_chg = None
    HY_latest = HY_chg = None
    BAA_latest = BAA_chg = None

    if fred:
        DGS10_latest, DGS10_chg = fred_change(fred, "DGS10")
        HY_latest, HY_chg = fred_change(fred, "BAMLH0A0HYM2")
        BAA_latest, BAA_chg = fred_change(fred, "BAA10YM")

    # ---- Breadth ----
    breadth = {"ADVDECL_RATIO": None, "ADV": None, "DECL": None, "UNCH": None, "BASKET_N": None}
    if not args.no_breadth:
        basket = DEFAULT_BREADTH_BASKET[: max(10, min(args.basket_size, len(DEFAULT_BREADTH_BASKET)))]
        # Pass t_end to cap breadth calculation
        b = compute_breadth(basket, t_end)
        breadth = {
            "ADVDECL_RATIO": b.get("ADVDECL_RATIO"),
            "ADV": b.get("ADV"),
            "DECL": b.get("DECL"),
            "UNCH": b.get("UNCH"),
            "BASKET_N": b.get("BASKET_N"),
        }

    # ---- Features dict for scoring ----
    features = {
        "ES_prem_ret": prem_rets.get("ES=F_prem_ret"),
        "NQ_prem_ret": prem_rets.get("NQ=F_prem_ret"),
        "YM_prem_ret": prem_rets.get("YM=F_prem_ret"),
        "RTY_prem_ret": prem_rets.get("RTY=F_prem_ret"),
        "SPY_prem_ret": prem_rets.get("SPY_prem_ret"),
        "QQQ_prem_ret": prem_rets.get("QQQ_prem_ret"),
        "DIA_prem_ret": prem_rets.get("DIA_prem_ret"),
        "IWM_prem_ret": prem_rets.get("IWM_prem_ret"),
        "XLK_minus_XLP": XLK_minus_XLP,
        "XLK_minus_XLI": XLK_minus_XLI,
        "VIX_pct_vs_prev": vix_pct_vs_prev,
        "VX1_pct_vs_prev": vx_pct_vs_prev,
        "DGS10_latest": DGS10_latest,
        "DGS10_chg": DGS10_chg,
        "HY_OAS_latest": HY_latest,
        "HY_OAS_chg": HY_chg,
        "BAA10YM_latest": BAA_latest,
        "BAA10YM_chg": BAA_chg,
        "BREADTH_ADVDECL_RATIO": breadth.get("ADVDECL_RATIO"),
    }

    score_0_100, comps = compute_risk_score(features)

    # ---- Print ----
    print("\n" + "=" * 92)
    print(f"PREMARKET CONTEXT  |  MODE: {mode}  |  Window: {t_start.strftime('%H:%M')} -> {t_end.strftime('%H:%M')} ET")
    print("=" * 92)

    # Quotes table
    dfq = pd.DataFrame([{
        "Symbol": r.symbol,
        "PrevClose": r.prev_close,
        "Last": r.last,
        "Chg": r.chg,
        "PctVsPrev": r.pct,
    } for r in quote_rows])

    # Sort for readability
    order = futures + proxies + sectors + vol
    dfq["__ord"] = dfq["Symbol"].apply(lambda s: order.index(s) if s in order else 999)
    dfq = dfq.sort_values("__ord").drop(columns="__ord")

    with pd.option_context("display.max_rows", 200, "display.width", 140):
        print("\n[RAW QUOTES] (Last price is capped at 09:30 ET if run later)")
        print(dfq.to_string(index=False, formatters={
            "PrevClose": lambda x: fmt(x, 4),
            "Last": lambda x: fmt(x, 4),
            "Chg": lambda x: fmt(x, 4),
            "PctVsPrev": lambda x: fmt(x, 2),
        }))

    # Premarket returns
    print(f"\n[PREMARKET RETURNS] ({t_start.strftime('%H:%M')} -> {t_end.strftime('%H:%M')})  %")
    for sym in futures + proxies + sectors:
        key = f"{sym}_prem_ret"
        print(f"  {sym:7s}: {fmt(prem_rets.get(key), 3)}")

    # ----------------------------------------------------
    # NEW: Range Position & Shock Detector Output
    # ----------------------------------------------------
    # Use QQQ/SPY as primary “tone” instruments. Add futures if you want.
    tone_syms = ["QQQ", "SPY", "NQ=F", "ES=F"]
    range_results = [compute_range_position(sym, t_start, t_end) for sym in tone_syms]
    shock_results = [compute_shock(sym, t_start, t_end) for sym in ["QQQ", "SPY"]]
    
    # We pass t_end as the 'asof' time for consistency
    print_range_position(tone_syms, t_start, t_end, results=range_results)
    print_shock(["QQQ", "SPY"], t_start, t_end, results=shock_results)
    # ----------------------------------------------------

    # Rotation + Vol
    print("\n[SECTOR ROTATION]")
    print(f"  XLK - XLP (pp): {fmt(XLK_minus_XLP, 3)}   (tech vs staples)")
    print(f"  XLK - XLI (pp): {fmt(XLK_minus_XLI, 3)}   (tech vs industrials)")

    print("\n[VOL]")
    print(f"  ^VIX pct vs prev close:    {fmt(vix_pct_vs_prev, 3)}")
    print(f"  ^VX1 pct vs prev close:    {fmt(vx_pct_vs_prev, 3)}  (futures proxy)")

    # FRED
    print("\n[FRED MACRO] (daily series)")
    print(f"  DGS10 (10Y):       latest={fmt(DGS10_latest,3)}  chg={fmt(DGS10_chg,3)}")
    print(f"  HY OAS:            latest={fmt(HY_latest,3)}     chg={fmt(HY_chg,3)}")
    print(f"  BAA10YM spread:    latest={fmt(BAA_latest,3)}     chg={fmt(BAA_chg,3)}")

    # Breadth
    print("\n[BREADTH PROXY]")
    if args.no_breadth:
        print("  (skipped)")
    else:
        print(f"  Basket N={breadth.get('BASKET_N')} | ADV={breadth.get('ADV')} DECL={breadth.get('DECL')} UNCH={breadth.get('UNCH')}")
        print(f"  ADV/DECL ratio ([-1..+1]): {fmt(breadth.get('ADVDECL_RATIO'), 3)}")

    # Score
    label = "RISK-OFF" if score_0_100 >= 60 else "RISK-ON" if score_0_100 <= 40 else "MIXED"
    print("\n" + "-" * 92)
    print(f"COMPOSITE SCORE (0..100, higher = more risk-off): {score_0_100:.1f}  =>  {label}")
    print("-" * 92)

    # Component contributions
    comp_sorted = sorted(comps.items(), key=lambda kv: kv[1], reverse=True)
    print("\n[COMPONENTS] (0..1 risk-off contribution)")
    for k, v in comp_sorted:
        print(f"  {k:18s}: {v:.3f}")

    # JSON output
    t_asof = t_end  # The simulated time is the end of the window
    raw_quotes = [
        {
            "symbol": r.symbol,
            "prev_close": r.prev_close,
            "last": r.last,
            "chg": r.chg,
            "pct_vs_prev": r.pct,
        }
        for r in quote_rows
    ]
    premarket_returns_pct = {sym: prem_rets.get(f"{sym}_prem_ret") for sym in (futures + proxies + sectors)}
    meta_date = ref_dt.strftime("%Y-%m-%d")
    meta_asof = t_asof.strftime("%H:%M")
    out_json_payload = {
        "meta": {
            "source_script": "macro_digits.py",
            "generated_at_et": now_et_iso(),
            "date": meta_date,
            "asof_et": meta_asof,
            "run_id": f"{meta_date}_{meta_asof}_macro_digits.py",
        },
        "run_meta": {
            "date": meta_date,
            "asof": meta_asof,
            "window": {"start_et": t_start.isoformat(), "end_et": t_end.isoformat()},
            "mode": mode,
        },
        "raw_quotes": raw_quotes,
        "premarket_returns_pct": premarket_returns_pct,
        "premarket_range_position": range_results,
        "open_shock": shock_results,
        "sector_rotation_pp": {
            "XLK_minus_XLP": XLK_minus_XLP,
            "XLK_minus_XLI": XLK_minus_XLI,
        },
        "vol": {
            "VIX_pct_vs_prev": vix_pct_vs_prev,
            "VX1_pct_vs_prev": vx_pct_vs_prev,
        },
        "fred_macro": {
            "DGS10_latest": DGS10_latest,
            "DGS10_chg": DGS10_chg,
            "HY_OAS_latest": HY_latest,
            "HY_OAS_chg": HY_chg,
            "BAA10YM_latest": BAA_latest,
            "BAA10YM_chg": BAA_chg,
        },
        "breadth_proxy": breadth,
        "composite_score": {"score_0_100": score_0_100, "label": label},
        "components_risk_off_0_1": comps,
    }
    payload = {
        "run_time_ny": t_run.isoformat(),
        "asof_time_ny": t_asof.isoformat(),
        "window_start_ny": t_start.isoformat(),
        "window_end_ny": t_end.isoformat(),
        "mode": mode,
        "quotes": dfq.to_dict(orient="records"),
        "premarket_returns": {k: v for k, v in prem_rets.items()},
        "features": features,
        "components": comps,
        "score": score_0_100,
        "label": label,
        "notes": [
            "Yahoo Finance data can be delayed/stale; use broker feed for execution decisions.",
            "FRED macro series are daily; intra-day regime shifts won't show up here.",
            "Breadth is a proxy over a basket, not true NYSE A/D.",
            "Analysis window is capped at 09:30 ET to simulate premarket conditions."
        ],
    }

    if args.json:
        fn = out_dir / f"premarket_context_{t_run.strftime('%Y%m%d_%H%M%S')}.json"
        fn.write_text(json.dumps(payload, indent=2))
        print(f"\n✅ Wrote: {fn}")

    if args.out_json:
        out_path = Path(args.out_json)
        write_json(out_path, out_json_payload)
        print(f"\n✅ Wrote: {out_path}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
