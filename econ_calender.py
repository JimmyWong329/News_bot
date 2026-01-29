#!/usr/bin/env python3
"""
econ_calendar.py
- Scrape official econ schedules with Crawl4AI (Fed, BLS, BEA, ISM)
- Pull latest/prior values from FRED for key macro series
- Print upcoming events (next N days) + current macro dashboard

Requires:
  pip install crawl4ai beautifulsoup4 requests python-dateutil tabulate python-dotenv

Env:
  export FRED_API_KEY="..."
"""
import os
import re
import argparse
import calendar
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date, time
from typing import List, Optional, Dict, Tuple
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup
from dateutil import parser as du_parser
from tabulate import tabulate
from dotenv import load_dotenv

# Crawl4AI imports
from crawl4ai import AsyncWebCrawler, BrowserConfig

from tools.io_utils import append_jsonl, ensure_out_dir, now_et_iso, write_json

# Load .env from the project folder (or wherever this script lives)
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

NY = ZoneInfo("America/New_York")

# -----------------------------
# FRED
# -----------------------------
class FREDClient:
    BASE = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str):
        if not api_key:
            print("Warning: FRED_API_KEY not found. FRED data will be skipped.")
            self.api_key = ""
        else:
            self.api_key = api_key
        self._cache = {}

    def _get(self, path: str, params: Dict) -> Dict:
        if not self.api_key:
            return {}
        params = dict(params)
        params["api_key"] = self.api_key
        params["file_type"] = "json"
        url = f"{self.BASE}/{path}"
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"FRED API Error: {e}")
            return {}

    def latest_two(self, series_id: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Returns (last, prev, last_date_str) from FRED series/observations.
        """
        if series_id in self._cache:
            return self._cache[series_id]

        js = self._get("series/observations", {
            "series_id": series_id,
            "sort_order": "desc",
            "limit": 2
        })
        obs = js.get("observations", [])
        if not obs:
            # Cache missing results so we don't retry failed lookups
            result = (None, None, None)
            self._cache[series_id] = result
            return result

        def to_float(v: str) -> Optional[float]:
            try:
                if v is None:
                    return None
                v = str(v).strip()
                if v in (".", "", "NaN", "nan"):
                    return None
                return float(v)
            except Exception:
                return None

        last = to_float(obs[0].get("value"))
        prev = to_float(obs[1].get("value")) if len(obs) > 1 else None
        last_date = obs[0].get("date")
        
        result = (last, prev, last_date)
        self._cache[series_id] = result
        return result


# -----------------------------
# Event model
# -----------------------------
@dataclass(order=True)
class EconEvent:
    sort_index: datetime = field(init=False, repr=False)
    dt: datetime
    name: str
    source: str
    tier: int = 3  # 1=High Impact (CPI, FOMC, Jobs), 2=Medium (PPI, ISM), 3=Low
    tags: List[str] = field(default_factory=list)
    fred_series: List[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        self.sort_index = self.dt


# -----------------------------
# Crawl helpers
# -----------------------------
async def crawl_page(crawler: AsyncWebCrawler, url: str) -> Tuple[str, str]:
    """
    Returns (html, markdown).
    """
    try:
        res = await crawler.arun(url=url)
        if not getattr(res, "success", False):
            print(f"Warning: Crawl failed for {url}")
            return "", ""
        
        html = getattr(res, "html", "") or ""
        md = ""
        m = getattr(res, "markdown", None)
        if isinstance(m, str):
            md = m
        elif m is not None:
            md = getattr(m, "fit_markdown", "") or getattr(m, "raw_markdown", "") or ""
        return html, md
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return "", ""


# -----------------------------
# Parsers
# -----------------------------

def _dt_from_strings(date_str: str, time_str: str, default_year: int) -> Optional[datetime]:
    try:
        if re.search(r"\b\d{4}\b", date_str) is None:
            date_str = f"{date_str} {default_year}"
        dt = du_parser.parse(f"{date_str} {time_str}", fuzzy=True)
        return dt.replace(tzinfo=NY)
    except Exception:
        return None

# --- FED: Monthly Page Parsing ---
def parse_fed_monthly_page(html: str, year: int, month: int) -> List[EconEvent]:
    """
    Parses a specific Fed monthly calendar page (e.g., 2026-january.htm).
    Strictly restricted to 'FOMC Meetings' and 'Beige Book' sections.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Get lines of text
    lines = [ln.strip() for ln in soup.get_text("\n", strip=True).splitlines() if ln.strip()]

    out: List[EconEvent] = []

    def parse_time(s: str) -> Optional[time]:
        # matches "2:30 p.m." / "11:00 a.m."
        m = re.match(r"^(\d{1,2}):(\d{2})\s*(a\.m\.|p\.m\.)$", s.lower())
        if not m:
            return None
        hh = int(m.group(1))
        mm = int(m.group(2))
        ampm = m.group(3)
        if ampm == "p.m." and hh != 12:
            hh += 12
        if ampm == "a.m." and hh == 12:
            hh = 0
        return time(hh, mm)

    def mk_dt(day: int, t: time) -> datetime:
        return datetime(year, month, day, t.hour, t.minute, tzinfo=NY)

    def parse_section(section_title: str, stop_titles: List[str]) -> None:
        # find section start
        try:
            start = next(i for i, ln in enumerate(lines) if ln.lower() == section_title.lower())
        except StopIteration:
            return

        # stop at next major heading or end of lines
        stop = len(lines)
        for i in range(start + 1, len(lines)):
            if any(lines[i].lower() == st.lower() for st in stop_titles):
                stop = i
                break

        pending_time: Optional[time] = None
        pending_event: Optional[str] = None

        # Iterate through lines in the section to find the pattern: Time -> Event -> Day
        for ln in lines[start:stop]:
            t = parse_time(ln)
            if t:
                pending_time = t
                pending_event = None
                continue

            # Event names check
            # We look for "FOMC", "Beige Book", or "Minutes"
            lower_ln = ln.lower()
            if pending_time and (("fomc" in lower_ln) or ("beige book" in lower_ln) or ("minutes" in lower_ln)):
                pending_event = ln
                continue

            # Day number line: e.g., "28"
            if pending_time and pending_event and re.fullmatch(r"\d{1,2}", ln):
                day = int(ln)
                try:
                    dt = mk_dt(day, pending_time)
                except ValueError:
                    continue

                low = pending_event.lower()
                
                # --- Categorization Logic ---
                if "press conference" in low:
                    out.append(EconEvent(
                        dt=dt, name="FOMC Press Conference", source="Fed Calendar",
                        tier=1, tags=["FED", "VOLATILITY"]
                    ))
                elif "fomc meeting" in low and "minutes" not in low:
                    out.append(EconEvent(
                        dt=dt, name="FOMC Rate Decision", source="Fed Calendar",
                        tier=1, tags=["FED", "RATES"], fred_series=["DFF", "DGS2"]
                    ))
                elif "minutes" in low and "fomc" in low:
                    out.append(EconEvent(
                        dt=dt, name="FOMC Minutes", source="Fed Calendar",
                        tier=1, tags=["FED", "VOLATILITY"]
                    ))
                elif "beige book" in low:
                    out.append(EconEvent(
                        dt=dt, name="Fed Beige Book", source="Fed Calendar",
                        tier=2, tags=["FED", "ECONOMY"]
                    ))

                pending_time = None
                pending_event = None

    # Only parse these specific sections to avoid noise
    parse_section("FOMC Meetings", stop_titles=["Beige Book", "Statistical Releases", "Other", "Testimony", "Speeches"])
    parse_section("Beige Book", stop_titles=["Statistical Releases", "Other", "Testimony", "Speeches", "FOMC Meetings"])

    return out

# --- ISM: Rule-Based (No Scraping) ---
def get_ism_events(year: int, month: int) -> List[EconEvent]:
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    month_days = c.monthdatescalendar(year, month)
    dates = [d for week in month_days for d in week if d.month == month]
    business_days = [d for d in dates if d.weekday() < 5]
    
    out = []
    # Manufacturing (1st business day)
    if len(business_days) >= 1:
        dt_man = datetime.combine(business_days[0], time(10, 0), tzinfo=NY)
        out.append(EconEvent(
            dt=dt_man, name="ISM Manufacturing PMI", source="ISM Rule",
            tier=2, tags=["ISM", "PMI"], fred_series=[] 
        ))
    # Services (3rd business day)
    if len(business_days) >= 3:
        dt_srv = datetime.combine(business_days[2], time(10, 0), tzinfo=NY)
        out.append(EconEvent(
            dt=dt_srv, name="ISM Services PMI", source="ISM Rule",
            tier=2, tags=["ISM", "PMI"]
        ))
    return out

# --- BLS: Parser with FRED Mapping ---
def parse_bls_schedule(html: str, year: int, keep_keywords: List[str]) -> List[EconEvent]:
    soup = BeautifulSoup(html, "html.parser")
    txt = soup.get_text("\n", strip=True)
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]

    def match_keep(release: str) -> bool:
        r = release.lower()
        return any(k.lower() in r for k in keep_keywords)

    out: List[EconEvent] = []
    i = 0
    date_re = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}$")
    time_re = re.compile(r"^\d{1,2}:\d{2}\s*(AM|PM)$", re.I)

    while i < len(lines) - 2:
        if date_re.match(lines[i]) and time_re.match(lines[i+1]):
            d = lines[i]
            t = lines[i+1]
            release = lines[i+2]
            
            if match_keep(release):
                dt = _dt_from_strings(d, t, year)
                if dt:
                    tags = ["BLS"]
                    series = []
                    tier = 3
                    r_low = release.lower()
                    
                    if "cpi" in r_low or "consumer price" in r_low: 
                        tags += ["INFLATION", "CPI"]
                        series = ["CPIAUCSL", "CPILFESL"] # Headline, Core
                        tier = 1
                    elif "employment situation" in r_low: 
                        tags += ["JOBS", "NFP"]
                        series = ["PAYEMS", "UNRATE"] # Payrolls, U-Rate
                        tier = 1
                    elif "ppi" in r_low or "producer price" in r_low: 
                        tags += ["INFLATION", "PPI"]
                        series = ["PPIACO"]
                        tier = 2
                    elif "job openings" in r_low or "jolts" in r_low: 
                        # FIX: Differentiate between National JOLTS and State JOLTS
                        if "state" in r_low:
                            tags += ["JOBS", "STATE"]
                            tier = 3 # State data is lower impact
                        else:
                            tags += ["JOBS"]
                            series = ["JTSJOL"]
                            tier = 2
                    elif "import and export" in r_low:
                        tags += ["INFLATION"]
                        tier = 3
                    elif "employment cost" in r_low:
                        tags += ["INFLATION", "WAGES"]
                        series = ["ECIALLCIV"]
                        tier = 1 # ECI is Fed favorite

                    out.append(EconEvent(dt=dt, name=release, source="BLS", tier=tier, tags=tags, fred_series=series))
            i += 3
        else:
            i += 1
    return out

# --- BEA: Parser with FRED Mapping ---
def parse_bea_schedule(html: str, default_year: int) -> List[EconEvent]:
    soup = BeautifulSoup(html, "html.parser")
    out: List[EconEvent] = []
    
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all(["td", "th"])
            row_text = [c.get_text(" ", strip=True) for c in cols]
            
            if len(row_text) < 3: continue
                
            d_str = row_text[0]
            t_str = row_text[1] if re.search(r"\d:\d{2}", row_text[1]) else "8:30 AM" 
            name = row_text[-1]
            
            if not (("GDP" in name or "Personal Income" in name) and "State" not in name):
                continue

            m_year = re.search(r"\b(20\d{2})\b", name)
            y = int(m_year.group(1)) if m_year else default_year
            
            if str(y) not in d_str: d_str = f"{d_str} {y}"
                
            dt = _dt_from_strings(d_str, t_str, y)
            if dt:
                tags = ["BEA"]
                series = []
                tier = 2
                if "GDP" in name: 
                    tags += ["GROWTH"]
                    series = ["GDPC1"] # Real GDP
                    tier = 1
                if "Personal Income" in name: 
                    tags += ["INFLATION", "PCE"]
                    series = ["PCEPI", "PCEPILFE"] # PCE Headline, Core
                    tier = 1
                
                out.append(EconEvent(dt=dt, name=name, source="BEA", tier=tier, tags=tags, fred_series=series))
                
    return out


# -----------------------------
# Macro “what to watch” series
# -----------------------------
MACRO_SERIES = {
    # Policy / Rates
    "DFF":      "Fed Funds Rate",
    "DGS2":     "2Y Treasury",
    "DGS10":    "10Y Treasury",
    
    # Growth / Jobs
    "UNRATE":   "Unemployment Rate",
    "PAYEMS":   "Nonfarm Payrolls",
    "JTSJOL":   "Job Openings (JOLTS)",
    "ICSA":     "Jobless Claims",
    
    # Inflation
    "CPIAUCSL": "CPI All Items",
    "CPILFESL": "Core CPI",
    "PCEPI":    "PCE Price Index",
    "PCEPILFE": "Core PCE",
}

def compute_spreads(vals: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    out = {}
    if vals.get("DGS10") is not None and vals.get("DGS2") is not None:
        out["10Y-2Y"] = vals["DGS10"] - vals["DGS2"]
    return out


# -----------------------------
# Main
# -----------------------------
async def build_calendar(days: int) -> List[EconEvent]:
    browser_cfg = BrowserConfig(headless=True)
    
    now = datetime.now(tz=NY)
    events: List[EconEvent] = []

    # 1. ISM (Rule-based)
    for i in range(3):
        m = now.month + i
        y = now.year
        if m > 12:
            m -= 12
            y += 1
        events += get_ism_events(y, m)

    # 2. Crawl Sources
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        # BLS - Smart Fallback Logic
        bls_year_used = now.year
        bls_url = f"https://www.bls.gov/schedule/{bls_year_used}/home.htm"
        print(f"Crawling BLS Schedule ({bls_url})...")
        bls_html, _ = await crawl_page(crawler, bls_url)
        
        # BLS Fallback (Year boundary safety)
        if not bls_html:
            bls_year_used = now.year + 1
            print(f"Warning: Current year BLS schedule empty. Trying fallback ({bls_year_used})...")
            bls_html, _ = await crawl_page(crawler, f"https://www.bls.gov/schedule/{bls_year_used}/home.htm")
            
        # BEA
        print("Crawling BEA Schedule...")
        bea_html, _ = await crawl_page(crawler, "https://www.bea.gov/news/schedule")

        # FED (Monthly Pages)
        fed_events = []
        for i in range(3):
            m = now.month + i
            y = now.year
            if m > 12: 
                m -= 12
                y += 1
            
            month_name = calendar.month_name[m].lower()
            fed_url = f"https://www.federalreserve.gov/newsevents/{y}-{month_name}.htm"
            print(f"Crawling Fed ({month_name})...")
            html, _ = await crawl_page(crawler, fed_url)
            if html:
                fed_events += parse_fed_monthly_page(html, y, m)
        events += fed_events

    # Parse
    keep_keywords = [
        "Consumer Price Index", "Employment Situation", "Producer Price Index", 
        "Job Openings", "Employment Cost Index", "Import and Export Price"
    ]
    
    # Use the specific year we successfully crawled
    events += parse_bls_schedule(bls_html, year=bls_year_used, keep_keywords=keep_keywords)
    events += parse_bea_schedule(bea_html, default_year=now.year)

    # Filter & Sort
    horizon = now + timedelta(days=days)
    events = [e for e in events if now <= e.dt <= horizon]
    events.sort()
    return events


def build_macro_dashboard(fred: FREDClient) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    vals: Dict[str, Optional[float]] = {}

    def f2(x):
        return "" if x is None else f"{x:.2f}"

    for sid, label in MACRO_SERIES.items():
        last, prev, last_date = fred.latest_two(sid)
        vals[sid] = last
        delta = None if (last is None or prev is None) else (last - prev)
        rows.append({
            "id": sid,
            "series": label,
            "last_date": last_date or "",
            "last": f2(last),
            "prev": f2(prev),
            "delta": f2(delta),
        })

    spreads = compute_spreads(vals)
    for k, v in spreads.items():
        rows.append({
            "id": k,
            "series": "Spread",
            "last_date": "",
            "last": f2(v),
            "prev": "",
            "delta": "",
        })
    return rows


def print_macro_dashboard(fred: FREDClient) -> List[Dict[str, str]]:
    rows = build_macro_dashboard(fred)
    print("\n" + "="*96)
    print("MACRO DASHBOARD (FRED latest)")
    print("="*96)
    print(tabulate(
        [
            [r["id"], r["series"], r["last_date"], r["last"], r["prev"], r["delta"]]
            for r in rows
        ],
        headers=["ID", "Series", "Last Date", "Last", "Prev", "Delta"],
        tablefmt="github",
    ))
    return rows


def build_upcoming_events(events: List[EconEvent], fred: Optional[FREDClient] = None) -> List[Dict[str, object]]:
    upcoming: List[Dict[str, object]] = []
    for e in events:
        latest_data: Dict[str, object] = {}
        if fred and e.fred_series:
            for sid in e.fred_series:
                last, prev, last_date = fred.latest_two(sid)
                latest_data[sid] = {
                    "last": last,
                    "prev": prev,
                    "last_date": last_date,
                }
        upcoming.append({
            "tier": e.tier,
            "time_et": e.dt.strftime("%Y-%m-%d %I:%M%p"),
            "event": e.name,
            "source": e.source,
            "tags": e.tags,
            "latest_data": latest_data,
        })
    return upcoming


def print_events(events: List[EconEvent], fred: Optional[FREDClient] = None) -> List[Dict[str, object]]:
    rows = []
    for e in events:
        series_str = ""
        if fred and e.fred_series:
            # Display up to 2 series to keep table clean
            items = []
            for sid in e.fred_series[:2]:
                last, _, _ = fred.latest_two(sid)
                if last is not None:
                    items.append(f"{sid}:{last:.2f}")
            series_str = " | ".join(items)

        rows.append([
            e.tier,
            e.dt.strftime("%Y-%m-%d %I:%M%p"),
            e.name,
            e.source,
            ", ".join(e.tags[:2]),
            series_str
        ])

    print("\n" + "="*96)
    print("UPCOMING EVENTS (Tier 1 = High Impact)")
    print("="*96)
    print(tabulate(rows, headers=["Tier", "Time (ET)", "Event", "Source", "Tags", "Latest Data"], tablefmt="github"))
    return build_upcoming_events(events, fred=fred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60, help="Lookahead window in days")
    ap.add_argument("--no-fred", action="store_true", help="Skip FRED value fetch")
    ap.add_argument("--today", action="store_true", help="Show events for the next 24 hours only")
    ap.add_argument("--out_dir", type=str, default="out", help="Output directory (default: out)")
    ap.add_argument("--out_json", type=str, default=None, help="Write calendar JSON to a path")
    ap.add_argument("--out_jsonl", type=str, default=None, help="Write calendar events JSONL to a path")
    args = ap.parse_args()

    out_dir = ensure_out_dir(args.out_dir)
    fred = None
    if not args.no_fred:
        fred_key = os.getenv("FRED_API_KEY", "").strip()
        fred = FREDClient(fred_key)

    # Determine window
    lookahead = 1 if args.today else args.days
    
    events = asyncio.run(build_calendar(lookahead))

    macro_dashboard = []
    if fred:
        macro_dashboard = print_macro_dashboard(fred)
    upcoming_events = print_events(events, fred=fred)

    if args.out_json or args.out_jsonl:
        meta_date = datetime.now(NY).strftime("%Y-%m-%d")
        payload = {
            "meta": {
                "source_script": "econ_calender.py",
                "generated_at_et": now_et_iso(),
                "date": meta_date,
                "asof_et": "",
                "run_id": f"{meta_date}_unknown_econ_calender.py",
                "days": lookahead,
            },
            "macro_dashboard": macro_dashboard,
            "upcoming_events": upcoming_events,
        }
        if args.out_json:
            write_json(Path(args.out_json), payload)
        if args.out_jsonl:
            for event in upcoming_events:
                append_target = Path(args.out_jsonl)
                append_jsonl(append_target, {"meta": payload["meta"], **event})


if __name__ == "__main__":
    main()
