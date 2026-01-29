import asyncio
import json_repair
from collections import Counter

import argparse
import json
import os
import re
import time
from html.parser import HTMLParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import pytz

from tools.io_utils import ensure_out_dir, now_et_iso, write_json


#(taenv) jordo@Gideon:~/news_crawler$ python fjc.py --mode day --fj_date 2026-01-23 --raw_only --raw_source fj
''') Default (LIVE mode, one-shot)
python fjc.py

B) Print raw headlines only (no Gemini calls)

FinancialJuice only:

python fjc.py --raw_only --raw_source fj --mode live


Finviz only:

python fjc.py --raw_only --raw_source finviz


All sources:

python fjc.py --raw_only --raw_source all

C) Historical day mode (scroll back for a specific date)
python fjc.py --mode day --fj_date 2026-01-23 --fj_scrolls 200 --fj_scroll_count 120

D) Loop it every 5 minutes
python fjc.py --loop --interval_min 5'''

# Crawl4AI
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    VirtualScrollConfig,
)

# Gemini
import google.generativeai as genai

# --- Timezone ---
NY_TZ = pytz.timezone("America/New_York")

OUT = Path("out")
OUT.mkdir(exist_ok=True)
LOG = OUT / "regime_log.jsonl"
# ITEMS_JSON = OUT / "regime_items.json" # Commented out to prevent JSON file creation

# --- Sources ---
FINVIZ_QUOTE_URL = "https://finviz.com/quote.ashx?t={}&p=d"
FINVIZ_NEWS_URL = "https://finviz.com/news.ashx"
FJ_HOME_URL = "https://www.financialjuice.com/home"

# --- Defaults (override with CLI) ---
DEFAULT_WATCHLIST = "SPY,QQQ,DIA,IWM,NVDA,AAPL,MSFT,AMZN,META,GOOGL,TSLA"
FJ_FEED_SELECTOR = "#mainFeed"
FJ_SUMMARY_SELECTOR = "#SummaryDiv"
FJ_SCOPE_SELECTOR = f"{FJ_SUMMARY_SELECTOR}, {FJ_FEED_SELECTOR}"
FJ_LAUNCH_JS = r"""
const el = Array.from(document.querySelectorAll('button,a,div,span'))
  .find(x => (x.textContent || '').trim().toUpperCase() === 'LAUNCH');
if (el) el.click();
"""
def fj_click_tab_js(tab_name: str) -> str:
    tab = (tab_name or "").strip().upper()
    return f"""
(() => {{
  const target = {json.dumps(tab)};
  const clickables = Array.from(document.querySelectorAll('button,a,div,span'))
    .filter(el => {{
      const t = (el.textContent || '').trim().toUpperCase();
      return t === target && typeof el.click === 'function';
    }});

  if (clickables.length) {{
    clickables[0].click();
    return true;
  }}
  return false;
}})();
"""
FJ_WAIT_TAB_ACTIVE = """
js:() => {
  const target = (window.__FJ_TARGET_TAB__ || "INDEXES").toUpperCase();
  const els = Array.from(document.querySelectorAll('button,a,div,span'))
    .filter(el => (el.textContent||"").trim().toUpperCase() === target);

  const active = els.some(el =>
    (el.getAttribute("aria-selected") === "true") ||
    ((""+el.className).toLowerCase().includes("active")) ||
    (el.parentElement && (""+el.parentElement.className).toLowerCase().includes("active"))
  );

  const feed = document.querySelector("#mainFeed");
  return !!feed && (feed.innerText || "").length > 50 && (active || els.length > 0);
}
"""
FJ_WAIT_FEED = (
    "js:() => { const f=document.querySelector('#mainFeed'); "
    "return f && f.innerText && f.innerText.length > 20; }"
)
FJ_WAIT_TIMEOUT_MS = 20000



# --- Domain quality filters (optional) ---
TRASH_DOMAINS = {"facebook.com", "twitter.com", "x.com", "finviz.com"}
MID_DOMAINS = {"seekingalpha.com", "benzinga.com", "investorplace.com", "motleyfool.com"}

KEY_EVENT_WORDS = [
    "earnings","guidance","revenue","profit","miss","beat","raises","cuts",
    "downgrade","upgrade","sec","doj","lawsuit","antitrust",
    "acquires","merger","partnership","contract",
    "forecast","outlook","margin","layoffs","buyback","dividend",
    "fed","rates","cpi","jobs","tariff","sanction","war","geopolitical","inflation","recession"
]


def infer_scope(item: Dict[str, Any]) -> str:
    category = (item.get("category") or item.get("ai_category") or "").lower()
    if "macro" in category:
        return "MACRO"
    if item.get("tickers"):
        return "TICKER"
    if item.get("sector"):
        return "SECTOR"
    return "UNKNOWN"

# -----------------------------
# Your regime prompt (embedded exactly, used for reference / schema)
# -----------------------------
MARKET_REGIME_PROMPT = """You are a disciplined market-regime classifier. Your job is to infer overall MARKET BIAS using ONLY the headline items I provide, each with scores and metadata.

Goal:
Return exactly ONE of these regime labels:
- RISK_ON
- RISK_OFF
- MIXED
- NONE

Inputs I will provide:
A JSON array called "items". Each item may include fields like:
timestamp_et, source, tickers, headline, category, direction (POS/NEG/NEU),
impact_score (0-200), certainty (0-100), tradable_now (true/false),
direct (true/false), priced_in_pct (0-100), mechanism, notes.

Rules (follow strictly):
1) This is MARKET-WIDE bias, not single-stock bias. Macro/systemic items matter most:
   - Systemic: tariffs, war/geopolitics, rates/Fed, inflation/jobs, credit stress, broad risk sentiment, major index moves, major regulation affecting many firms.
   - Idiosyncratic: single-company earnings/product/one-off lawsuit = lower weight unless it clearly spreads across the market (e.g., “US bans advanced AI chips broadly”).
2) De-duplicate near-identical stories. If multiple items describe the same event, count it once (keep the strongest/most certain one).
3) Weight each item by:
   - strength = normalized(impact_score) * (certainty/100)
   - macro_weight = 1.0 for systemic, 0.6 for sector-wide, 0.3 for single-name
   - freshness_weight = 1.0 if within last 2 hours, 0.7 if 2–6h, 0.4 if older
   - priced_in_discount = (1 - priced_in_pct/100)
   final_weight = strength * macro_weight * freshness_weight * priced_in_discount
4) Compute:
   - total_neg = sum(final_weight for NEG)
   - total_pos = sum(final_weight for POS)
   - net = total_pos - total_neg
   - magnitude = total_pos + total_neg
5) Classification thresholds:
   - If magnitude < 0.60  -> NONE
   - Else if abs(net) < 0.20 * magnitude -> MIXED
   - Else if net >= 0.20 * magnitude -> RISK_ON
   - Else -> RISK_OFF
6) Be skeptical: don’t hallucinate extra context. Use only my items.
"""

# -----------------------------
# Helpers
# -----------------------------

def print_raw_items(rows: List[Dict[str, Any]], source_filter: str = "all"):
    def key_ts(it):
        ts = it.get("timestamp_et")
        try:
            return datetime.fromisoformat(ts) if ts else datetime.min
        except Exception:
            return datetime.min

    # optional filtering
    sf = (source_filter or "all").lower()
    if sf != "all":
        if sf == "fj":
            rows = [r for r in rows if str(r.get("source", "")).startswith("financialjuice")]
        elif sf == "finviz":
            rows = [r for r in rows if str(r.get("source", "")).startswith("finviz")]
        else:
            rows = [r for r in rows if str(r.get("source", "")).lower() == sf]

    # group by source
    by_src: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_src.setdefault(r.get("source", "?"), []).append(r)

    for src in sorted(by_src.keys()):
        group = sorted(by_src[src], key=key_ts, reverse=True)
        print("\n" + "=" * 110)
        print(f"RAW SOURCE: {src} | count={len(group)}")
        print("=" * 110)

        for i, it in enumerate(group, 1):
            ts = it.get("timestamp_et") or "NA"
            url = it.get("url") or ""
            headline = it.get("headline") or ""

            print(f"{i:3d}. {ts}  {headline}")
            if url:
                print(f"     {url}")

            # NEW: print expanded paragraph(s) if present
            body = it.get("body")
            if body:
                print("     ---")
                print("     " + body.replace("\n", "\n     "))



def print_counts(label: str, rows: List[Dict[str, Any]]):
    c = Counter([r.get("source","?") for r in rows])
    total = len(rows)
    parts = ", ".join([f"{k}={v}" for k,v in sorted(c.items())])
    print(f"[COVERAGE] {label}: total={total} | {parts}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SummaryDivTextParser(HTMLParser):
    def __init__(self, target_id: str) -> None:
        super().__init__()
        self.target_id = target_id
        self.capture_depth = 0
        self.in_target = False
        self.parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "div" and attrs_dict.get("id") == self.target_id:
            self.in_target = True
            self.capture_depth = 1
            return
        if self.in_target:
            self.capture_depth += 1

    def handle_endtag(self, tag):
        if self.in_target:
            self.capture_depth -= 1
            if self.capture_depth <= 0:
                self.in_target = False

    def handle_data(self, data):
        if self.in_target and data:
            self.parts.append(data)


STOP_PHRASES = [
    "join us and go real-time",
    "this feed is delayed",
    "don't like ads",
    "go pro",
]

def _norm(s: str) -> str:
    s = (s or "").strip().lower().replace("×", "")
    s = re.sub(r"[\*\_`]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def extract_need_to_know_risk_from_text(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None

    target = "need to know market risk"
    start = None
    for i, ln in enumerate(lines):
        if target in _norm(ln):
            start = i
            break
    if start is None:
        return None

    out = []
    for j in range(start + 1, len(lines)):
        n = _norm(lines[j])
        if any(p in n for p in STOP_PHRASES):
            break

    txt = "\n".join(out).strip()
    return txt if txt else None


def extract_need_to_know_risk(md: str, html: Optional[str] = None) -> Optional[str]:
    if html:
        parser = SummaryDivTextParser("SummaryDiv")
        parser.feed(html)
        summary_text = "\n".join(parser.parts).strip()
        got = extract_need_to_know_risk_from_text(summary_text)
        if got:
            return got

    return extract_need_to_know_risk_from_text(md or "")


def safe_load_json_array(text: str) -> list:
    if not text:
        raise ValueError("Empty model output")

    t = text.strip()

    # Strip ```json ... ``` fences
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        t = m.group(1).strip()

    # 1) Normal strict JSON
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for k in ("items", "headlines", "data"):
                if k in obj and isinstance(obj[k], list):
                    return obj[k]
    except json.JSONDecodeError:
        pass

    # 2) LLM repair JSON (fix missing commas, quotes, etc.)
    try:
        obj = json_repair.loads(t)  # can replace json.loads for LLM outputs
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for k in ("items", "headlines", "data"):
                if k in obj and isinstance(obj[k], list):
                    return obj[k]
    except Exception:
        pass

    # 3) Fallback: slice first '[' ... last ']'
    i = t.find("[")
    j = t.rfind("]")
    if i != -1 and j != -1 and j > i:
        sliced = t[i:j+1]
        try:
            return json.loads(sliced)
        except json.JSONDecodeError:
            return json_repair.loads(sliced)

    raise ValueError("Could not extract/repair JSON array from model output")


def now_ny() -> datetime:
    return datetime.now(NY_TZ)

def append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def domain(url: str) -> str:
    return urlparse(url).netloc.lower()

def mask_key(k: str, show_last: int = 4) -> str:
    if not k:
        return "MISSING"
    return "****" + k[-show_last:]

def looks_like_nav(title: str) -> bool:
    t = (title or "").strip()
    bad = {
        "Screener","Portfolio","Calendar","Backtests","Register","Market News",
        "Stocks News","ETF News","Crypto News","Forex News","Futures News",
        "Login","Elite","Home","News","Submit"
    }
    if t in bad: return True
    if t.lower().startswith("skip to"): return True
    if t.startswith("English"): return True
    return False

def is_probable_headline(title: str, url: str) -> bool:
    t = " ".join((title or "").split()).strip()
    if len(t) < 15: return False
    if looks_like_nav(t): return False
    d = domain(url)
    if d.endswith("finviz.com") and "quote.ashx" not in url:
        # allow quote.ashx page itself, but reject internal finviz link spam
        return False
    if "utm_" in (url or ""): return False
    return True

def quality_score(title: str, url: str) -> float:
    t = (title or "").lower()
    d = domain(url)
    score = 0.0
    score += 1.0 * sum(1 for w in KEY_EVENT_WORDS if w in t)
    if d in TRASH_DOMAINS: score -= 3.0
    if d in MID_DOMAINS: score -= 0.5
    if re.search(r"[\d\.\%$]", title or ""): score += 0.5
    if len((title or "").strip()) < 25: score -= 0.5
    return score

def is_today_ny(published_iso: Optional[str]) -> bool:
    if not published_iso:
        return False
    try:
        dt = datetime.fromisoformat(published_iso)
        if dt.tzinfo is None:
            dt = NY_TZ.localize(dt)
        else:
            dt = dt.astimezone(NY_TZ)
        return dt.date() == now_ny().date()
    except Exception:
        return False

def parse_time_today_ny(hhmm_ampm: str) -> Optional[str]:
    """
    For feeds that only show bare times (e.g., 11:23AM), anchor to NY "today".
    """
    try:
        m = re.match(r"^(\d{1,2}):(\d{2})(AM|PM)$", (hhmm_ampm or "").strip().upper())
        if not m:
            return None
        hh = int(m.group(1))
        mm = int(m.group(2))
        ap = m.group(3)
        if hh == 12:
            hh = 0
        if ap == "PM":
            hh += 12
        d = now_ny().date()
        dt = NY_TZ.localize(datetime(d.year, d.month, d.day, hh, mm, 0))
        return dt.isoformat()
    except Exception:
        return None

def normalize_headline_key(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    words = [w for w in s.split() if w not in {"the","a","an","to","of","in","on","for","and","or","as","at","by","with"}]
    return " ".join(words[:10]).strip()

# -----------------------------
# Markdown link extraction (works with Crawl4AI result.markdown)
# -----------------------------
def extract_markdown_links(md: str) -> List[Tuple[str, str, str]]:
    out = []
    link_pat = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
    for raw_line in (md or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for title, url in link_pat.findall(line):
            out.append((line, " ".join(title.split()).strip(), url))
    return out

# -----------------------------
# Your quote-page extractor (kept intact style)
# -----------------------------
def extract_quote_page_news_items(md: str) -> List[Dict[str, Any]]:
    lines = (md or "").split("\n")
    items = []
    seen = set()
    link_pat = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
    ts_pat = re.compile(r"((?:[A-Z][a-z]{2}-\d{2}-\d{2}\s+|Today\s+)?\d{2}:\d{2}[AP]M)")

    current_date_context = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        links = link_pat.findall(line)
        if not links:
            continue

        match = ts_pat.search(line)
        published_iso = None
        raw_date_debug = None

        if match:
            raw_ts = match.group(1)
            raw_date_debug = raw_ts

            if "Today" in raw_ts:
                current_date_context = datetime.now().date()
            elif re.match(r"^[A-Z][a-z]{2}-\d{2}-\d{2}", raw_ts):
                try:
                    d_str = raw_ts.split(" ")[0]
                    current_date_context = datetime.strptime(d_str, "%b-%d-%y").date()
                except Exception:
                    pass

            if current_date_context:
                try:
                    t_str = raw_ts[-7:]
                    t_obj = datetime.strptime(t_str, "%I:%M%p").time()
                    dt = datetime.combine(current_date_context, t_obj)
                    published_iso = dt.isoformat()
                except Exception:
                    pass

        for title, url in links:
            title = " ".join(title.split()).strip()
            key = (title, url)
            if key in seen:
                continue
            if not is_probable_headline(title, url):
                continue
            seen.add(key)
            items.append({
                "headline": title,
                "url": url,
                "published_raw": raw_date_debug,
                "timestamp_et": published_iso,  # note: naive ISO; we treat as NY local
                "source": "finviz_quote",
            })
    return items

# -----------------------------
# Finviz market feed extractor (fixes the "bare times" issue)
# -----------------------------
def extract_finviz_market_items(md: str) -> List[Dict[str, Any]]:
    items = []
    seen = set()
    for line, title, url in extract_markdown_links(md):
        if not is_probable_headline(title, url):
            continue
        # Find bare time in the line (Finviz market feed often uses this)
        m = re.search(r"\b(\d{1,2}:\d{2}[AP]M)\b", line.upper())
        ts = parse_time_today_ny(m.group(1)) if m else None

        key = (title, url)
        if key in seen:
            continue
        seen.add(key)

        items.append({
            "headline": title,
            "url": url,
            "timestamp_et": ts,
            "source": "finviz_market",
        })
    return items
async def crawl_financialjuice_for_date(
    crawler: AsyncWebCrawler,
    target_date,
    limit: int = 200,                  # <-- number of headlines to collect
    scroll_count: int = 80,            # <-- virtual scroll count
    max_scroll_attempts: int = 120,    # <-- JS scroll attempts (fallback loop)
    session_id: Optional[str] = None,
    close_session: bool = True,
    tab: str = "Indexes",
) -> tuple[list[dict], Optional[str]]:

    fj_wait = (
        'js:() => {'
        '  const t = document.body && document.body.innerText ? document.body.innerText : "";'
        '  return (t.includes("Jan") || t.includes("Feb") || t.includes("Mar") || t.includes("Apr") || '
        '          t.includes("May") || t.includes("Jun") || t.includes("Jul") || t.includes("Aug") || '
        '          t.includes("Sep") || t.includes("Oct") || t.includes("Nov") || t.includes("Dec"));'
        '}'
    )

    session_id = session_id or f"fj_{target_date.isoformat()}"

    def to_ny_date(ts_iso: str):
        dt = datetime.fromisoformat(ts_iso)
        if dt.tzinfo is None:
            dt = NY_TZ.localize(dt)
        else:
            dt = dt.astimezone(NY_TZ)
        return dt.date()

    def is_target(it: dict) -> bool:
        ts = it.get("timestamp_et")
        return bool(ts) and to_ny_date(ts) == target_date

    def is_older(it: dict) -> bool:
        ts = it.get("timestamp_et")
        return bool(ts) and to_ny_date(ts) < target_date

    virtual_cfg = VirtualScrollConfig(
        container_selector="#mainFeed",
        scroll_count=scroll_count,
        scroll_by="container_height",
        wait_after_scroll=0.35,
    )

    cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=90000,
        wait_for=FJ_WAIT_FEED,
        wait_for_timeout=FJ_WAIT_TIMEOUT_MS,
        session_id=session_id,
        virtual_scroll_config=virtual_cfg,
        css_selector=FJ_SCOPE_SELECTOR,
        js_code=[
            FJ_LAUNCH_JS,
            "await new Promise(r => setTimeout(r, 800));",
            f"window.__FJ_TARGET_TAB__ = {json.dumps(tab.upper())};",
            fj_click_tab_js(tab),
            "await new Promise(r => setTimeout(r, 600));",
            "const t=document.querySelector('#SummaryDiv .headline-title-nolink'); if(t){t.click();}",
            "await new Promise(r => setTimeout(r, 700));",
        ],
    )
    out: list[dict] = []
    seen = set()
    def add_batch(items: list[dict]) -> bool:
        for it in items:
            if not str(it.get("source", "")).startswith("financialjuice"):
                continue
            key = (it.get("headline"), it.get("timestamp_et"))
            if key in seen:
                continue
            seen.add(key)
            if is_target(it):
                out.append(it)
        out.sort(key=lambda it: it.get("timestamp_et") or "", reverse=True)
        if limit and len(out) > limit:
            out[:] = out[:limit]
        return bool(limit) and len(out) >= limit

    res = await crawler.arun(url=FJ_HOME_URL, config=cfg)
    html = getattr(res, "cleaned_html", None) or getattr(res, "html", None)
    summary_text = extract_need_to_know_risk(res.markdown or "", html=html)
    batch = extract_financialjuice_items(res.markdown or "")
    if add_batch(batch):
        if close_session:
            try:
                await crawler.kill_session(session_id)
            except Exception:
                pass
        return (out, summary_text)

    no_progress = 0

    if max_scroll_attempts > 0:
        # Scroll loop (keep same session/tab; run JS only)
        for _ in range(max_scroll_attempts):
            scroll_cfg = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                page_timeout=90000,
                session_id=session_id,
                js_only=True,
                js_code=[
        """
        const feed = document.querySelector('#mainFeed');
        if (feed) {
          feed.scrollTop = feed.scrollHeight;
          feed.dispatchEvent(new Event('scroll', {bubbles:true}));
        }
        window.scrollTo(0, document.body.scrollHeight);
        window.dispatchEvent(new Event('scroll'));
        """,
        "await new Promise(r => setTimeout(r, 1200));",
    ],

                css_selector=FJ_FEED_SELECTOR,
                word_count_threshold=0,
            )

            res2 = await crawler.arun(url=FJ_HOME_URL, config=scroll_cfg)
            batch2 = extract_financialjuice_items(res2.markdown or "")

            before = len(out)
            reached = add_batch(batch2)

            # stop once we hit the target number of headlines
            if reached:
                break

            # today-only stop: if we see older-than-target items, we're done
            if any(is_older(x) for x in batch2):
                break

            if len(out) == before:
                no_progress += 1
            else:
                no_progress = 0

            if no_progress >= 12:
                break

    if close_session:
        try:
            await crawler.kill_session(session_id)
        except Exception:
            pass

    out.sort(key=lambda it: it.get("timestamp_et") or "", reverse=True)
    return (out[:limit] if limit else out, summary_text)

# -----------------------------
# FinancialJuice extractor
# -----------------------------
def extract_financialjuice_items(md: str, html: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    FinancialJuice center feed rows can be:
      - collapsed:  <headline> then <timestamp line "HH:MM Mon DD" + tags>
      - expanded:   <headline> + multiple paragraph lines + <timestamp line + tags>
    Parse from Crawl4AI markdown.
    """

    if (not md) and html:
        html_text = HTMLStripper.strip(html)
        lines = [ln.strip() for ln in (html_text or "").splitlines() if ln.strip()]
    else:
        lines = [ln.strip() for ln in (md or "").splitlines() if ln.strip()]
    items: List[Dict[str, Any]] = []
    seen = set()

    ts_pat = re.compile(r"\b(\d{2}:\d{2})\s+([A-Z][a-z]{2})\s+(\d{1,2})\b")
    link_pat = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    img_only_pat = re.compile(r"^!\[\]\(https?://[^\)]+\)$")

    def is_image_only_line(s: str) -> bool:
        s = (s or "").strip()
        return bool(img_only_pat.match(s))

    def is_ad_or_promo_line(s: str) -> bool:
        u = (s or "").strip().upper()
        low = (s or "").lower()
        return (
            "DON'T LIKE ADS" in u
            or "GO PRO" in u
            or "THIS FEED IS DELAYED" in u
            or "GO REAL-TIME" in u
            or "FREE ACCESS" in u
            or "UNUSUAL ACTIVITY" in u
            or "historyfeed/" in low
            or "comp-images/" in low
        )

    def looks_like_tag_row(s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return True
        if len(s) <= 4:
            return True
        # glued tags
        if " " not in s and s.isalnum() and len(s) <= 80:
            return True
        # short ASCII tag bundles
        if len(s) <= 55 and s.replace(" ", "").isalnum() and s.isascii():
            if any(ch.isdigit() for ch in s) or "%" in s:
                return False
            return True
        if s.startswith("Forex") and len(s.split()) <= 8 and not any(ch.isdigit() for ch in s):
            return True
        return False

    def is_metric_line(s: str) -> bool:
        s = (s or "").strip()
        return (":" in s) and any(ch.isdigit() for ch in s)

    def looks_like_short_title(s: str) -> bool:
        s = (s or "").strip()
        if not (5 <= len(s) <= 30):
            return False
        if any(ch.isdigit() for ch in s):
            return False
        if looks_like_tag_row(s):
            return False
        if is_ad_or_promo_line(s) or is_image_only_line(s):
            return False
        return True

    def looks_like_headline(s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return False
        u = s.upper()

        # UI / promo noise
        if "COPIED" in u:
            return False
        if "JAVASCRIPT:VOID" in u or "JAVASCRIPT:VOID(0)" in u:
            return False
        if is_ad_or_promo_line(s) or is_image_only_line(s):
            return False

        # tag rows
        if looks_like_tag_row(s):
            return False

        # allow short titles like "MOO Imbalance"
        if looks_like_short_title(s):
            return True

        # relaxed headline shape
        if len(s) < 15 or len(s) > 280:
            return False
        if s.count(".") >= 4:
            return False
        if len(s.split()) > 45:
            return False

        return True

    # -----------------------------
    # A) Middle-feed parser (headline + optional body + timestamp)
    # -----------------------------
    for i in range(1, len(lines)):
        m = ts_pat.search(lines[i])
        if not m:
            continue

        # Convert "HH:MM Mon DD" -> ISO (assume current year, NY time)
        hhmm = m.group(1)
        mon = m.group(2)
        day = int(m.group(3))
        try:
            dt = datetime.strptime(f"{mon} {day} {now_ny().year} {hhmm}", "%b %d %Y %H:%M")
            dt = NY_TZ.localize(dt)
            ts_iso = dt.isoformat()
        except Exception:
            ts_iso = None

        # Find the previous timestamp (start of this item's block)
        prev_ts_i = None
        for j in range(i - 1, -1, -1):
            if ts_pat.search(lines[j]):
                prev_ts_i = j
                break
        block_start = (prev_ts_i + 1) if prev_ts_i is not None else 0

        # Pick headline within block
        h_i = None
        block_has_risk = any(
            "need to know market risk" in _norm(lines[k]) for k in range(block_start, i)
        )

        if block_has_risk:
            for j in range(i - 1, block_start - 1, -1):
                cand = lines[j].strip()
                if not cand or cand.startswith("*"):
                    continue

                if is_image_only_line(cand) or is_ad_or_promo_line(cand) or looks_like_tag_row(cand):
                    continue

                mm = link_pat.search(cand)
                title = " ".join(mm.group(1).split()).strip() if mm else cand
                if looks_like_headline(title) and "need to know market risk" not in _norm(title):
                    h_i = j
                    break
            if h_i is None:
                continue
        else:
            for j in range(block_start, i):
                cand = lines[j].strip()
                if not cand:
                    continue

                # ✅ A) skip ads/images during headline candidate scan
                if is_image_only_line(cand) or is_ad_or_promo_line(cand):
                    continue
                if looks_like_tag_row(cand):
                    continue

                # Prefer short-title + metric blocks (MOO Imbalance style)
                if looks_like_short_title(cand):
                    next1 = lines[j + 1].strip() if j + 1 < i else ""
                    next2 = lines[j + 2].strip() if j + 2 < i else ""
                    if is_metric_line(next1) or is_metric_line(next2):
                        h_i = j
                        break

                mm = link_pat.search(cand)
                if mm:
                    title = " ".join(mm.group(1).split()).strip()
                    link_url = (mm.group(2) or "").strip()
                    if link_url.lower().startswith("javascript:") or title.lower() == "copied":
                        continue
                    if looks_like_headline(title):
                        h_i = j
                        break
                else:
                    if looks_like_headline(cand):
                        h_i = j
                        break

        if h_i is None:
            continue

        raw_headline_line = lines[h_i].strip()

        # Extract headline/url
        url = None
        mm = link_pat.search(raw_headline_line)
        if mm:
            headline = " ".join(mm.group(1).split()).strip()
            url = (mm.group(2) or "").strip()
        else:
            headline = raw_headline_line

        # ✅ B) skip ads/images after headline finalized
        if is_image_only_line(headline) or is_ad_or_promo_line(headline):
            continue
        if url and url.lower().startswith("javascript:"):
            continue
        if headline.lower() == "copied":
            continue
        if _norm(headline) == "need to know market risk":
            continue

        # promo filters
        if "help us test" in headline.lower():
            continue
        if url and "unusualactivity" in url.lower():
            continue

        if not looks_like_headline(headline):
            continue

        # Body is everything between headline and timestamp line
        body_lines = []
        for k in range(h_i + 1, i):
            t = lines[k].strip()
            if not t:
                continue

            # ✅ C) skip ads/images in body
            if is_image_only_line(t) or is_ad_or_promo_line(t):
                continue

            if looks_like_tag_row(t):
                continue
            if t == headline:
                continue
            if t.lower() == "copied":
                continue
            body_lines.append(t)

        body = "\n".join(body_lines).strip() if body_lines else None

        key = ("fj_mid", headline, ts_iso)
        if key in seen:
            continue
        seen.add(key)

        items.append({
            "headline": headline,
            "body": body,
            "url": url,
            "timestamp_et": ts_iso,
            "source": "financialjuice_mid",
        })

    # -----------------------------
    # B) Fallback: /News/ link harvesting
    # -----------------------------
    for line, title, url in extract_markdown_links(md):
        if "financialjuice.com/News/" not in (url or ""):
            continue
        if "unusualactivity" in (url or "").lower():
            continue
        if "help us test" in (title or "").lower():
            continue
        if url.lower().startswith("javascript:"):
            continue
        if is_ad_or_promo_line(title) or is_image_only_line(title):
            continue
        if not is_probable_headline(title, url):
            continue

        m2 = re.search(r"\b(\d{1,2}:\d{2}[AP]M)\b", (line or "").upper())
        ts = parse_time_today_ny(m2.group(1)) if m2 else None

        key = ("fj_link", title, url)
        if key in seen:
            continue
        seen.add(key)

        items.append({
            "headline": title,
            "body": None,
            "url": url,
            "timestamp_et": ts,
            "source": "financialjuice_link",
        })

    return items

# -----------------------------
# Crawl wrappers
# -----------------------------
async def crawl_url_markdown(crawler: AsyncWebCrawler, url: str, wait_for: Optional[str] = None) -> str:
    cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=60000,
        wait_for=wait_for,
        delay_before_return_html=1.0,
        word_count_threshold=0
    )
    res = await crawler.arun(url=url, config=cfg)
    return res.markdown or ""


async def fj_live_warm(
    crawler: AsyncWebCrawler,
    session_id: str,
    tab: str = "Indexes",
) -> tuple[str, Optional[str]]:
    cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=60000,
        session_id=session_id,
        wait_for=FJ_WAIT_FEED,
        wait_for_timeout=FJ_WAIT_TIMEOUT_MS,
        css_selector=FJ_SCOPE_SELECTOR,
        js_code=[
            FJ_LAUNCH_JS,
            "await new Promise(r => setTimeout(r, 800));",
            f"window.__FJ_TARGET_TAB__ = {json.dumps(tab.upper())};",
            fj_click_tab_js(tab),
            "await new Promise(r => setTimeout(r, 600));",
        ],
        word_count_threshold=0,
    )
    res = await crawler.arun(url=FJ_HOME_URL, config=cfg)
    html = getattr(res, "cleaned_html", None) or getattr(res, "html", None)
    return res.markdown or "", html


async def fj_live_refresh(
    crawler: AsyncWebCrawler,
    session_id: str,
    tab: str = "Indexes",
) -> tuple[str, Optional[str]]:
    cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=30000,
        session_id=session_id,
        js_only=True,
        wait_for=FJ_WAIT_FEED,
        wait_for_timeout=FJ_WAIT_TIMEOUT_MS,
        css_selector=FJ_SCOPE_SELECTOR,
        js_code=[
            f"window.__FJ_TARGET_TAB__ = {json.dumps(tab.upper())};",
            fj_click_tab_js(tab),
            "window.scrollTo(0, 0);",
            "await new Promise(r => setTimeout(r, 300));",
        ],
        word_count_threshold=0,
    )
    res = await crawler.arun(url=FJ_HOME_URL, config=cfg)
    html = getattr(res, "cleaned_html", None) or getattr(res, "html", None)
    return res.markdown or "", html


async def fj_day_collect(
    crawler: AsyncWebCrawler,
    scroll_count: int,
    session_id: str,
    tab: str = "Indexes",
) -> tuple[str, Optional[str]]:
    virtual_cfg = VirtualScrollConfig(
        container_selector="#mainFeed",
        scroll_count=scroll_count,
        scroll_by="container_height",
        wait_after_scroll=0.35,
    )
    cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=90000,
        session_id=session_id,
        wait_for=FJ_WAIT_FEED,
        wait_for_timeout=FJ_WAIT_TIMEOUT_MS,
        virtual_scroll_config=virtual_cfg,
        css_selector=FJ_SCOPE_SELECTOR,
        js_code=[
            FJ_LAUNCH_JS,
            "await new Promise(r => setTimeout(r, 800));",
            f"window.__FJ_TARGET_TAB__ = {json.dumps(tab.upper())};",
            fj_click_tab_js(tab),
            "await new Promise(r => setTimeout(r, 600));",
        ],
        word_count_threshold=0,
    )
    res = await crawler.arun(url=FJ_HOME_URL, config=cfg)
    html = getattr(res, "cleaned_html", None) or getattr(res, "html", None)
    return res.markdown or "", html

async def crawl_finviz_quote_pages(crawler: AsyncWebCrawler, tickers: List[str]) -> List[Dict[str, Any]]:
    async def crawl_one(ticker: str) -> List[Dict[str, Any]]:
        url = FINVIZ_QUOTE_URL.format(ticker)
        try:
            res = await crawler.arun(url=url)  # matches your style
            md = res.markdown or ""
            items = extract_quote_page_news_items(md)
            for it in items:
                it["ticker_page"] = ticker
            return items
        except Exception:
            return []

    tasks = [crawl_one(t) for t in tickers]
    results = await asyncio.gather(*tasks)
    flat = []
    for sub in results:
        flat.extend(sub)
    return flat

# -----------------------------
# Gemini: tag raw headlines into your regime-schema items
# -----------------------------
def init_gemini(api_key: str):
    if not api_key:
        raise RuntimeError("ERROR: No Google API Key provided. Set GEMINI_API_KEY in .env or pass --api-key.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")

def batch_tag_for_regime(model, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert raw scraped headlines into the exact fields your regime classifier expects.
    Gemini only fills metadata; regime label is computed deterministically below.
    """
    if not raw:
        return []

    # Assign stable IDs
    for i, it in enumerate(raw):
        it["_gid"] = i

    prompt_lines = [
        "You are a market news metadata tagger.",
        "Return ONLY raw JSON array (no markdown).",
        "",
        "For each headline, output an object with EXACT keys:",
        "id(int), direction(POS/NEG/NEU), impact_score(0-200), certainty(0-100), priced_in_pct(0-100),",
        "category(one of: MACRO_LIQUIDITY, GEOPOLITICS, REGULATION_GOV, EARNINGS_GUIDANCE, SUPPLY_PRODUCTION, LEGAL_LITIGATION, MNA_DEAL, OTHER),",
        "tradable_now(bool), direct(bool), mechanism(string <= 12 words), notes(string <= 18 words), tickers(array of tickers or empty).",
        "",
        "Interpretation rules:",
        "- Systemic items (Fed/rates/tariffs/geopolitics/broad index moves/credit stress) should have higher impact_score.",
        "- priced_in_pct is your estimate of how much markets already reflect (0 none, 100 fully).",
        "- direct=true only if headline implies immediate change (not rumors).",
        "- tradable_now=true if it plausibly moves prices within minutes-hours.",
        "",
        "HEADLINES:"
    ]

    for it in raw:
        prompt_lines.append(
            f"{it['_gid']}. [{it.get('source')}] {it.get('headline')} (domain: {domain(it.get('url',''))})"
        )

    resp = model.generate_content(
    "\n".join(prompt_lines),
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0,
        "max_output_tokens": 4096,
    }
)

    rows = safe_load_json_array(resp.text)
    mp = {r["id"]: r for r in rows if isinstance(r, dict) and "id" in r}

    out = []
    for it in raw:
        r = mp.get(it["_gid"])
        if not r:
            continue

        out.append({
            "timestamp_et": it.get("timestamp_et"),
            "source": it.get("source"),
            "tickers": r.get("tickers", []) or [],
            "headline": it.get("headline"),
            "category": (r.get("category") or "OTHER"),
            "direction": (r.get("direction") or "NEU"),
            "impact_score": int(max(0, min(200, float(r.get("impact_score", 0))))),
            "certainty": int(max(0, min(100, float(r.get("certainty", 0))))),
            "tradable_now": bool(r.get("tradable_now", False)),
            "direct": bool(r.get("direct", False)),
            "priced_in_pct": int(max(0, min(100, float(r.get("priced_in_pct", 0))))),
            "mechanism": (r.get("mechanism") or "")[:200],
            "notes": (r.get("notes") or "")[:240],
            "url": it.get("url"),
        })
    return out

# -----------------------------
# Deterministic regime classifier (implements your rules)
# -----------------------------
def macro_weight(item: Dict[str, Any]) -> float:
    cat = (item.get("category") or "OTHER").upper()
    tickers = item.get("tickers") or []

    # Systemic
    if cat in {"MACRO_LIQUIDITY", "GEOPOLITICS"}:
        return 1.0

    # Regulation can be systemic or sector; infer by breadth
    if cat == "REGULATION_GOV":
        if len(tickers) == 0:
            return 1.0
        if len(tickers) >= 3:
            return 0.6
        return 0.3

    # Sector-wide heuristic
    if len(tickers) >= 3:
        return 0.6

    return 0.3

def freshness_weight(item: Dict[str, Any]) -> float:
    ts = item.get("timestamp_et")
    if not ts:
        return 0.4
    try:
        dt = datetime.fromisoformat(ts)
        # Treat naive as NY local (matches your pipeline)
        if dt.tzinfo is None:
            dt = NY_TZ.localize(dt)
        else:
            dt = dt.astimezone(NY_TZ)
        age_h = (now_ny() - dt).total_seconds() / 3600.0
        if age_h <= 2: return 1.0
        if age_h <= 6: return 0.7
        return 0.4
    except Exception:
        return 0.4

def dedupe_near_identical(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for it in items:
        key = normalize_headline_key(it.get("headline") or "")
        if not key:
            continue
        cur = best.get(key)
        if not cur:
            best[key] = it
            continue
        cur_strength = (cur.get("impact_score", 0) / 200.0) * (cur.get("certainty", 0) / 100.0)
        it_strength = (it.get("impact_score", 0) / 200.0) * (it.get("certainty", 0) / 100.0)
        if it_strength > cur_strength:
            best[key] = it
    return list(best.values())

def classify_regime(items: List[Dict[str, Any]]) -> str:
    items = dedupe_near_identical(items)

    total_neg = 0.0
    total_pos = 0.0

    for it in items:
        direction = (it.get("direction") or "NEU").upper()
        impact_score = float(it.get("impact_score", 0))
        certainty = float(it.get("certainty", 0))
        priced_in_pct = float(it.get("priced_in_pct", 0))

        strength = (impact_score / 200.0) * (certainty / 100.0)
        mw = macro_weight(it)
        fw = freshness_weight(it)
        priced_in_discount = (1.0 - (priced_in_pct / 100.0))

        final_weight = strength * mw * fw * priced_in_discount

        if direction == "NEG":
            total_neg += final_weight
        elif direction == "POS":
            total_pos += final_weight

    net = total_pos - total_neg
    magnitude = total_pos + total_neg

    if magnitude < 0.60:
        return "NONE"
    if abs(net) < 0.20 * magnitude:
        return "MIXED"
    if net >= 0.20 * magnitude:
        return "RISK_ON"
    return "RISK_OFF"

# -----------------------------
# Main loop
# -----------------------------
async def run_once(
    args,
    crawler: AsyncWebCrawler,
    fj_session_id: Optional[str] = None,
    close_fj_session: bool = True,
) -> str:
    """
    Returns a regime label in normal mode.
    In --raw_only mode, prints raw items and returns "NONE".
    Also prints FinancialJuice 'Need to know market risk' panel whenever FJ is crawled.
    """

    watch = [t.strip().upper() for t in args.watchlist.split(",") if t.strip()]
    raw: List[Dict[str, Any]] = []

    sf = str(args.raw_source).lower()
    only_fj = args.raw_only and sf in {"fj", "financialjuice_mid", "financialjuice_link"}
    only_finviz = args.raw_only and sf in {"finviz", "finviz_market", "finviz_quote"}

    # --- FinancialJuice live/raw fast path (no virtual scroll) ---
    if args.raw_only and args.mode == "live" and (only_fj or sf == "all"):
        session_id = fj_session_id or "fj_home_raw"
        fj_md, fj_html = await fj_live_warm(
            crawler,
            session_id=session_id,
            tab=args.fj_tab,
        )
        raw = extract_financialjuice_items(fj_md, html=fj_html)
        print_counts("RAW (before dedupe)", raw)
        print_raw_items(raw, source_filter=args.raw_source)
        return "NONE"

    # --- FinancialJuice (only if needed) ---
    if (not args.raw_only) or only_fj or sf == "all":
        fj_items, risk_text = await crawl_financialjuice_for_date(
            crawler,
            target_date=args.fj_target_date,
            limit=args.fj_scrolls,   # 0 = unlimited (if you set it that way)
            scroll_count=args.fj_scroll_count,
            max_scroll_attempts=args.fj_scroll_attempts,
            session_id=fj_session_id,
            close_session=close_fj_session,
            tab=args.fj_tab,
        )
        raw.extend(fj_items)
        if risk_text:
            print("\n=== NEED TO KNOW MARKET RISK ===")
            print(risk_text)
            print("================================\n")

    # --- If raw_only + fj filter, stop here (do NOT touch finviz) ---
    if only_fj:
        print_counts("RAW (before dedupe)", raw)
        print_raw_items(raw, source_filter=args.raw_source)
        return "NONE"

    # --- Finviz (only if needed) ---
    if (not args.raw_only) or only_finviz or sf == "all":
        finviz_news_md = await crawl_url_markdown(crawler, FINVIZ_NEWS_URL)
        raw.extend(extract_finviz_market_items(finviz_news_md))

        if args.include_quote_pages:
            quote_items = await crawl_finviz_quote_pages(crawler, watch)
            quote_items = [x for x in quote_items if is_today_ny(x.get("timestamp_et"))]
            raw.extend(quote_items)

    # --- If raw_only (finviz or all), print and stop here ---
    if args.raw_only:
        print_counts("RAW (before dedupe)", raw)
        print_raw_items(raw, source_filter=args.raw_source)
        return "NONE"

    # -----------------------------
    # Normal regime mode below here
    # -----------------------------

    api_key = args.api_key or os.getenv("GEMINI_API_KEY", "")
    if args.debug:
        print("GEMINI_API_KEY present:", bool(api_key))
        print("GEMINI_API_KEY masked :", mask_key(api_key))
    model = init_gemini(api_key)

    # ===== A) Coverage: right after crawl =====
    print_counts("RAW (before dedupe)", raw)

    # De-dupe raw by (headline,url)
    seen = set()
    raw2: List[Dict[str, Any]] = []
    for it in raw:
        key = (it.get("headline"), it.get("url"))
        if key in seen:
            continue
        seen.add(key)
        it["qscore"] = quality_score(it.get("headline", ""), it.get("url", "") or "")
        raw2.append(it)

    # ===== B) Coverage: after dedupe =====
    print_counts("RAW2 (after dedupe)", raw2)
    print(f"[COVERAGE] dropped_by_dedupe={len(raw) - len(raw2)}")

    # Sort + trim
    raw2.sort(key=lambda x: x.get("qscore", 0.0), reverse=True)
    raw2_kept = raw2[: args.max_raw]

    # ===== C) Coverage: after trim =====
    print_counts(f"RAW2 kept (top max_raw={args.max_raw})", raw2_kept)
    print(f"[COVERAGE] dropped_by_max_raw={max(0, len(raw2) - len(raw2_kept))}")
    raw2 = raw2_kept

    # Gemini -> items
    items = batch_tag_for_regime(model, raw2)

    # ===== D) Coverage: after Gemini =====
    print_counts("ITEMS (after Gemini tagging)", items)

    # Rank + trim
    items.sort(
        key=lambda x: (x.get("impact_score", 0) / 200.0) * (x.get("certainty", 0) / 100.0),
        reverse=True,
    )
    items = items[: args.max_items]

    # Print analyzed items
    print(f"\n--- Analyzed {len(items)} Items ---")
    for i, it in enumerate(items, 1):
        src = it.get("source", "?")
        ts = it.get("timestamp_et") or "NA"
        url = it.get("url") or ""
        print(
            f"{i:2d}. [{src}] [{it.get('direction')}] "
            f"(imp={it.get('impact_score')}, cert={it.get('certainty')}, pin={it.get('priced_in_pct')})"
        )
        print(f"    {ts}  {it.get('headline')}")
        if url:
            print(f"    {url}")
    print("-" * 30)

    if args.out_json or args.out_jsonl:
        meta_date = args.fj_target_date.strftime("%Y-%m-%d") if args.fj_target_date else now_ny().strftime("%Y-%m-%d")
        meta = {
            "source_script": "US_News.py",
            "generated_at_et": now_et_iso(),
            "date": meta_date,
            "asof_et": "",
            "run_id": f"{meta_date}_unknown_US_News.py",
        }
        headlines_payload = []
        for it in items:
            tags = []
            if it.get("category"):
                tags.append(it.get("category"))
            if it.get("source"):
                tags.append(it.get("source"))
            if it.get("tickers"):
                tags.extend(it.get("tickers"))
            tags = [t for t in tags if t]
            headlines_payload.append({
                "meta": meta,
                "time_et": it.get("timestamp_et") or "",
                "source": it.get("source") or "",
                "text": it.get("headline") or "",
                "tags": tags,
                "scope": infer_scope(it),
            })
        if args.out_json:
            write_json(Path(args.out_json), {"meta": meta, "headlines": headlines_payload})
        if args.out_jsonl:
            for entry in headlines_payload:
                append_jsonl(Path(args.out_jsonl), entry)

    regime = classify_regime(items)

    append_jsonl(
        LOG,
        {
            "fetched_utc": utc_now_iso(),
            "regime": regime,
            "n_raw": len(raw),
            "n_scored": len(items),
        },
    )

    if args.debug:
        print("\nDEBUG top items:")
        for it in items[: min(10, len(items))]:
            print(
                f"- {it.get('direction')} imp={it.get('impact_score')} cert={it.get('certainty')} "
                f"pin={it.get('priced_in_pct')} cat={it.get('category')} src={it.get('source')} "
                f"| {it.get('headline','')[:140]}"
            )

    return regime



async def main():
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--mode",
        type=str,
        default="live",
        choices=["live", "day"],
        help="live=fast warm session refreshes, day=scroll to a specific date (slower).",
    )

    ap.add_argument(
        "--fj_date",
        type=str,
        default=None,
        help="FinancialJuice date YYYY-MM-DD (default: today NY).",
    )
    ap.add_argument(
        "--fj_tab",
        type=str,
        default="Indexes",
        help=(
            "FinancialJuice top tab to click: My News | Bonds | Commodities | Crypto | "
            "Equities | Forex | Indexes | Macro"
        ),
    )

    # ✅ Remapped: this number now means HEADLINES (limit), not scroll steps
    ap.add_argument(
        "--fj_scrolls",
        type=int,
        default=0,
        help="Max FinancialJuice headlines to collect for the target day (0 = no limit; default 0).",
    )
    ap.add_argument(
        "--fj_scroll_count",
        type=int,
        default=80,
        help="FinancialJuice virtual scroll count (default 80).",
    
    )
    ap.add_argument(
        "--fj_scroll_attempts",
        type=int,
        default=120,
        help="FinancialJuice JS scroll attempts for fallback loop (default 120).",
    )
    ap.add_argument("--out_dir", type=str, default="out", help="Output directory (default: out)")
    ap.add_argument("--out_json", type=str, default=None, help="Write headlines JSON to a path")
    ap.add_argument("--out_jsonl", type=str, default=None, help="Write headlines JSONL to a path")


    ap.add_argument("--api-key", type=str, default=os.getenv("GEMINI_API_KEY", ""))
    ap.add_argument("--watchlist", type=str, default=DEFAULT_WATCHLIST)
    ap.add_argument("--include_quote_pages", action="store_true")
    ap.add_argument("--max_raw", type=int, default=120)
    ap.add_argument("--max_items", type=int, default=60)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval_min", type=int, default=5)
    ap.add_argument("--debug", action="store_true")

    ap.add_argument(
        "--raw_only",
        action="store_true",
        help="Print all scraped raw items and exit (no Gemini).",
    )
    ap.add_argument(
        "--raw_source",
        type=str,
        default="all",
        help="Filter raw print: all | finviz | fj | finviz_market | financialjuice_mid | financialjuice_link",
    )

    args = ap.parse_args()
    out_dir = ensure_out_dir(args.out_dir)

    # target date for FinancialJuice (default: today NY)
    args.fj_target_date = (
        datetime.strptime(args.fj_date, "%Y-%m-%d").date()
        if args.fj_date
        else now_ny().date()
    )

    if args.mode == "live" and args.fj_date:
        print("WARNING: --fj_date is ignored in live mode (use --mode day for historical dates).")

    if args.raw_only and args.loop:
        print("ERROR: --raw_only does not support --loop (disable one).")
        return

    browser_cfg = BrowserConfig(
        headless=True,
        text_mode=True,
        light_mode=True,
        viewport_width=1280,
        viewport_height=720,
        extra_args=["--disable-extensions"],
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        if args.raw_only:
            await run_once(args, crawler)  # run_once prints raw and exits early in raw_only mode
            return

        if not args.loop:
            regime = await run_once(args, crawler)
            print(regime)
            return

        fj_session_id = "fj_home"
        while True:
            regime = await run_once(
                args,
                crawler,
                fj_session_id=fj_session_id,
                close_fj_session=False,
            )
            print(regime)
            time.sleep(max(5, args.interval_min * 60))



if __name__ == "__main__":
    asyncio.run(main())
