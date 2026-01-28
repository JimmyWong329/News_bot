import asyncio
import argparse
import json
import re
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
'''(taenv) jordo@Gideon:~/news_crawler$ python test_finbert.py 
GEMINI_API_KEY present: True
GEMINI_API_KEY masked : ****3kQc
✓ Gemini API Key loaded (AIzaS...)
[INIT].... → Crawl4AI 0.7.8 

==============================================================================================================
Starting parallel crawl for 1 tickers: ['NVDA']
[FETCH]... ↓ https://finviz.com/quote.ashx?t=NVDA&p=d                                                             | ✓ | ⏱:
3.39s 
[SCRAPE].. ◆ https://finviz.com/quote.ashx?t=NVDA&p=d                                                             | ✓ | ⏱:
0.85s 
[COMPLETE] ● https://finviz.com/quote.ashx?t=NVDA&p=d                                                             | ✓ | ⏱:
4.25s 
Total unique headlines (History + Today): 120
Today's Headlines (NY Time): 5
Sending 5 headlines to Gemini in batches...
Processing batch 0 to 5...

TOP TRADABLE HEADLINES (Gemini Powered):
 1. ⚪ ⚡ [Score:145] [TRADABLE] [09:39AM] [NVDA] China Opens Door to Nvidia H200 Chips But Questions Outnumber Answers
    Type: REGULATION_GOV | Mat: 60 | Imm: 70 | PricedIn: 30%
    Mech: Potential access to China market with H200 chips.
    Watch: Actual sales numbers in China.
    Link: https://finance.yahoo.com/m/d433cf59-e29a-3dda-9559-2abc70b3a29c/china-opens-door-to-nvidia.html

 2. 🔴 ⚡ [Score:94] [TRADABLE] [10:00AM] [NVDA] Investors Hedge China, Tech Risks Amid Trump TACO Trade Drama
    Type: REGULATION_GOV | Mat: 40 | Imm: 50 | PricedIn: 40%
    Mech: Hedge China/Tech risks amid Trump trade drama.
    Watch: Political developments; trade policy changes.
    Link: https://finance.yahoo.com/news/investors-hedge-china-tech-risks-150000906.html


Appended -> out/news_log.jsonl
(taenv) jordo@Gideon:~/news_crawler$ '''



# --- NEW: Timezone handling ---
import pytz
NY_TZ = pytz.timezone("America/New_York")

load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)
key = os.getenv("GEMINI_API_KEY", "")

def mask_key(k: str, show_last: int = 4) -> str:
    if not k:
        return "MISSING"
    return "****" + k[-show_last:]

print("GEMINI_API_KEY present:", bool(key))
print("GEMINI_API_KEY masked :", mask_key(key))

# --- Changed: Replaced Torch/Transformers with Gemini ---
import google.generativeai as genai
from google.api_core import retry
from crawl4ai import AsyncWebCrawler

OUT = Path("out")
OUT.mkdir(exist_ok=True)
LOG = OUT / "news_log.jsonl"

FINVIZ_QUOTE_URL = "https://finviz.com/quote.ashx?t={}&p=d"

# SPY + MAG7 (plus GOOG alias)
WATCH = { "NVDA"}

# --- Scoring Constants ---
TRASH_DOMAINS = {
    "facebook.com", "twitter.com", "x.com", "finviz.com",
}
MID_DOMAINS = {"seekingalpha.com", "benzinga.com", "investorplace.com", "motleyfool.com"}

# --- NEW: Roundup / liveblog filters (kills “movers”, “stock market today”, etc.) ---
GENERIC_ROUNDUP_PATTERNS = [
    r"these stocks are (today'?s )?movers",
    r"stock market today",
    r"live coverage",
    r"stocks? to watch",
    r"world stocks",
    r"market(s)? (sink|sinks|dive|dives|fall|falls|slide|slides)",
    r"why the stock market",
]

ANALYST_ACTION_WORDS = [
    "downgrade","upgraded","upgrade","initiated","initiation",
    "price target","pt ","raises target","cuts target",
    "buy rating","sell rating","underperform","outperform",
    "overweight","underweight"
]

# If you want “DIRECT EVENT ONLY” behavior:
DIRECT_ONLY = True

# If you want to exclude vague OTHER category unless it’s an analyst action:
ALLOW_OTHER_ONLY_IF_ANALYST_ACTION = True

KEY_EVENT_WORDS = [
    "earnings","guidance","revenue","profit","miss","beat","raises","cuts",
    "downgrade","upgrade","sec","doj","lawsuit","antitrust",
    "acquires","merger","partnership","contract",
    "forecast","outlook","margin","layoffs","buyback","dividend"
]

# --- Explicit High-Impact Keywords for Rank Boosting ---
HIGH_IMPACT_KEYWORDS = {
    "earnings","guidance","revenue","profit","miss","beat",
    "regulation","ban","tariff","sanction","approval","restriction",
    "lawsuit","investigation","antitrust","merger","acquire","acquisition",
    "shutdown","production","supply","macro","fed","cpi","gdp","geopolitical"
}

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def domain(url: str) -> str:
    return urlparse(url).netloc.lower()

def is_roundup_or_liveblog(title: str) -> bool:
    t = (title or "").lower()
    if any(re.search(p, t) for p in GENERIC_ROUNDUP_PATTERNS):
        return True
    # Title listing many tickers tends to be roundup content
    if (title or "").count(",") >= 3:
        return True
    return False

def has_analyst_action(title: str) -> bool:
    t = (title or "").lower()
    return any(w in t for w in ANALYST_ACTION_WORDS)

def story_key(title: str) -> str:
    # crude but effective dedupe key for repeated same-story headlines
    t = re.sub(r"[^a-z0-9\s]", " ", (title or "").lower())
    toks = [w for w in t.split() if w not in {"the","a","an","after","as","says","said","stock","shares","is","are"}]
    
    # Specific keywords for the China/H200 cluster (as requested)
    keep = [w for w in toks if w in {"h200","china","block","blocked","ban","customs","approval","stalled","stuck","shipment","shipments","export","restriction"}]
    
    # Fallback to general tokens if specific keywords miss, to avoid over-collapsing unrelated news
    if not keep:
        return " ".join(toks[:6])
        
    return " ".join(keep[:10])

def looks_like_nav(title: str) -> bool:
    t = title.strip()
    bad = {
        "Screener","Portfolio","Calendar","Backtests","Register","Market News",
        "Stocks News","ETF News","Crypto News","Forex News","Futures News",
        "Login","Elite","Home","News", "Submit"
    }
    if t in bad: return True
    if t.lower().startswith("skip to"): return True
    if t.startswith("English"): return True
    return False

def is_probable_headline(title: str, url: str) -> bool:
    t = " ".join(title.split()).strip()
    if len(t) < 15: return False
    if looks_like_nav(t): return False
    
    # --- NEW: Apply Roundup Filter ---
    # Drop generic market noise immediately
    if is_roundup_or_liveblog(t): 
        return False
        
    d = domain(url)
    if d.endswith("finviz.com"): return False
    if "utm_" in url: return False
    return True

# --- Heuristic Quality Score (Pre-Filter) ---
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

def is_today_ny(published_iso: str) -> bool:
    """
    Checks if the ISO date string represents 'Today' in New York time.
    """
    if not published_iso:
        return False
    try:
        # Parse ISO (naive or aware)
        dt = datetime.fromisoformat(published_iso)
        # Convert to NY time
        if dt.tzinfo is None:
            dt = NY_TZ.localize(dt)
        else:
            dt = dt.astimezone(NY_TZ)
            
        now = datetime.now(NY_TZ)
        return dt.date() == now.date()
    except Exception:
        return False

def parse_finviz_date(date_str: str) -> str | None:
    """
    Strict date parsing. Returns ISO string ONLY if the row explicitly
    states 'Today' or a full date. Bare times are dropped to ensure
    we don't mistakenly treat old news as today's news.
    """
    date_str = date_str.strip()
    now = datetime.now()

    try:
        # ✅ Trust explicit Today
        if date_str.startswith("Today"):
            time_part = date_str.split(" ", 1)[1]
            dt = datetime.strptime(time_part, "%I:%M%p")
            dt = dt.replace(year=now.year, month=now.month, day=now.day)
            return dt.isoformat()

        # ✅ Trust full date like Jan-13-26 03:58PM
        if re.match(r"^[A-Z][a-z]{2}-\d{2}-\d{2} \d{2}:\d{2}[AP]M$", date_str):
            dt = datetime.strptime(date_str, "%b-%d-%y %I:%M%p")
            return dt.isoformat()

        # ❌ If it's only "03:58PM", DO NOT assume today. 
        # This prevents "everything is today" bugs on quote pages.
        return None
    except Exception:
        return None

def extract_news_items(md: str):
    lines = md.split('\n')
    items = []
    seen = set()
    link_pat = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
    
    # Regex to capture timestamp. Matches: "Jan-13-26 09:30AM", "Today 09:30AM", or just "09:30AM"
    ts_pat = re.compile(r"((?:[A-Z][a-z]{2}-\d{2}-\d{2}\s+|Today\s+)?\d{2}:\d{2}[AP]M)")

    # STATE: Track the date context across lines (The "Finviz Mental Model")
    current_date_context = None 

    for line in lines:
        line = line.strip()
        if not line: continue
        
        links = link_pat.findall(line)
        if not links: continue

        # 1. Parsing Date/Time with Context
        match = ts_pat.search(line)
        published_iso = None
        raw_date_debug = None
        
        if match:
            raw_ts = match.group(1)
            raw_date_debug = raw_ts
            
            # Check if this row establishes a new date context
            if "Today" in raw_ts:
                current_date_context = datetime.now().date()
            elif re.match(r"^[A-Z][a-z]{2}-\d{2}-\d{2}", raw_ts):
                try:
                    # Parse "Jan-13-26"
                    d_str = raw_ts.split(' ')[0]
                    current_date_context = datetime.strptime(d_str, "%b-%d-%y").date()
                except:
                    pass
            
            # If we have a date context (either new or carried over), parse the time
            if current_date_context:
                try:
                    # Last 7 chars are always "HH:MMAM" e.g. "09:30AM"
                    t_str = raw_ts[-7:]
                    t_obj = datetime.strptime(t_str, "%I:%M%p").time()
                    
                    # Combine context date + row time
                    dt = datetime.combine(current_date_context, t_obj)
                    published_iso = dt.isoformat()
                except:
                    pass

        for title, url in links:
            title = " ".join(title.split()).strip()
            key = (title, url)
            if key in seen: continue
            if not is_probable_headline(title, url): continue
            seen.add(key)
            items.append({
                "title": title,
                "url": url,
                "published_raw": raw_date_debug, # For debug printing
                "published_iso": published_iso
            })
    return items

async def crawl_one(crawler, ticker: str):
    url = FINVIZ_QUOTE_URL.format(ticker)
    try:
        res = await crawler.arun(url=url)
        md = res.markdown or ""
        items = extract_news_items(md)
        for it in items:
            it["ticker_page"] = ticker
        return url, md, items
    except Exception as e:
        print(f"Error crawling {ticker}: {e}")
        return url, "", []

async def crawl_watchlist(crawler, tickers: list[str]):
    print(f"Starting parallel crawl for {len(tickers)} tickers: {tickers}")
    tasks = [crawl_one(crawler, t) for t in tickers]
    return await asyncio.gather(*tasks)

# --- GEMINI AI INTEGRATION ---
def init_gemini(api_key: str):
    if not api_key:
        print("ERROR: No Google API Key provided. Use --api-key or set GEMINI_API_KEY env var.")
        return None
    
    # Confirm loaded key for user peace of mind
    safe_key = api_key[:5] + "..." if len(api_key) > 5 else "***"
    print(f"✓ Gemini API Key loaded ({safe_key})")
    
    genai.configure(api_key=api_key)
    # Use stable Gemini 2.0 Flash
    return genai.GenerativeModel('gemini-2.0-flash') 


def batch_analyze_with_gemini(model, items: list) -> dict:
    """
    Sends a batch of headlines to Gemini to get JSON scores.
    Returns a dict { index_id: {sentiment, impact, reason, is_catalyst} }
    """
    if not items: return {}
    
    prompt_lines = [
        "You are a specific 'Trade Catalyst' classifier for stock news.",
        "Your job is to identify news that acts as a DIRECT CATALYST for the specific ticker symbols involved.",
        "",
        "CRITICAL INSTRUCTIONS FOR 'DIRECTNESS' AND 'MECHANISM':",
        "1. **DIRECT EVENT (Tier 1)**: The company IS the subject or object of the verb. (e.g., 'Nvidia banned', 'Nvidia earnings').",
        "2. **READ-THROUGH (Tier 3)**: The company is a shadow beneficiary of a peer/supplier. (e.g., 'TSMC earnings up' implies Nvidia demand). YOU MUST MARK THESE AS 'is_read_through': true.",
        "",
        "SCORING FACTORS:",
        "- Materiality (0-100): How significant is the revenue/price impact?",
        "- Immediacy (0-100): Is the effect confirmed now vs theoretical future?",
        "- PricedIn (0-100): Is this old news or widely expected?",
        "",
        "OUTPUT JSON ARRAY ONLY. Each object:",
        "{",
        "  'id': int,",
        "  'sentiment': 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL',",
        "  'materiality': 0-100,",
        "  'immediacy': 0-100,",
        "  'priced_in': 0-100,",
        "  'mechanism': string (max 10 words, e.g. 'Reduced China revenue due to ban'),",
        "  'watch_next': string (max 10 words, e.g. 'China ministry response'),",
        "  'category': 'REGULATION_GOV' | 'EARNINGS_GUIDANCE' | 'SUPPLY_PRODUCTION' | 'MNA_DEAL' | 'OTHER',",
        "  'is_direct_event': boolean (True if company is subject/object, False if peer/sector read-through),",
        "  'is_tradable': boolean",
        "}",
        "",
        "HEADLINES:"
    ]
    
    for it in items:
        # Include context (found on which ticker page) to help decide directness
        found_context = ",".join(list(it.get("found_on", [])))
        prompt_lines.append(f"{it['_gid']}. {it['title']} (Context: {found_context})")
    
    prompt = "\n".join(prompt_lines)
    
    try:
        # Generate
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        data = json.loads(response.text)
        
        # Map back to dict
        results = {}
        rows = data if isinstance(data, list) else data.get("headlines", [])
        
        for row in rows:
            results[row['id']] = row
        return results
            
    except Exception as e:
        print(f"Gemini Batch Error: {e}")
        return {}

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default=None, help="Crawl specific ticker(s).")
    ap.add_argument("--print", type=int, default=30, help="How many ranked headlines to print.")
    ap.add_argument("--api-key", type=str, default=os.getenv("GEMINI_API_KEY"), help="Google Gemini API Key. Defaults to env var.")
    args = ap.parse_args()

    # Determine targets
    if args.ticker:
        tickers = [t.strip().upper() for t in args.ticker.split(',')]
    else:
        tickers = sorted([t for t in WATCH if t != "GOOG"])

    # Init Gemini
    model = init_gemini(args.api_key)

    async with AsyncWebCrawler() as crawler:
        print("\n" + "=" * 110)
        results = await crawl_watchlist(crawler, tickers)

        # Flatten & Deduplicate
        unique_map = {}
        for url, md, items in results:
            for it in items:
                key = (it["title"], it["url"])
                if key not in unique_map:
                    it["found_on"] = {it["ticker_page"]} if "ticker_page" in it else set()
                    unique_map[key] = it
                else:
                    if "ticker_page" in it: unique_map[key]["found_on"].add(it["ticker_page"])
        
        candidates = list(unique_map.values())
        print(f"Total unique headlines (History + Today): {len(candidates)}")

        # --- 1. Assign Global IDs (_gid) ---
        for i, h in enumerate(candidates):
            h["_gid"] = i

        # --- 2. Filter for Today Only (NY Time) ---
        today_candidates = [h for h in candidates if is_today_ny(h.get("published_iso"))]
        
        print(f"Today's Headlines (NY Time): {len(today_candidates)}")
        
        candidates = today_candidates

        # 3. Heuristic Scoring (Pre-sort to prioritize what we send to Gemini)
        for h in candidates:
            h["qscore"] = quality_score(h["title"], h["url"])
        
        # Sort by heuristic first
        candidates.sort(key=lambda x: x["qscore"], reverse=True)
        
        if model:
            print(f"Sending {len(candidates)} headlines to Gemini in batches...")
            BATCH_SIZE = 20
            
            for i in range(0, len(candidates), BATCH_SIZE):
                batch = candidates[i : i + BATCH_SIZE]
                print(f"Processing batch {i} to {i+len(batch)}...")
                
                scores = batch_analyze_with_gemini(model, batch)
                
                # Apply scores to candidates
                for item in batch:
                    gid = item["_gid"]
                    if gid in scores:
                        s = scores[gid]
                        item["ai_sentiment"] = s.get("sentiment", "NEUTRAL")
                        
                        # New Fields
                        item["materiality"] = s.get("materiality", 0)
                        item["immediacy"] = s.get("immediacy", 0)
                        item["priced_in"] = s.get("priced_in", 50)
                        item["mechanism"] = s.get("mechanism", "Unknown")
                        item["watch_next"] = s.get("watch_next", "None")
                        item["ai_category"] = s.get("category", "OTHER")
                        item["is_direct"] = s.get("is_direct_event", False)
                        item["is_tradable"] = s.get("is_tradable", False)
                        
                    else:
                        item["ai_sentiment"] = "NEUTRAL"
                        item["materiality"] = 0
                        item["immediacy"] = 0
                        item["priced_in"] = 100
                        item["is_direct"] = False
                        item["is_tradable"] = False
                
                time.sleep(0.2) 

        # --- NEW: HARD GATES (direct-only catalyst mode) ---
        if DIRECT_ONLY:
            candidates = [h for h in candidates if h.get("is_direct") and h.get("is_tradable")]

        if ALLOW_OTHER_ONLY_IF_ANALYST_ACTION:
            filtered = []
            for h in candidates:
                cat = h.get("ai_category", "OTHER")
                if cat != "OTHER":
                    filtered.append(h)
                    continue
                # Keep OTHER only if it’s clearly an analyst action (downgrade/upgrade/PT/etc.)
                if has_analyst_action(h.get("title", "")):
                    filtered.append(h)
            candidates = filtered

        # Final Rank: Revised Trade Score Formula
        for h in candidates:
            mat = h.get("materiality", 0)
            imm = h.get("immediacy", 0)
            priced = h.get("priced_in", 0)
            is_direct = h.get("is_direct", False)
            
            # Base Score: Sum of Mat and Imm (Max 200)
            # Subtract a portion of 'PricedIn' (e.g., up to 30 points if 100% priced in)
            base_score = (mat + imm) - (priced * 0.3)
            
            # --- THE DIRECTNESS GATE ---
            # If it's a "Read-Through" (Not Direct), apply heavy penalty
            # Unless explicitly marked tradable by AI (but even then, discount it)
            if not is_direct:
                directness_mult = 0.6  # 40% Penalty for read-throughs
            else:
                directness_mult = 1.0

            # Category Multiplier
            cat_mult_map = {
                "REGULATION_GOV": 1.2,
                "EARNINGS_GUIDANCE": 1.2,
                "SUPPLY_PRODUCTION": 1.1,
                "MNA_DEAL": 1.1,
                "OTHER": 0.8
            }
            cat_mult = cat_mult_map.get(h.get("ai_category"), 1.0)
            
            final_score = base_score * directness_mult * cat_mult

            # Cap min score
            h["trade_score"] = max(0, final_score)

        # --- NEW: DEDUPE SAME-STORY HEADLINES (keep best scoring per story) ---
        dedup = {}
        for h in candidates:
            k = story_key(h.get("title", ""))
            if k not in dedup or h.get("trade_score", 0) > dedup[k].get("trade_score", 0):
                dedup[k] = h

        candidates = list(dedup.values())
        
        # Sort by Trade Score
        candidates.sort(key=lambda x: x.get("trade_score", -999), reverse=True)

        print("\nTOP TRADABLE HEADLINES (Gemini Powered):")
        if not candidates:
            print("No news found for TODAY (New York Time).")

        for i, h in enumerate(candidates[: args.print], 1):
            # Parse time for display
            pub_time = ""
            if h.get("published_iso"):
                try:
                    dt = datetime.fromisoformat(h["published_iso"])
                    pub_time = dt.strftime("%I:%M%p")
                except: pass
            
            found_on = sorted(list(h.get("found_on", [])))
            ticker_tag = f"[{found_on[0]}]" if found_on else "[--]"
            
            # Display format
            sent = h.get("ai_sentiment", "NEUTRAL")
            score = h.get("trade_score", 0)
            tradable_tag = "[TRADABLE]" if h.get("is_tradable") else ""
            
            # Icons
            sym = "⚪"
            if sent == "POSITIVE": sym = "🟢"
            if sent == "NEGATIVE": sym = "🔴"
            
            cat_icon = "⚡" if score > 80 else "  "

            # User Requested Format
            print(f"{i:2d}. {sym} {cat_icon} [Score:{score:.0f}] {tradable_tag} [{pub_time}] {ticker_tag} {h['title']}")
            print(f"    Type: {h.get('ai_category')} | Mat: {h.get('materiality')} | Imm: {h.get('immediacy')} | PricedIn: {h.get('priced_in')}%")
            print(f"    Mech: {h.get('mechanism')}")
            print(f"    Watch: {h.get('watch_next')}")
            print(f"    Link: {h['url']}\n")

        event = {
            "fetched_utc": utc_now_iso(),
            "mode": "gemini_crawl",
            "tickers_crawled": tickers,
            "headlines": [{**x, "found_on": list(x.get("found_on", []))} for x in candidates]
        }
        append_jsonl(LOG, event)
        print(f"\nAppended -> {LOG}")

if __name__ == "__main__":
    asyncio.run(main())
