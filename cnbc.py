#!/usr/bin/env python3
"""
cnbc_full_liveblog.py
- Finds today's (or given date's) CNBC "Stock Market Today: Live updates" URL
- Opens it with Crawl4AI
- Clicks "Load more" repeatedly + scrolls to load posts
- Extracts FULL intro + FULL live updates (no truncation)
- Scrapes relative timestamps ("40 Min Ago") from DOM
- OPTIONAL: Computes relative timestamps relative to an --asof time (great for historical replays)
- Prints everything to terminal (and optionally saves a .txt)

Notes:
- If you run for a past date, the "Ago" labels on the page might be absolute dates or broken.
  Use --asof "YYYY-MM-DD HH:MM" to force computed "Ago" labels.

Usage:
  # Live run (uses DOM "40 Min Ago" labels)
  python cnbc_full_liveblog.py --headless 1

  # Historical run (strict date match)
  python cnbc_full_liveblog.py --date 2026-01-23 --strict_date 1 --headless 0

  # Historical run with computed "Ago" times
  python cnbc_full_liveblog.py --date 2023-06-15 --query "stock market live updates June 15 2023" --asof "2023-06-15 16:00"
"""

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

NY = ZoneInfo("America/New_York")

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# CNBC uses Queryly for site search; we use it to find the daily link reliably.
QUERYLY_ENDPOINT = "https://api.queryly.com/cnbc/json.aspx"
QUERYLY_KEY = "31a35d40a9a64ab3"
ADDITIONAL_INDEXES = "4cd6f71fbf22424d,937d600b0d0d4e23,3bfbe40caee7443e,626fdfcd96444f28"


@dataclass
class LiveUpdate:
    time_iso: str
    headline: str
    text: str
    url: Optional[str] = None


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _date_from_cnbc_url(url: str) -> Optional[date]:
    """Extracts date object from standard CNBC URL structure /YYYY/MM/DD/."""
    m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
    if not m:
        return None
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def discover_live_updates_url(target_date: str, query: str, max_results: int = 50, strict_date: bool = True) -> str:
    params = {
        "queryly_key": QUERYLY_KEY,
        "query": query,
        "endindex": "0",
        "batchsize": str(max_results),
        "callback": "",
        "showfaceted": "false",
        "timezoneoffset": "-300",
        "facetedfields": "formats",
        "facetedkey": "formats|",
        "facetedvalue": "!Press Release|",
        "needtoptickers": "1",
        "additionalindexes": ADDITIONAL_INDEXES,
    }

    r = requests.get(QUERYLY_ENDPOINT, params=params, timeout=25)
    r.raise_for_status()
    data = r.json()

    results = data.get("results", [])
    if not results:
        raise RuntimeError("No results from CNBC search backend.")

    target = date.fromisoformat(target_date)

    # Keep only the live-updates URLs, then choose the closest by date
    live_urls: List[Tuple[int, str]] = []
    for item in results:
        url = (
            item.get("cn:liveURL")
            or item.get("cn:liveUrl")
            or item.get("url")
            or item.get("link")
            or ""
        )
        url = str(url)
        if "cnbc.com" not in url:
            continue

        # HARD filter: only the daily live-updates series
        if "stock-market-today-live-updates" not in url:
            continue

        d = _date_from_cnbc_url(url)
        if not d:
            continue

        # smaller diff = better
        diff_days = abs((d - target).days)
        score = 10_000 - diff_days  # highest = closest date
        # bonus if exact date
        if d == target:
            score += 1_000_000

        live_urls.append((score, url))

    if not live_urls:
        raise RuntimeError("No 'stock-market-today-live-updates' URLs found in search results.")

    live_urls.sort(key=lambda x: x[0], reverse=True)
    best = live_urls[0][1]
    best_date = _date_from_cnbc_url(best)

    if strict_date and best_date != target:
        raise RuntimeError(f"No exact match for {target_date}. Closest found: {best} (date={best_date})")

    return best


def pick_liveblog_node(json_obj: Any) -> Optional[Dict[str, Any]]:
    """Find a dict node whose @type includes LiveBlogPosting."""
    nodes: List[Dict[str, Any]] = []

    def add_candidate(x: Any):
        if isinstance(x, dict):
            nodes.append(x)
            if "@graph" in x and isinstance(x["@graph"], list):
                for g in x["@graph"]:
                    if isinstance(g, dict):
                        nodes.append(g)

    if isinstance(json_obj, dict):
        add_candidate(json_obj)
    elif isinstance(json_obj, list):
        for x in json_obj:
            add_candidate(x)

    for n in nodes:
        t = n.get("@type")
        if isinstance(t, list) and "LiveBlogPosting" in t:
            return n
        if isinstance(t, str) and t == "LiveBlogPosting":
            return n
    return None


def extract_dom_intro_and_image(html: str) -> Tuple[str, str]:
    """
    Fallback extraction from DOM if JSON-LD description is too short.
    Extracts the full article intro paragraphs and the main figure caption.
    """
    soup = BeautifulSoup(html, "lxml")

    # Anchor around the main headline
    h1 = soup.find("h1")
    if not h1:
        return "", ""

    # Usually CNBC wraps content in an article tag or similar container
    article = h1.find_parent("article") or soup

    # Image caption/credit (best effort)
    caption = ""
    fig = article.find("figure")
    if fig:
        cap = fig.find("figcaption")
        if cap:
            caption = cap.get_text("\n", strip=True)

    # Intro paragraphs: grab <p> blocks until the live updates section starts
    paras = []
    for p in article.find_all("p"):
        t = p.get_text(" ", strip=True)
        if not t:
            continue
        # stop once we hit the "Min Ago / Hour Ago" style live-update list
        # CNBC live update timestamps often look like "32 Min Ago" or "1 Hour Ago"
        if re.search(r"\b(\d+)\s+(Min|Hour)s?\s+Ago\b", t):
            break
        paras.append(t)

    intro = "\n\n".join(paras).strip()
    return intro, caption


def extract_dom_relative_times(html: str) -> Dict[str, Dict[str, str]]:
    """
    Returns { "108257243-post": {"text": "40 Min Ago", "datetime": "2026-..."}, ... }
    scraped from <time data-testid="lastpublished-timestamp"> inside div[id$="-post"]
    """
    soup = BeautifulSoup(html, "lxml")
    rel = {}

    # Each live update post is a div with id like "108257243-post"
    for post in soup.select('div[id$="-post"]'):
        pid = post.get("id")
        if not pid:
            continue
        t = post.select_one('time[data-testid="lastpublished-timestamp"]')
        if t:
            # We grab both the visible text and the machine attribute
            text_val = t.get_text(" ", strip=True)
            dt_val = t.get("datetime") or ""
            rel[pid] = {"text": text_val, "datetime": str(dt_val)}

    return rel


def post_id_from_update_url(update_url: str) -> str:
    # update_url looks like ...html#108257243-post
    if not update_url:
        return ""
    frag = urlparse(update_url).fragment
    return frag or ""


def extract_from_jsonld(html: str) -> Tuple[Dict[str, str], List[LiveUpdate]]:
    soup = BeautifulSoup(html, "lxml")
    meta: Dict[str, str] = {}

    # Title fallback
    if soup.title and soup.title.text:
        meta["title"] = clean(soup.title.text)

    scripts = soup.select('script[type="application/ld+json"]')

    for s in scripts:
        raw = (s.string or s.get_text() or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue

        node = pick_liveblog_node(obj)
        if not node:
            continue

        # Meta fields (best effort; CNBC may vary)
        meta["title"] = clean(str(node.get("headline") or node.get("name") or meta.get("title", "")))
        meta["date_modified"] = clean(str(node.get("dateModified") or node.get("datePublished") or ""))
        meta["date_published"] = clean(str(node.get("datePublished") or ""))
        # author can be dict or list
        author = node.get("author")
        if isinstance(author, dict):
            meta["author"] = clean(str(author.get("name") or ""))
        elif isinstance(author, list) and author:
            if isinstance(author[0], dict):
                meta["author"] = clean(str(author[0].get("name") or ""))

        # Sometimes intro/description is here
        desc = node.get("description") or node.get("articleBody") or ""
        meta["intro"] = clean(str(desc))

        # Image caption/credit sometimes accessible via image/name (varies)
        img = node.get("image")
        if isinstance(img, dict):
            meta["image_url"] = clean(str(img.get("url") or ""))
            meta["image_caption"] = clean(str(img.get("caption") or img.get("description") or ""))
        elif isinstance(img, list) and img:
            if isinstance(img[0], dict):
                meta["image_url"] = clean(str(img[0].get("url") or ""))
                meta["image_caption"] = clean(str(img[0].get("caption") or img[0].get("description") or ""))

        updates = node.get("liveBlogUpdate") or []
        if isinstance(updates, dict):
            updates = [updates]

        out: List[LiveUpdate] = []
        if isinstance(updates, list):
            for u in updates:
                if not isinstance(u, dict):
                    continue
                time_iso = str(u.get("datePublished") or u.get("dateModified") or "")
                headline = clean(str(u.get("headline") or u.get("name") or ""))
                text = clean(str(u.get("articleBody") or u.get("text") or ""))
                url = u.get("url")
                if headline or text:
                    out.append(LiveUpdate(time_iso=time_iso, headline=headline, text=text, url=url))

        # Sort chronologically if time strings exist
        out.sort(key=lambda x: x.time_iso or "")
        return meta, out

    return meta, []


def iso_to_et_date(iso_str: str) -> Optional[str]:
    if not iso_str:
        return None
    # CNBC uses formats like 2026-01-26T23:18:07+0000
    m = re.match(r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([+-]\d{4})$", iso_str)
    if not m:
        # try plain prefix
        if re.match(r"^\d{4}-\d{2}-\d{2}", iso_str):
            return iso_str[:10]
        return None
    dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%S%z")
    return dt.astimezone(NY).strftime("%Y-%m-%d")


def parse_iso_datetime(iso_str: str) -> Optional[datetime]:
    if not iso_str:
        return None
    try:
        # 2026-01-26T20:02:00+0000
        return datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        pass
    # Fallback/other formats could go here
    return None


def compute_ago(update_dt: datetime, asof_dt: datetime) -> str:
    """Computes a human-friendly 'Xh Ym ago' string."""
    diff = asof_dt - update_dt
    seconds = diff.total_seconds()
    
    if seconds < 0:
        return "in the future" # Should not happen if asof is correct
    
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} Min Ago"
    
    hours = int(minutes // 60)
    minutes_rem = minutes % 60
    return f"{hours}h {minutes_rem}m Ago"


def filter_updates_by_et_day(updates: List[LiveUpdate], target_date: str) -> List[LiveUpdate]:
    kept = []
    for u in updates:
        d = iso_to_et_date(u.time_iso)
        if d is None or d == target_date:
            kept.append(u)
    return kept


async def crawl(url: str, headless: bool, debug_html: Optional[Path], load_more_clicks: int, scroll_rounds: int) -> str:
    browser_cfg = BrowserConfig(
        browser_type="chromium",
        headless=headless,
        viewport_width=1365,
        viewport_height=768,
        user_agent=DEFAULT_UA,
    )

    # This JS tries to:
    # - accept cookies
    # - click "Load more" repeatedly
    # - scroll to force lazy-load
    js_code = f"""
(async () => {{
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  const clickByText = (regex) => {{
    const els = Array.from(document.querySelectorAll('button, a, div[role="button"]'));
    const hit = els.find(el => regex.test((el.textContent || '').trim()));
    if (hit) {{ hit.click(); return true; }}
    return false;
  }};

  // consent / overlays
  clickByText(/accept/i);
  clickByText(/agree/i);
  clickByText(/continue/i);
  const closeBtn = document.querySelector('[aria-label="Close"], button[aria-label="close"], .close, .CloseButton');
  if (closeBtn) closeBtn.click();

  // initial scroll
  window.scrollBy(0, Math.floor(window.innerHeight * 0.9));
  await sleep(800);

  // click load more posts
  for (let i = 0; i < {load_more_clicks}; i++) {{
    const clicked = clickByText(/load more|more posts|show more/i);
    window.scrollTo(0, document.body.scrollHeight);
    await sleep(clicked ? 1100 : 700);
    if (!clicked) break;
  }}

  // extra scrolling rounds to fill content
  for (let j = 0; j < {scroll_rounds}; j++) {{
    window.scrollTo(0, document.body.scrollHeight);
    await sleep(800);
  }}
}})();
"""

    wait_for = r"""js:() => {
      const jsonLd = document.querySelectorAll('script[type="application/ld+json"]').length;
      return jsonLd > 0;
    }"""

    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        js_code=js_code,
        wait_for=wait_for,
        page_timeout=90000,
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)

    if not result or not result.success or not result.html:
        err = getattr(result, "error_message", None) or "Unknown failure (possible block)"
        raise RuntimeError(f"Crawl failed: {err}")

    if debug_html:
        debug_html.write_text(result.html, encoding="utf-8")
        print(f"[debug] wrote HTML -> {debug_html}")

    return result.html


def format_output(meta: Dict[str, str], url: str, updates: List[LiveUpdate], rel_map: Dict[str, Dict[str, str]], asof_str: str = "") -> str:
    lines: List[str] = []

    title = meta.get("title") or "CNBC Live Updates"
    lines.append(title)
    lines.append(url)
    lines.append("-" * 88)

    # Prepare asof time if provided
    asof_dt = None
    if asof_str:
        try:
            # Flexible parsing (simple YYYY-MM-DD HH:MM usually enough)
            # We assume the input is NY time for convenience
            dt_naive = datetime.strptime(asof_str, "%Y-%m-%d %H:%M")
            asof_dt = dt_naive.replace(tzinfo=NY)
            lines.append(f"Times calculated relative to: {asof_str} ET")
        except ValueError:
            lines.append(f"[Warning] Could not parse --asof '{asof_str}'. Using scraped labels.")

    # Header block
    if meta.get("author"):
        lines.append(f"Author: {meta['author']}")
    if meta.get("date_modified") or meta.get("date_published"):
        lines.append(f"Updated: {meta.get('date_modified','')}".strip())
        if meta.get("date_published"):
            lines.append(f"Published: {meta.get('date_published','')}".strip())
    
    if meta.get("image_caption"):
         lines.append(f"Image Caption: {meta['image_caption']}")

    intro = meta.get("intro", "")
    if intro:
        lines.append("")
        lines.append(intro)
        lines.append("")

    # Live updates
    for i, u in enumerate(updates, start=1):
        pid = post_id_from_update_url(u.url)
        
        # Determine the "ago" label
        ago_str = ""
        
        if asof_dt and u.time_iso:
            # 1. Computed preference if --asof is set
            update_dt = parse_iso_datetime(u.time_iso)
            if update_dt:
                # convert update_dt to NY for consistent diff (though total_seconds works across zones)
                update_dt = update_dt.astimezone(NY)
                ago_str = compute_ago(update_dt, asof_dt)
        
        if not ago_str:
            # 2. Fallback to DOM text (scraped "40 Min Ago" or "UPDATED ...")
            dom_data = rel_map.get(pid, {})
            ago_str = dom_data.get("text", "")
        
        prefix = f"[{ago_str}] " if ago_str else ""

        t = u.time_iso or ""
        h = u.headline or "(no headline)"
        lines.append(f"{i:03d}. {prefix}{t}  {h}")
        if u.text:
            lines.append(u.text)
        if u.url:
            lines.append(u.url)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default="")
    ap.add_argument("--query", type=str, default="stock market today live updates")
    ap.add_argument("--headless", type=int, default=1)
    ap.add_argument("--debug_html", type=str, default="")
    ap.add_argument("--out_txt", type=str, default="")
    ap.add_argument("--load_more_clicks", type=int, default=25)
    ap.add_argument("--scroll_rounds", type=int, default=6)
    ap.add_argument("--asof", type=str, default="", help="Calculate relative times from this date/time (YYYY-MM-DD HH:MM)")
    ap.add_argument("--strict_date", type=int, default=1, help="1=require exact date match, 0=allow closest")
    args = ap.parse_args()

    target_date = args.date.strip() or datetime.now(NY).strftime("%Y-%m-%d")

    url = discover_live_updates_url(target_date=target_date, query=args.query, strict_date=bool(args.strict_date))
    print(f"[pick] {target_date} -> {url}")

    debug_path = Path(args.debug_html) if args.debug_html else None
    html = await crawl(
        url=url,
        headless=bool(args.headless),
        debug_html=debug_path,
        load_more_clicks=args.load_more_clicks,
        scroll_rounds=args.scroll_rounds,
    )

    meta, updates = extract_from_jsonld(html)
    
    # --- Upgrade A: DOM Fallback Logic ---
    if len(meta.get("intro", "")) < 120:
        dom_intro, dom_caption = extract_dom_intro_and_image(html)
        if dom_intro:
            meta["intro"] = dom_intro
        if dom_caption:
            meta["image_caption"] = dom_caption
    
    # --- Upgrade B: Relative Time Logic (with robust datetime scraping) ---
    rel_map = extract_dom_relative_times(html)
    # --------------------------------------------------------------------

    updates = filter_updates_by_et_day(updates, target_date)

    out = format_output(meta, url, updates, rel_map, asof_str=args.asof)
    print("\n" + out)

    if args.out_txt:
        p = Path(args.out_txt)
        p.write_text(out, encoding="utf-8")
        print(f"[saved] {p.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
