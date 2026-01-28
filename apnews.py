#!/usr/bin/env python3
"""
ap_today_article.py
- Crawl AP "Financial Markets" hub
- Pick latest story posted/updated for a TARGET DATE (default: today)
- Open that story
- Print full article text (RichTextStoryBody paragraphs) in terminal

Run:
  python ap_financial_markets_crawl.py --headless 0 --debug 1
  python ap_financial_markets_crawl.py --date 2026-01-23
"""

import argparse
import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin

from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

HUB_URL = "https://apnews.com/hub/financial-markets"
NY = ZoneInfo("America/New_York")

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

@dataclass
class HubItem:
    title: str
    url: str
    posted_ms: int
    updated_ms: int

def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def to_int(s: str) -> int:
    try:
        return int(s)
    except Exception:
        return 0

def ms_to_ny_date(ms: int) -> Optional[str]:
    if not ms:
        return None
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(NY)
    return dt.strftime("%Y-%m-%d")

def score_ms(it: HubItem) -> int:
    return it.updated_ms or it.posted_ms

async def crawl(url: str, headless: bool, debug_path: Optional[Path], wait_for: str, scroll_rounds: int) -> str:
    # Crawl4AI docs: wait_for must be "css:..." or "js:..."
    browser_cfg = BrowserConfig(
        browser_type="chromium",
        headless=headless,
        user_agent=DEFAULT_UA,
        viewport_width=1365,
        viewport_height=768,
        enable_stealth=True,  # helps with bot detection
    )

    js_code = f"""
(async () => {{
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  // ONLY click consent-like BUTTONS (never <a>, to avoid clicking article text like "agreed")
  const clickByText = (regex) => {{
    const els = Array.from(document.querySelectorAll(
      'button, div[role="button"], input[type="button"], input[type="submit"]'
    ));
    const hit = els.find(el => regex.test((el.textContent || '').trim()));
    if (hit) {{ hit.click(); return true; }}
    return false;
  }};

  clickByText(/^(accept|accept all|i agree|agree)$/i);

  for (let i = 0; i < {scroll_rounds}; i++) {{
    window.scrollTo(0, document.body.scrollHeight);
    await sleep(650);
  }}
}})();
"""

    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        js_code=js_code,
        wait_for=wait_for,                 # IMPORTANT: use css:/js: prefix
        page_timeout=90000,
        remove_overlay_elements=True,      # removes popups/overlays when possible
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        res = await crawler.arun(url=url, config=run_cfg)

    if not res or not res.success or not res.html:
        err = getattr(res, "error_message", None) or "Unknown failure"
        raise RuntimeError(f"Crawl failed for {url}: {err}")

    if debug_path:
        debug_path.write_text(res.html, encoding="utf-8")
        print(f"[debug] wrote HTML -> {debug_path}")

    return res.html

def parse_hub(html: str, base_url: str, max_items: int) -> List[HubItem]:
    soup = BeautifulSoup(html, "lxml")
    out: List[HubItem] = []

    for promo in soup.select("div.PagePromo"):
        a = promo.select_one("h3.PagePromo-title a.Link, h3.PagePromo-title a")
        if not a:
            continue

        title = clean(a.get_text(" ", strip=True))
        href = (a.get("href") or "").strip()
        if not title or not href:
            continue

        url = urljoin(base_url, href)
        posted_ms = to_int(promo.get("data-posted-date-timestamp", "0"))
        updated_ms = to_int(promo.get("data-updated-date-timestamp", "0"))

        out.append(HubItem(title=title, url=url, posted_ms=posted_ms, updated_ms=updated_ms))
        if len(out) >= max_items:
            break

    # de-dupe by url
    seen = set()
    dedup = []
    for it in out:
        if it.url in seen:
            continue
        seen.add(it.url)
        dedup.append(it)
    return dedup

def pick_latest_for_date(items: List[HubItem], target_yyyy_mm_dd: str) -> Tuple[HubItem, bool]:
    """
    Pick latest item matching the target date (NY time).
    Returns (item, is_exact_match).
    If no match for date, returns latest item found with False.
    """
    target_items = []
    for it in items:
        d = ms_to_ny_date(score_ms(it))
        if d == target_yyyy_mm_dd:
            target_items.append(it)

    if target_items:
        target_items.sort(key=score_ms, reverse=True)
        return target_items[0], True

    # fallback (still give you something, but tell you it wasn't the target date)
    if not items:
        raise RuntimeError("No hub items found.")
    items_sorted = sorted(items, key=score_ms, reverse=True)
    return items_sorted[0], False

def extract_article_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")

    # AP article pages usually have a story body container (RichTextStoryBody)
    h1 = soup.find("h1")
    title = clean(h1.get_text(" ", strip=True)) if h1 else ""

    # Prefer RichTextStoryBody paragraphs if present; fall back to main/article p tags
    paras = []
    body_root = soup.select_one("div.RichTextStoryBody, div.RichTextBody")
    if body_root:
        for p in body_root.select("p"):
            t = clean(p.get_text(" ", strip=True))
            if t:
                paras.append(t)
    else:
        main = soup.find("main") or soup.find("article") or soup
        for p in main.find_all("p"):
            t = clean(p.get_text(" ", strip=True))
            if t:
                paras.append(t)

    body = "\n\n".join(paras).strip()
    return title, body

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", type=int, default=1)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--max_items", type=int, default=60)
    # New argument for filtering by date
    ap.add_argument("--date", type=str, default="", help="Target date YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    # Determine target date
    if args.date.strip():
        target_date = args.date.strip()
    else:
        target_date = datetime.now(NY).strftime("%Y-%m-%d")

    hub_dbg = Path("out_ap_hub.html") if args.debug else None
    art_dbg = Path("out_ap_article.html") if args.debug else None

    print(f"[crawl] Hub: {HUB_URL}")
    hub_html = await crawl(
        HUB_URL,
        headless=bool(args.headless),
        debug_path=hub_dbg,
        wait_for="css:div.PagePromo",   # correct Crawl4AI wait_for format
        scroll_rounds=4,
    )

    items = parse_hub(hub_html, base_url=HUB_URL, max_items=args.max_items)
    
    # Use the date-specific picker
    chosen, exact_match = pick_latest_for_date(items, target_date)

    chosen_date = ms_to_ny_date(score_ms(chosen)) or "unknown"
    print(f"\n[picked] {chosen.title}")
    print(f"        {chosen.url}")
    print(f"        hub_date(NY)={chosen_date}  target(NY)={target_date}  exact_match={exact_match}")
    print("-" * 88)

    # Wait for actual story body (RichTextStoryBody + many <p> tags)
    article_html = await crawl(
        chosen.url,
        headless=bool(args.headless),
        debug_path=art_dbg,
        wait_for="css:div.RichTextStoryBody p",  # stronger than meta/h1, matches DOM
        scroll_rounds=1,
    )

    title, body = extract_article_text(article_html)
    print(title or chosen.title)
    print()
    print(body if body else "[No body parsed — open out_ap_article.html and we’ll adjust selectors.]")
    print()

if __name__ == "__main__":
    asyncio.run(main())
