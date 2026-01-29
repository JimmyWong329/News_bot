#!/usr/bin/env python3
"""
fidelity_one_article.py

Scrape ONE article from Fidelity "US Markets" and print the article text in the terminal.

Why it works on this site:
- Fidelity article pages are JS-rendered (raw HTML has {{news.*}} placeholders until JS runs),
  so we use Crawl4AI's `wait_for` with a JS condition.

Install:
  pip install crawl4ai beautifulsoup4 lxml
  # if Playwright browsers aren't installed yet:
  python -m playwright install

Run:
  python fidelity_one_article.py --headless 1
  python fidelity_one_article.py --headless 0 --debug_html out_listing.html --debug_article_html out_article.html
"""

import argparse
import asyncio
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

from tools.io_utils import append_jsonl, ensure_out_dir, now_et_iso, write_json


LISTING_URL = "https://www.fidelity.com/news/us-markets"
BASE_URL = "https://www.fidelity.com"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _best_html(result) -> str:
    # Crawl4AI returns multiple HTML variants; cleaned_html is usually best for parsing.
    return (result.cleaned_html or result.html or "").strip()


def extract_first_article_url(listing_html: str) -> str:
    soup = BeautifulSoup(listing_html, "lxml")

    # Most specific selector (matches what you saw in DevTools)
    a = soup.select_one("td[data-title='Title'] a[href^='/news/article/']")
    if not a:
        # Fallback: first internal news/article link anywhere
        a = soup.select_one("a[href^='/news/article/']")

    if not a or not a.get("href"):
        # Last-resort: regex scan
        m = re.search(r'href="(/news/article/[^"]+)"', listing_html)
        if not m:
            raise RuntimeError("Could not find an article link on the listing page.")
        href = m.group(1)
    else:
        href = a["href"]

    return urljoin(BASE_URL, href)


def extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    root = soup.select_one("#scl-news-article") or soup.select_one("div.scl-news-article")
    if not root:
        return ""

    title_el = root.select_one("h1.scl-news-article--heading") or root.find("h1")
    meta_el  = root.select_one("p.scl-news-article--author") or root.find("p")

    # pull only paragraphs + headings from the body block (avoids footer/share junk)
    body_block = root.select_one("div.scl-news-article--description")
    if body_block:
        chunks = []
        for el in body_block.select("p, h2, h3, li"):
            t = el.get_text(" ", strip=True)
            if t:
                chunks.append(t)
        body = "\n\n".join(chunks).strip()
    else:
        body = ""

    title = title_el.get_text(" ", strip=True) if title_el else ""
    meta  = meta_el.get_text(" ", strip=True) if meta_el else ""

    parts = [p for p in [title, meta, body] if p]
    return "\n\n".join(parts).strip()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=LISTING_URL)
    ap.add_argument("--headless", type=int, default=1)
    ap.add_argument("--debug_html", default="")
    ap.add_argument("--debug_article_html", default="")
    ap.add_argument("--out_dir", type=str, default="out", help="Output directory (default: out)")
    ap.add_argument("--out_json", type=str, default=None, help="Write article JSON to a path")
    ap.add_argument("--out_jsonl", type=str, default=None, help="Write article JSONL to a path")
    ap.add_argument("--out_raw_html", type=str, default="", help="Write raw HTML to a path")
    ap.add_argument("--date", type=str, default="", help="Override date YYYY-MM-DD for metadata")
    args = ap.parse_args()

    out_dir = ensure_out_dir(args.out_dir)
    browser_cfg = BrowserConfig(
        headless=bool(args.headless),
        browser_type="chromium",
        viewport_width=1400,
        viewport_height=900,
        user_agent=UA,
        text_mode=True,
        enable_stealth=True,
    )

    # 1) Load listing page (wait until the table has article links)
    listing_run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_until="domcontentloaded",
        page_timeout=90000,
        wait_for="css:a[href^='/news/article/']", 
        wait_for_timeout=90000,
        remove_overlay_elements=True,
        scan_full_page=True,
        scroll_delay=0.2,
        excluded_tags=["nav", "footer", "header"],
        delay_before_return_html=0.2,
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        listing_res = await crawler.arun(args.url, config=listing_run_cfg)
        if not listing_res.success:
            raise RuntimeError(f"Listing crawl failed: {listing_res.error_message}")

        listing_html = _best_html(listing_res)
        if args.debug_html:
            Path(args.debug_html).write_text(listing_html, encoding="utf-8")

        article_url = extract_first_article_url(listing_html)

        # 2) Load the article page (wait until JS replaces {{news.*}} placeholders)
        article_run_cfg = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            wait_until="domcontentloaded",
            page_timeout=90000,
            # Wait for the specific article body container
            wait_for="css:div.scl-news-article--description",
            wait_for_timeout=90000,
            remove_overlay_elements=True,
            excluded_tags=["nav", "footer", "header"],
            delay_before_return_html=0.2,
        )

        article_res = await crawler.arun(article_url, config=article_run_cfg)
        if not article_res.success:
            raise RuntimeError(f"Article crawl failed: {article_res.error_message}")

        # Use raw HTML because cleaned_html often strips specific JS-rendered containers like this one
        article_html = (article_res.html or "").strip()
        
        if args.debug_article_html:
            Path(args.debug_article_html).write_text(article_html, encoding="utf-8")

        text = extract_article_text(article_html)

        print("=" * 100)
        print("Fidelity | US Markets | Article")
        print(article_url)
        print("=" * 100)
        print(text if text else "[No article text extracted -- run with --headless 0 and --debug_article_html to inspect HTML.]")
        print("=" * 100)

        raw_html_path = ""
        if args.out_raw_html:
            raw_path = Path(args.out_raw_html)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(article_html, encoding="utf-8")
            raw_html_path = str(raw_path)
        elif args.out_json or args.out_jsonl:
            meta_date = args.date.strip() or datetime.now().strftime("%Y-%m-%d")
            raw_path = out_dir / "raw" / f"fid_{meta_date}_0000.html"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(article_html, encoding="utf-8")
            raw_html_path = str(raw_path)

        if args.out_json or args.out_jsonl:
            meta_date = args.date.strip() or datetime.now().strftime("%Y-%m-%d")
            meta = {
                "source_script": "fid.py",
                "generated_at_et": now_et_iso(),
                "date": meta_date,
                "asof_et": "",
                "run_id": f"{meta_date}_unknown_fid.py",
            }
            article_obj = {
                "meta": meta,
                "source": "fid",
                "phase": "UNKNOWN",
                "published_at_et": "",
                "url": article_url,
                "title": "",
                "key_points": [],
                "tickers_mentioned": [],
                "raw_html_path": raw_html_path,
                "raw_text_excerpt": text[:500],
            }
            if args.out_json:
                write_json(Path(args.out_json), article_obj)
            if args.out_jsonl:
                append_jsonl(Path(args.out_jsonl), article_obj)


if __name__ == "__main__":
    asyncio.run(main())
