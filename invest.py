#!/usr/bin/env python3
"""
investopedia_live_today.py

- Open Investopedia Markets News hub
- Click "LIVE MARKETS NEWS" tab (best-effort)
- Grab the first "Live Markets News" article card (today’s top item)
- Open the article and print the text to the terminal
- ALSO prints: top updated time + each liveblog entry's time + headline ("subtitle")
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

LISTING_URL = "https://www.investopedia.com/markets-news-4427704"
BASE_URL = "https://www.investopedia.com"


def save_text(path: str | None, text: str) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def pick_first_live_markets_article(rendered_html: str) -> str:
    soup = BeautifulSoup(rendered_html, "lxml")

    live_cards = soup.select('div.card__content[data-tag="Live Markets News"]')
    for card in live_cards:
        a = card.find_parent("a", href=True)
        if a and a["href"]:
            href = a["href"].strip()
            return href if href.startswith("http") else urljoin(BASE_URL, href)

    for a in soup.select("a[href]"):
        txt = a.get_text(" ", strip=True).lower()
        if "live markets news" in txt:
            href = a["href"].strip()
            if href:
                return href if href.startswith("http") else urljoin(BASE_URL, href)

    raise RuntimeError("Could not find a 'Live Markets News' article card on the listing page.")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def extract_article_with_times_and_subtitles(rendered_html: str) -> str:
    soup = BeautifulSoup(rendered_html, "lxml")

    def _clean(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def _has_class(tag, cls: str) -> bool:
        classes = tag.get("class") or []
        return cls in classes

    def _inside_liveblog(tag) -> bool:
        # robust: walk parents and look for any class containing mntl-liveblog-item
        for parent in tag.parents:
            if getattr(parent, "get", None):
                classes = parent.get("class") or []
                if any(c == "mntl-liveblog-item" for c in classes):
                    return True
        return False

    # --- Title ---
    h1 = soup.find("h1")
    title = _clean(h1.get_text(" ", strip=True)) if h1 else None

    # --- Top time: "Updated January 26, 2026" + "06:11 PM EST" ---
    date_el = soup.select_one("div.mntl-attribution__item-date")
    ts_el = soup.select_one('div.comp.timestamp, div[id^="timestamp_"]')

    date_part = _clean(date_el.get_text(" ", strip=True)) if date_el else ""
    time_part = _clean(ts_el.get_text(" ", strip=True)) if ts_el else ""

    top_time = ""
    if date_part and time_part:
        top_time = f"{date_part} {time_part}"
    else:
        top_time = date_part or time_part

    # --- Collect LEAD paragraphs (intro above liveblog items) ---
    lead_paras = []
    for p in soup.select("p.mntl-sc-block, p[id^='mntl-sc-block']"):
        if _inside_liveblog(p):
            continue
        t = _clean(p.get_text(" ", strip=True))
        if not t:
            continue
        # filter tiny UI junk
        if t.lower() in {"close"}:
            continue
        if len(t) >= 25:
            lead_paras.append(t)

    # de-dupe lead
    lead_seen = set()
    lead_clean = []
    for t in lead_paras:
        if t not in lead_seen:
            lead_seen.add(t)
            lead_clean.append(t)

    # --- Liveblog items ---
    live_items = soup.select("div.mntl-liveblog-item")

    out = []
    if title:
        out.append(title)
        out.append("-" * min(len(title), 80))
    if top_time:
        out.append(f"TOP TIME: {top_time}")
        out.append("")

    # ✅ Print the missing intro section
    if lead_clean:
        out.append("LEAD:")
        out.append("")
        out.extend(lead_clean)
        out.append("\n" + ("=" * 80) + "\n")

    if live_items:
        entry_num = 0
        for item in live_items:
            rel = item.select_one('div[class*="relativePublishedDate"]')
            abs_time = item.select_one('div[class*="publishedDate"], time')

            rel_txt = _clean(rel.get_text(" ", strip=True)) if rel else ""
            abs_txt = _clean(abs_time.get_text(" ", strip=True)) if abs_time else ""

            # headline/subtitle container (your DOM: id="mntl-blogpost__headline_X-0")
            head_el = item.select_one('[id^="mntl-blogpost__headline"], .mntl-blogpost__headline')
            headline = ""
            if head_el:
                strong = head_el.find("strong")
                headline = _clean(strong.get_text(" ", strip=True)) if strong else _clean(head_el.get_text(" ", strip=True))

            # body paragraphs inside this liveblog entry (skip headline area)
            paras = []
            for p in item.select("p.mntl-sc-block, p"):
                # skip the headline paragraph if it lives under head_el
                if head_el and p.find_parent() and p.find_parent() == head_el:
                    continue
                if head_el:
                    # skip if any ancestor is the headline container
                    for parent in p.parents:
                        if parent == head_el:
                            break
                    else:
                        parent = None
                    if parent == head_el:
                        continue

                t = _clean(p.get_text(" ", strip=True))
                if t and len(t) >= 25:
                    paras.append(t)

            # de-dupe body
            seen = set()
            body = []
            for t in paras:
                if t not in seen:
                    seen.add(t)
                    body.append(t)

            if not headline and not body and not (rel_txt or abs_txt):
                continue

            entry_num += 1
            out.append(f"[{entry_num}]")
            if abs_txt:
                out.append(f"TIME: {abs_txt}")
            if rel_txt and rel_txt.lower() != abs_txt.lower():
                out.append(f"REL : {rel_txt}")
            if headline:
                out.append(f"SUBTITLE: {headline}")
            out.append("")
            out.extend(body if body else ["(no body text found for this entry)"])
            out.append("\n" + ("-" * 80) + "\n")

        return "\n".join(out).strip()

    # fallback (non-liveblog)
    if lead_clean:
        out.extend(lead_clean)
        return "\n\n".join(out).strip()

    raise RuntimeError("No readable content found (lead or liveblog).")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", type=int, default=1, help="1=headless, 0=visible browser")
    ap.add_argument("--debug_listing", type=str, default="", help="Save rendered listing HTML to this file")
    ap.add_argument("--debug_article", type=str, default="", help="Save rendered article HTML to this file")
    ap.add_argument("--out_dir", type=str, default="out", help="Output directory (default: out)")
    ap.add_argument("--out_json", type=str, default=None, help="Write article JSON to a path")
    ap.add_argument("--out_jsonl", type=str, default=None, help="Write article JSONL to a path")
    ap.add_argument("--out_raw_html", type=str, default="", help="Write raw HTML to a path")
    ap.add_argument("--date", type=str, default="", help="Override date YYYY-MM-DD for metadata")
    args = ap.parse_args()

    out_dir = ensure_out_dir(args.out_dir)
    browser_cfg = BrowserConfig(
        headless=bool(args.headless),
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        viewport_width=1400,
        viewport_height=900,
    )

    js_prep = r"""
    (() => {
      const tryClick = (sel) => {
        const el = document.querySelector(sel);
        if (el) { el.click(); return true; }
        return false;
      };
      tryClick('#onetrust-accept-btn-handler');

      // Click tab if present
      const candidates = Array.from(document.querySelectorAll('a,button,span,div'));
      const tab = candidates.find(el => (el.textContent || '').trim().toLowerCase() === 'live markets news');
      if (tab) tab.click();
    })();
    """

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        listing_cfg = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            js_code=js_prep,
            wait_for="js:() => document.querySelectorAll('a[class*=\"mntl-card-list-items\"], a.mntl-card-list-items').length > 0",
        )
        listing = await crawler.arun(url=LISTING_URL, config=listing_cfg)
        listing_html = listing.html or ""
        if not listing_html.strip():
            raise RuntimeError("Listing page returned empty HTML (possible block or timeout).")
        save_text(args.debug_listing or None, listing_html)

        article_url = pick_first_live_markets_article(listing_html)
        print(f"[picked] {article_url}")

        article_cfg = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            js_code=js_prep,
            # Liveblog pages: wait for liveblog items OR plenty of paragraphs
            wait_for=(
                "js:() => document.querySelectorAll('div.mntl-liveblog-item').length > 0 "
                "|| document.querySelectorAll('p.mntl-sc-block, article p').length > 5"
            ),
        )
        article = await crawler.arun(url=article_url, config=article_cfg)
        article_html = article.html or ""
        if not article_html.strip():
            raise RuntimeError("Article page returned empty HTML (possible block or timeout).")
        save_text(args.debug_article or None, article_html)

        text = extract_article_with_times_and_subtitles(article_html)
        print("\n" + "=" * 100)
        print(text)
        print("=" * 100 + "\n")

        raw_html_path = ""
        if args.out_raw_html:
            raw_path = Path(args.out_raw_html)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(article_html, encoding="utf-8")
            raw_html_path = str(raw_path)
        elif args.out_json or args.out_jsonl:
            meta_date = args.date.strip() or datetime.now().strftime("%Y-%m-%d")
            raw_path = out_dir / "raw" / f"invest_{meta_date}_0000.html"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(article_html, encoding="utf-8")
            raw_html_path = str(raw_path)

        if args.out_json or args.out_jsonl:
            meta_date = args.date.strip() or datetime.now().strftime("%Y-%m-%d")
            meta = {
                "source_script": "invest.py",
                "generated_at_et": now_et_iso(),
                "date": meta_date,
                "asof_et": "",
                "run_id": f"{meta_date}_unknown_invest.py",
            }
            article_obj = {
                "meta": meta,
                "source": "invest",
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
