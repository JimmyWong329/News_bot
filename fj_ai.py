#!/usr/bin/env python3
import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Optional, Dict

from bs4 import BeautifulSoup

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode


FJ_HOME_URL = "https://www.financialjuice.com/home"


def click_text_js(label: str) -> str:
    lab = (label or "").strip().upper()
    return f"""
(() => {{
  const target = {json.dumps(lab)};
  const el = Array.from(document.querySelectorAll('button,a,div,span'))
    .find(x => (x.textContent || '').trim().toUpperCase() === target);
  if (el && typeof el.click === 'function') {{ el.click(); return true; }}
  return false;
}})();
"""


WAIT_RISK_READY = """js:() => {
  // Prefer #SummaryDiv if present; otherwise search for the "need to know" block anywhere.
  const root = document.querySelector('#SummaryDiv') || document;
  const titleEl = root.querySelector('.headline-title-nolink, .headline-title');
  const bodyEl  = root.querySelector('.headline-content-container, .headline-content');

  const t = (titleEl && titleEl.innerText) ? titleEl.innerText.trim() : "";
  const b = (bodyEl  && bodyEl.innerText)  ? bodyEl.innerText.trim()  : "";

  // Make sure we're actually on the right block
  const okTitle = t.toLowerCase().includes("need to know market risk");
  const okBody  = b.length > 30;

  return okTitle && okBody;
}"""


def parse_need_to_know_risk(html: str) -> Optional[Dict[str, str]]:
    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")

    # If css selection removed wrapper, search globally.
    root = soup.select_one("#SummaryDiv") or soup

    title_el = root.select_one(".headline-title-nolink, .headline-title")
    body_el  = root.select_one(".headline-content-container, .headline-content")

    if not title_el or not body_el:
        return None

    title = title_el.get_text(" ", strip=True)
    body = body_el.get_text("\n", strip=True)

    # sanity check
    if "need to know market risk" not in title.lower():
        return None

    # clean obvious promo noise if it leaks in
    bad = ("this feed is delayed", "get 3 months", "go elite", "go real-time")
    body_lines = []
    for ln in (body or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if any(x in s.lower() for x in bad):
            continue
        body_lines.append(s)
    body = "\n".join(body_lines).strip()

    return {"title": title, "body": body} if body else None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", type=int, default=1)
    ap.add_argument("--debug_html", type=str, default="")
    args = ap.parse_args()

    browser_cfg = BrowserConfig(
        headless=bool(args.headless),
        text_mode=True,
        light_mode=True,
        viewport_width=1280,
        viewport_height=720,
        # important: no extensions / less weirdness
        extra_args=["--disable-extensions"],
    )

    cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=60000,
        wait_for=WAIT_RISK_READY,
        wait_for_timeout=30000,
        # DON’T click LAUNCH (it’s voice news + can mess up the page)
        js_code=[
            "await new Promise(r => setTimeout(r, 900));",
            click_text_js("Risk"),
            "await new Promise(r => setTimeout(r, 900));",
        ],
        word_count_threshold=0,
        delay_before_return_html=0.8,
        # do NOT use css_selector here; we want full html for reliable parsing/debug
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        res = await crawler.arun(url=FJ_HOME_URL, config=cfg)

        # Always prefer raw html for DOM parsing
        html = (getattr(res, "html", None) or "")
        out = parse_need_to_know_risk(html)

        if args.debug_html:
            Path(args.debug_html).write_text(html, encoding="utf-8")

        if not out:
            # print real debug info from CrawlResult
            print("FAILED to extract risk summary.")
            print("success:", getattr(res, "success", None))
            print("status_code:", getattr(res, "status_code", None))
            print("error_message:", getattr(res, "error_message", None))
            print("html_len:", len(html))
            if args.debug_html:
                print("Saved debug html to:", args.debug_html)
            return

        print("\n=== NEED TO KNOW MARKET RISK ===")
        print(out["title"])
        print(out["body"])
        print("================================\n")


if __name__ == "__main__":
    asyncio.run(main())
