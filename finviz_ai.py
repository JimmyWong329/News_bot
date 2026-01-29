import argparse
import asyncio
import re
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from playwright.async_api import async_playwright

from tools.io_utils import ensure_out_dir, now_et_iso, write_json

FINVIZ_HOME = "https://finviz.com/"
NY_TZ = ZoneInfo("America/New_York")

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="out", help="Output directory (default: out)")
    ap.add_argument("--out_json", type=str, default=None, help="Write digest JSON to a path")
    ap.add_argument("--date", type=str, default="", help="Override date YYYY-MM-DD for metadata")
    args = ap.parse_args()

    out_dir = ensure_out_dir(args.out_dir)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=bool(args.headless))
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        # 1) Load
        print(f"Loading {FINVIZ_HOME}...")
        await page.goto(FINVIZ_HOME, wait_until="domcontentloaded")
        await page.wait_for_selector(".js-why-stock-moving-root", timeout=30000)

        # 2) Kill common overlays if they exist
        for txt in ["Accept", "I Agree", "Agree", "OK", "Got it"]:
            loc = page.get_by_role("button", name=re.compile(txt, re.I))
            if await loc.count():
                try:
                    await loc.first.click(timeout=1500)
                    print(f"Dismissed overlay: {txt}")
                except:
                    pass

        # 3) Click the *actual* More button (your screenshot shows "More/" sometimes)
        print("Looking for 'More' button...")
        root = page.locator(".js-why-stock-moving-root")
        more_btn = root.locator("button", has_text=re.compile(r"^\s*More", re.I)).first
        
        # Fallback if button isn't found inside root, try scanning page (less precise but safer)
        if not await more_btn.count():
             more_btn = page.locator("button", has_text=re.compile(r"^\s*More", re.I)).first

        if await more_btn.count():
            await more_btn.click(timeout=5000)
            print("Clicked 'More' button.")
        else:
            print("Could not find 'More' button. Clicking root container as fallback.")
            await root.click(timeout=5000)

        # 4) Wait until the digest is visible (don’t trust class names)
        print("Waiting for 'Daily Digest' text...")
        try:
            await page.wait_for_selector("text=Daily Digest", timeout=15000)
        except Exception:
            print("Timed out waiting for 'Daily Digest' text. Taking screenshot.")

        # Optional: proof screenshot
        screenshot_path = out_dir / "finviz_after_click.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"Saved screenshot to {screenshot_path}")

        # 5) Extract from the LIVE DOM via evaluate (most reliable)
        data = await page.evaluate(r"""
() => {
  const textMatch = (node, re) => node && node.textContent && re.test(node.textContent);

  // Find the node containing "Daily Digest"
  const all = Array.from(document.querySelectorAll("*"));
  const dd = all.find(n => textMatch(n, /\bDaily Digest\b/i));
  if (!dd) return null;

  // Walk up to a container that has bullets
  let c = dd;
  for (let i = 0; i < 12; i++) {
    if (!c) break;
    const lis = c.querySelectorAll("ul li");
    if (lis && lis.length) {
      const h = c.querySelector("h1,h2,h3,h4");
      const title = h ? h.textContent.trim() : "";
      const bullets = Array.from(lis).map(li => li.textContent.trim()).filter(Boolean);
      return { title, bullets };
    }
    c = c.parentElement;
  }
  return { title: "", bullets: [] };
}
""")

        if not data or not data.get("bullets"):
            print(f"Clicked, but couldn't find bullets. Check screenshot: {screenshot_path}")
        else:
            print("\nDAILY DIGEST")
            print("-----------")
            if data.get("title"):
                print(data["title"])
            for b in data["bullets"]:
                print("-", b)

            legacy_path = out_dir / "digest.json"
            legacy_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"\nSaved: {legacy_path}")

            if args.out_json:
                meta_date = args.date.strip() or datetime.now(NY_TZ).strftime("%Y-%m-%d")
                payload = {
                    "meta": {
                        "source_script": "finviz_ai.py",
                        "generated_at_et": now_et_iso(),
                        "date": meta_date,
                        "asof_et": "",
                        "run_id": f"{meta_date}_unknown_finviz_ai.py",
                    },
                    "title": data.get("title") or "DAILY DIGEST",
                    "bullets": data.get("bullets") or [],
                    "raw_text": "\n".join(data.get("bullets") or []),
                }
                write_json(args.out_json, payload)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
