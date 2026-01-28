import asyncio
import re
import json
from pathlib import Path
from playwright.async_api import async_playwright

FINVIZ_HOME = "https://finviz.com/"

async def main(headless: bool = False):
    Path("out").mkdir(exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
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
        await page.screenshot(path="out/finviz_after_click.png", full_page=True)
        print("Saved screenshot to out/finviz_after_click.png")

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
            print("Clicked, but couldn't find bullets. Check screenshot: out/finviz_after_click.png")
        else:
            print("\nDAILY DIGEST")
            print("-----------")
            if data.get("title"):
                print(data["title"])
            for b in data["bullets"]:
                print("-", b)

            Path("out/digest.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            print("\nSaved: out/digest.json")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main(headless=False))
