import argparse
import os
from pathlib import Path
from datetime import datetime, time
from zoneinfo import ZoneInfo
from dotenv import load_dotenv, find_dotenv

from google import genai
from google.genai import types

from tools.io_utils import ensure_out_dir, now_et_iso, write_json

NY = ZoneInfo("America/New_York")


def market_mode_and_window(now_et: datetime) -> tuple[str, str]:
    wd = now_et.weekday()  # Mon=0 ... Sun=6
    t = now_et.time()

    # Cash hours (rough)
    cash_open = time(9, 30)
    cash_close = time(16, 0)

    # Futures reopen Sunday ~6pm ET
    fut_open_sun = time(18, 0)

    if wd in (5,):  # Saturday
        return ("CLOSED", "since Friday 4:00pm ET (last cash close) + weekend macro developments")

    if wd == 6:  # Sunday
        if t < fut_open_sun:
            return ("CLOSED", "since Friday 4:00pm ET (last cash close) + weekend macro developments")
        else:
            return ("FUTURES_LIVE", "since Sunday 6:00pm ET futures open")

    # Mon-Fri
    if cash_open <= t <= cash_close:
        return ("CASH_LIVE", "last 4 hours (intraday)")
    return ("OFF_HOURS", "since prior cash close + any major overnight futures moves")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="out", help="Output directory (default: out)")
    ap.add_argument("--out_json", type=str, default=None, help="Write summary JSON to a path")
    ap.add_argument("--date", type=str, default="", help="Override date YYYY-MM-DD for metadata")
    args = ap.parse_args()

    ensure_out_dir(args.out_dir)

    # --- Load .env safely ---
    dotenv_path = find_dotenv(usecwd=True) or (Path.cwd() / ".env")
    load_dotenv(dotenv_path=dotenv_path, override=True)

    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(f"GEMINI_API_KEY not found. Checked: {dotenv_path}")

    client = genai.Client(api_key=key)

    # --- Enable web search grounding ---
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=0.2,
    )

    # --- Calculate Context ---
    now_et = datetime.now(NY)
    mode, window = market_mode_and_window(now_et)
    as_of_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")

    print(f"MODE: {mode} | WINDOW: {window} | AS OF: {as_of_et}")

    # --- Dynamic Prompt ---
    prompt_template = """
You are a disciplined market-regime classifier.

AS OF: {as_of_et}
WINDOW: {window}
MODE: {mode}

Use Google Search. Use AT MOST 5 searches total.
Prefer authoritative market recaps (AP/Reuters/WSJ/FT) for index closes and major catalysts.

Task:
- Decide ONE: RISK_ON / RISK_OFF / MIXED / NONE

Strict Rules:
1) **CLOSED MODE GATE**: If MODE is CLOSED (weekend/holiday) and there are no live ES/NQ futures prints in the window:
   - **Label MUST be NONE**.
   - Instead of a single label, provide:
     Label: NONE
     Open Bias: [RISK_ON / RISK_OFF / MIXED] (based on weekend catalysts)
     Confidence: [Low / Medium / High]
   - Do NOT claim cross-asset confirmation without current prices. Use "Hypothetical impact" instead.

2) **LIVE MODE GATE**: If MODE is CASH_LIVE or FUTURES_LIVE:
   - Require explicit prints for Cross-asset check: ES/NQ % change, VIX level, 2Y/10Y yield change, DXY, Gold, WTI.
   - If you cannot find a specific live print, explicitly state "Unavailable".

3) **VERIFICATION**: Any claim involving military action, abductions, or major policy shocks (e.g., tariffs) must be supported by **2 independent sources** or labeled "UNVERIFIED".

4) **TIMESTAMPS**: Provide earliest headline time for the top catalyst.

Output Format:
1) Label section: [Label] (or Label + Open Bias + Confidence if CLOSED)
2) 1 tight paragraph describing the tape (or expected open)
3) Evidence: 6-10 bullets (macro/systemic > single-stock) with sources
4) Cross-asset check: 3-6 bullets (Actual prints if LIVE, Projected impact if CLOSED)
5) Watch next: 3-6 bullets (upcoming events)
"""

    final_prompt = prompt_template.format(as_of_et=as_of_et, window=window, mode=mode)

    # --- Generate ---
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=final_prompt,
        config=config,
    )

    print("-" * 40)
    print(resp.text)
    print("-" * 40)

    # --- Sanity check ---
    cand = resp.candidates[0]
    gm = getattr(cand, "grounding_metadata", None)
    if gm:
        queries = getattr(gm, "web_search_queries", None) or []
        print(f"\nSEARCH QUERIES USED ({len(queries)}):", queries)
        chunks = getattr(gm, "grounding_chunks", None) or []
        print("NUM SOURCES:", len(chunks))
    else:
        print("\nNo grounding_metadata returned (likely no search was used).")

    if args.out_json:
        meta_date = args.date.strip() or now_et.strftime("%Y-%m-%d")
        payload = {
            "meta": {
                "source_script": "gemini_web.py",
                "generated_at_et": now_et_iso(),
                "date": meta_date,
                "asof_et": now_et.strftime("%H:%M"),
                "run_id": f"{meta_date}_{now_et.strftime('%H:%M')}_gemini_web.py",
            },
            "prompt": final_prompt,
            "model": "gemini-2.5-flash",
            "summary": resp.text,
            "structured": {},
        }
        write_json(args.out_json, payload)


if __name__ == "__main__":
    main()
