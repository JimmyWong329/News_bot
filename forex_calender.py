#!/usr/bin/env python3
"""
ff_calendar_json_usd.py

Pull ForexFactory calendar JSON (this week), apply your filter:
- Currency: USD only
- Impact: High, Medium, Non-Economic (and optionally Holiday if you want)
- Event Types: (JSON feed doesn't include type buckets like "Growth", so we don't filter those)

Then print a clean table to the terminal.

Why this is best:
- No UI clicking
- Much less likely to break
- Much faster

Source feed example:
https://nfs.faireconomy.media/ff_calendar_thisweek.json
"""

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from tabulate import tabulate

from tools.io_utils import append_jsonl, ensure_out_dir, now_et_iso, write_json


FF_WEEK_JSON = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Your screenshot: High + Medium + Non-Economic; USD only
DEFAULT_IMPACTS = {"High", "Medium", "Non-Economic"}  # add "Holiday" if you want


def to_local(dt_str: str, tz: str) -> str:
    # dt_str like "2026-01-29T08:30:00-05:00"
    dt = datetime.fromisoformat(dt_str)
    return dt.astimezone(ZoneInfo(tz)).strftime("%Y-%m-%d %I:%M%p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tz", default="America/New_York", help="Display timezone (IANA), e.g. America/New_York")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--include_holiday", action="store_true", help="Also include impact=Holiday")
    ap.add_argument("--out_dir", type=str, default="out", help="Output directory (default: out)")
    ap.add_argument("--out_json", type=str, default=None, help="Write calendar JSON to a path")
    ap.add_argument("--out_jsonl", type=str, default=None, help="Write calendar JSONL to a path")
    args = ap.parse_args()

    out_dir = ensure_out_dir(args.out_dir)
    impacts = set(DEFAULT_IMPACTS)
    if args.include_holiday:
        impacts.add("Holiday")

    r = requests.get(
        FF_WEEK_JSON,
        timeout=args.timeout,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    items = r.json()

    rows = []
    for e in items:
        if e.get("country") != "USD":
            continue
        if e.get("impact") not in impacts:
            continue

        rows.append(
            [
                to_local(e["date"], args.tz),
                e.get("country", ""),
                e.get("impact", ""),
                e.get("title", ""),
                e.get("actual", ""),     # sometimes present later in the day
                e.get("forecast", ""),
                e.get("previous", ""),
            ]
        )

    rows.sort(key=lambda x: x[0])
    print(tabulate(rows, headers=["Time", "CCY", "Impact", "Event", "Actual", "Forecast", "Previous"], tablefmt="github"))

    if args.out_json or args.out_jsonl:
        meta_date = datetime.now(ZoneInfo(args.tz)).strftime("%Y-%m-%d")
        events = [
            {
                "time_et": row[0],
                "ccy": row[1],
                "impact": row[2],
                "event": row[3],
                "actual": row[4],
                "forecast": row[5],
                "previous": row[6],
            }
            for row in rows
        ]
        payload = {
            "meta": {
                "source_script": "forex_calender.py",
                "generated_at_et": now_et_iso(),
                "date": meta_date,
                "asof_et": "",
                "run_id": f"{meta_date}_unknown_forex_calender.py",
            },
            "events": events,
        }
        if args.out_json:
            write_json(args.out_json, payload)
        if args.out_jsonl:
            for event in events:
                append_jsonl(args.out_jsonl, {"meta": payload["meta"], **event})


if __name__ == "__main__":
    main()
