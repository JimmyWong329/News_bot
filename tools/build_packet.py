#!/usr/bin/env python3
"""Build a merged market/NVDA packet JSON from existing outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")


def load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def load_jsonl(path: Path) -> List[Any]:
    if not path.exists():
        return []
    items: List[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def find_files(out_dir: Path, patterns: Iterable[str]) -> List[Path]:
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(out_dir.glob(pattern))
    return [p for p in matches if p.is_file()]


def dedupe_headlines(items: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in items:
        headline = item.get("headline") or item.get("title") or item.get("text") or ""
        key = headline.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def run_script(name: str, cmd: List[str]) -> Dict[str, Any]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "script": name,
        "cmd": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-2000:],
        "ok": result.returncode == 0,
    }


def load_market_headlines(out_dir: Path, date: str) -> List[Any]:
    path = out_dir / f"market_headlines_{date}.jsonl"
    return load_jsonl(path)


def load_nvda_headlines(out_dir: Path, date: str) -> List[Any]:
    path = out_dir / f"nvda_headlines_{date}.jsonl"
    return load_jsonl(path)


def load_calendar(out_dir: Path, date: str, calendar_json: Optional[str]) -> List[Any]:
    if calendar_json:
        data = load_json(Path(calendar_json))
        if data is None:
            return []
        if isinstance(data, list):
            return data
        return [data]

    data = load_json(out_dir / f"calendar_{date}.json")
    if data is None:
        return []
    if isinstance(data, list):
        return data
    return [data]


def load_forex_calendar(out_dir: Path, date: str) -> Optional[Any]:
    return load_json(out_dir / f"forex_calendar_{date}.json")


def load_market_summaries(out_dir: Path, date: str, asof: str) -> Dict[str, Any]:
    summaries = {
        "finviz_digest": load_json(out_dir / f"digest_{date}.json"),
        "ai_summary_1": load_json(out_dir / f"market_summary_{date}_{asof}.json"),
        "ai_summary_2": load_json(out_dir / f"market_summary2_{date}_{asof}.json"),
    }
    return summaries


def load_market_snapshot(out_dir: Path, date: str, asof: str) -> Optional[Any]:
    return load_json(out_dir / f"market_snapshot_{date}_{asof}.json")


def load_articles(out_dir: Path, date: str) -> Dict[str, Any]:
    article_files = find_files(out_dir, [f"articles_*_{date}.jsonl"])
    digests: List[Any] = []
    for path in article_files:
        digests.extend(load_jsonl(path))
    raw_sources = [str(p) for p in find_files(out_dir / "raw", [f"*_{date}_*.html"])]
    return {"digests": digests, "raw_sources": raw_sources}


def build_packet(out_dir: Path, date: str, asof: str, calendar_json: Optional[str]) -> Dict[str, Any]:
    missing_inputs: List[str] = []

    market_snapshot = load_market_snapshot(out_dir, date, asof)
    if market_snapshot is None:
        missing_inputs.append("market_snapshot")

    calendar_items = load_calendar(out_dir, date, calendar_json)
    if not calendar_items:
        missing_inputs.append("calendar_next_24h")

    market_summaries = load_market_summaries(out_dir, date, asof)
    if not market_summaries.get("finviz_digest"):
        missing_inputs.append("market_summaries.finviz_digest")
    if not market_summaries.get("ai_summary_1"):
        missing_inputs.append("market_summaries.ai_summary_1")
    if not market_summaries.get("ai_summary_2"):
        missing_inputs.append("market_summaries.ai_summary_2")

    market_headlines = load_market_headlines(out_dir, date)
    if not market_headlines:
        missing_inputs.append("market_headlines")

    nvda_headlines = load_nvda_headlines(out_dir, date)
    if not nvda_headlines:
        missing_inputs.append("nvda_headlines")

    nvda_snapshot = load_json(out_dir / f"nvda_snapshot_{date}_{asof}.json")
    if nvda_snapshot is None:
        missing_inputs.append("nvda_snapshot")

    market_articles = load_articles(out_dir, date)
    if not market_articles["digests"] and not market_articles["raw_sources"]:
        missing_inputs.append("market_articles")

    packet = {
        "meta": {
            "date": date,
            "asof_et": asof,
            "generated_at_et": datetime.now(NY_TZ).isoformat(),
            "repo": "news_crawler",
        },
        "market_snapshot": market_snapshot,
        "calendar_next_24h": calendar_items,
        "market_summaries": market_summaries,
        "market_headlines": market_headlines,
        "market_articles": market_articles,
        "nvda_snapshot": nvda_snapshot,
        "nvda_headlines": dedupe_headlines(nvda_headlines, limit=20),
        "missing_inputs": missing_inputs,
    }
    return packet


def main() -> int:
    ap = argparse.ArgumentParser(description="Build merged market packet")
    ap.add_argument("--date", required=True, help="Date YYYY-MM-DD")
    ap.add_argument("--asof", required=True, help="As-of time HH:MM")
    ap.add_argument("--out", required=True, help="Output packet path")
    ap.add_argument("--calendar_json", default=None, help="Optional calendar JSON path")
    ap.add_argument("--out_dir", default="out", help="Output directory (default: out)")
    ap.add_argument("--run_all", action="store_true", help="Run all scripts before building packet")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    script_runs: List[Dict[str, Any]] = []
    if args.run_all:
        script_runs.append(run_script(
            "market_snapshot",
            [
                sys.executable,
                "crawl_news.py",
                "--date",
                args.date,
                "--asof",
                args.asof,
                "--out_dir",
                args.out_dir,
                "--out_json",
                str(out_dir / f"market_snapshot_{args.date}_{args.asof}.json"),
            ],
        ))
        script_runs.append(run_script(
            "econ_calendar",
            [
                sys.executable,
                "econ_calender.py",
                "--today",
                "--out_dir",
                args.out_dir,
                "--out_json",
                str(out_dir / f"calendar_{args.date}.json"),
            ],
        ))
        script_runs.append(run_script(
            "forex_calendar",
            [
                sys.executable,
                "forex_calender.py",
                "--out_dir",
                args.out_dir,
                "--out_json",
                str(out_dir / f"forex_calendar_{args.date}.json"),
            ],
        ))
        script_runs.append(run_script(
            "finviz_digest",
            [
                sys.executable,
                "finviz_ai.py",
                "--out_dir",
                args.out_dir,
                "--out_json",
                str(out_dir / f"digest_{args.date}.json"),
                "--date",
                args.date,
            ],
        ))
        script_runs.append(run_script(
            "market_headlines",
            [
                sys.executable,
                "US_News.py",
                "--mode",
                "day",
                "--fj_date",
                args.date,
                "--out_dir",
                args.out_dir,
                "--out_jsonl",
                str(out_dir / f"market_headlines_{args.date}.jsonl"),
            ],
        ))
        script_runs.append(run_script(
            "fj_risk_summary",
            [
                sys.executable,
                "fj_ai.py",
                "--out_dir",
                args.out_dir,
                "--out_jsonl",
                str(out_dir / f"market_headlines_{args.date}.jsonl"),
                "--date",
                args.date,
            ],
        ))
        script_runs.append(run_script(
            "nvda_headlines",
            [
                sys.executable,
                "news_catalysts.py",
                "--out_dir",
                args.out_dir,
                "--out_jsonl",
                str(out_dir / f"nvda_headlines_{args.date}.jsonl"),
            ],
        ))
        script_runs.append(run_script(
            "market_summary",
            [
                sys.executable,
                "gemini_web.py",
                "--out_dir",
                args.out_dir,
                "--out_json",
                str(out_dir / f"market_summary_{args.date}_{args.asof}.json"),
                "--date",
                args.date,
            ],
        ))
        script_runs.append(run_script(
            "apnews",
            [
                sys.executable,
                "apnews.py",
                "--date",
                args.date,
                "--out_dir",
                args.out_dir,
                "--out_jsonl",
                str(out_dir / f"articles_apnews_{args.date}.jsonl"),
            ],
        ))
        script_runs.append(run_script(
            "cnbc",
            [
                sys.executable,
                "cnbc.py",
                "--date",
                args.date,
                "--out_dir",
                args.out_dir,
                "--out_jsonl",
                str(out_dir / f"articles_cnbc_{args.date}.jsonl"),
            ],
        ))
        script_runs.append(run_script(
            "fid",
            [
                sys.executable,
                "fid.py",
                "--out_dir",
                args.out_dir,
                "--out_jsonl",
                str(out_dir / f"articles_fid_{args.date}.jsonl"),
                "--date",
                args.date,
            ],
        ))
        script_runs.append(run_script(
            "invest",
            [
                sys.executable,
                "invest.py",
                "--out_dir",
                args.out_dir,
                "--out_jsonl",
                str(out_dir / f"articles_invest_{args.date}.jsonl"),
                "--date",
                args.date,
            ],
        ))

    packet = build_packet(out_dir, args.date, args.asof, args.calendar_json)
    if script_runs:
        packet["script_runs"] = script_runs
        for run in script_runs:
            if not run.get("ok"):
                packet.setdefault("missing_inputs", []).append(f"script:{run.get('script')}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")

    runs_path = out_dir / "packets" / "runs.jsonl"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with runs_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(packet) + "\n")

    if not out_path.exists():
        return 1
    try:
        json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 1
    if not packet.get("market_snapshot"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
