#!/usr/bin/env python3
"""Build a merged market/NVDA packet JSON from existing outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")


def asof_to_tag(asof: str) -> str:
    return asof.replace(":", "")


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


def run_cmd(cmd: List[str], out_raw_dir: Path, tag: str, timeout_s: int = 180) -> Dict[str, Any]:
    out_raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_raw_dir / f"{tag}.txt"
    start = time.time()
    error = None
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        ok = cp.returncode == 0
    except subprocess.TimeoutExpired:
        cp = None
        ok = False
        error = f"TIMEOUT after {timeout_s}s"
    duration_s = round(time.time() - start, 3)

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"CMD: {' '.join(cmd)}\n")
        handle.write(f"OK: {ok}\n")
        handle.write(f"DURATION_S: {duration_s}\n")
        if error:
            handle.write(f"ERROR: {error}\n")
        handle.write("\n===== STDOUT =====\n")
        if cp and cp.stdout:
            handle.write(cp.stdout)
        handle.write("\n===== STDERR =====\n")
        if cp and cp.stderr:
            handle.write(cp.stderr)

    return {
        "cmd": cmd,
        "ok": ok,
        "returncode": None if cp is None else cp.returncode,
        "duration_s": duration_s,
        "log_path": str(log_path),
        "timeout_s": timeout_s,
        "error": error,
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


def load_market_summaries(out_dir: Path, date: str, asof_tag: str) -> Dict[str, Any]:
    summaries = {
        "finviz_digest": load_json(out_dir / f"digest_{date}.json"),
        "ai_summary_1": load_json(out_dir / f"market_summary_{date}_{asof_tag}.json"),
        "ai_summary_2": load_json(out_dir / f"market_summary2_{date}_{asof_tag}.json"),
    }
    return summaries


def load_market_snapshot(out_dir: Path, date: str, asof_tag: str) -> Optional[Any]:
    return load_json(out_dir / f"market_snapshot_{date}_{asof_tag}.json")


def load_articles(out_dir: Path, date: str) -> Dict[str, Any]:
    article_files = find_files(out_dir, [f"articles_*_{date}.jsonl"])
    digests: List[Any] = []
    for path in article_files:
        digests.extend(load_jsonl(path))
    raw_sources = [str(p) for p in find_files(out_dir / "raw", [f"*_{date}_*.html"])]
    return {"digests": digests, "raw_sources": raw_sources}


def build_packet(out_dir: Path, date: str, asof: str, calendar_json: Optional[str]) -> Dict[str, Any]:
    asof_tag = asof_to_tag(asof)
    missing_inputs: List[str] = []

    market_snapshot = load_market_snapshot(out_dir, date, asof_tag)
    if market_snapshot is None:
        missing_inputs.append("market_snapshot")

    calendar_items = load_calendar(out_dir, date, calendar_json)
    if not calendar_items:
        missing_inputs.append("calendar_next_24h")

    market_summaries = load_market_summaries(out_dir, date, asof_tag)
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

    nvda_snapshot = load_json(out_dir / f"nvda_snapshot_{date}_{asof_tag}.json")
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


def run_all(date: str, asof: str, out_dir: Path) -> Dict[str, Any]:
    asof_tag = asof_to_tag(asof)
    raw_dir = out_dir / "raw"
    script_runs: Dict[str, Any] = {}

    script_runs["crawl_news"] = run_cmd(
        [
            sys.executable,
            "crawl_news.py",
            "--date",
            date,
            "--asof",
            asof,
            "--out_json",
            str(out_dir / f"market_snapshot_{date}_{asof_tag}.json"),
        ],
        raw_dir,
        f"crawl_news_{date}_{asof_tag}",
        timeout_s=300,
    )

    return script_runs


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

    script_runs: Dict[str, Any] = {}
    if args.run_all:
        script_runs = run_all(args.date, args.asof, out_dir)

    packet = build_packet(out_dir, args.date, args.asof, args.calendar_json)
    if script_runs:
        packet["script_runs"] = script_runs
        for name, run in script_runs.items():
            if not run.get("ok"):
                packet.setdefault("missing_inputs", []).append(f"script:{name}")

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
