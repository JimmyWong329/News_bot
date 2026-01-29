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
OUT_DIR = Path("out")


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


def run_crawl_news(date: str, asof: str, out_json: Path) -> bool:
    cmd = [
        sys.executable,
        "crawl_news.py",
        "--date",
        date,
        "--asof",
        asof,
        "--out_json",
        str(out_json),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and out_json.exists()


def load_market_headlines(out_dir: Path) -> List[Any]:
    candidates = find_files(out_dir, ["*headline*.json", "*headline*.jsonl", "*regime_log*.jsonl"])
    items: List[Any] = []
    for path in candidates:
        if path.suffix == ".jsonl":
            items.extend(load_jsonl(path))
        else:
            data = load_json(path)
            if data is None:
                continue
            if isinstance(data, list):
                items.extend(data)
            else:
                items.append(data)
    return items


def load_nvda_headlines(out_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    log_path = out_dir / "news_log.jsonl"
    for entry in load_jsonl(log_path):
        if isinstance(entry, dict):
            items.append(entry)

    nvda_files = find_files(out_dir, ["*nvda*.json", "*nvda*.jsonl"])
    for path in nvda_files:
        if path == log_path:
            continue
        if path.suffix == ".jsonl":
            for entry in load_jsonl(path):
                if isinstance(entry, dict):
                    items.append(entry)
        else:
            data = load_json(path)
            if isinstance(data, list):
                items.extend([d for d in data if isinstance(d, dict)])
            elif isinstance(data, dict):
                items.append(data)
    return dedupe_headlines(items, limit=20)


def load_calendar(out_dir: Path, calendar_json: Optional[str]) -> List[Any]:
    if calendar_json:
        path = Path(calendar_json)
        data = load_json(path)
        if data is None:
            return []
        if isinstance(data, list):
            return data
        return [data]

    candidates = find_files(out_dir, ["*calendar*.json", "*calender*.json", "*calendar*.jsonl", "*calender*.jsonl"])
    for path in candidates:
        if path.suffix == ".jsonl":
            return load_jsonl(path)
        data = load_json(path)
        if data is None:
            continue
        if isinstance(data, list):
            return data
        return [data]
    return []


def load_nvda_snapshot(out_dir: Path) -> Optional[Any]:
    candidates = find_files(out_dir, ["*nvda*snapshot*.json", "*nvda*qqq*.json"]) + list(out_dir.glob("*nvda*dominance*.json"))
    for path in candidates:
        data = load_json(path)
        if data is not None:
            return data
    return None


def load_article_sources(out_dir: Path) -> List[str]:
    patterns = [
        "*apnews*.html",
        "*apnews*.json",
        "*cnbc*.html",
        "*cnbc*.json",
        "*fid*.html",
        "*fid*.json",
        "*invest*.html",
        "*invest*.json",
    ]
    sources = find_files(out_dir, patterns)
    return [str(p) for p in sources]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build merged market packet")
    ap.add_argument("--date", required=True, help="Date YYYY-MM-DD")
    ap.add_argument("--asof", required=True, help="As-of time HH:MM")
    ap.add_argument("--out", required=True, help="Output packet path")
    ap.add_argument("--calendar_json", default=None, help="Optional calendar JSON path")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tmp_market_snapshot = OUT_DIR / "tmp_market_snapshot.json"
    market_snapshot: Optional[Any] = None
    missing_inputs: List[str] = []

    if run_crawl_news(args.date, args.asof, tmp_market_snapshot):
        market_snapshot = load_json(tmp_market_snapshot)
    if market_snapshot is None:
        missing_inputs.append("market_snapshot")

    finviz_digest = load_json(OUT_DIR / "digest.json")
    if finviz_digest is None:
        missing_inputs.append("market_summaries.finviz_digest")

    calendar_items = load_calendar(OUT_DIR, args.calendar_json)
    if not calendar_items:
        missing_inputs.append("calendar_next_24h")

    market_headlines = load_market_headlines(OUT_DIR)
    if not market_headlines:
        missing_inputs.append("market_headlines")

    nvda_headlines = load_nvda_headlines(OUT_DIR)
    if not nvda_headlines:
        missing_inputs.append("nvda_headlines")

    nvda_snapshot = load_nvda_snapshot(OUT_DIR)
    if nvda_snapshot is None:
        missing_inputs.append("nvda_snapshot")

    article_sources = load_article_sources(OUT_DIR)
    market_articles = {
        "digests": [],
        "raw_sources": article_sources,
    }
    if not market_articles["digests"] and not market_articles["raw_sources"]:
        missing_inputs.append("market_articles")

    ai_summary_1 = None
    ai_summary_2 = None
    if ai_summary_1 is None:
        missing_inputs.append("market_summaries.ai_summary_1")
    if ai_summary_2 is None:
        missing_inputs.append("market_summaries.ai_summary_2")

    generated_at = datetime.now(NY_TZ).isoformat()
    packet = {
        "meta": {
            "date": args.date,
            "asof_et": args.asof,
            "generated_at_et": generated_at,
            "repo": "news_crawler",
        },
        "market_snapshot": market_snapshot,
        "calendar_next_24h": calendar_items,
        "market_summaries": {
            "finviz_digest": finviz_digest,
            "ai_summary_1": ai_summary_1,
            "ai_summary_2": ai_summary_2,
        },
        "market_headlines": market_headlines,
        "market_articles": market_articles,
        "nvda_snapshot": nvda_snapshot,
        "nvda_headlines": nvda_headlines,
        "missing_inputs": missing_inputs,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")

    runs_path = OUT_DIR / "packets" / "runs.jsonl"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with runs_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(packet) + "\n")

    if not out_path.exists():
        return 1
    try:
        json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
