#!/usr/bin/env bash
set -euo pipefail

PACKET="${1:-out/packet.json}"

python tools/build_packet.py --run_all --date 2026-01-28 --asof 09:25 --out "${PACKET}"
python - <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit("packet.json not created")
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    raise SystemExit(f"invalid JSON: {exc}")
if not data.get("market_snapshot"):
    raise SystemExit("market_snapshot missing from packet")
print("OK: market_snapshot present")
PY "${PACKET}"
